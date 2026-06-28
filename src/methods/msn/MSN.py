from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import os

from src.utils import write_on_log, plot_fig, write_on_csv, save_json, is_main_process, \
    recreate_csv_log, get_last_epoch, load_last_values, AllReduceSum, make_param_groups
from src.schedulers import WarmupCosineSchedule, CosineWDSchedule, EMACosineSchedule, \
    LinearWarmupTemperatureSchedule
from src.datasets import datasets
from .models import deit_tiny, deit_small, deit_small_p8, deit_small_p7, vitc_4gf, deit_small_convstem, \
    deit_base_p8, deit_base_p7, deit_base_p4, deit_base, deit_large_p7, deit_large_p8, deit_large, deit_huge, deit_huge_p8, deit_huge_p7, deit_huge_p10
from .msn_loss import msn_loss

class MSN():
    def __init__(self,
                 opened_config,
                 output_folder,
                 device,
                 rank,
                 world_size,
                 continue_training,
                ):

        self.config = opened_config
        self.output_folder = output_folder
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.continue_training = continue_training

        self._load_config()
        self._load_models()
        self._load_transform()
        self._load_dataloader()
        self._load_prototypes()
        self._load_optimizer()
        self._load_schedulers()

        self.scaler = torch.amp.GradScaler()
        self.msn_loss = msn_loss(
            num_views=self.data_global_views_num + self.data_local_views_num - 1,
            me_max=True,
            return_preds=True,
            softmax_temperature=self.optimization_softmax_temperature,
        )

        self.train_loss = []
        self.lr_values = []
        self.wd_values = []
        self.ema_values = []

        if self.continue_training:
            self.last_epoch = get_last_epoch(self.output_folder)
            self.optimizer.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "optimizer.pth"), map_location=self.device))
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "lr_scheduler.pth"), map_location=self.device))
            self.wd_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "wd_scheduler.pth"), map_location=self.device))
            self.ema_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "ema_scheduler.pth"), map_location=self.device))
            self.target_temp_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "target_temp_scheduler.pth"), map_location=self.device))
            self.scaler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "scaler.pth"), map_location=self.device))
            loaded = torch.load(os.path.join(self.output_folder, "models", "prototypes.pth"), map_location=self.device)
            with torch.no_grad():
                self.prototypes.data.copy_(loaded.to(self.device))
            recreate_csv_log(self.output_folder, self.last_epoch)
            self.lr_values, self.wd_values, self.ema_values, self.train_loss = load_last_values(self.output_folder, self.last_epoch)

            write_on_log(f"Continuing training from epoch {self.last_epoch}...", self.output_folder)

    def train(self):
        write_on_log("Starting training...", self.output_folder)

        self.proto_labels = self.one_hot(torch.tensor([i for i in range(self.optimization_num_prototypes)]), self.optimization_num_prototypes)

        for epoch in range(1, self.optimization_num_epochs + 1):
            if self.continue_training and epoch <= self.last_epoch:
                continue

            write_on_log(f"Epoch {epoch}/{self.optimization_num_epochs}", self.output_folder)
            self.train_sampler.set_epoch(epoch)

            self.train_loss.append(0.0)
            num_samples = 0

            self.prototypes.requires_grad = True if epoch > self.optimization_freeze_prototypes_epochs else False

            for iteration, (images, _) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                images = [img.to(self.device, non_blocking=True) for img in images]

                with torch.amp.autocast(device_type=self.device.type):
                    h_student, z_student = self.encoder(
                        images[1:],
                        return_before_head=True,
                        patch_drop=self.data_global_views_mask_ratio,
                    )
                    with torch.no_grad():
                        h_teacher, _ = self.target_encoder(images[0], return_before_head=True)

                h_student = h_student.float()
                z_student = z_student.float()
                h_teacher = h_teacher.float()
                prototypes = self.prototypes.float()
                proto_labels = self.proto_labels.float()

                with torch.amp.autocast(device_type=self.device.type, enabled=False):
                    ploss, me_max, ent, _ = self.msn_loss.compute_loss(
                        T=float(self.target_temp_scheduler.get_value()),
                        use_sinkhorn=self.optimization_use_sinkhorn,
                        use_entropy=self.optimization_use_entropy,
                        anchor_views=z_student,
                        target_views=h_teacher.detach(),
                        proto_labels=proto_labels,
                        prototypes=prototypes,
                    )

                    if self.optimization_use_entropy:
                        loss = (
                            ploss
                            + self.optimization_memax_regularization_weight * me_max
                            + self.optimization_entropy_regularization_weight * ent
                        )
                    else:
                        loss = ploss + self.optimization_memax_regularization_weight * me_max

                    loss_value = loss.item()
                    self.train_loss[-1] += loss_value * images[0].size(0)
                    num_samples += images[0].size(0)

                self.scaler.scale(loss).backward()
                with torch.no_grad():
                    if self.prototypes.grad is not None:
                        self.prototypes.grad.data = AllReduceSum.apply(self.prototypes.grad.data)
                        self.prototypes.grad.data /= self.world_size

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.encoder.module.parameters() if self.world_size > 1 else self.encoder.parameters(),
                    max_norm=3.0,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.lr_values.append(self.lr_scheduler.get_value())
                self.wd_values.append(self.wd_scheduler.get_value())
                self.ema_values.append(self.ema_scheduler.get_value())
                write_on_csv(self.output_folder, epoch, iteration, loss_value, self.lr_values[-1], self.wd_values[-1], self.ema_values[-1])

                self.update_target_network(self.ema_scheduler.get_value())

                self.lr_scheduler.step()
                self.wd_scheduler.step()
                self.ema_scheduler.step()
                self.target_temp_scheduler.step()
            
            self.train_loss[-1] /= num_samples

            self.save_models(epoch)

            write_on_log(f"Loss: {self.train_loss[-1]}", self.output_folder)

            plot_fig(range(len(self.train_loss)), "Epoch", self.train_loss, "Loss", f"loss", self.output_folder)
            plot_fig(range(len(self.lr_values)), "Iteration", self.lr_values, "Learning Rate", f"learning_rate", self.output_folder)
            plot_fig(range(len(self.wd_values)), "Iteration", self.wd_values, "Weight Decay", f"weight_decay", self.output_folder)
            plot_fig(range(len(self.ema_values)), "Iteration", self.ema_values, "EMA", f"ema", self.output_folder)
            
            save_json({"train_loss": self.train_loss}, self.output_folder, "training_info")
            save_json({"last_epoch": epoch}, self.output_folder, "last_epoch")

            write_on_log("", self.output_folder)     

    def save_models(self, epoch):
        if not is_main_process():
            return
        
        os.makedirs(os.path.join(self.output_folder, "models"), exist_ok=True)
        
        encoder_state_dict = self.encoder.module.state_dict() if self.world_size > 1 else self.encoder.state_dict()
        target_encoder_state_dict = self.target_encoder.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        lr_scheduler_state_dict = self.lr_scheduler.state_dict()
        wd_scheduler_state_dict = self.wd_scheduler.state_dict()
        ema_scheduler_state_dict = self.ema_scheduler.state_dict()
        scaler_state_dict = self.scaler.state_dict()
        target_temp_scheduler_state_dict = self.target_temp_scheduler.state_dict()

        torch.save(encoder_state_dict, os.path.join(self.output_folder, "models", f"encoder.pth"))
        torch.save(target_encoder_state_dict, os.path.join(self.output_folder, "models", f"target_encoder.pth"))
        torch.save(optimizer_state_dict, os.path.join(self.output_folder, "models", f"optimizer.pth"))
        torch.save(lr_scheduler_state_dict, os.path.join(self.output_folder, "models", f"lr_scheduler.pth"))
        torch.save(wd_scheduler_state_dict, os.path.join(self.output_folder, "models", f"wd_scheduler.pth"))
        torch.save(ema_scheduler_state_dict, os.path.join(self.output_folder, "models", f"ema_scheduler.pth"))
        torch.save(scaler_state_dict, os.path.join(self.output_folder, "models", f"scaler.pth"))
        torch.save(target_temp_scheduler_state_dict, os.path.join(self.output_folder, "models", f"target_temp_scheduler.pth"))
        torch.save(self.prototypes.detach().cpu(), os.path.join(self.output_folder, "models", "prototypes.pth"))
        
        if self.meta_save_every > 0 and epoch % self.meta_save_every == 0:
            torch.save(encoder_state_dict, os.path.join(self.output_folder, "models", f"encoder_{epoch}.pth"))
            torch.save(target_encoder_state_dict, os.path.join(self.output_folder, "models", f"target_encoder_{epoch}.pth"))
            torch.save(optimizer_state_dict, os.path.join(self.output_folder, "models", f"optimizer_{epoch}.pth"))
            torch.save(lr_scheduler_state_dict, os.path.join(self.output_folder, "models", f"lr_scheduler_{epoch}.pth"))
            torch.save(wd_scheduler_state_dict, os.path.join(self.output_folder, "models", f"wd_scheduler_{epoch}.pth"))
            torch.save(ema_scheduler_state_dict, os.path.join(self.output_folder, "models", f"ema_scheduler_{epoch}.pth"))
            torch.save(scaler_state_dict, os.path.join(self.output_folder, "models", f"scaler_{epoch}.pth"))
            torch.save(target_temp_scheduler_state_dict, os.path.join(self.output_folder, "models", f"target_temp_scheduler_{epoch}.pth"))
            torch.save(self.prototypes.detach().cpu(), os.path.join(self.output_folder, "models", f"prototypes_{epoch}.pth"))

    def update_target_network(self, ema):
        encoder_module = self.encoder.module if self.world_size > 1 else self.encoder
        target_module = self.target_encoder

        with torch.no_grad():
            for param_q, param_k in zip(encoder_module.parameters(), target_module.parameters()):
                param_k.data.mul_(ema).add_(param_q.detach().data, alpha=1 - ema)

    def one_hot(self, targets, num_classes, smoothing=0.0):
        off_value = smoothing / num_classes
        on_value = 1. - smoothing + off_value
        targets = targets.long().view(-1, 1).to(self.device)
        return torch.full((len(targets), num_classes), off_value, device=self.device).scatter_(1, targets, on_value)

    def _load_prototypes(self):
        prototypes = torch.empty(
            (self.optimization_num_prototypes, self.meta_projection_head_output_dim),
            device=self.device
        )
        _sqrt_k = (1. / self.meta_projection_head_output_dim) ** 0.5
        torch.nn.init.uniform_(prototypes, -_sqrt_k, _sqrt_k)

        self.prototypes = torch.nn.Parameter(prototypes) 

    def _load_schedulers(self):
        self.lr_scheduler = WarmupCosineSchedule(
            optimizer=self.optimizer,
            warmup_steps=self.optimization_warmup_epochs * len(self.train_dataloader),
            start_lr=self.optimization_lr[0],
            middle_lr=self.optimization_lr[1],
            final_lr=self.optimization_lr[2],
            T_max=self.optimization_num_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
        )

        self.wd_scheduler = CosineWDSchedule(
            optimizer=self.optimizer,
            start_wd=self.optimization_weight_decay[0],
            final_wd=self.optimization_weight_decay[1],
            T_max=self.optimization_num_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
        )

        self.ema_scheduler = EMACosineSchedule(
            start_ema=self.optimization_ema[0],
            final_ema=self.optimization_ema[1],
            T_max=self.optimization_num_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
        )

        self.target_temp_scheduler = LinearWarmupTemperatureSchedule(
            start_temp=self.optimization_temperature_target[0],
            middle_temp=self.optimization_temperature_target[1],
            final_temp=self.optimization_temperature_target[2],
            warmup_steps=self.optimization_tempereature_warmup * len(self.train_dataloader),
            T_max=self.optimization_num_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
        )
    
    def _load_optimizer(self):
        match self.optimization_optimizer:
            case "adamw":
                encoder = self.encoder if self.world_size == 1 else self.encoder.module

                param_groups = []

                param_groups.extend(
                    make_param_groups(
                        model=encoder,
                        weight_decay=self.optimization_weight_decay[0],
                        decay_bias=self.optimization_decay_bias,
                        decay_norm=self.optimization_decay_norm,
                        lr=self.optimization_lr[0],
                    )
                )

                param_groups.extend(
                    [
                        {
                            "params": [self.prototypes],
                            "lr": self.optimization_lr[1],
                            "initial_lr": self.optimization_lr[1],
                            "weight_decay": self.optimization_weight_decay[0],
                            "is_bias": False,
                            "is_norm": False,
                            "decay_bias": self.optimization_decay_bias,
                            "decay_norm": self.optimization_decay_norm,
                            "adapt_bias": False,
                            "adapt_norm": False,
                            "fix_lr": False,
                        }
                    ]
                )

                self.optimizer = optim.AdamW(
                    param_groups,
                    lr=self.optimization_lr[0],
                )

            case _:
                raise ValueError(
                    f"Unsupported optimizer: {self.optimization_optimizer}"
                )

    def _load_dataloader(self):
        self.train_dataset = datasets(
            operation="train",
            datasets_folder_path=self.data_datasets_path,
            dataset_name=self.data_train_dataset,
            separate_val_subset=self.data_separate_val_subset_use,
            val_size=self.data_separate_val_subset_size,
            transforms=[self.transform_global, self.transform_local],
            times=[self.data_global_views_num, self.data_local_views_num]
        )

        self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.data_batch_size,
            sampler=self.train_sampler,
            num_workers=self.data_num_workers,
            prefetch_factor=self.data_prefetch_factor,
            pin_memory=self.data_pin_memory,
            drop_last=self.data_drop_last
        )

    def _load_transform(self):
        # Pseudo code of Apendix A from SimCLR paper
        def __get_color_distortion(strength=0.5):
            collor_jitter = v2.ColorJitter(0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength)
            rnd_color_jitter = v2.RandomApply([collor_jitter], p=0.8)
            rnd_gray = v2.RandomGrayscale(p=0.2)

            return v2.Compose([rnd_color_jitter, rnd_gray])

        self.transform_global = v2.Compose([
            v2.RandomResizedCrop(self.data_global_views_crop_size, scale=tuple(self.data_global_views_crop_scale), ratio=tuple(self.data_global_views_crop_ratio)),
            v2.RandomHorizontalFlip(p=0.5) if self.data_global_views_horizontal_flip else v2.RandomHorizontalFlip(p=0.0),
            __get_color_distortion(strength=0.5) if self.data_global_views_color_jitter else v2.Identity(),
            v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)) if self.data_global_views_gaussian_blur else v2.Identity(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

        self.transform_local = v2.Compose([
            v2.RandomResizedCrop(self.data_local_views_crop_size, scale=tuple(self.data_local_views_crop_scale), ratio=tuple(self.data_local_views_crop_ratio)),
            v2.RandomHorizontalFlip(p=0.5) if self.data_local_views_horizontal_flip else v2.RandomHorizontalFlip(p=0.0),
            __get_color_distortion(strength=0.5) if self.data_local_views_color_jitter else v2.Identity(),
            v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)) if self.data_local_views_gaussian_blur else v2.Identity(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

    def _load_models(self):
        match self.meta_model_name:
            case "vit_tiny":
                self.encoder = deit_tiny(patch_size=self.meta_patch_size, drop_path_rate=self.optimization_drop_path_rate, use_checkpoint=self.meta_checkpoint)
                self.target_encoder = deit_tiny(patch_size=self.meta_patch_size)
            
            case "vit_small":
                self.encoder = deit_small(patch_size=self.meta_patch_size, drop_path_rate=self.optimization_drop_path_rate, use_checkpoint=self.meta_checkpoint)
                self.target_encoder = deit_small(patch_size=self.meta_patch_size)

            case "vit_base":
                self.encoder = deit_base(patch_size=self.meta_patch_size, drop_path_rate=self.optimization_drop_path_rate, use_checkpoint=self.meta_checkpoint)
                self.target_encoder = deit_base(patch_size=self.meta_patch_size)
            
            case "vit_large":
                self.encoder = deit_large(patch_size=self.meta_patch_size, drop_path_rate=self.optimization_drop_path_rate, use_checkpoint=self.meta_checkpoint)
                self.target_encoder = deit_large(patch_size=self.meta_patch_size)

            case "vit_huge":
                self.encoder = deit_huge(patch_size=self.meta_patch_size, drop_path_rate=self.optimization_drop_path_rate, use_checkpoint=self.meta_checkpoint)
                self.target_encoder = deit_huge(patch_size=self.meta_patch_size)

        if self.meta_pretrained_weights is not None:
            if os.path.exists(self.meta_pretrained_weights):
                self.encoder.load_weights(
                    weight_path=self.meta_pretrained_weights,
                    device=self.device
                )
            else:
                raise FileNotFoundError(f"Pretrained weights file not found at {self.meta_pretrained_weights}.")
        
        self.target_encoder.load_state_dict(self.encoder.state_dict())

        if self.continue_training:
            if os.path.exists(os.path.join(self.output_folder, "models")):
                self.encoder.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"encoder.pth"), map_location=self.device))
                self.target_encoder.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"target_encoder.pth"), map_location=self.device))
            else:
                raise FileNotFoundError(f"Model checkpoint files not found in {os.path.join(self.output_folder, 'models')}.")

        self.encoder.to(self.device)
        self.target_encoder.to(self.device)

        self.encoder.unfreeze()
        self.target_encoder.freeze()

        if self.world_size > 1:
            self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)

            self.encoder = DDP(self.encoder, device_ids=[self.rank], output_device=self.rank)
        
        self.encoder.train()
        self.target_encoder.train()

    def _load_config(self):
        self.data_datasets_path =  str(self.config["data"]["datasets_path"])
        self.data_train_dataset = str(self.config["data"]["train_dataset"])
        self.data_batch_size = int(self.config["data"]["batch_size"])
        self.data_num_workers = int(self.config["data"]["num_workers"])
        self.data_prefetch_factor = int(self.config["data"]["prefetch_factor"])
        self.data_pin_memory = bool(self.config["data"]["pin_memory"])
        self.data_drop_last = bool(self.config["data"]["drop_last"])
        self.data_global_views_num = int(self.config["data"]["global_views"]["num"])
        self.data_global_views_crop_scale = list(self.config["data"]["global_views"]["crop_scale"])
        self.data_global_views_crop_ratio = list(self.config["data"]["global_views"]["crop_ratio"])
        self.data_global_views_color_jitter = bool(self.config["data"]["global_views"]["color_jitter"])
        self.data_global_views_gaussian_blur = bool(self.config["data"]["global_views"]["gaussian_blur"])
        self.data_global_views_horizontal_flip = bool(self.config["data"]["global_views"]["horizontal_flip"])
        self.data_global_views_crop_size = int(self.config["data"]["global_views"]["crop_size"])
        self.data_global_views_mask_ratio = float(self.config["data"]["global_views"]["mask_ratio"])
        self.data_local_views_num = int(self.config["data"]["local_views"]["num"])
        self.data_local_views_crop_scale = list(self.config["data"]["local_views"]["crop_scale"])
        self.data_local_views_crop_ratio = list(self.config["data"]["local_views"]["crop_ratio"])
        self.data_local_views_color_jitter = bool(self.config["data"]["local_views"]["color_jitter"])
        self.data_local_views_gaussian_blur = bool(self.config["data"]["local_views"]["gaussian_blur"])
        self.data_local_views_horizontal_flip = bool(self.config["data"]["local_views"]["horizontal_flip"])
        self.data_local_views_crop_size = int(self.config["data"]["local_views"]["crop_size"])
        self.data_normalize_mean = list(map(float, self.config["data"]["normalize"]["mean"]))
        self.data_normalize_std = list(map(float, self.config["data"]["normalize"]["std"]))
        self.data_separate_val_subset_use = bool(self.config["data"]["separate_val_subset"]["use"])
        self.data_separate_val_subset_size = float(self.config["data"]["separate_val_subset"]["size"])

        self.meta_model_name = str(self.config["meta"]["model_name"])
        self.meta_checkpoint = bool(self.config["meta"]["checkpoint"])
        self.meta_pretrained_weights = self.config["meta"]["pretrained_weights"]
        self.meta_save_every = int(self.config["meta"]["save_every"])
        self.meta_patch_size = int(self.config["meta"]["patch_size"])
        self.meta_projection_head_hidden_dim = int(self.config["meta"]["projection_head"]["hidden_dim"])
        self.meta_projection_head_output_dim = int(self.config["meta"]["projection_head"]["output_dim"])

        self.optimization_ipe_scale = float(self.config["optimization"]["ipe_scale"])
        self.optimization_lr = list(map(float, self.config["optimization"]["lr"]))
        self.optimization_weight_decay = list(map(float, self.config["optimization"]["weight_decay"]))
        self.optimization_ema = list(map(float, self.config["optimization"]["ema"]))
        self.optimization_temperature_target = list(map(float, self.config["optimization"]["temperature_target"]))
        self.optimization_tempereature_warmup = int(self.config["optimization"]["tempereature_warmup"])
        self.optimization_num_epochs = int(self.config["optimization"]["num_epochs"])
        self.optimization_warmup_epochs = int(self.config["optimization"]["warmup_epochs"])
        self.optimization_optimizer = str(self.config["optimization"]["optimizer"])
        self.optimization_memax_regularization_weight = float(self.config["optimization"]["memax_regularization_weight"])
        self.optimization_num_prototypes = int(self.config["optimization"]["num_prototypes"])
        self.optimization_freeze_prototypes_epochs = int(self.config["optimization"]["freeze_prototypes_epochs"])
        self.optimization_use_sinkhorn = bool(self.config["optimization"]["use_sinkhorn"])
        self.optimization_drop_path_rate = float(self.config["optimization"]["drop_path_rate"])
        self.optimization_use_entropy = bool(self.config["optimization"]["use_entropy"])
        self.optimization_entropy_regularization_weight = float(self.config["optimization"]["entropy_regularization_weight"])
        self.optimization_softmax_temperature = float(self.config["optimization"]["softmax_temperature"])
        self.optimization_decay_bias = bool(self.config["optimization"]["decay_bias"])
        self.optimization_decay_norm = bool(self.config["optimization"]["decay_norm"])
    
        self.data_datasets_path += "/" if not self.data_datasets_path.endswith("/") else ""