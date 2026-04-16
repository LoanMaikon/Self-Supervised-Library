from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import os

from src.utils import write_on_log, plot_fig, write_on_csv, save_json, is_main_process, concat_all_gather, \
    recreate_csv_log, get_last_epoch, step_schedulers_to_epoch, load_last_values
from src.schedulers import WarmupCosineSchedule, CosineWDSchedule
from .resnet import resnet50, resnet50w2, resnet50w4, resnet50w5, projection_head, prototypes
from src.datasets import datasets
from src.lars import LARS
from src.methods.swav.sinkhorn import sinkhorn

class SwAV():
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
        self._load_optimizer()
        self._load_schedulers()
        self._load_queue()

        self.train_loss = []

        self.lr_values = []
        self.wd_values = []

        if self.continue_training:
            self.last_epoch = get_last_epoch(self.output_folder)
            step_schedulers_to_epoch(self.last_epoch, len(self.train_dataloader), self.lr_scheduler, self.wd_scheduler)
            recreate_csv_log(self.output_folder, self.last_epoch)
            self.lr_values, self.wd_values, _, self.train_loss = load_last_values(self.output_folder, self.last_epoch)
            self.load_queue_from_last_epoch()

            write_on_log(f"Continuing training from epoch {self.last_epoch}...", self.output_folder)

    def train(self):
        write_on_log("Starting training...", self.output_folder)
        scaler = torch.amp.GradScaler()

        for epoch in range(1, self.optimization_num_epochs + 1):
            if self.continue_training and epoch <= self.last_epoch:
                continue

            write_on_log(f"Epoch {epoch}/{self.optimization_num_epochs}", self.output_folder)
            self.train_sampler.set_epoch(epoch)

            self.train_loss.append(0.0)
            num_samples = 0

            for iteration, (images, _) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                # Normalize prototypes
                with torch.no_grad():
                    w = self.prototypes.module.prototypes.weight.data.clone() if self.world_size > 1 else self.prototypes.prototypes.weight.data.clone()
                    w = nn.functional.normalize(w, dim=1, p=2)
                    self.prototypes.module.prototypes.weight.copy_(w) if self.world_size > 1 else self.prototypes.prototypes.weight.copy_(w)

                images = [img.to(self.device, non_blocking=True) for img in images]
                bs = images[0].size(0)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    feats = self.encoder(images)
                    emb = self.projection_head(feats)
                    emb = nn.functional.normalize(emb, dim=1, p=2)
                    out = self.prototypes(emb)

                    loss = 0
                    for i, crop_id in enumerate(range(self.data_global_views_num)):
                        with torch.no_grad():
                            out_i = out[bs * crop_id : bs * (crop_id + 1)]

                            if epoch >= self.optimization_queue_start_epoch:
                                proto_w = self.prototypes.module.prototypes.weight if self.world_size > 1 else self.prototypes.prototypes.weight
                                queue_logits = torch.mm(self.queue[i], proto_w.t())
                                out_i = torch.cat((queue_logits, out_i), dim=0)

                                self.queue[i, bs:] = self.queue[i, :-bs].clone()
                                self.queue[i, :bs] = emb[bs * crop_id : bs * (crop_id + 1)]

                            q = sinkhorn(out_i, self.optimization_sinkhorn_epsilon, self.optimization_sinkhorn_iterations, self.world_size)
                            q = q[-bs:] if epoch >= self.optimization_queue_start_epoch else q

                        subloss = 0
                        for v in np.delete(np.arange(self.data_global_views_num + self.data_local_views_num), crop_id):
                            x = out[bs * v : bs * (v + 1)] / self.optimization_temperature
                            subloss -= torch.mean(torch.sum(q * torch.log_softmax(x, dim=1), dim=1))
                        loss += subloss / (self.data_global_views_num + self.data_local_views_num - 1)
                    loss /= self.data_global_views_num

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                loss_value = loss.item()
                self.train_loss[-1] += loss_value * bs
                num_samples += bs

                self.lr_values.append(self.lr_scheduler.get_value())
                self.wd_values.append(self.wd_scheduler.get_value())

                write_on_csv(self.output_folder, epoch, iteration, loss_value, self.lr_values[-1], self.wd_values[-1])

                self.lr_scheduler.step()
                self.wd_scheduler.step()

            self.train_loss[-1] /= num_samples

            self.save_models(epoch)

            write_on_log(f"Loss: {self.train_loss[-1]}", self.output_folder)

            plot_fig(range(len(self.train_loss)), "Epoch", self.train_loss, "Loss", "loss", self.output_folder)
            plot_fig(range(len(self.lr_values)), "Iteration", self.lr_values, "Learning Rate", "learning_rate", self.output_folder)
            plot_fig(range(len(self.wd_values)), "Iteration", self.wd_values, "Weight Decay", "weight_decay", self.output_folder)

            save_json({"train_loss": self.train_loss}, self.output_folder, "training_info")
            save_json({"last_epoch": epoch}, self.output_folder, "last_epoch")

            write_on_log("", self.output_folder)

    def save_models(self, epoch):
        if not is_main_process():
            return
    
        os.makedirs(os.path.join(self.output_folder, "models"), exist_ok=True)
        os.makedirs(os.path.join(self.output_folder, "queue"), exist_ok=True)

        encoder_state_dict = self.encoder.module.state_dict() if self.world_size > 1 else self.encoder.state_dict()
        projection_head_state_dict = self.projection_head.module.state_dict() if self.world_size > 1 else self.projection_head.state_dict()
        prototypes_state_dict = self.prototypes.module.state_dict() if self.world_size > 1 else self.prototypes.state_dict()

        torch.save(encoder_state_dict, os.path.join(self.output_folder, "models", "encoder.pth"))
        torch.save(projection_head_state_dict, os.path.join(self.output_folder, "models", "projection_head.pth"))
        torch.save(prototypes_state_dict, os.path.join(self.output_folder, "models", "prototypes.pth"))
        torch.save(self.queue, os.path.join(self.output_folder, "queue", "queue.pth"))

        if self.meta_save_every > 0 and epoch % self.meta_save_every == 0:
            torch.save(encoder_state_dict, os.path.join(self.output_folder, "models", f"encoder_epoch_{epoch}.pth"))
            torch.save(projection_head_state_dict, os.path.join(self.output_folder, "models", f"projection_head_epoch_{epoch}.pth"))
            torch.save(prototypes_state_dict, os.path.join(self.output_folder, "models", f"prototypes_epoch_{epoch}.pth"))
            torch.save(self.queue, os.path.join(self.output_folder, "queue", f"queue_epoch_{epoch}.pth"))

    def _load_queue(self):
        self.queue = torch.zeros(self.data_global_views_num, self.optimization_queue_length, self.meta_projection_dim).to(self.device)

    def load_queue_from_last_epoch(self):
        if self.last_epoch < self.optimization_queue_start_epoch:
            return

        queue_path = os.path.join(self.output_folder, "queue", f"queue.pth")
        if os.path.exists(queue_path):
            self.queue = torch.load(queue_path, map_location=self.device)
        else:
            raise FileNotFoundError(f"Queue file not found at {queue_path} for loading queue from last epoch.")

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
    
    def _load_optimizer(self):
        match self.optimization_optimizer:
            case "lars":
                param_groups = [
                    {
                        'params': (p for n, p in self.encoder.named_parameters()
                                if ('bias' not in n) and (len(p.shape) != 1)),
                        'layer_adaptation': True,
                        'weight_decay': self.optimization_weight_decay[0],
                    }, 
                    {
                        'params': (p for n, p in self.projection_head.named_parameters()
                                if ('bias' not in n) and (len(p.shape) != 1)),
                        'layer_adaptation': True,
                        'weight_decay': self.optimization_weight_decay[0],
                    },
                    {
                        'params': (p for n, p in self.prototypes.named_parameters()
                                if ('bias' not in n) and (len(p.shape) != 1)),
                        'layer_adaptation': True,
                        'weight_decay': self.optimization_weight_decay[0],
                    },
                    {
                        'params': (p for n, p in self.encoder.named_parameters()
                                if ('bias' in n) or (len(p.shape) == 1)),
                        'WD_exclude': True,
                        'weight_decay': 0,
                    },
                    {
                        'params': (p for n, p in self.projection_head.named_parameters()
                                if ('bias' in n) or (len(p.shape) == 1)),
                        'WD_exclude': True,
                        'weight_decay': 0,
                    },
                    {
                        'params': (p for n, p in self.prototypes.named_parameters()
                                if ('bias' in n) or (len(p.shape) == 1)),
                        'WD_exclude': True,
                        'weight_decay': 0,
                    },
                ]
                
                self.base_optimizer = optim.SGD(param_groups, lr=self.optimization_lr[0], momentum=0.9)
                self.optimizer = LARS(
                    self.base_optimizer,
                    trust_coefficient=0.001
                )
            
            case _:
                raise ValueError(f"Unsupported optimizer: {self.optimization_optimizer}")

    def _load_dataloader(self):
        train_dataset = datasets(
            operation="train",
            datasets_folder_path=self.data_datasets_path,
            dataset_name=self.data_train_dataset,
            separate_val_subset=self.data_separate_val_subset_use,
            val_size=self.data_separate_val_subset_size,
            transforms=self.transforms,
            times=[self.data_global_views_num, self.data_local_views_num]
        )
        self.train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)

        self.train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.data_batch_size,
            num_workers=self.data_num_workers,
            prefetch_factor=self.data_prefetch_factor,
            pin_memory=self.data_pin_memory,
            drop_last=self.data_drop_last,
            sampler=self.train_sampler
        )

    def _load_transform(self):
        # Pseudo code of Apendix A from SimCLR paper
        def __get_color_distortion(strength=1.0):
            collor_jitter = v2.ColorJitter(0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength)
            rnd_color_jitter = v2.RandomApply([collor_jitter], p=0.8)
            rnd_gray = v2.RandomGrayscale(p=0.2)

            return v2.Compose([rnd_color_jitter, rnd_gray])
        
        self.global_transform = v2.Compose([
            v2.RandomResizedCrop(self.data_global_views_crop_size, scale=tuple(self.data_global_views_crop_scale), ratio=tuple(self.data_global_views_crop_ratio)),
            v2.RandomHorizontalFlip() if self.data_global_views_horizontal_flip else v2.Identity(),
            __get_color_distortion() if self.data_global_views_color_jitter else v2.Identity(),
            v2.GaussianBlur(kernel_size=int(0.1 * self.data_global_views_crop_size) * 2 + 1) if self.data_global_views_gaussian_blur else v2.Identity(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std)
        ])

        self.local_transform = v2.Compose([
            v2.RandomResizedCrop(self.data_local_views_crop_size, scale=tuple(self.data_local_views_crop_scale), ratio=tuple(self.data_local_views_crop_ratio)),
            v2.RandomHorizontalFlip() if self.data_local_views_horizontal_flip else v2.Identity(),
            __get_color_distortion() if self.data_local_views_color_jitter else v2.Identity(),
            v2.GaussianBlur(kernel_size=int(0.1 * self.data_local_views_crop_size) * 2 + 1) if self.data_local_views_gaussian_blur else v2.Identity(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std)
        ])

        self.transforms = [self.global_transform, self.local_transform]

    def _load_models(self):
        match self.meta_model_name:
            case "resnet50":
                self.encoder = resnet50(use_checkpoint=self.meta_checkpoint)

            case "resnet50w2":
                self.encoder = resnet50w2(use_checkpoint=self.meta_checkpoint)

            case "resnet50w4":
                self.encoder = resnet50w4(use_checkpoint=self.meta_checkpoint)

            case "resnet50w5":
                self.encoder = resnet50w5(use_checkpoint=self.meta_checkpoint)

            case _:
                raise ValueError(f"Unsupported model name: {self.meta_model_name}")
        
        self.projection_head = projection_head(self.encoder.get_output_dim(), self.encoder.get_output_dim(), self.meta_projection_dim)
        self.prototypes = prototypes(self.meta_projection_dim, self.optimization_num_prototypes)

        self.encoder.remove_unnecessary_modules()
        
        if self.meta_pretrained_weights is not None:
            if os.path.exists(self.meta_pretrained_weights):
                self.encoder.load_weights(
                    weight_path=self.meta_pretrained_weights,
                    device=self.device
                )
            else:
                raise FileNotFoundError(f"Pretrained weights file not found at {self.meta_pretrained_weights}.")
        
        if self.continue_training:
            if os.path.exists(os.path.join(self.output_folder, "models")):
                self.encoder.load_weights(
                    weight_path=os.path.join(self.output_folder, "models", "encoder.pth"),
                    device=self.device
                )
                self.projection_head.load_weights(
                    weight_path=os.path.join(self.output_folder, "models", "projection_head.pth"),
                    device=self.device
                )
                self.prototypes.load_weights(
                    weight_path=os.path.join(self.output_folder, "models", "prototypes.pth"),
                    device=self.device
                )
            else:
                raise FileNotFoundError("Checkpoint files not found for continuing training.")

        self.encoder = self.encoder.to(self.device)
        self.projection_head = self.projection_head.to(self.device)
        self.prototypes = self.prototypes.to(self.device)

        self.encoder.unfreeze()
        self.projection_head.unfreeze()
        self.prototypes.unfreeze()

        if self.world_size > 1:
            self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
            self.projection_head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.projection_head)
            self.prototypes = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.prototypes)

            self.encoder = DDP(self.encoder, device_ids=[self.rank], output_device=self.rank)
            self.projection_head = DDP(self.projection_head, device_ids=[self.rank], output_device=self.rank)
            self.prototypes = DDP(self.prototypes, device_ids=[self.rank], output_device=self.rank)

        self.encoder.train()
        self.projection_head.train()
        self.prototypes.train()

    def _load_config(self):
        self.data_datasets_path = str(self.config["data"]["datasets_path"])
        self.data_train_dataset = str(self.config["data"]["train_dataset"])
        self.data_batch_size = int(self.config["data"]["batch_size"])
        self.data_num_workers = int(self.config["data"]["num_workers"])
        self.data_prefetch_factor = int(self.config["data"]["prefetch_factor"])
        self.data_pin_memory = bool(self.config["data"]["pin_memory"])
        self.data_drop_last = bool(self.config["data"]["drop_last"])
        self.data_global_views_num = int(self.config["data"]["global_views"]["num"])
        self.data_global_views_crop_scale = list(map(float, self.config["data"]["global_views"]["crop_scale"]))
        self.data_global_views_crop_ratio = list(map(float, self.config["data"]["global_views"]["crop_ratio"]))
        self.data_global_views_color_jitter = bool(self.config["data"]["global_views"]["color_jitter"])
        self.data_global_views_gaussian_blur = bool(self.config["data"]["global_views"]["gaussian_blur"])
        self.data_global_views_horizontal_flip = bool(self.config["data"]["global_views"]["horizontal_flip"])
        self.data_global_views_crop_size = int(self.config["data"]["global_views"]["crop_size"])
        self.data_local_views_num = int(self.config["data"]["local_views"]["num"])
        self.data_local_views_crop_scale = list(map(float, self.config["data"]["local_views"]["crop_scale"]))
        self.data_local_views_crop_ratio = list(map(float, self.config["data"]["local_views"]["crop_ratio"]))
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
        self.meta_projection_dim = int(self.config["meta"]["projection_dim"])
        self.meta_pretrained_weights = self.config["meta"]["pretrained_weights"]
        self.meta_save_every = int(self.config["meta"]["save_every"])

        self.optimization_ipe_scale = float(self.config["optimization"]["ipe_scale"])
        self.optimization_lr = list(map(float, self.config["optimization"]["lr"]))
        self.optimization_weight_decay = list(map(float, self.config["optimization"]["weight_decay"]))
        self.optimization_num_epochs = int(self.config["optimization"]["num_epochs"])
        self.optimization_warmup_epochs = int(self.config["optimization"]["warmup_epochs"])
        self.optimization_optimizer = str(self.config["optimization"]["optimizer"])
        self.optimization_temperature = float(self.config["optimization"]["temperature"])
        self.optimization_sinkhorn_epsilon = float(self.config["optimization"]["sinkhorn_epsilon"])
        self.optimization_sinkhorn_iterations = int(self.config["optimization"]["sinkhorn_iterations"])
        self.optimization_num_prototypes = int(self.config["optimization"]["num_prototypes"])
        self.optimization_queue_length = int(self.config["optimization"]["queue_length"])
        self.optimization_queue_start_epoch = int(self.config["optimization"]["queue_start_epoch"])

        self.data_datasets_path += "/" if not self.data_datasets_path.endswith("/") else ""
