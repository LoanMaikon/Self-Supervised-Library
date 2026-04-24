from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import copy
import os

from src.utils import write_on_log, plot_fig, write_on_csv, save_json, is_main_process, \
    recreate_csv_log, get_last_epoch, load_last_values, repeat_interleave_batch
from .models import vit_base, vit_large, vit_small, vit_tiny, projection_head
from src.schedulers import WarmupCosineSchedule, CosineWDSchedule, EMACosineSchedule, \
    LinearWarmupTemperatureSchedule
from torch import distributed as dist
from src.datasets import datasets

class iBOT():
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
        self._load_centers()

        self.scaler = torch.amp.GradScaler()

        self.train_loss = []
        self.lr_values = []
        self.wd_values = []
        self.ema_values = []

        if self.continue_training:
            self.last_epoch = get_last_epoch(self.output_folder)
            self.optimizer.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"optimizer.pth"), map_location=self.device))
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"lr_scheduler.pth"), map_location=self.device))
            self.wd_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"wd_scheduler.pth"), map_location=self.device))
            self.ema_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"ema_scheduler.pth"), map_location=self.device))
            self.scaler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "scaler.pth"), map_location=self.device))
            self.student_cls_temperature_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"student_cls_temperature_scheduler.pth"), map_location=self.device))
            self.student_patch_temperature_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"student_patch_temperature_scheduler.pth"), map_location=self.device))
            self.teacher_cls_temperature_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"teacher_cls_temperature_scheduler.pth"), map_location=self.device))
            self.teacher_patch_temperature_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"teacher_patch_temperature_scheduler.pth"), map_location=self.device))
            self.center_cls = torch.load(os.path.join(self.output_folder, "models", f"center_cls.pth"), map_location=self.device)
            self.center_patch = torch.load(os.path.join(self.output_folder, "models", f"center_patch.pth"), map_location=self.device)
            recreate_csv_log(self.output_folder, self.last_epoch)
            self.lr_values, self.wd_values, self.ema_values, self.train_loss = load_last_values(self.output_folder, self.last_epoch)

            write_on_log(f"Continuing training from epoch {self.last_epoch}...", self.output_folder)

    def train(self):
        pass

    def save_models(self, epoch):
        if not is_main_process():
            return
        
        os.makedirs(os.path.join(self.output_folder, "models"), exist_ok=True)
        
        encoder_state_dict = self.encoder.module.state_dict() if self.world_size > 1 else self.encoder.state_dict()
        projection_head_state_dict = self.projection_head.module.state_dict() if self.world_size > 1 else self.projection_head.state_dict()
        target_encoder_state_dict = self.target_encoder.state_dict()
        target_projection_head_state_dict = self.target_projection_head.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        lr_scheduler_state_dict = self.lr_scheduler.state_dict()
        wd_scheduler_state_dict = self.wd_scheduler.state_dict()
        ema_scheduler_state_dict = self.ema_scheduler.state_dict()
        scaler_state_dict = self.scaler.state_dict()
        student_cls_temperature_scheduler_state_dict = self.student_cls_temperature_scheduler.state_dict()
        student_patch_temperature_scheduler_state_dict = self.student_patch_temperature_scheduler.state_dict()
        teacher_cls_temperature_scheduler_state_dict = self.teacher_cls_temperature_scheduler.state_dict()
        teacher_patch_temperature_scheduler_state_dict = self.teacher_patch_temperature_scheduler.state_dict()

        torch.save(encoder_state_dict, os.path.join(self.output_folder, "models", f"encoder.pth"))
        torch.save(projection_head_state_dict, os.path.join(self.output_folder, "models", f"projection_head.pth"))
        torch.save(target_encoder_state_dict, os.path.join(self.output_folder, "models", f"target_encoder.pth"))
        torch.save(target_projection_head_state_dict, os.path.join(self.output_folder, "models", f"target_projection_head.pth"))
        torch.save(optimizer_state_dict, os.path.join(self.output_folder, "models", f"optimizer.pth"))
        torch.save(lr_scheduler_state_dict, os.path.join(self.output_folder, "models", f"lr_scheduler.pth"))
        torch.save(wd_scheduler_state_dict, os.path.join(self.output_folder, "models", f"wd_scheduler.pth"))
        torch.save(ema_scheduler_state_dict, os.path.join(self.output_folder, "models", f"ema_scheduler.pth"))
        torch.save(scaler_state_dict, os.path.join(self.output_folder, "models", f"scaler.pth"))
        torch.save(student_cls_temperature_scheduler_state_dict, os.path.join(self.output_folder, "models", f"student_cls_temperature_scheduler.pth"))
        torch.save(student_patch_temperature_scheduler_state_dict, os.path.join(self.output_folder, "models", f"student_patch_temperature_scheduler.pth"))
        torch.save(teacher_cls_temperature_scheduler_state_dict, os.path.join(self.output_folder, "models", f"teacher_cls_temperature_scheduler.pth"))
        torch.save(teacher_patch_temperature_scheduler_state_dict, os.path.join(self.output_folder, "models", f"teacher_patch_temperature_scheduler.pth"))
        torch.save(self.center_cls, os.path.join(self.output_folder, "models", f"center_cls.pth"))
        torch.save(self.center_patch, os.path.join(self.output_folder, "models", f"center_patch.pth"))

        if self.meta_save_every > 0 and epoch % self.meta_save_every == 0:
            torch.save(encoder_state_dict, os.path.join(self.output_folder, "models", f"encoder_epoch_{epoch}.pth"))
            torch.save(projection_head_state_dict, os.path.join(self.output_folder, "models", f"projection_head_epoch_{epoch}.pth"))
            torch.save(target_encoder_state_dict, os.path.join(self.output_folder, "models", f"target_encoder_epoch_{epoch}.pth"))
            torch.save(target_projection_head_state_dict, os.path.join(self.output_folder, "models", f"target_projection_head_epoch_{epoch}.pth"))
            torch.save(optimizer_state_dict, os.path.join(self.output_folder, "models", f"optimizer_epoch_{epoch}.pth"))
            torch.save(lr_scheduler_state_dict, os.path.join(self.output_folder, "models", f"lr_scheduler_epoch_{epoch}.pth"))
            torch.save(wd_scheduler_state_dict, os.path.join(self.output_folder, "models", f"wd_scheduler_epoch_{epoch}.pth"))
            torch.save(ema_scheduler_state_dict, os.path.join(self.output_folder, "models", f"ema_scheduler_epoch_{epoch}.pth"))
            torch.save(scaler_state_dict, os.path.join(self.output_folder, "models", f"scaler_epoch_{epoch}.pth"))
            torch.save(student_cls_temperature_scheduler_state_dict, os.path.join(self.output_folder, "models", f"student_cls_temperature_scheduler_epoch_{epoch}.pth"))
            torch.save(student_patch_temperature_scheduler_state_dict, os.path.join(self.output_folder, "models", f"student_patch_temperature_scheduler_epoch_{epoch}.pth"))
            torch.save(teacher_cls_temperature_scheduler_state_dict, os.path.join(self.output_folder, "models", f"teacher_cls_temperature_scheduler_epoch_{epoch}.pth"))
            torch.save(teacher_patch_temperature_scheduler_state_dict, os.path.join(self.output_folder, "models", f"teacher_patch_temperature_scheduler_epoch_{epoch}.pth"))
            torch.save(self.center_cls, os.path.join(self.output_folder, "models", f"center_cls_epoch_{epoch}.pth"))
            torch.save(self.center_patch, os.path.join(self.output_folder, "models", f"center_patch_epoch_{epoch}.pth"))
    
    def update_target_network(self, ema):
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(ema).add_(param_q.data, alpha=1 - ema)

            for param_q, param_k in zip(self.projection_head.parameters(), self.target_projection_head.parameters()):
                param_k.data.mul_(ema).add_(param_q.data, alpha=1 - ema)

    def update_centers(self, teacher_cls, teacher_patch):
        with torch.no_grad():
            cls_center = torch.sum(teacher_cls, dim=0, keepdim=True)
            dist.all_reduce(cls_center)
            cls_center = cls_center / (len(teacher_cls) * dist.get_world_size())
            self.center_cls = self.center_cls * self.optimization_center_momentum_cls + cls_center * (1 - self.optimization_center_momentum_cls)

            patch_center = torch.sum(teacher_patch.mean(1), dim=0, keepdim=True)
            dist.all_reduce(patch_center)
            patch_center = patch_center / (len(teacher_patch) * dist.get_world_size())
            self.center_patch = self.center_patch * self.optimization_center_momentum_patch + patch_center * (1 - self.optimization_center_momentum_patch)

    def _load_centers(self):
        self.center_cls = torch.zeros(self.meta_projection_head_output_dim).to(self.device)
        self.center_patch = torch.zeros(self.meta_projection_head_output_dim).to(self.device)

    def _load_schedulers(self):
        self.lr_scheduler = WarmupCosineSchedule(
            optimizer=self.optimizer,
            warmup_steps=self.optimization_warmup_epochs * len(self.train_dataloader),
            start_lr=self.optimization_lr[0],
            middle_lr=self.optimization_lr[1],
            final_lr=self.optimization_lr[2],
            T_max=self.optimization_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
        )

        self.wd_scheduler = CosineWDSchedule(
            optimizer=self.optimizer,
            start_wd=self.optimization_weight_decay[0],
            final_wd=self.optimization_weight_decay[1],
            T_max=self.optimization_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
        )

        self.ema_scheduler = EMACosineSchedule(
            start_ema=self.optimization_ema[0],
            final_ema=self.optimization_ema[1],
            T_max=self.optimization_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
        )

        self.student_cls_temperature_scheduler = LinearWarmupTemperatureSchedule(
            start_temp=self.optimization_temperature_student_cls[0],
            middle_temp=self.optimization_temperature_student_cls[1],
            final_temp=self.optimization_temperature_student_cls[2],
            warmup_steps=self.optimization_tempereature_warmup * len(self.train_dataloader),
            T_max=self.optimization_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
        )

        self.student_patch_temperature_scheduler = LinearWarmupTemperatureSchedule(
            start_temp=self.optimization_temperature_student_patch[0],
            middle_temp=self.optimization_temperature_student_patch[1],
            final_temp=self.optimization_temperature_student_patch[2],
            warmup_steps=self.optimization_tempereature_warmup * len(self.train_dataloader),
            T_max=self.optimization_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
        )

        self.teacher_cls_temperature_scheduler = LinearWarmupTemperatureSchedule(
            start_temp=self.optimization_temperature_teacher_cls[0],
            middle_temp=self.optimization_temperature_teacher_cls[1],
            final_temp=self.optimization_temperature_teacher_cls[2],
            warmup_steps=self.optimization_tempereature_warmup * len(self.train_dataloader),
            T_max=self.optimization_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
        )

        self.teacher_patch_temperature_scheduler = LinearWarmupTemperatureSchedule(
            start_temp=self.optimization_temperature_teacher_patch[0],
            middle_temp=self.optimization_temperature_teacher_patch[1],
            final_temp=self.optimization_temperature_teacher_patch[2],
            warmup_steps=self.optimization_tempereature_warmup * len(self.train_dataloader),
            T_max=self.optimization_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
        )

    def _load_optimizer(self):
        match self.optimization_optimizer:
            case "adamw":
                target_modules = [self.encoder, self.projection_head] if self.world_size == 1 else [self.encoder.module, self.projection_head.module]

                decay_params = []
                no_decay_params = []

                for module in target_modules:
                    for name, p in module.named_parameters():
                        if not p.requires_grad:
                            continue

                        if p.ndim > 1 and "bias" not in name:
                            decay_params.append(p)
                        else:
                            no_decay_params.append(p)

                param_groups = [
                    {
                        "params": decay_params,
                        "weight_decay": self.optimization_weight_decay[0],
                        "WD_exclude": False
                    },
                    {
                        "params": no_decay_params,
                        "weight_decay": 0.0,
                        "WD_exclude": True
                    },
                ]

                self.optimizer = optim.AdamW(
                    param_groups,
                    lr=self.optimization_lr[0],
                )

            case _:
                raise ValueError(f"Unsupported optimizer: {self.optimization_optimizer}")

    def _load_dataloader(self):
        self.train_dataset = datasets(
            operation="train",
            datasets_folder_path=self.data_datasets_path,
            dataset_name=self.data_train_dataset,
            separate_val_subset=self.data_separate_val_subset_use,
            val_size=self.data_separate_val_subset_size,
            transforms=[self.global_transform_1, self.global_transform_2, self.local_transform],
            times=[1, 1, self.data_local_views_num]
        )

        self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)

        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.data_batch_size,
            sampler=self.train_sampler,
            num_workers=self.data_num_workers,
            prefetch_factor=self.data_prefetch_factor,
            pin_memory=self.data_pin_memory,
            drop_last=self.data_drop_last,
        )

    def _load_transform(self):
        # Pseudo code of Apendix A from SimCLR paper
        def __get_color_distortion(strength=1.0):
            collor_jitter = v2.ColorJitter(0.4 * strength, 0.4 * strength, 0.4 * strength, 0.1 * strength)
            rnd_color_jitter = v2.RandomApply([collor_jitter], p=0.8)
            rnd_gray = v2.RandomGrayscale(p=0.2)

            return v2.Compose([rnd_color_jitter, rnd_gray])
    
        self.global_transform1 = v2.Compose([
            v2.RandomResizedCrop(self.data_global_views_crop_size, scale=tuple(self.data_global_views_crop_scale), ratio=tuple(self.data_global_views_crop_ratio)),
            v2.RandomHorizontalFlip(p=0.5),
            __get_color_distortion(strength=1.0),
            v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

        self.global_transform2 = v2.Compose([
            v2.RandomResizedCrop(self.data_global_views_crop_size, scale=tuple(self.data_global_views_crop_scale), ratio=tuple(self.data_global_views_crop_ratio)),
            v2.RandomHorizontalFlip(p=0.5),
            __get_color_distortion(strength=0.4),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
            v2.RandomApply([v2.Solarize(threshold=128)], p=0.2),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

        self.local_transform = v2.Compose([
            v2.RandomResizedCrop(self.data_local_views_crop_size, scale=tuple(self.data_local_views_crop_scale), ratio=tuple(self.data_local_views_crop_ratio)),
            v2.RandomHorizontalFlip(p=0.5),
            __get_color_distortion(strength=0.4),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

    def _load_models(self):
        match self.meta_model_name:
            case "vit_base":
                self.encoder = vit_base(patch_size=self.meta_patch_size, use_checkpoint=self.meta_checkpoint, drop_path_rate=self.meta_drop_path_rate)

            case "vit_large":
                self.encoder = vit_large(patch_size=self.meta_patch_size, use_checkpoint=self.meta_checkpoint, drop_path_rate=self.meta_drop_path_rate)

            case "vit_small":
                self.encoder = vit_small(patch_size=self.meta_patch_size, use_checkpoint=self.meta_checkpoint, drop_path_rate=self.meta_drop_path_rate)

            case "vit_tiny":
                self.encoder = vit_tiny(patch_size=self.meta_patch_size, use_checkpoint=self.meta_checkpoint, drop_path_rate=self.meta_drop_path_rate)

            case _:
                raise ValueError(f"Model {self.meta_model_name} not recognized.")
        
        self.projection_head = projection_head(
            in_dim=self.encoder.embed_dim,
            hidden_dim=self.meta_projection_head_hidden_dim,
            bottleneck_dim=self.meta_projection_head_bottleneck_dim,
            out_dim=self.meta_projection_head_output_dim,
            use_bn=self.meta_projection_head_use_bn,
            norm_last_layer=self.meta_projection_head_norm_last_layer,
            n_layers=self.meta_projection_head_n_layers,
            use_checkpoint=self.meta_checkpoint,
        )

        if self.meta_pretrained_weights is not None:
            if os.path.exists(self.meta_pretrained_weights):
                self.encoder.load_weights(
                    weight_path=self.meta_pretrained_weights,
                    device=self.device
                )
            else:
                raise FileNotFoundError(f"Pretrained weights file not found at {self.meta_pretrained_weights}.")
        
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.use_checkpoint = False # Target model should not use checkpointing
        self.target_projection_head = copy.deepcopy(self.projection_head)
        self.target_projection_head.use_checkpoint = False # Target projection head should not use checkpointing

        if self.continue_training:
            if os.path.exists(os.path.join(self.output_folder, "models")):
                self.encoder.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"encoder.pth"), map_location=self.device))
                self.projection_head.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"projection_head.pth"), map_location=self.device))
                self.target_encoder.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"target_encoder.pth"), map_location=self.device))
                self.target_projection_head.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"target_projection_head.pth"), map_location=self.device))
            else:
                raise FileNotFoundError(f"Model checkpoint files not found in {os.path.join(self.output_folder, 'models')}.")
        
        self.encoder.unfreeze()
        self.projection_head.unfreeze()
        self.target_encoder.freeze()
        self.target_projection_head.freeze()

        self.encoder.to(self.device)
        self.projection_head.to(self.device)
        self.target_encoder.to(self.device)
        self.target_projection_head.to(self.device)

        if self.world_size > 1:
            self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
            self.projection_head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.projection_head)
            self.encoder = DDP(self.encoder, device_ids=[self.rank], output_device=self.rank)
            self.projection_head = DDP(self.projection_head, device_ids=[self.rank], output_device=self.rank)
        
        self.encoder.train()
        self.projection_head.train()
        self.target_encoder.train()
        self.target_projection_head.train()

    def _load_config(self):
        self.data_datasets_path = str(self.config["data"]["datasets_path"])
        self.data_train_dataset = str(self.config["data"]["train_dataset"])
        self.data_batch_size = int(self.config["data"]["batch_size"])
        self.data_num_workers = int(self.config["data"]["num_workers"])
        self.data_prefetch_factor = int(self.config["data"]["prefetch_factor"])
        self.data_pin_memory = bool(self.config["data"]["pin_memory"])
        self.data_drop_last = bool(self.config["data"]["drop_last"])
        self.data_normalize_mean = list(map(float, self.config["data"]["normalize"]["mean"]))
        self.data_normalize_std = list(map(float, self.config["data"]["normalize"]["std"]))
        self.data_separate_val_subset_use = bool(self.config["data"]["separate_val_subset"]["use"])
        self.data_separate_val_subset_size = float(self.config["data"]["separate_val_subset"]["size"])
        self.data_global_views_num = int(self.config["data"]["global_views"]["num"])
        self.data_global_views_crop_scale = list(map(float, self.config["data"]["global_views"]["crop_scale"]))
        self.data_global_views_crop_ratio = list(map(float, self.config["data"]["global_views"]["crop_ratio"]))
        self.data_global_views_crop_size = int(self.config["data"]["global_views"]["crop_size"])
        self.data_local_views_num = int(self.config["data"]["local_views"]["num"])
        self.data_local_views_crop_scale = list(map(float, self.config["data"]["local_views"]["crop_scale"]))
        self.data_local_views_crop_ratio = list(map(float, self.config["data"]["local_views"]["crop_ratio"]))
        self.data_local_views_crop_size = int(self.config["data"]["local_views"]["crop_size"])

        self.meta_model_name = str(self.config["meta"]["model_name"])
        self.meta_drop_path_rate = float(self.config["meta"]["drop_path_rate"])
        self.meta_checkpoint = bool(self.config["meta"]["checkpoint"])
        self.meta_pretrained_weights = str(self.config["meta"]["pretrained_weights"])
        self.meta_save_every = int(self.config["meta"]["save_every"])
        self.meta_patch_size = int(self.config["meta"]["patch_size"])
        self.meta_projection_head_hidden_dim = int(self.config["meta"]["projection_head"]["hidden_dim"])
        self.meta_projection_head_bottleneck_dim = int(self.config["meta"]["projection_head"]["bottleneck_dim"])
        self.meta_projection_head_output_dim = int(self.config["meta"]["projection_head"]["output_dim"])
        self.meta_projection_head_use_bn = bool(self.config["meta"]["projection_head"]["use_bn"])
        self.meta_projection_head_norm_last_layer = bool(self.config["meta"]["projection_head"]["norm_last_layer"])
        self.meta_projection_head_n_layers = int(self.config["meta"]["projection_head"]["n_layers"])
        self.meta_mask_ratio = list(map(float, self.config["meta"]["mask_ratio"]))
        self.meta_mask_ratio_var = list(map(float, self.config["meta"]["mask_ratio_var"]))

        self.optimization_ipe_scale = float(self.config["optimization"]["ipe_scale"])
        self.optimization_ema = list(map(float, self.config["optimization"]["ema"]))
        self.optimization_lr = list(map(float, self.config["optimization"]["lr"]))
        self.optimization_weight_decay = list(map(float, self.config["optimization"]["weight_decay"]))
        self.optimization_epochs = int(self.config["optimization"]["epochs"])
        self.optimization_warmup_epochs = int(self.config["optimization"]["warmup_epochs"])
        self.optimization_optimizer = str(self.config["optimization"]["optimizer"])
        self.optimization_temperature_student_patch = list(map(float, self.config["optimization"]["temperature_student_patch"]))
        self.optimization_temperature_student_cls = list(map(float, self.config["optimization"]["temperature_student_cls"]))
        self.optimization_temperature_teacher_patch = list(map(float, self.config["optimization"]["temperature_teacher_patch"]))
        self.optimization_temperature_teacher_cls = list(map(float, self.config["optimization"]["temperature_teacher_cls"]))
        self.optimization_tempereature_warmup = int(self.config["optimization"]["tempereature_warmup"])
        self.optimization_center_momentum_cls = float(self.config["optimization"]["center_momentum_cls"])
        self.optimization_center_momentum_patch = float(self.config["optimization"]["center_momentum_patch"])

        self.data_datasets_path += "/" if not self.data_datasets_path.endswith("/") else ""
