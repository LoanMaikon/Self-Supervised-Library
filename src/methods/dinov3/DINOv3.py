from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import torch.nn.functional as F
from functools import partial
import torch.optim as optim
import torch
import copy
import os

from src.utils import write_on_log, plot_fig, write_on_csv, save_json, is_main_process, \
    recreate_csv_log, get_last_epoch, load_last_values, make_param_groups
from src.schedulers import WarmupCosineSchedule, CosineWDSchedule, EMACosineSchedule, \
    LinearWarmupTemperatureSchedule
from .models import vit_base, vit_small, vit_large, vit_so400m, vit_giant2, vit_huge2, vit_7b, dino_head
from src.datasets import datasets
from .mask_collator import MaskingGenerator, collate_data_and_cast

class DINOv2():
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

        self.scaler = torch.amp.GradScaler()

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
            self.student_cls_temperature_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "student_cls_temperature_scheduler.pth"), map_location=self.device))
            self.student_patch_temperature_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "student_patch_temperature_scheduler.pth"), map_location=self.device))
            self.teacher_cls_temperature_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "teacher_cls_temperature_scheduler.pth"), map_location=self.device))
            self.teacher_patch_temperature_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "teacher_patch_temperature_scheduler.pth"), map_location=self.device))
            self.scaler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "scaler.pth"), map_location=self.device))
            recreate_csv_log(self.output_folder, self.last_epoch)
            self.lr_values, self.wd_values, self.ema_values, self.train_loss = load_last_values(self.output_folder, self.last_epoch)

            self.global_iteration = len(self.lr_values)

            write_on_log(f"Continuing training from epoch {self.last_epoch}...", self.output_folder)

    def train(self):
        pass

    def save_models(self, epoch):
        if not is_main_process():
            return
        
        os.makedirs(os.path.join(self.output_folder, "models"), exist_ok=True)
        
        encoder_state_dict = self.encoder.module.state_dict() if self.world_size > 1 else self.encoder.state_dict()
        projection_head_cls_state_dict = self.projection_head_cls.module.state_dict() if self.world_size > 1 else self.projection_head_cls.state_dict()
        projection_head_patch_state_dict = self.projection_head_patch.module.state_dict() if self.world_size > 1 else self.projection_head_patch.state_dict()
        target_encoder_state_dict = self.target_encoder.state_dict()
        target_projection_head_cls_state_dict = self.target_projection_head_cls.state_dict()
        target_projection_head_patch_state_dict = self.target_projection_head_patch.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        lr_scheduler_state_dict = self.lr_scheduler.state_dict()
        wd_scheduler_state_dict = self.wd_scheduler.state_dict()
        ema_scheduler_state_dict = self.ema_scheduler.state_dict()
        scaler_state_dict = self.scaler.state_dict()
        student_temp_scheduler_cls_state_dict = self.student_cls_temperature_scheduler.state_dict()
        teacher_temp_scheduler_cls_state_dict = self.teacher_cls_temperature_scheduler.state_dict()
        student_temp_scheduler_patch_state_dict = self.student_patch_temperature_scheduler.state_dict()
        teacher_temp_scheduler_patch_state_dict = self.teacher_patch_temperature_scheduler.state_dict()

        torch.save(encoder_state_dict, os.path.join(self.output_folder, "models", f"encoder.pth"))
        torch.save(projection_head_cls_state_dict, os.path.join(self.output_folder, "models", f"projection_head_cls.pth"))
        torch.save(projection_head_patch_state_dict, os.path.join(self.output_folder, "models", f"projection_head_patch.pth"))
        torch.save(target_encoder_state_dict, os.path.join(self.output_folder, "models", f"target_encoder.pth"))
        torch.save(target_projection_head_cls_state_dict, os.path.join(self.output_folder, "models", f"target_projection_head_cls.pth"))
        torch.save(target_projection_head_patch_state_dict, os.path.join(self.output_folder, "models", f"target_projection_head_patch.pth"))
        torch.save(optimizer_state_dict, os.path.join(self.output_folder, "models", f"optimizer.pth"))
        torch.save(lr_scheduler_state_dict, os.path.join(self.output_folder, "models", f"lr_scheduler.pth"))
        torch.save(wd_scheduler_state_dict, os.path.join(self.output_folder, "models", f"wd_scheduler.pth"))
        torch.save(ema_scheduler_state_dict, os.path.join(self.output_folder, "models", f"ema_scheduler.pth"))
        torch.save(scaler_state_dict, os.path.join(self.output_folder, "models", f"scaler.pth"))
        torch.save(student_temp_scheduler_cls_state_dict, os.path.join(self.output_folder, "models", f"student_cls_temperature_scheduler.pth"))
        torch.save(teacher_temp_scheduler_cls_state_dict, os.path.join(self.output_folder, "models", f"teacher_cls_temperature_scheduler.pth"))
        torch.save(student_temp_scheduler_patch_state_dict, os.path.join(self.output_folder, "models", f"student_patch_temperature_scheduler.pth"))
        torch.save(teacher_temp_scheduler_patch_state_dict, os.path.join(self.output_folder, "models", f"teacher_patch_temperature_scheduler.pth"))

        if self.meta_save_every > 0 and epoch % self.meta_save_every == 0:
            torch.save(encoder_state_dict, os.path.join(self.output_folder, "models", f"encoder_epoch_{epoch}.pth"))
            torch.save(projection_head_cls_state_dict, os.path.join(self.output_folder, "models", f"projection_head_cls_epoch_{epoch}.pth"))
            torch.save(projection_head_patch_state_dict, os.path.join(self.output_folder, "models", f"projection_head_patch_epoch_{epoch}.pth"))
            torch.save(target_encoder_state_dict, os.path.join(self.output_folder, "models", f"target_encoder_epoch_{epoch}.pth"))
            torch.save(target_projection_head_cls_state_dict, os.path.join(self.output_folder, "models", f"target_projection_head_cls_epoch_{epoch}.pth"))
            torch.save(target_projection_head_patch_state_dict, os.path.join(self.output_folder, "models", f"target_projection_head_patch_epoch_{epoch}.pth"))
            torch.save(optimizer_state_dict, os.path.join(self.output_folder, "models", f"optimizer_epoch_{epoch}.pth"))
            torch.save(lr_scheduler_state_dict, os.path.join(self.output_folder, "models", f"lr_scheduler_epoch_{epoch}.pth"))
            torch.save(wd_scheduler_state_dict, os.path.join(self.output_folder, "models", f"wd_scheduler_epoch_{epoch}.pth"))
            torch.save(ema_scheduler_state_dict, os.path.join(self.output_folder, "models", f"ema_scheduler_epoch_{epoch}.pth"))
            torch.save(scaler_state_dict, os.path.join(self.output_folder, "models", f"scaler_epoch_{epoch}.pth"))
            torch.save(student_temp_scheduler_cls_state_dict, os.path.join(self.output_folder, "models", f"student_cls_temperature_scheduler_epoch_{epoch}.pth"))
            torch.save(teacher_temp_scheduler_cls_state_dict, os.path.join(self.output_folder, "models", f"teacher_cls_temperature_scheduler_epoch_{epoch}.pth"))
            torch.save(student_temp_scheduler_patch_state_dict, os.path.join(self.output_folder, "models", f"student_patch_temperature_scheduler_epoch_{epoch}.pth"))
            torch.save(teacher_temp_scheduler_patch_state_dict, os.path.join(self.output_folder, "models", f"teacher_patch_temperature_scheduler_epoch_{epoch}.pth"))

    def _update_ema(self):
        with torch.no_grad():
            ema = self.ema_scheduler.get_value()
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
                param_k.data.mul_(ema).add_(param_q.data, alpha=1 - ema)
            for param_q, param_k in zip(self.projection_head_cls.parameters(), self.target_projection_head_cls.parameters()):
                param_k.data.mul_(ema).add_(param_q.data, alpha=1 - ema)
            for param_q, param_k in zip(self.projection_head_patch.parameters(), self.target_projection_head_patch.parameters()):
                param_k.data.mul_(ema).add_(param_q.data, alpha=1 - ema)

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
            warmup_steps=self.optimization_temperature_warmup * len(self.train_dataloader),
            T_max=self.optimization_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
        )

        self.student_patch_temperature_scheduler = LinearWarmupTemperatureSchedule(
            start_temp=self.optimization_temperature_student_patch[0],
            middle_temp=self.optimization_temperature_student_patch[1],
            final_temp=self.optimization_temperature_student_patch[2],
            warmup_steps=self.optimization_temperature_warmup * len(self.train_dataloader),
            T_max=self.optimization_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
        )

        self.teacher_cls_temperature_scheduler = LinearWarmupTemperatureSchedule(
            start_temp=self.optimization_temperature_teacher_cls[0],
            middle_temp=self.optimization_temperature_teacher_cls[1],
            final_temp=self.optimization_temperature_teacher_cls[2],
            warmup_steps=self.optimization_temperature_warmup * len(self.train_dataloader),
            T_max=self.optimization_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
        )

        self.teacher_patch_temperature_scheduler = LinearWarmupTemperatureSchedule(
            start_temp=self.optimization_temperature_teacher_patch[0],
            middle_temp=self.optimization_temperature_teacher_patch[1],
            final_temp=self.optimization_temperature_teacher_patch[2],
            warmup_steps=self.optimization_temperature_warmup * len(self.train_dataloader),
            T_max=self.optimization_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
        )

    def _load_optimizer(self):
        match self.optimization_optimizer:
            case "adamw":
                param_groups = []

                param_groups.extend(
                    make_param_groups(
                        model=self.encoder if self.world_size == 1 else self.encoder.module,
                        weight_decay=self.optimization_weight_decay[0],
                        decay_bias=self.optimization_decay_bias,
                        decay_norm=self.optimization_decay_norm,
                        lr=self.optimization_lr[0],
                    )
                )

                param_groups.extend(
                    make_param_groups(
                        model=self.projection_head_cls if self.world_size == 1 else self.projection_head_cls.module,
                        weight_decay=self.optimization_weight_decay[0],
                        decay_bias=self.optimization_decay_bias,
                        decay_norm=self.optimization_decay_norm,
                        lr=self.optimization_lr[0],
                    )
                )

                param_groups.extend(
                    make_param_groups(
                        model=self.projection_head_patch if self.world_size == 1 else self.projection_head_patch.module,
                        weight_decay=self.optimization_weight_decay[0],
                        decay_bias=self.optimization_decay_bias,
                        decay_norm=self.optimization_decay_norm,
                        lr=self.optimization_lr[0],
                    )
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
        masking_generator = MaskingGenerator(
            input_size=(self.data_global_views_crop_size // self.meta_patch_size, self.data_global_views_crop_size // self.meta_patch_size),
            max_num_patches=0.5 * self.data_global_views_crop_size // self.meta_patch_size * self.data_global_views_crop_size // self.meta_patch_size,
        )

        collate_fn = partial(
            collate_data_and_cast,
            mask_ratio_tuple=self.meta_mask_ratio,
            mask_probability=self.meta_mask_probability,
            n_tokens=(self.data_global_views_crop_size // self.meta_patch_size) ** 2,
            mask_generator=masking_generator,
            dtype=torch.float16,
            n_global_crops=self.data_global_views_num,
            n_local_crops=self.data_local_views_num,
        )

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
            collate_fn=collate_fn,
        )

    def _load_transform(self):
        # Pseudo code of Apendix A from SimCLR paper
        def __get_color_distortion(strength=1.0):
            collor_jitter = v2.ColorJitter(0.4 * strength, 0.4 * strength, 0.2 * strength, 0.1 * strength)
            rnd_color_jitter = v2.RandomApply([collor_jitter], p=0.8)
            rnd_gray = v2.RandomGrayscale(p=0.2)

            return v2.Compose([rnd_color_jitter, rnd_gray])
    
        self.global_transform_1 = v2.Compose([
            v2.RandomResizedCrop(self.data_global_views_crop_size, scale=tuple(self.data_global_views_crop_scale), ratio=tuple(self.data_global_views_crop_ratio), interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(p=0.5),
            __get_color_distortion(strength=1.0),
            v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

        self.global_transform_2 = v2.Compose([
            v2.RandomResizedCrop(self.data_global_views_crop_size, scale=tuple(self.data_global_views_crop_scale), ratio=tuple(self.data_global_views_crop_ratio), interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(p=0.5),
            __get_color_distortion(strength=0.4),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

        self.local_transform = v2.Compose([
            v2.RandomResizedCrop(self.data_local_views_crop_size, scale=tuple(self.data_local_views_crop_scale), ratio=tuple(self.data_local_views_crop_ratio), interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(p=0.5),
            __get_color_distortion(strength=0.4),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

    def _load_models(self):
        pass

    def _load_config(self):
        self.data_datasets_path = str(self.config['data']['datasets_path'])
        self.data_train_dataset = str(self.config['data']['train_dataset'])
        self.data_batch_size = int(self.config['data']['batch_size'])
        self.data_num_workers = int(self.config['data']['num_workers'])
        self.data_prefetch_factor = int(self.config['data']['prefetch_factor'])
        self.data_pin_memory = bool(self.config['data']['pin_memory'])
        self.data_drop_last = bool(self.config['data']['drop_last'])
        self.data_normalize_mean = self.config['data']['normalize']['mean']
        self.data_normalize_std = self.config['data']['normalize']['std']
        self.data_separate_val_subset_use = bool(self.config['data']['separate_val_subset']['use'])
        self.data_separate_val_subset_size = float(self.config['data']['separate_val_subset']['size'])
        self.data_global_views_num = int(self.config['data']['global_views']['num'])
        self.data_global_views_crop_scale = list(map(float, self.config['data']['global_views']['crop_scale']))
        self.data_global_views_crop_ratio = list(map(float, self.config['data']['global_views']['crop_ratio']))
        self.data_global_views_crop_size = int(self.config['data']['global_views']['crop_size'])
        self.data_local_views_num = int(self.config['data']['local_views']['num'])
        self.data_local_views_crop_scale = list(map(float, self.config['data']['local_views']['crop_scale']))
        self.data_local_views_crop_ratio = list(map(float, self.config['data']['local_views']['crop_ratio']))
        self.data_local_views_crop_size = int(self.config['data']['local_views']['crop_size'])

        self.meta_model_name = str(self.config['meta']['model_name'])
        self.meta_drop_path_rate = float(self.config['meta']['drop_path_rate'])
        self.meta_checkpoint = bool(self.config['meta']['checkpoint'])
        self.meta_pretrained_weights = self.config['meta']['pretrained_weights']
        self.meta_save_every = int(self.config['meta']['save_every'])
        self.meta_patch_size = int(self.config['meta']['patch_size'])
        self.meta_projection_head_dino_hidden_dim = int(self.config['meta']['projection_head_dino']['hidden_dim'])
        self.meta_projection_head_dino_bottleneck_dim = int(self.config['meta']['projection_head_dino']['bottleneck_dim'])
        self.meta_projection_head_dino_output_dim = int(self.config['meta']['projection_head_dino']['output_dim'])
        self.meta_projection_head_dino_use_bn = bool(self.config['meta']['projection_head_dino']['use_bn'])
        self.meta_projection_head_dino_n_layers = int(self.config['meta']['projection_head_dino']['n_layers'])
        self.meta_projection_head_ibot_hidden_dim = int(self.config['meta']['projection_head_ibot']['hidden_dim'])
        self.meta_projection_head_ibot_bottleneck_dim = int(self.config['meta']['projection_head_ibot']['bottleneck_dim'])
        self.meta_projection_head_ibot_output_dim = int(self.config['meta']['projection_head_ibot']['output_dim'])
        self.meta_projection_head_ibot_use_bn = bool(self.config['meta']['projection_head_ibot']['use_bn'])
        self.meta_projection_head_ibot_n_layers = int(self.config['meta']['projection_head_ibot']['n_layers'])
        self.meta_mask_ratio = list(map(float, self.config['meta']['mask_ratio']))
        self.meta_mask_probability = float(self.config['meta']['mask_probability'])

        self.optimization_ipe_scale = float(self.config['optimization']['ipe_scale'])
        self.optimization_ema = list(map(float, self.config['optimization']['ema']))
        self.optimization_lr = list(map(float, self.config['optimization']['lr']))
        self.optimization_weight_decay = list(map(float, self.config['optimization']['weight_decay']))
        self.optimization_epochs = int(self.config['optimization']['epochs'])
        self.optimization_warmup_epochs = int(self.config['optimization']['warmup_epochs'])
        self.optimization_optimizer = str(self.config['optimization']['optimizer'])
        self.optimization_temperature_student_patch = list(map(float, self.config['optimization']['temperature_student_patch']))
        self.optimization_temperature_student_cls = list(map(float, self.config['optimization']['temperature_student_cls']))
        self.optimization_temperature_teacher_cls = list(map(float, self.config['optimization']['temperature_teacher_cls']))
        self.optimization_temperature_teacher_patch = list(map(float, self.config['optimization']['temperature_teacher_patch']))
        self.optimization_temperature_warmup = int(self.config['optimization']['temperature_warmup'])
        self.optimization_dino_loss_weight = float(self.config['optimization']['dino_loss_weight'])
        self.optimization_ibot_loss_weight = float(self.config['optimization']['ibot_loss_weight'])
        self.optimization_koleo_loss_weight = float(self.config['optimization']['koleo_loss_weight'])
        self.optimization_freeze_last_layer_epochs = int(self.config['optimization']['freeze_last_layer_epochs'])
        self.optimization_sinkhorn_iterations = int(self.config['optimization']['sinkhorn_iterations'])
        self.optimization_num_register_tokens = int(self.config['optimization']['num_register_tokens'])
        self.optimization_decay_bias = bool(self.config['optimization']['decay_bias'])
        self.optimization_decay_norm = bool(self.config['optimization']['decay_norm'])

        self.data_datasets_path += "/" if not self.data_datasets_path.endswith("/") else ""
