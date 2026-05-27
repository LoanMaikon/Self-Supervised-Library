from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import torch.nn.functional as F
import torch.optim as optim
import torch
import copy
import os

from src.utils import write_on_log, plot_fig, write_on_csv, save_json, is_main_process, \
    recreate_csv_log, get_last_epoch, load_last_values
from src.schedulers import WarmupCosineSchedule, CosineWDSchedule, EMACosineSchedule, \
    LinearWarmupTemperatureSchedule
from .models import vit_base, vit_small, vit_tiny, vit_large, vit_giant2
from src.methods.ibot.mask_collator import MaskCollator
from src.methods.ibot.models import projection_head
from src.datasets import datasets

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

        # . . .

    def train(self):
        pass        

    def save_models(self, epoch):
        pass

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
        mask_collator = MaskCollator(
            patch_size=self.meta_patch_size,
            global_crop_size=self.data_global_views_crop_size,
            local_crop_size=self.data_local_views_crop_size,
            pred_ratio=self.meta_mask_ratio,
            pred_ratio_var=self.meta_mask_ratio_var,
            pred_aspect_ratio=self.meta_mask_aspect_ratio,
            num_global_crops=self.data_global_views_num,
            num_local_crops=self.data_local_views_num,
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
            collate_fn=mask_collator,
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
        match self.meta_model_name:
            case "vit_tiny":
                self.encoder = vit_tiny(drop_path_rate=self.meta_drop_path_rate, num_register_tokens=self.optimization_num_register_tokens, patch_size=self.meta_patch_size, use_checkpoint=self.meta_checkpoint)
                self.target_encoder = vit_tiny(num_register_tokens=self.optimization_num_register_tokens, patch_size=self.meta_patch_size)

            case "vit_small":
                self.encoder = vit_small(drop_path_rate=self.meta_drop_path_rate, num_register_tokens=self.optimization_num_register_tokens, patch_size=self.meta_patch_size, use_checkpoint=self.meta_checkpoint)
                self.target_encoder = vit_small(num_register_tokens=self.optimization_num_register_tokens, patch_size=self.meta_patch_size)

            case "vit_base":
                self.encoder = vit_base(drop_path_rate=self.meta_drop_path_rate, num_register_tokens=self.optimization_num_register_tokens, patch_size=self.meta_patch_size, use_checkpoint=self.meta_checkpoint)
                self.target_encoder = vit_base(num_register_tokens=self.optimization_num_register_tokens, patch_size=self.meta_patch_size)

            case "vit_large":
                self.encoder = vit_large(drop_path_rate=self.meta_drop_path_rate, num_register_tokens=self.optimization_num_register_tokens, patch_size=self.meta_patch_size, use_checkpoint=self.meta_checkpoint)
                self.target_encoder = vit_large(num_register_tokens=self.optimization_num_register_tokens, patch_size=self.meta_patch_size)

            case "vit_giant2":
                self.encoder = vit_giant2(drop_path_rate=self.meta_drop_path_rate, num_register_tokens=self.optimization_num_register_tokens, patch_size=self.meta_patch_size, use_checkpoint=self.meta_checkpoint)
                self.target_encoder = vit_giant2(num_register_tokens=self.optimization_num_register_tokens, patch_size=self.meta_patch_size)

        self.projection_head_cls = projection_head(
            in_dim=self.encoder.get_embed_dim(),
            out_dim=self.meta_projection_head_dino_output_dim,
            use_bn=self.meta_projection_head_dino_use_bn,
            norm_last_layer=self.meta_projection_head_dino_norm_last_layer,
            nlayers=self.meta_projection_head_dino_n_layers,
            hidden_dim=self.meta_projection_head_dino_hidden_dim,
            bottleneck_dim=self.meta_projection_head_dino_bottleneck_dim,
            use_checkpoint=self.meta_checkpoint,
        )

        self.projection_head_patch = projection_head(
            in_dim=self.encoder.get_embed_dim(),
            out_dim=self.meta_projection_head_ibot_output_dim,
            use_bn=self.meta_projection_head_ibot_use_bn,
            norm_last_layer=self.meta_projection_head_ibot_norm_last_layer,
            nlayers=self.meta_projection_head_ibot_n_layers,
            hidden_dim=self.meta_projection_head_ibot_hidden_dim,
            bottleneck_dim=self.meta_projection_head_ibot_bottleneck_dim,
            use_checkpoint=self.meta_checkpoint,
        )
        
        self.target_projection_head_cls = projection_head(
            in_dim=self.encoder.get_embed_dim(),
            out_dim=self.meta_projection_head_dino_output_dim,
            use_bn=self.meta_projection_head_dino_use_bn,
            norm_last_layer=self.meta_projection_head_dino_norm_last_layer,
            nlayers=self.meta_projection_head_dino_n_layers,
            hidden_dim=self.meta_projection_head_dino_hidden_dim,
            bottleneck_dim=self.meta_projection_head_dino_bottleneck_dim,
            use_checkpoint=self.meta_checkpoint,
        )

        self.target_projection_head_patch = projection_head(
            in_dim=self.encoder.get_embed_dim(),
            out_dim=self.meta_projection_head_ibot_output_dim,
            use_bn=self.meta_projection_head_ibot_use_bn,
            norm_last_layer=self.meta_projection_head_ibot_norm_last_layer,
            nlayers=self.meta_projection_head_ibot_n_layers,
            hidden_dim=self.meta_projection_head_ibot_hidden_dim,
            bottleneck_dim=self.meta_projection_head_ibot_bottleneck_dim,
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
        
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_projection_head_cls.load_state_dict(self.projection_head_cls.state_dict())
        self.target_projection_head_patch.load_state_dict(self.projection_head_patch.state_dict())

        if self.continue_training:
            if os.path.exists(os.path.join(self.output_folder, "models")):
                self.encoder.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"encoder.pth"), map_location=self.device))
                self.target_encoder.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"target_encoder.pth"), map_location=self.device))
                self.projection_head_cls.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"projection_head_cls.pth"), map_location=self.device))
                self.projection_head_patch.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"projection_head_patch.pth"), map_location=self.device))
                self.target_projection_head_cls.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"target_projection_head_cls.pth"), map_location=self.device))
                self.target_projection_head_patch.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"target_projection_head_patch.pth"), map_location=self.device))
            else:
                raise FileNotFoundError(f"Model checkpoint files not found in {os.path.join(self.output_folder, 'models')}.")

        self.encoder.to(self.device)
        self.target_encoder.to(self.device)
        self.projection_head_cls.to(self.device)
        self.projection_head_patch.to(self.device)
        self.target_projection_head_cls.to(self.device)
        self.target_projection_head_patch.to(self.device)

        self.encoder.unfreeze()
        self.target_encoder.unfreeze()
        self.projection_head_cls.unfreeze()
        self.projection_head_patch.unfreeze()
        self.target_projection_head_cls.unfreeze()
        self.target_projection_head_patch.unfreeze()

        if self.world_size > 1:
            self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
            self.projection_head_cls = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.projection_head_cls)
            self.projection_head_patch = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.projection_head_patch)

            self.encoder = DDP(self.encoder, device_ids=[self.rank], output_device=self.rank)
            self.projection_head_cls = DDP(self.projection_head_cls, device_ids=[self.rank], output_device=self.rank)
            self.projection_head_patch = DDP(self.projection_head_patch, device_ids=[self.rank], output_device=self.rank)
        
        self.encoder.train()
        self.target_encoder.train()
        self.projection_head_cls.train()
        self.projection_head_patch.train()
        self.target_projection_head_cls.train()
        self.target_projection_head_patch.train()

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
        self.meta_projection_head_dino_norm_last_layer = bool(self.config['meta']['projection_head_dino']['norm_last_layer'])
        self.meta_projection_head_dino_n_layers = int(self.config['meta']['projection_head_dino']['n_layers'])
        self.meta_projection_head_ibot_hidden_dim = int(self.config['meta']['projection_head_ibot']['hidden_dim'])
        self.meta_projection_head_ibot_bottleneck_dim = int(self.config['meta']['projection_head_ibot']['bottleneck_dim'])
        self.meta_projection_head_ibot_output_dim = int(self.config['meta']['projection_head_ibot']['output_dim'])
        self.meta_projection_head_ibot_use_bn = bool(self.config['meta']['projection_head_ibot']['use_bn'])
        self.meta_projection_head_ibot_norm_last_layer = bool(self.config['meta']['projection_head_ibot']['norm_last_layer'])
        self.meta_projection_head_ibot_n_layers = int(self.config['meta']['projection_head_ibot']['n_layers'])
        self.meta_mask_ratio = list(map(float, self.config['meta']['mask_ratio']))
        self.meta_mask_ratio_var = list(map(float, self.config['meta']['mask_ratio_var']))
        self.meta_mask_aspect_ratio = list(map(float, self.config['meta']['mask_aspect_ratio']))
        self.meta_mask_sample_probability = float(self.config['meta']['mask_sample_probability'])

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
        self.optimization_center_momentum_cls = float(self.config['optimization']['center_momentum_cls'])
        self.optimization_center_momentum_patch = float(self.config['optimization']['center_momentum_patch'])
        self.optimization_dino_loss_weight = float(self.config['optimization']['dino_loss_weight'])
        self.optimization_ibot_loss_weight = float(self.config['optimization']['ibot_loss_weight'])
        self.optimization_koleo_loss_weight = float(self.config['optimization']['koleo_loss_weight'])
        self.optimization_freeze_last_layer_epochs = int(self.config['optimization']['freeze_last_layer_epochs'])
        self.optimization_sinkhorn_epsilon = float(self.config['optimization']['sinkhorn_epsilon'])
        self.optimization_sinkhorn_iterations = int(self.config['optimization']['sinkhorn_iterations'])
        self.optimization_num_register_tokens = int(self.config['optimization']['num_register_tokens'])
