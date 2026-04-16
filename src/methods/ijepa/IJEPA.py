from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import copy
import os

from .models import repeat_interleave_batch, vit_predictor, vit_tiny, vit_small, vit_base, vit_large, vit_huge, vit_giant, apply_masks, repeat_interleave_batch
from src.utils import write_on_log, plot_fig, write_on_csv, save_json, is_main_process, \
    recreate_csv_log, get_last_epoch, load_last_values
from src.schedulers import WarmupCosineSchedule, CosineWDSchedule, EMACosineSchedule
from .mask_collator import MaskCollator
from src.datasets import datasets

class IJEPA():
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
        self._load_criterion()
        self._load_optimizer()
        self._load_schedulers()

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
            recreate_csv_log(self.output_folder, self.last_epoch)
            self.lr_values, self.wd_values, self.ema_values, self.train_loss = load_last_values(self.output_folder, self.last_epoch)

            write_on_log(f"Continuing training from epoch {self.last_epoch}...", self.output_folder)

    def train(self):
        write_on_log("Starting training...", self.output_folder)

        for epoch in range(1, self.optimization_epochs + 1):
            if self.continue_training and epoch <= self.last_epoch:
                continue

            write_on_log(f"Epoch {epoch}/{self.optimization_epochs}", self.output_folder)
            self.train_sampler.set_epoch(epoch)

            self.train_loss.append(0.0)
            num_samples = 0

            for iteration, ((images, _), masks_context, masks_pred) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                images = images[0].to(self.device, non_blocking=True)
                masks_context = [m.to(self.device) for m in masks_context]
                masks_pred = [m.to(self.device) for m in masks_pred]

                with torch.amp.autocast("cuda", dtype=torch.float16):
                    z = self.encoder(images, masks_context)
                    z_pred = self.predictor(z, masks_context, masks_pred)

                    with torch.no_grad():
                        z_target = self.target_encoder(images)
                        z_target = F.layer_norm(z_target, (z_target.size(-1),))
                        B = len(z_target)
                        z_target = apply_masks(z_target, masks_pred)
                        z_target = repeat_interleave_batch(z_target, B, repeat=len(masks_context))

                    loss = self.apply_criterion(z_pred, z_target)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_value = loss.item()
                self.train_loss[-1] += loss_value * images.size(0)
                num_samples += images.size(0)

                self.lr_values.append(self.lr_scheduler.get_value())
                self.wd_values.append(self.wd_scheduler.get_value())
                self.ema_values.append(self.ema_scheduler.get_value())
                write_on_csv(self.output_folder, epoch, iteration, loss_value, self.lr_values[-1], self.wd_values[-1], self.ema_values[-1])

                self.update_target_network(ema=self.ema_scheduler.get_value())
                
                self.lr_scheduler.step()
                self.wd_scheduler.step()
                self.ema_scheduler.step()
            
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
        predictor_state_dict = self.predictor.module.state_dict() if self.world_size > 1 else self.predictor.state_dict()
        target_encoder_state_dict = self.target_encoder.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        lr_scheduler_state_dict = self.lr_scheduler.state_dict()
        wd_scheduler_state_dict = self.wd_scheduler.state_dict()
        ema_scheduler_state_dict = self.ema_scheduler.state_dict()
        scaler_state_dict = self.scaler.state_dict()

        torch.save(encoder_state_dict, os.path.join(self.output_folder, "models", "encoder.pth"))
        torch.save(predictor_state_dict, os.path.join(self.output_folder, "models", "predictor.pth"))
        torch.save(target_encoder_state_dict, os.path.join(self.output_folder, "models", "target_encoder.pth"))
        torch.save(optimizer_state_dict, os.path.join(self.output_folder, "models", f"optimizer.pth"))
        torch.save(lr_scheduler_state_dict, os.path.join(self.output_folder, "models", f"lr_scheduler.pth"))
        torch.save(wd_scheduler_state_dict, os.path.join(self.output_folder, "models", f"wd_scheduler.pth"))
        torch.save(ema_scheduler_state_dict, os.path.join(self.output_folder, "models", f"ema_scheduler.pth"))
        torch.save(scaler_state_dict, os.path.join(self.output_folder, "models", "scaler.pth"))

        if self.meta_save_every > 0 and epoch % self.meta_save_every == 0:
            torch.save(encoder_state_dict, os.path.join(self.output_folder, "models", f"encoder_epoch_{epoch}.pth"))
            torch.save(predictor_state_dict, os.path.join(self.output_folder, "models", f"predictor_epoch_{epoch}.pth"))
            torch.save(target_encoder_state_dict, os.path.join(self.output_folder, "models", f"target_encoder_epoch_{epoch}.pth"))
            torch.save(optimizer_state_dict, os.path.join(self.output_folder, "models", f"optimizer_epoch_{epoch}.pth"))
            torch.save(lr_scheduler_state_dict, os.path.join(self.output_folder, "models", f"lr_scheduler_epoch_{epoch}.pth"))
            torch.save(wd_scheduler_state_dict, os.path.join(self.output_folder, "models", f"wd_scheduler_epoch_{epoch}.pth"))
            torch.save(ema_scheduler_state_dict, os.path.join(self.output_folder, "models", f"ema_scheduler_epoch_{epoch}.pth"))
            torch.save(scaler_state_dict, os.path.join(self.output_folder, "models", f"scaler_epoch_{epoch}.pth"))
    
    def update_target_network(self, ema):
        with torch.no_grad():
            for param_q, param_k in zip(self.encoder.parameters(), self.target_encoder.parameters()):
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

    def _load_optimizer(self):
        match self.optimization_optimizer:
            case "adamw":
                target_modules = [self.encoder, self.predictor] if self.world_size == 1 else [self.encoder.module, self.predictor.module]

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

    def _load_criterion(self):
        match self.optimization_criterion:
            case "mse_loss":
                self.criterion = nn.MSELoss()

            case "l1_smooth_loss":
                self.criterion = nn.SmoothL1Loss()

            case _:
                raise ValueError(f"Unsupported criterion: {self.optimization_criterion}")

    def apply_criterion(self, z1, z2):
        match self.optimization_criterion:
            case "mse_loss":
                return self.criterion(z1, z2)
            
            case "l1_smooth_loss"   :
                return self.criterion(z1, z2)

            case _:
                raise ValueError(f"Unsupported criterion: {self.optimization_criterion}")

    def _load_dataloader(self):
        mask_collator = MaskCollator(
            crop_size=self.data_crop_size,
            patch_size=self.mask_patch_size,
            n_targets=self.mask_num_target_masks,
            min_keep=self.mask_min_context_patches,
            context_mask_scale=self.mask_context_mask_scale,
            pred_aspect_ratio=self.mask_target_aspect_ratio,
            pred_mask_scale=self.mask_target_mask_scale
        )

        self.train_dataset = datasets(
            operation="train",
            datasets_folder_path=self.data_datasets_path,
            dataset_name=self.data_train_dataset,
            separate_val_subset=self.data_separate_val_subset_use,
            val_size=self.data_separate_val_subset_size,
            transforms=[self.transform],
            times=[1]
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
        self.transform = v2.Compose([
            v2.Resize((self.data_crop_size, self.data_crop_size)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std)
        ])

    def _load_models(self):
        match self.meta_model_name:
            case "vit_tiny":
                self.encoder = vit_tiny(patch_size=self.mask_patch_size, checkpoint=self.meta_checkpoint)
            case "vit_small":
                self.encoder = vit_small(patch_size=self.mask_patch_size, checkpoint=self.meta_checkpoint)
            case "vit_base":
                self.encoder = vit_base(patch_size=self.mask_patch_size, checkpoint=self.meta_checkpoint)
            case "vit_large":
                self.encoder = vit_large(patch_size=self.mask_patch_size, checkpoint=self.meta_checkpoint)
            case "vit_huge":
                self.encoder = vit_huge(patch_size=self.mask_patch_size, checkpoint=self.meta_checkpoint)
            case "vit_giant":
                self.encoder = vit_giant(patch_size=self.mask_patch_size, checkpoint=self.meta_checkpoint)
        
        self.predictor = vit_predictor(num_patches=self.encoder.get_num_patches(),
                                       embed_dim=self.encoder.get_embed_dim(),
                                       depth=self.meta_predictor_depth,
                                       predictor_embed_dim=self.meta_predictor_emb_dim,
                                       num_heads=self.meta_predictor_num_heads,    
                                       checkpoint=self.meta_checkpoint,                 
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
        self.target_encoder.checkpoint = False # Target model should not use checkpointing

        if self.continue_training:
            if os.path.exists(os.path.join(self.output_folder, "models")):
                self.encoder.load_weights(os.path.join(self.output_folder, "models", "encoder.pth"), device=self.device)
                self.predictor.load_weights(os.path.join(self.output_folder, "models", "predictor.pth"), device=self.device)
                self.target_encoder.load_weights(os.path.join(self.output_folder, "models", "target_encoder.pth"), device=self.device)
            else:
                raise FileNotFoundError("Checkpoint files not found for continuing training.")

        self.encoder.unfreeze()
        self.predictor.unfreeze()
        self.target_encoder.freeze()

        self.encoder.to(self.device)
        self.predictor.to(self.device)
        self.target_encoder.to(self.device)

        if self.world_size > 1:
            self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
            self.predictor = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.predictor)

            self.encoder = DDP(self.encoder, device_ids=[self.rank], output_device=self.rank)
            self.predictor = DDP(self.predictor, device_ids=[self.rank], output_device=self.rank)

        self.encoder.train()
        self.predictor.train()
        self.target_encoder.train()

    def _load_config(self):
        self.data_datasets_path = str(self.config["data"]["datasets_path"])
        self.data_train_dataset = str(self.config["data"]["train_dataset"])
        self.data_batch_size = int(self.config["data"]["batch_size"])
        self.data_crop_size = int(self.config["data"]["crop_size"])
        self.data_num_workers = int(self.config["data"]["num_workers"])
        self.data_prefetch_factor = int(self.config["data"]["prefetch_factor"])
        self.data_pin_memory = bool(self.config["data"]["pin_memory"])
        self.data_drop_last = bool(self.config["data"]["drop_last"])
        self.data_normalize_mean = list(map(float, self.config["data"]["normalize"]["mean"]))
        self.data_normalize_std = list(map(float, self.config["data"]["normalize"]["std"]))
        self.data_separate_val_subset_use = bool(self.config["data"]["separate_val_subset"]["use"])
        self.data_separate_val_subset_size = float(self.config["data"]["separate_val_subset"]["size"])

        self.mask_target_aspect_ratio = list(map(float, self.config["mask"]["target_aspect_ratio"]))
        self.mask_context_mask_scale = list(map(float, self.config["mask"]["context_mask_scale"]))
        self.mask_min_context_patches = int(self.config["mask"]["min_context_patches"])
        self.mask_num_target_masks = int(self.config["mask"]["num_target_masks"])
        self.mask_patch_size = int(self.config["mask"]["patch_size"])
        self.mask_target_mask_scale = list(map(float, self.config["mask"]["target_mask_scale"]))

        self.meta_model_name = str(self.config["meta"]["model_name"])
        self.meta_checkpoint = bool(self.config["meta"]["checkpoint"])
        self.meta_predictor_depth = int(self.config["meta"]["predictor_depth"])
        self.meta_predictor_emb_dim = int(self.config["meta"]["predictor_emb_dim"])
        self.meta_predictor_num_heads = int(self.config["meta"]["predictor_num_heads"])
        self.meta_pretrained_weights = self.config["meta"]["pretrained_weights"]
        self.meta_save_every = int(self.config["meta"]["save_every"])

        self.optimization_ipe_scale = float(self.config["optimization"]["ipe_scale"])
        self.optimization_ema = list(map(float, self.config["optimization"]["ema"]))
        self.optimization_lr = list(map(float, self.config["optimization"]["lr"]))
        self.optimization_weight_decay = list(map(float, self.config["optimization"]["weight_decay"]))
        self.optimization_epochs = int(self.config["optimization"]["epochs"])
        self.optimization_warmup_epochs = int(self.config["optimization"]["warmup_epochs"])
        self.optimization_optimizer = str(self.config["optimization"]["optimizer"])
        self.optimization_criterion = str(self.config["optimization"]["criterion"])

        self.data_datasets_path += "/" if not self.data_datasets_path.endswith("/") else ""
