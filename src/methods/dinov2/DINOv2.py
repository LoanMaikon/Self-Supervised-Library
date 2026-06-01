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
from src.koleo_loss import KoLeoLoss
from src.sinkhorn import sinkhorn
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

        self.scaler = torch.amp.GradScaler()

        self.train_loss = []
        self.lr_values = []
        self.wd_values = []
        self.ema_values = []

        self.koleo_loss = KoLeoLoss().to(self.device)

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
        write_on_log("Starting training...", self.output_folder)


        for epoch in range(1, self.optimization_epochs + 1):
            if self.continue_training and epoch <= self.last_epoch:
                continue

            write_on_log(f"Epoch {epoch}/{self.optimization_epochs}", self.output_folder)
            self.train_sampler.set_epoch(epoch)
            self.train_dataloader.collate_fn.set_epoch(epoch)

            self.train_loss.append(0.0)
            num_samples = 0

            for iteration, ((images, _), masks) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad(set_to_none=True)

                images = [img.to(self.device, non_blocking=True) for img in images]
                masks = [mask.to(self.device, non_blocking=True) for mask in masks]
                batch_size = images[0].size(0)

                n_global_crops = self.data_global_views_num
                n_local_crops = len(images) - n_global_crops

                global_images = images[:n_global_crops]
                local_images = images[n_global_crops:]

                global_masks = masks[:n_global_crops]
                if global_masks:
                    global_masks = [mask.flatten(1) for mask in global_masks]
                    global_masks_cat = torch.cat(global_masks, dim=0)
                else:
                    global_masks_cat = None

                global_images_cat = torch.cat(global_images, dim=0)
                local_images_cat = torch.cat(local_images, dim=0) if n_local_crops > 0 else None

                student_temp_cls = self.student_cls_temperature_scheduler.get_value()
                student_temp_patch = self.student_patch_temperature_scheduler.get_value()
                teacher_temp_cls = self.teacher_cls_temperature_scheduler.get_value()
                teacher_temp_patch = self.teacher_patch_temperature_scheduler.get_value()

                with torch.amp.autocast(device_type=self.device.type):
                    with torch.no_grad():
                        teacher_out = self.target_encoder(global_images_cat, masks=None, is_training=True)
                        teacher_cls_tokens = teacher_out["x_norm_clstoken"]
                        teacher_patch_tokens = teacher_out["x_norm_patchtokens"]

                        teacher_cls_chunks = teacher_cls_tokens.chunk(n_global_crops)
                        teacher_cls_reordered = torch.cat(list(teacher_cls_chunks)[::-1], dim=0)
                        teacher_cls_after_head = self.target_projection_head_cls(teacher_cls_reordered).float()

                        teacher_patch_after_head = self.target_projection_head_patch(
                            teacher_patch_tokens.flatten(0, 1)
                        ).view(teacher_patch_tokens.shape[0], teacher_patch_tokens.shape[1], -1).float()

                        teacher_cls_targets_list = [
                            sinkhorn(
                                chunk / teacher_temp_cls,
                                self.optimization_sinkhorn_epsilon,
                                self.optimization_sinkhorn_iterations,
                                self.world_size,
                            )
                            for chunk in teacher_cls_after_head.chunk(n_global_crops)
                        ]

                        teacher_patch_targets_list = None
                        if global_masks:
                            teacher_patch_chunks = teacher_patch_after_head.chunk(n_global_crops)
                            teacher_patch_targets_list = []

                            for iq in range(n_global_crops):
                                mask = global_masks[iq].bool()
                                if mask.sum() == 0:
                                    teacher_patch_targets_list.append((None, mask))
                                    continue

                                teacher_patch_masked = teacher_patch_chunks[iq][mask]
                                t = sinkhorn(
                                    teacher_patch_masked / teacher_temp_patch,
                                    self.optimization_sinkhorn_epsilon,
                                    self.optimization_sinkhorn_iterations,
                                    self.world_size,
                                )
                                teacher_patch_targets_list.append((t, mask))

                    student_out_global = self.encoder(global_images_cat, masks=global_masks_cat, is_training=True)
                    student_cls_tokens_global = student_out_global["x_norm_clstoken"]
                    student_patch_tokens_global = student_out_global["x_norm_patchtokens"]

                    student_cls_after_head_global = self.projection_head_cls(student_cls_tokens_global).float()
                    student_patch_after_head_global = self.projection_head_patch(
                        student_patch_tokens_global.flatten(0, 1)
                    ).view(student_patch_tokens_global.shape[0], student_patch_tokens_global.shape[1], -1).float()

                    student_cls_after_head_local = None
                    if n_local_crops > 0:
                        local_out = self.encoder(local_images_cat, masks=None, is_training=True)
                        student_cls_after_head_local = self.projection_head_cls(local_out["x_norm_clstoken"]).float()

                    n_local_terms = max(n_local_crops * n_global_crops, 1)
                    n_global_terms = (n_global_crops - 1) * n_global_crops
                    dino_den = n_local_terms + n_global_terms

                    dino_local_loss = torch.tensor(0.0, device=self.device)
                    if n_local_crops > 0:
                        for s in student_cls_after_head_local.chunk(n_local_crops):
                            for t in teacher_cls_targets_list:
                                dino_local_loss += torch.sum(
                                    -t * F.log_softmax(s / student_temp_cls, dim=-1),
                                    dim=-1,
                                ).mean()
                        dino_local_loss = dino_local_loss / dino_den

                    teacher_global_all = torch.cat(teacher_cls_targets_list, dim=0)
                    dino_global_loss = torch.sum(
                        -teacher_global_all * F.log_softmax(student_cls_after_head_global / student_temp_cls, dim=-1),
                        dim=-1,
                    ).mean()
                    dino_global_loss = dino_global_loss * 2.0 / dino_den
                    dino_loss = dino_local_loss + dino_global_loss

                    ibot_loss = torch.tensor(0.0, device=self.device)
                    if global_masks:
                        teacher_patch_chunks = teacher_patch_after_head.chunk(n_global_crops)
                        student_patch_chunks = student_patch_after_head_global.chunk(n_global_crops)

                        for iq in range(n_global_crops):
                            mask = global_masks[iq].bool()
                            if mask.sum() == 0:
                                continue

                            teacher_patch_target, _ = teacher_patch_targets_list[iq]
                            if teacher_patch_target is None:
                                continue

                            student_patch_masked = student_patch_chunks[iq][mask]
                            loss_per_patch = torch.sum(
                                -teacher_patch_target * F.log_softmax(student_patch_masked / student_temp_patch, dim=-1),
                                dim=-1,
                            )

                            mask_indices = mask.nonzero(as_tuple=False)
                            per_image_sum = torch.zeros(batch_size, device=loss_per_patch.device)
                            per_image_count = torch.zeros(batch_size, device=loss_per_patch.device)

                            per_image_sum.index_add_(0, mask_indices[:, 0], loss_per_patch)
                            per_image_count.index_add_(0, mask_indices[:, 0], torch.ones_like(loss_per_patch))

                            ibot_loss += (per_image_sum / per_image_count.clamp(min=1.0)).mean()

                        ibot_loss = ibot_loss * (2.0 / n_global_crops)

                    koleo_loss = torch.tensor(0.0, device=self.device)
                    if self.optimization_koleo_loss_weight > 0:
                        koleo_loss = sum(self.koleo_loss(p) for p in student_cls_tokens_global.chunk(n_global_crops))

                    total_loss = (
                        self.optimization_dino_loss_weight * dino_loss
                        + self.optimization_ibot_loss_weight * ibot_loss
                        + self.optimization_koleo_loss_weight * koleo_loss
                    )

                loss_value = total_loss.item()
                self.train_loss[-1] += loss_value * batch_size
                num_samples += batch_size

                self.scaler.scale(total_loss).backward()

                if self.world_size > 1:
                    self.projection_head_cls.module.cancel_gradients_last_layer(
                        epoch, self.optimization_freeze_last_layer_epochs
                    )
                    self.projection_head_patch.module.cancel_gradients_last_layer(
                        epoch, self.optimization_freeze_last_layer_epochs
                    )
                else:
                    self.projection_head_cls.cancel_gradients_last_layer(
                        epoch, self.optimization_freeze_last_layer_epochs
                    )
                    self.projection_head_patch.cancel_gradients_last_layer(
                        epoch, self.optimization_freeze_last_layer_epochs
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()

                self.lr_values.append(self.lr_scheduler.get_value())
                self.wd_values.append(self.wd_scheduler.get_value())
                self.ema_values.append(self.ema_scheduler.get_value())
                write_on_csv(
                    self.output_folder, epoch, iteration, loss_value,
                    self.lr_values[-1], self.wd_values[-1], self.ema_values[-1]
                )

                self._update_ema()

                self.lr_scheduler.step()
                self.wd_scheduler.step()
                self.ema_scheduler.step()
                self.student_cls_temperature_scheduler.step()
                self.student_patch_temperature_scheduler.step()
                self.teacher_cls_temperature_scheduler.step()
                self.teacher_patch_temperature_scheduler.step()

            self.train_loss[-1] /= num_samples

            self.save_models(epoch)
            write_on_log(f"Loss: {self.train_loss[-1]}", self.output_folder)

            plot_fig(range(len(self.train_loss)), "Epoch", self.train_loss, "Loss", "loss", self.output_folder)
            plot_fig(range(len(self.lr_values)), "Iteration", self.lr_values, "Learning Rate", "learning_rate", self.output_folder)
            plot_fig(range(len(self.wd_values)), "Iteration", self.wd_values, "Weight Decay", "weight_decay", self.output_folder)
            plot_fig(range(len(self.ema_values)), "Iteration", self.ema_values, "EMA", "ema", self.output_folder)

            save_json({"train_loss": self.train_loss}, self.output_folder, "training_info")
            save_json({"last_epoch": epoch}, self.output_folder, "last_epoch")

            write_on_log("", self.output_folder)

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
                target_modules = [self.encoder, self.projection_head_cls, self.projection_head_patch] if self.world_size == 1 else [self.encoder.module, self.projection_head_cls.module, self.projection_head_patch.module]

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
        self.projection_head_cls.unfreeze()
        self.projection_head_patch.unfreeze()
        self.target_encoder.freeze()
        self.target_projection_head_cls.freeze()
        self.target_projection_head_patch.freeze()

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
        self.optimization_sinkhorn_epsilon = float(self.config['optimization']['sinkhorn_epsilon'])
        self.optimization_sinkhorn_iterations = int(self.config['optimization']['sinkhorn_iterations'])
        self.optimization_num_register_tokens = int(self.config['optimization']['num_register_tokens'])
