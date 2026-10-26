from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import torch.nn.functional as F
from functools import partial
import torch.optim as optim
import torch
import os

from src.utils import write_on_log, plot_fig, write_on_csv, save_json, is_main_process, \
    recreate_csv_log, get_last_epoch, load_last_values, make_param_groups
from src.schedulers import WarmupCosineSchedule, CosineWDSchedule, EMACosineSchedule, \
    LinearWarmupTemperatureSchedule
from .models import vit_base, vit_small, vit_large, vit_so400m, vit_giant2, vit_huge2, vit_7b, dino_head
from src.datasets import datasets
from .mask_collator import MaskingGenerator, collate_data_and_cast
from .dino_clstoken_loss import DINOLoss
from .gram_loss import GramLoss
from .ibot_patch_loss import iBOTPatchLoss
from .koleo_loss import KoLeoLoss

class GlobalCropWithGram:
    def __init__(self, transform, global_size, gram_size):
        self.transform = transform
        self.resize_global = v2.Resize((global_size, global_size), interpolation=v2.InterpolationMode.BICUBIC)
        self.resize_gram = v2.Resize((gram_size, gram_size), interpolation=v2.InterpolationMode.BICUBIC)

    def __call__(self, image):
        # Both teachers must see the same spatial crop as the student.
        crop = self.transform(image)
        return self.resize_global(crop), self.resize_gram(crop)

class DINOv3():
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
        try:
            self.device_type = torch.device(device).type
        except Exception:
            self.device_type = "cuda" if torch.cuda.is_available() else "cpu"
        self.amp_enabled = self.device_type == "cuda"
        self.amp_dtype = torch.float16 if self.amp_enabled else torch.float32
        self.global_iteration = 0

        self._load_config()
        self._load_models()
        self._load_transform()
        self._load_dataloader()
        self._load_optimizer()
        self._load_schedulers()

        self.dino_loss = DINOLoss(
            student_temp = self.optimization_temperature_student_cls[1],
            out_dim = self.meta_projection_head_dino_output_dim
        )
        self.gram_loss = GramLoss(
            apply_norm=self.gram_normalized,
            img_level=self.gram_img_level,
            remove_neg=self.gram_remove_neg,
            remove_only_teacher_neg=self.gram_remove_only_teacher_neg,
        )
        self.ibot_patch_loss = iBOTPatchLoss(
            student_temp=self.optimization_temperature_student_patch[1],
            patch_out_dim = self.meta_projection_head_ibot_output_dim
        )
        self.koleo_loss = KoLeoLoss()
        self.dino_loss.init_weights()
        self.ibot_patch_loss.init_weights()

        self.scaler = torch.amp.GradScaler(device=self.device_type, enabled=self.amp_enabled)

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
        write_on_log("Starting training...", self.output_folder)

        for epoch in range(1, self.optimization_epochs + 1):
            if self.continue_training and epoch <= self.last_epoch:
                continue

            write_on_log(f"Epoch {epoch}/{self.optimization_epochs}", self.output_folder)
            self.train_sampler.set_epoch(epoch)

            self.train_loss.append(0.0)
            num_samples = 0

            for iteration, images_dict in enumerate(self.train_dataloader):
                self.optimizer.zero_grad(set_to_none=True)

                global_crops = images_dict["collated_global_crops"].to(self.device, non_blocking=True) # [B * global_crops, C, H, W]
                local_crops = images_dict["collated_local_crops"].to(self.device, non_blocking=True) # [B * local_crops, C, H, W]
                masks = images_dict["collated_masks"].to(self.device, non_blocking=True) # [B * global_crops, num_patches]
                mask_indices_list = images_dict["mask_indices_list"].to(self.device, non_blocking=True) # [total_masked_patches], ex: [[1, 4, 6, ...], [0, 2, 5, ...], ...]
                n_masked_patches_tensor = images_dict["n_masked_patches"].to(self.device, non_blocking=True) # int, total number of masked patches in the batch
                n_masked_patches = mask_indices_list.shape[0] # int, total number of masked patches in the batch
                upperbound = images_dict["upperbound"] # int, max number of masked patches in a single image
                masks_weight = images_dict["masks_weight"].to(self.device, non_blocking=True) # [total_masked_patches], for each image contribute equally to the loss regardless of the number of masked patches
                gram_teacher_crops = images_dict.get("collated_gram_teacher_crops")
                if gram_teacher_crops is not None:
                    gram_teacher_crops = gram_teacher_crops.to(self.device, non_blocking=True)

                B = images_dict["collated_local_crops"].shape[0] // self.data_local_views_num

                with torch.amp.autocast(device_type=self.device_type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                    teacher_global = self.get_teacher_output(
                        global_crops.unflatten(0, (self.data_global_views_num, B)),
                        teacher_cls_temp=self.teacher_cls_temperature_scheduler.get_value(),
                        teacher_patch_temp=self.teacher_patch_temperature_scheduler.get_value(),
                        n_masked_patches_tensor=n_masked_patches_tensor,
                        mask_indices_list=mask_indices_list,
                        upperbound=upperbound,
                    )

                    student_global, student_local = self.get_student_output(
                        global_crops=global_crops.unflatten(0, (self.data_global_views_num, B)),
                        local_crops=local_crops.unflatten(0, (self.data_local_views_num, B)),
                        upperbound=upperbound,
                        masks=masks,
                        mask_indices_list=mask_indices_list,
                    )

                    if self.gram_use_loss:
                        gram_global = self.get_gram_teacher_output(
                            gram_teacher_crops.unflatten(0, (self.data_global_views_num, B)) if gram_teacher_crops is not None else None,
                            masks=masks,
                            teacher_global=teacher_global,
                            student_global=student_global,
                            student_global_crops_size=global_crops.shape[-1],
                        )
                    else:
                        gram_global = {}

                    self.dino_loss.student_temp = self.student_cls_temperature_scheduler.get_value()
                    self.ibot_patch_loss.student_temp = self.student_patch_temperature_scheduler.get_value()

                    loss_accumulator, loss_dict = self.compute_losses(
                        teacher_global=teacher_global,
                        student_global=student_global,
                        student_local=student_local,
                        gram_global=gram_global,
                        masks=masks,
                        mask_indices_list=mask_indices_list,
                        masks_weight=masks_weight,
                        iteration=iteration,
                    )

                total_loss = loss_accumulator
                batch_size = global_crops.shape[0] // self.data_global_views_num

                loss_value = total_loss.item()
                self.train_loss[-1] += loss_value * batch_size
                num_samples += batch_size

                self.scaler.scale(total_loss).backward()

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
        gram_state_dict = self.gram.state_dict() if self.gram is not None else None

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
        if gram_state_dict is not None:
            torch.save(gram_state_dict, os.path.join(self.output_folder, "models", f"gram.pth"))

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
            if gram_state_dict is not None:
                torch.save(gram_state_dict, os.path.join(self.output_folder, "models", f"gram_epoch_{epoch}.pth"))

    @torch.no_grad()
    def get_teacher_output(
        self,
        images,
        *,
        upperbound,
        mask_indices_list,
        teacher_cls_temp=None,
        teacher_patch_temp=None,
        teacher_temp=None,
        n_masked_patches_tensor=None,
    ):
        if teacher_temp is not None and teacher_cls_temp is None:
            teacher_cls_temp = teacher_temp
        if teacher_patch_temp is None:
            teacher_patch_temp = teacher_cls_temp
        n_crops, B, rgb, H, W = images.shape
        images = images.flatten(0, 1)

        backbone_out = self.target_encoder(images, is_training=True)
        cls = backbone_out["x_norm_clstoken"]  # [n_crops * B, D]
        reg = backbone_out["x_storage_tokens"]  # [n_crops * B, R, D]
        ibot_patch = backbone_out["x_norm_patchtokens"]  # [n_crops * B, P, D]

        # IBOT head only on patches that are masked for the student
        buffer = torch.index_select(ibot_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        masked_patch_after_head = self.target_projection_head_patch(buffer)

        # DINO head on CLS tokens
        cls_after_head = self.target_projection_head_cls(cls)  # [n_crops * B, K]

        # Center with sinkhorn-knopp
        cls_centered = self.dino_loss.sinkhorn_knopp_teacher(
            cls_after_head, teacher_temp=teacher_cls_temp, n_iterations=self.optimization_sinkhorn_iterations
        )  # [n_crops * B, K]
        cls_centered = cls_centered.unflatten(0, (n_crops, B))  # [n_crops, B, K]
        masked_patch_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
            masked_patch_after_head,
            teacher_temp=teacher_patch_temp,
            n_masked_patches_tensor=n_masked_patches_tensor,
            n_iterations=self.optimization_sinkhorn_iterations,
        )  # [n_masked_patches, K]

        return {
            "cls_pre_head": cls.unflatten(0, [n_crops, B]),  # [n_crops, B, D]
            "reg_pre_head": reg.unflatten(0, [n_crops, B]),  # [n_crops, B, R, D]
            "patch_pre_head": ibot_patch.unflatten(0, [n_crops, B]),  # [n_crops, B, P, D]
            "cls_after_head": cls_after_head.unflatten(0, [n_crops, B]),  # [n_crops, B, K]
            "cls_centered": cls_centered,  # [n_crops, B, K]
            "masked_patch_centered": masked_patch_centered,  # [n_masked_patches, K]
        }

    def get_gram_teacher_output(self, images, *, masks, teacher_global, student_global, student_global_crops_size):
        """
        1. EMA-teacher Gram target: reuse the already computed teacher global patches.
        2. Separate Gram teacher: forward self.gram backbone on collated_gram_teacher_crops.
        """
        student_patches = student_global["patch_pre_head"].flatten(0, 1)  # [2B, P, D]

        if self.gram_ema_teacher or self.gram is None:
            teacher_patches = teacher_global["patch_pre_head"].flatten(0, 1)
        else:
            if images is None:
                raise ValueError(
                    "Gram loss is configured with a separate Gram teacher, but collated_gram_teacher_crops "
                    "is missing. Either provide gram teacher crops or set gram.ema_teacher/use_ema_teacher=True."
                )
            n_crops, B, rgb, H, W = images.shape
            images = images.flatten(0, 1)
            with torch.no_grad():
                backbone_out = self.gram(images, is_training=True)
            teacher_patches = backbone_out["x_norm_patchtokens"]

            if teacher_patches.shape[1] != student_patches.shape[1]:
                N = H // self.meta_patch_size
                assert teacher_patches.shape[1] == N**2
                N_student = student_global_crops_size // self.meta_patch_size
                assert student_patches.shape[1] == N_student**2
                patches_hw = teacher_patches.transpose(-2, -1).unflatten(-1, (N, N))
                patches_hw = torch.nn.functional.interpolate(
                    patches_hw,
                    size=(N_student, N_student),
                    mode=self.gram_global_teacher_resize_method,
                    align_corners=False if self.gram_global_teacher_resize_method in {"linear", "bilinear", "bicubic", "trilinear"} else None,
                    antialias=self.gram_global_teacher_resize_antialias,
                )
                teacher_patches = patches_hw.flatten(-2, -1).transpose(-2, -1)
                assert teacher_patches.shape == student_patches.shape

        orig_student_patches = student_patches
        orig_teacher_patches = teacher_patches
        if self.gram_tokens_used == "masked":
            student_patches = student_patches[masks]
            teacher_patches = teacher_patches[masks]
        elif self.gram_tokens_used == "unmasked":
            student_patches = student_patches[~masks]
            teacher_patches = teacher_patches[~masks]

        return {
            "student_patches": student_patches,
            "teacher_patches": teacher_patches,
            "orig_student_patches": orig_student_patches,
            "orig_teacher_patches": orig_teacher_patches,
        }

    def get_student_output(self, *, global_crops, local_crops, upperbound, masks, mask_indices_list):
        n_global_crops, B, rgb, H, W = global_crops.shape
        n_local_crops, B, rgb, H, W = local_crops.shape

        global_crops = global_crops.flatten(0, 1)

        # Forward global and local crops through the student backbone jointly
        global_out, local_out = self.encoder(
            [global_crops, local_crops.flatten(0, 1)],
            masks=[masks, None],
            is_training=True,
        )
        g_cls, g_reg, g_patch = (
            global_out["x_norm_clstoken"],
            global_out["x_storage_tokens"],
            global_out["x_norm_patchtokens"],
        )
        l_cls, l_reg, l_patch = (
            local_out["x_norm_clstoken"],
            local_out["x_storage_tokens"],
            local_out["x_norm_patchtokens"],
        )

        # IBOT head only on masked patches
        masked_patches_pre_head = torch.index_select(g_patch.flatten(0, 1), dim=0, index=mask_indices_list)
        global_masked_patch_after_head = self.projection_head_patch(masked_patches_pre_head)

        # DINO head on CLS tokens (all in one pass)
        buffer = [
            g_cls,  # [n_global_crops * B, D]
            l_cls,  # [n_local_crops * B, D]
        ]
        sizes = [x.shape[0] for x in buffer]
        buffer = torch.cat(buffer, dim=0)  # [n_global_crops * B + n_local_crops * B, D]
        buffer = self.projection_head_cls(buffer)  # [n_global_crops * B + n_local_crops * B, K]
        buffer = torch.split_with_sizes(buffer, sizes, dim=0)

        global_out = {
            "cls_pre_head": g_cls.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, D]
            "reg_pre_head": g_reg.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, R, D]
            "patch_pre_head": g_patch.unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, P, D]
            "cls_after_head": buffer[0].unflatten(0, [n_global_crops, B]),  # [n_global_crops, B, K],
            "masked_patch_after_head": global_masked_patch_after_head,  # [n_masked_patches, K]
            "masked_patch_pre_head": masked_patches_pre_head,  # [n_masked_patches, D]
        }
        local_out = {
            "cls_pre_head": l_cls.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, D]
            "reg_pre_head": l_reg.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, R, D]
            "patch_pre_head": l_patch.unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, P, D]
            "cls_after_head": buffer[1].unflatten(0, [n_local_crops, B]),  # [n_local_crops, B, K],
        }

        return global_out, local_out

    def compute_losses(
        self,
        *,
        teacher_global,
        student_global,
        student_local,
        gram_global,
        masks,
        mask_indices_list,
        masks_weight,
        iteration,
    ):
        n_global_crops = student_global["cls_after_head"].shape[0]
        n_local_crops = student_local["cls_after_head"].shape[0]
        loss_dict = {}
        loss_accumulator = 0.0

        # Loss scales like in DINOv2, these are multiplied with the loss weights from the config
        dino_global_terms = (
            n_global_crops * (n_global_crops - 1) if self.dino_global_ignore_diagonal else n_global_crops**2
        )
        dino_local_terms = n_global_crops * n_local_crops
        dino_global_scale = dino_global_terms / (dino_global_terms + dino_local_terms)
        dino_local_scale = dino_local_terms / (dino_global_terms + dino_local_terms)
        koleo_scale = n_global_crops

        # DINO local loss: compare post-head CLS tokens: student(local crops) vs. teacher(global crops)
        dino_local_crops_loss = self.dino_loss(
            student_logits=student_local["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
        )
        loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

        local_weight = 1.0

        loss_dict["dino_local_loss_weight"] = local_weight
        loss_accumulator += self.optimization_dino_loss_weight * dino_local_scale * local_weight * dino_local_crops_loss

        # DINO global loss: compare post-head CLS tokens: student(global crops) vs. teacher(global crops)
        dino_global_crops_loss = self.dino_loss(
            student_logits=student_global["cls_after_head"],
            teacher_probs=teacher_global["cls_centered"],
            ignore_diagonal=self.dino_global_ignore_diagonal,
        )
        loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
        loss_accumulator += self.optimization_dino_loss_weight * dino_global_scale * dino_global_crops_loss

        # Koleo: regularize pre-head CLS tokens of student(global crops)
        koleo_loss = sum(self.koleo_loss(x) for x in student_global["cls_pre_head"]) / n_global_crops
        loss_dict["koleo_loss"] = koleo_loss
        loss_accumulator += self.optimization_koleo_loss_weight * koleo_scale * koleo_loss

        # IBOT loss
        ibot_patch_loss = self.ibot_patch_loss.forward_masked(
            student_global["masked_patch_after_head"],
            teacher_global["masked_patch_centered"],
            student_masks_flat=masks,
            n_masked_patches=mask_indices_list.shape[0],
            masks_weight=masks_weight,
        )
        loss_dict["ibot_loss"] = ibot_patch_loss
        loss_accumulator += self.optimization_ibot_loss_weight * ibot_patch_loss

        if self.gram_use_loss:
            gram_loss = self.gram_loss(
                gram_global["student_patches"],
                gram_global["teacher_patches"],
                img_level=self.gram_img_level,
            )

            gram_loss_weight = self.optimization_gram_loss_weight
            loss_dict["gram_loss_weight"] = gram_loss_weight
            loss_accumulator += gram_loss * gram_loss_weight
            loss_dict["gram_loss"] = gram_loss

            if self.gram_compute_stats:
                with torch.no_grad():
                    gram_loss_masked = self.gram_loss(
                        gram_global["orig_student_patches"][masks].detach(),
                        gram_global["orig_teacher_patches"][masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/masked_gram_loss"] = gram_loss_masked
                    gram_loss_unmasked = self.gram_loss(
                        gram_global["orig_student_patches"][~masks].detach(),
                        gram_global["orig_teacher_patches"][~masks],
                        img_level=False,
                    )
                    loss_dict["stats_only/unmasked_gram_loss"] = gram_loss_unmasked

        return loss_accumulator, loss_dict

    @torch.no_grad()
    def _update_ema(self, m=0):
        ema = self.ema_scheduler.get_value()

        student_encoder = self.encoder.module if self.world_size > 1 else self.encoder
        student_cls_head = self.projection_head_cls.module if self.world_size > 1 else self.projection_head_cls
        student_patch_head = self.projection_head_patch.module if self.world_size > 1 else self.projection_head_patch

        for p_t, p_s in zip(self.target_encoder.parameters(), student_encoder.parameters()):
            p_t.data.mul_(ema).add_(p_s.data, alpha=1 - ema)

        for p_t, p_s in zip(self.target_projection_head_cls.parameters(), student_cls_head.parameters()):
            p_t.data.mul_(ema).add_(p_s.data, alpha=1 - ema)

        for p_t, p_s in zip(self.target_projection_head_patch.parameters(), student_patch_head.parameters()):
            p_t.data.mul_(ema).add_(p_s.data, alpha=1 - ema)

        if self.gram is not None and self.gram_rep_update:
            for p_g, p_t in zip(self.gram.parameters(), self.target_encoder.parameters()):
                p_g.data.mul_(self.gram_update_momentum).add_(p_t.data, alpha=1 - self.gram_update_momentum)

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
            dtype=self.batch_dtype,
            n_global_crops=self.data_global_views_num,
            n_local_crops=self.data_local_views_num,
            n_gram_crops=(self.data_global_views_num if (self.gram_use_loss and not self.gram_ema_teacher) else 0),
            random_circular_shift=self.mask_random_circular_shift,
        )

        transforms = [self.global_transform_1, self.global_transform_2, self.local_transform]
        times = [1, 1, self.data_local_views_num]

        self.train_dataset = datasets(
            operation="train",
            datasets_folder_path=self.data_datasets_path,
            dataset_name=self.data_train_dataset,
            separate_val_subset=self.data_separate_val_subset_use,
            val_size=self.data_separate_val_subset_size,
            transforms=transforms,
            times=times,
        )

        self.train_sampler = DistributedSampler(self.train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)

        loader_kwargs = dict(
            dataset=self.train_dataset,
            batch_size=self.data_batch_size,
            sampler=self.train_sampler,
            num_workers=self.data_num_workers,
            pin_memory=self.data_pin_memory,
            drop_last=self.data_drop_last,
            collate_fn=collate_fn,
        )
        if self.data_num_workers > 0:
            loader_kwargs["prefetch_factor"] = self.data_prefetch_factor
        self.train_dataloader = torch.utils.data.DataLoader(**loader_kwargs)

    def _load_transform(self):
        # Pseudo code of Apendix A from SimCLR paper
        def __get_color_distortion(strength=1.0):
            collor_jitter = v2.ColorJitter(0.4 * strength, 0.4 * strength, 0.2 * strength, 0.1 * strength)
            rnd_color_jitter = v2.RandomApply([collor_jitter], p=0.8)
            rnd_gray = v2.RandomGrayscale(p=0.2)

            return v2.Compose([rnd_color_jitter, rnd_gray])
    
        global_crop_size = self.data_global_views_crop_size
        if self.gram_use_loss and not self.gram_ema_teacher:
            global_crop_size = max(global_crop_size, self.data_gram_teacher_crops_size)

        self.global_transform_1 = v2.Compose([
            v2.RandomResizedCrop(global_crop_size, scale=tuple(self.data_global_views_crop_scale), ratio=tuple(self.data_global_views_crop_ratio), interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(p=0.5),
            __get_color_distortion(strength=1.0),
            v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

        self.global_transform_2 = v2.Compose([
            v2.RandomResizedCrop(global_crop_size, scale=tuple(self.data_global_views_crop_scale), ratio=tuple(self.data_global_views_crop_ratio), interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(p=0.5),
            __get_color_distortion(strength=1.0),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
            v2.RandomSolarize(threshold=128, p=0.2),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

        self.local_transform = v2.Compose([
            v2.RandomResizedCrop(self.data_local_views_crop_size, scale=tuple(self.data_local_views_crop_scale), ratio=tuple(self.data_local_views_crop_ratio), interpolation=v2.InterpolationMode.BICUBIC),
            v2.RandomHorizontalFlip(p=0.5),
            __get_color_distortion(strength=1.0),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

        if self.gram_use_loss and not self.gram_ema_teacher:
            self.global_transform_1 = GlobalCropWithGram(
                self.global_transform_1, self.data_global_views_crop_size, self.data_gram_teacher_crops_size
            )
            self.global_transform_2 = GlobalCropWithGram(
                self.global_transform_2, self.data_global_views_crop_size, self.data_gram_teacher_crops_size
            )

    def _load_models(self):
        builders = {
            "vit_small": vit_small,
            "vit_base": vit_base,
            "vit_large": vit_large,
            "vit_so400m": vit_so400m,
            "vit_huge2": vit_huge2,
            "vit_giant2": vit_giant2,
            "vit_7b": vit_7b,
        }
        if self.meta_model_name not in builders:
            raise ValueError(f"Unsupported DINOv3 model_name: {self.meta_model_name}. Available: {sorted(builders)}")
        builder = builders[self.meta_model_name]

        self.encoder = builder(drop_path_rate=self.meta_drop_path_rate, patch_size=self.meta_patch_size, use_checkpoint=self.meta_checkpoint)
        self.target_encoder = builder(drop_path_rate=0.0, patch_size=self.meta_patch_size, use_checkpoint=False)
        self.gram = None if self.gram_ema_teacher else builder(drop_path_rate=0.0, patch_size=self.meta_patch_size, use_checkpoint=False)

        self.projection_head_cls = dino_head(
            in_dim=self.encoder.get_embed_dim(),
            out_dim=self.meta_projection_head_dino_output_dim,
            use_bn=self.meta_projection_head_dino_use_bn,
            nlayers=self.meta_projection_head_dino_n_layers,
            hidden_dim=self.meta_projection_head_dino_hidden_dim,
            bottleneck_dim=self.meta_projection_head_dino_bottleneck_dim,
        )
        self.projection_head_patch = dino_head(
            in_dim=self.encoder.get_embed_dim(),
            out_dim=self.meta_projection_head_ibot_output_dim,
            use_bn=self.meta_projection_head_ibot_use_bn,
            nlayers=self.meta_projection_head_ibot_n_layers,
            hidden_dim=self.meta_projection_head_ibot_hidden_dim,
            bottleneck_dim=self.meta_projection_head_ibot_bottleneck_dim,
        )
        self.target_projection_head_cls = dino_head(
            in_dim=self.encoder.get_embed_dim(),
            out_dim=self.meta_projection_head_dino_output_dim,
            use_bn=self.meta_projection_head_dino_use_bn,
            nlayers=self.meta_projection_head_dino_n_layers,
            hidden_dim=self.meta_projection_head_dino_hidden_dim,
            bottleneck_dim=self.meta_projection_head_dino_bottleneck_dim,
        )
        self.target_projection_head_patch = dino_head(
            in_dim=self.encoder.get_embed_dim(),
            out_dim=self.meta_projection_head_ibot_output_dim,
            use_bn=self.meta_projection_head_ibot_use_bn,
            nlayers=self.meta_projection_head_ibot_n_layers,
            hidden_dim=self.meta_projection_head_ibot_hidden_dim,
            bottleneck_dim=self.meta_projection_head_ibot_bottleneck_dim,
        )

        if self.meta_pretrained_weights is not None:
            if os.path.exists(self.meta_pretrained_weights):
                self.encoder.load_weights(weight_path=self.meta_pretrained_weights, device=self.device)
            else:
                raise FileNotFoundError(f"Pretrained weights file not found at {self.meta_pretrained_weights}.")

        self.target_encoder.load_state_dict(self.encoder.state_dict())
        self.target_projection_head_cls.load_state_dict(self.projection_head_cls.state_dict())
        self.target_projection_head_patch.load_state_dict(self.projection_head_patch.state_dict())
        if self.gram is not None:
            self.gram.load_state_dict(self.encoder.state_dict())

        if self.continue_training:
            model_dir = os.path.join(self.output_folder, "models")
            if os.path.exists(model_dir):
                self.encoder.load_state_dict(torch.load(os.path.join(model_dir, "encoder.pth"), map_location=self.device))
                self.target_encoder.load_state_dict(torch.load(os.path.join(model_dir, "target_encoder.pth"), map_location=self.device))
                self.projection_head_cls.load_state_dict(torch.load(os.path.join(model_dir, "projection_head_cls.pth"), map_location=self.device))
                self.projection_head_patch.load_state_dict(torch.load(os.path.join(model_dir, "projection_head_patch.pth"), map_location=self.device))
                self.target_projection_head_cls.load_state_dict(torch.load(os.path.join(model_dir, "target_projection_head_cls.pth"), map_location=self.device))
                self.target_projection_head_patch.load_state_dict(torch.load(os.path.join(model_dir, "target_projection_head_patch.pth"), map_location=self.device))
                gram_path = os.path.join(model_dir, "gram.pth")
                if self.gram is not None and os.path.exists(gram_path):
                    self.gram.load_state_dict(torch.load(gram_path, map_location=self.device))
            else:
                raise FileNotFoundError(f"Model checkpoint files not found in {model_dir}.")

        for module in [self.encoder, self.target_encoder, self.projection_head_cls, self.projection_head_patch, self.target_projection_head_cls, self.target_projection_head_patch, self.gram]:
            if module is not None:
                module.to(self.device)

        self.encoder.unfreeze()
        self.projection_head_cls.unfreeze()
        self.projection_head_patch.unfreeze()
        self.target_encoder.freeze()
        self.target_projection_head_cls.freeze()
        self.target_projection_head_patch.freeze()
        if self.gram is not None:
            self.gram.freeze()

        if self.world_size > 1:
            self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
            self.projection_head_cls = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.projection_head_cls)
            self.projection_head_patch = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.projection_head_patch)
            self.encoder = DDP(self.encoder, device_ids=[self.rank], output_device=self.rank)
            self.projection_head_cls = DDP(self.projection_head_cls, device_ids=[self.rank], output_device=self.rank)
            self.projection_head_patch = DDP(self.projection_head_patch, device_ids=[self.rank], output_device=self.rank)
            for target in (self.target_encoder, self.target_projection_head_cls, self.target_projection_head_patch, self.gram):
                if target is not None:
                    for value in target.state_dict().values():
                        torch.distributed.broadcast(value, src=0)

        self.encoder.train()
        self.projection_head_cls.train()
        self.projection_head_patch.train()
        self.target_encoder.eval()
        self.target_projection_head_cls.eval()
        self.target_projection_head_patch.eval()
        if self.gram is not None:
            self.gram.eval()

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
        self.optimization_gram_loss_weight = float(self.config['optimization']['gram_loss_weight'])
        self.optimization_sinkhorn_iterations = int(self.config['optimization']['sinkhorn_iterations'])
        self.optimization_num_register_tokens = int(self.config['optimization']['num_register_tokens'])
        self.optimization_decay_bias = bool(self.config['optimization']['decay_bias'])
        self.optimization_decay_norm = bool(self.config['optimization']['decay_norm'])

        opt_cfg = self.config.get('optimization', {})
        data_cfg = self.config.get('data', {})
        gram_cfg = self.config.get('gram', {})
        self.dino_global_ignore_diagonal = bool(opt_cfg.get('dino_global_ignore_diagonal', True))
        self.mask_random_circular_shift = bool(opt_cfg.get('mask_random_circular_shift', False))
        self.gram_use_loss = bool(gram_cfg.get('use_loss', self.optimization_gram_loss_weight > 0))
        self.gram_ema_teacher = bool(gram_cfg.get('ema_teacher', gram_cfg.get('use_ema_teacher', True)))
        self.gram_img_level = bool(gram_cfg.get('img_level', False))
        self.gram_tokens_used = str(gram_cfg.get('tokens_used', 'all'))
        if self.gram_tokens_used not in {'all', 'masked', 'unmasked'}:
            raise ValueError("gram.tokens_used must be one of: all, masked, unmasked")
        self.gram_normalized = bool(gram_cfg.get('normalized', True))
        self.gram_remove_neg = bool(gram_cfg.get('remove_neg', True))
        self.gram_remove_only_teacher_neg = bool(gram_cfg.get('remove_only_teacher_neg', False))
        self.gram_compute_stats = bool(gram_cfg.get('compute_stats', False))
        self.gram_rep_update = bool(gram_cfg.get('rep_update', False))
        self.gram_update_momentum = float(gram_cfg.get('update_momentum', 0.0))
        self.gram_global_teacher_resize_method = str(gram_cfg.get('global_teacher_resize_method', 'bicubic'))
        self.gram_global_teacher_resize_antialias = bool(gram_cfg.get('global_teacher_resize_antialias', False))
        gram_crop_cfg = data_cfg.get('gram_teacher_crops', {}) if isinstance(data_cfg.get('gram_teacher_crops', {}), dict) else {}
        self.data_gram_teacher_crops_size = int(gram_crop_cfg.get('crop_size', self.data_global_views_crop_size))
        dtype_name = str(opt_cfg.get('batch_dtype', opt_cfg.get('amp_dtype', 'fp16'))).lower()
        self.batch_dtype = {'fp32': torch.float32, 'float32': torch.float32, 'fp16': torch.float16, 'float16': torch.float16, 'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16}.get(dtype_name, torch.float16)

        self.data_datasets_path += "/" if not self.data_datasets_path.endswith("/") else ""
