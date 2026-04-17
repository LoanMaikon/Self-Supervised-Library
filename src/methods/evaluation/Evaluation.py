from unittest import case

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import torch.distributed as dist
import torch.optim as optim
import torch
import json
import os

from src.methods.byol.resnet import resnet50 as byol_resnet50, resnet200 as byol_resnet200
from src.methods.simclr.resnet import resnet50 as simclr_resnet50
from src.methods.swav.resnet import resnet50 as swav_resnet50
from src.methods.ijepa.models import vit_tiny as ijepa_vit_tiny, vit_small as ijepa_vit_small, vit_base as ijepa_vit_base, \
    vit_large as ijepa_vit_large, vit_huge as ijepa_vit_huge, vit_giant as ijepa_vit_giant
from src.methods.mae.models import mae_vit_base_patch16, mae_vit_large_patch16, mae_vit_huge_patch14, mae_vit_small_patch16, mae_vit_tiny_patch16
from src.methods.dinov1.models import vit_tiny as dinov1_vit_tiny, vit_small as dinov1_vit_small, vit_base as dinov1_vit_base
from .resnet50 import resnet50 as resnet50_eval

from src.utils import write_on_log, plot_fig, write_on_csv, save_json, is_main_process, \
    recreate_csv_log, get_last_epoch, load_last_values
from src.schedulers import WarmupCosineSchedule, CosineWDSchedule
from .linear_head import LinearHead
from src.datasets import datasets

class Evaluation():
    def __init__(self,
                 opened_config,
                 output_folder,
                 device,
                 rank,
                 world_size,
                 evaluate_weights,
                 continue_training,
                ):
    
        self.config = opened_config
        self.output_folder = output_folder
        self.device = device
        self.rank = rank
        self.world_size = world_size
        self.evaluate_weights = evaluate_weights
        self.continue_training = continue_training

        self._load_config()
        self._load_transform()
        self._load_dataloader()
        self._load_models()
        self._load_criterion()
        self._load_optimizer()
        self._load_schedulers()

        self.scaler = torch.amp.GradScaler()

        self.train_loss = []
        if self.has_val():
            self.val_loss = []
            self.val_accuracy = []
        self.lr_values = []
        self.wd_values = []

        if self.continue_training:
            self.last_epoch = get_last_epoch(self.output_folder)
            self.optimizer.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"optimizer.pth"), map_location=self.device))
            self.lr_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"lr_scheduler.pth"), map_location=self.device))
            self.wd_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"wd_scheduler.pth"), map_location=self.device))
            self.scaler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "scaler.pth"), map_location=self.device))
            recreate_csv_log(self.output_folder, self.last_epoch)
            self.lr_values, self.wd_values, _, self.train_loss = load_last_values(self.output_folder, self.last_epoch)

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

            self.encoder.train() if self.meta_mode == "fine_tuning" else self.encoder.eval()
            self.linear_head.train()
            for iteration, (images, labels) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                images = images[0].to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    if self.meta_mode == "linear_eval":
                        with torch.no_grad():
                            features = self.encoder(images)
                            features = self.encoder.get_features(features)
                    else:
                        features = self.encoder(images)
                        features = self.encoder.get_features(features)
                    output = self.linear_head(features)
                    loss = self.apply_criterion(output, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_value = loss.item()
                self.train_loss[-1] += loss_value * images.size(0)
                num_samples += images.size(0)

                self.lr_values.append(self.lr_scheduler.get_value())
                self.wd_values.append(self.wd_scheduler.get_value())

                write_on_csv(self.output_folder, epoch, iteration, loss_value, self.lr_values[-1], self.wd_values[-1])

                self.lr_scheduler.step()
                self.wd_scheduler.step()

            self.train_loss[-1] /= num_samples

            write_on_log(f"Train loss: {self.train_loss[-1]}", self.output_folder)

            plot_fig(range(len(self.train_loss)), "Epoch", self.train_loss, "Loss", f"loss", self.output_folder)
            plot_fig(range(len(self.lr_values)), "Iteration", self.lr_values, "Learning Rate", f"learning_rate", self.output_folder)
            plot_fig(range(len(self.wd_values)), "Iteration", self.wd_values, "Weight Decay", f"weight_decay", self.output_folder)

            if self.has_val():
                self.val_loss.append(0.0)
                self.val_accuracy.append(0.0)

                self.encoder.eval()
                self.linear_head.eval()

                num_samples = 0

                with torch.no_grad():
                    for (images, labels) in self.val_dataloader:
                        images = images[0].to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)

                        features = self.encoder(images)
                        features = self.encoder.get_features(features)
                        output = self.linear_head(features)

                        loss = self.apply_criterion(output, labels)
                        accuracy = (output.argmax(dim=1) == labels).float().mean().item()

                        num_samples += images.size(0)

                        self.val_loss[-1] += loss.item() * images.size(0)
                        self.val_accuracy[-1] += accuracy * images.size(0)

                if dist.is_available() and dist.is_initialized():
                    val_metrics = torch.tensor(
                        [self.val_loss[-1], self.val_accuracy[-1], float(num_samples)],
                        device=self.device,
                        dtype=torch.float64,
                    )
                    dist.all_reduce(val_metrics, op=dist.ReduceOp.SUM)
                    self.val_loss[-1], self.val_accuracy[-1], num_samples = val_metrics.tolist()
                    num_samples = int(num_samples)

                self.val_loss[-1] /= num_samples
                self.val_accuracy[-1] /= num_samples

                write_on_log(f"Validation loss: {self.val_loss[-1]}", self.output_folder)
                write_on_log(f"Validation accuracy: {self.val_accuracy[-1]}", self.output_folder)

                plot_fig(range(len(self.val_loss)), "Epoch", self.val_loss, "Loss", f"val_loss", self.output_folder)
                plot_fig(range(len(self.val_accuracy)), "Epoch", self.val_accuracy, "Accuracy", f"val_accuracy", self.output_folder)

                save_json({
                    "train_loss": self.train_loss,
                    "val_loss": self.val_loss,
                    "val_accuracy": self.val_accuracy
                }, self.output_folder, "training_info")
            else:
                save_json({
                    "train_loss": self.train_loss,
                }, self.output_folder, "training_info")

            self.save_models(epoch)

            save_json({"last_epoch": epoch}, self.output_folder, "last_epoch")

            write_on_log("", self.output_folder)
    
    def test(self):
        write_on_log("Starting testing...", self.output_folder)

        if os.path.exists(os.path.join(self.output_folder, "test_results.json")):
            write_on_log("Test results already exist. Skipping testing.", self.output_folder)
            return

        self.encoder.eval()
        self.linear_head.eval()

        test_loss = 0.0
        test_accuracy = 0.0
        num_samples = 0

        with torch.no_grad():
            for (images, labels) in self.test_dataloader:
                images = images[0].to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                features = self.encoder(images)
                features = self.encoder.get_features(features)
                output = self.linear_head(features)

                loss = self.apply_criterion(output, labels)
                accuracy = (output.argmax(dim=1) == labels).float().mean().item()

                test_loss += loss.item() * images.size(0)
                test_accuracy += accuracy * images.size(0)
                num_samples += images.size(0)

        if dist.is_available() and dist.is_initialized():
            metrics = torch.tensor([test_loss, test_accuracy, float(num_samples)], device=self.device)
            dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
            test_loss, test_accuracy, num_samples = metrics.tolist()
            num_samples = int(num_samples)

        test_loss /= num_samples
        test_accuracy /= num_samples

        write_on_log(f"Test loss: {test_loss}", self.output_folder)
        write_on_log(f"Test accuracy: {test_accuracy}", self.output_folder)

        save_json({"test_loss": test_loss, "test_accuracy": test_accuracy}, self.output_folder, "test_results")

    def save_models(self, epoch):
        if not is_main_process():
            return
    
        os.makedirs(os.path.join(self.output_folder, "models"), exist_ok=True)

        encoder_state_dict = self.encoder.module.state_dict() if self.world_size > 1 and self.meta_mode == "fine_tuning" else self.encoder.state_dict()
        linear_head_state_dict = self.linear_head.module.state_dict() if self.world_size > 1 else self.linear_head.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        lr_scheduler_state_dict = self.lr_scheduler.state_dict()
        wd_scheduler_state_dict = self.wd_scheduler.state_dict()
        scaler_state_dict = self.scaler.state_dict()

        torch.save(encoder_state_dict, os.path.join(self.output_folder, "models", "encoder.pth"))
        torch.save(linear_head_state_dict, os.path.join(self.output_folder, "models", "linear_head.pth"))
        torch.save(optimizer_state_dict, os.path.join(self.output_folder, "models", "optimizer.pth"))
        torch.save(lr_scheduler_state_dict, os.path.join(self.output_folder, "models", "lr_scheduler.pth"))
        torch.save(wd_scheduler_state_dict, os.path.join(self.output_folder, "models", "wd_scheduler.pth"))
        torch.save(scaler_state_dict, os.path.join(self.output_folder, "models", "scaler.pth"))

        if self.meta_save_every > 0 and epoch % self.meta_save_every == 0:
            torch.save(encoder_state_dict, os.path.join(self.output_folder, "models", f"encoder_epoch_{epoch}.pth"))
            torch.save(linear_head_state_dict, os.path.join(self.output_folder, "models", f"linear_head_epoch_{epoch}.pth"))
            torch.save(optimizer_state_dict, os.path.join(self.output_folder, "models", f"optimizer_epoch_{epoch}.pth"))
            torch.save(lr_scheduler_state_dict, os.path.join(self.output_folder, "models", f"lr_scheduler_epoch_{epoch}.pth"))
            torch.save(wd_scheduler_state_dict, os.path.join(self.output_folder, "models", f"wd_scheduler_epoch_{epoch}.pth"))
            torch.save(scaler_state_dict, os.path.join(self.output_folder, "models", f"scaler_epoch_{epoch}.pth"))

    def _load_models(self):
        def __try_load_models():
            errors = []

            if self.evaluate_weights == "supervised_resnet50":
                self.encoder = resnet50_eval(use_checkpoint=self.meta_checkpoint, pretrained=True)
                self.model_type = "supervised_resnet50"
                return
            elif self.evaluate_weights == "random_resnet50":
                self.encoder = resnet50_eval(use_checkpoint=self.meta_checkpoint, pretrained=False)
                self.model_type = "random_resnet50"
                return

            match self.meta_framework:
                case "simclr":
                    try:
                        self.encoder = simclr_resnet50(self.meta_checkpoint)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "simclr_resnet50"
                        return
                    except Exception as e:
                        errors.append(("simclr_resnet50", str(e)))

                case "byol":
                    try:
                        self.encoder = byol_resnet50(self.meta_checkpoint)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "byol_resnet50"
                        return
                    except Exception as e:
                        errors.append(("byol_resnet50", str(e)))
                    
                    try:
                        self.encoder = byol_resnet200(self.meta_checkpoint)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "byol_resnet200"
                        return
                    except Exception as e:
                        errors.append(("byol_resnet200", str(e)))
            
                case "ijepa":
                    try:
                        self.encoder = ijepa_vit_tiny(checkpoint=self.meta_checkpoint, patch_size=16)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "ijepa_vit_tiny_16"
                        return
                    except Exception as e:
                        errors.append(("ijepa_vit_tiny_16", str(e)))
                    
                    try:
                        self.encoder = ijepa_vit_tiny(checkpoint=self.meta_checkpoint, patch_size=14)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "ijepa_vit_tiny_14"
                        return
                    except Exception as e:
                        errors.append(("ijepa_vit_tiny_14", str(e)))

                    try:
                        self.encoder = ijepa_vit_small(checkpoint=self.meta_checkpoint, patch_size=16)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "ijepa_vit_small_16"
                        return
                    except Exception as e:
                        errors.append(("ijepa_vit_small_16", str(e)))
                    
                    try:
                        self.encoder = ijepa_vit_small(checkpoint=self.meta_checkpoint, patch_size=14)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "ijepa_vit_small_14"
                        return
                    except Exception as e:
                        errors.append(("ijepa_vit_small_14", str(e)))

                    try:
                        self.encoder = ijepa_vit_base(checkpoint=self.meta_checkpoint, patch_size=16)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "ijepa_vit_base_16"
                        return
                    except Exception as e:
                        errors.append(("ijepa_vit_base_16", str(e)))
                    
                    try:
                        self.encoder = ijepa_vit_base(checkpoint=self.meta_checkpoint, patch_size=14)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "ijepa_vit_base_14"
                        return
                    except Exception as e:
                        errors.append(("ijepa_vit_base_14", str(e)))

                    try:
                        self.encoder = ijepa_vit_large(checkpoint=self.meta_checkpoint, patch_size=16)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "ijepa_vit_large_16"
                        return
                    except Exception as e:
                        errors.append(("ijepa_vit_large_16", str(e)))

                    try:
                        self.encoder = ijepa_vit_large(checkpoint=self.meta_checkpoint, patch_size=14)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "ijepa_vit_large_14"
                        return
                    except Exception as e:
                        errors.append(("ijepa_vit_large_14", str(e)))

                    try:
                        self.encoder = ijepa_vit_huge(checkpoint=self.meta_checkpoint, patch_size=16)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "ijepa_vit_huge_16"
                        return
                    except Exception as e:
                        errors.append(("ijepa_vit_huge_16", str(e)))
                    
                    try:
                        self.encoder = ijepa_vit_huge(checkpoint=self.meta_checkpoint, patch_size=14)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "ijepa_vit_huge_14"
                        return
                    except Exception as e:
                        errors.append(("ijepa_vit_huge_14", str(e)))

                    try:
                        self.encoder = ijepa_vit_giant(checkpoint=self.meta_checkpoint, patch_size=16)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "ijepa_vit_giant_16"
                        return
                    except Exception as e:
                        errors.append(("ijepa_vit_giant_16", str(e)))
                    
                    try:
                        self.encoder = ijepa_vit_giant(checkpoint=self.meta_checkpoint, patch_size=14)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "ijepa_vit_giant_14"
                        return
                    except Exception as e:
                        errors.append(("ijepa_vit_giant_14", str(e)))
            
                case "swav":
                    try:
                        self.encoder = swav_resnet50(self.meta_checkpoint)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "swav_resnet50"
                        return
                    except Exception as e:
                        errors.append(("swav_resnet50", str(e)))

                case "mae":
                    try:
                        self.encoder = mae_vit_tiny_patch16(image_size=self.data_crop_size, use_checkpoint=self.meta_checkpoint)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "mae_vit_tiny_patch16"
                        return
                    except Exception as e:
                        errors.append(("mae_vit_tiny_patch16", str(e)))

                    try:
                        self.encoder = mae_vit_small_patch16(image_size=self.data_crop_size, use_checkpoint=self.meta_checkpoint)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "mae_vit_small_patch16"
                        return
                    except Exception as e:
                        errors.append(("mae_vit_small_patch16", str(e)))

                    try:
                        self.encoder = mae_vit_base_patch16(image_size=self.data_crop_size, use_checkpoint=self.meta_checkpoint)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "mae_vit_base_patch16"
                        return
                    except Exception as e:
                        errors.append(("mae_vit_base_patch16", str(e)))

                    try:
                        self.encoder = mae_vit_large_patch16(image_size=self.data_crop_size, use_checkpoint=self.meta_checkpoint)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "mae_vit_large_patch16"
                        return
                    except Exception as e:
                        errors.append(("mae_vit_large_patch16", str(e)))

                    try:
                        self.encoder = mae_vit_huge_patch14(image_size=self.data_crop_size, use_checkpoint=self.meta_checkpoint)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "mae_vit_huge_patch14"
                        return
                    except Exception as e:
                        errors.append(("mae_vit_huge_patch14", str(e)))

                case "dinov1":
                    try:
                        self.encoder = dinov1_vit_tiny(checkpoint=self.meta_checkpoint, patch_size=16)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "dinov1_vit_tiny_16"
                        return
                    except Exception as e:
                        errors.append(("dinov1_vit_tiny_16", str(e)))

                    try:
                        self.encoder = dinov1_vit_small(use_checkpoint=self.meta_checkpoint, patch_size=16)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "dinov1_vit_small_16"
                        return
                    except Exception as e:
                        errors.append(("dinov1_vit_small_16", str(e)))

                    try:
                        self.encoder = dinov1_vit_base(checkpoint=self.meta_checkpoint, patch_size=16)
                        self.encoder.load_weights(self.evaluate_weights, device=self.device)
                        self.model_type = "dinov1_vit_base_16"
                        return
                    except Exception as e:
                        errors.append(("dinov1_vit_base_16", str(e)))
                
                case _:
                    errors.append(("unknown_framework", f"Unsupported framework: {self.meta_framework}"))

            raise ValueError(
                f"Failed to load weights from {self.evaluate_weights}. Errors: {errors}."
            )

        __try_load_models()

        if self.meta_framework == "mae":
            self.linear_head = LinearHead(self.encoder.get_output_dim(), self.train_dataset.get_num_classes(), batch_norm=True).to(self.device)
        else:
            self.linear_head = LinearHead(self.encoder.get_output_dim(), self.train_dataset.get_num_classes()).to(self.device)
        self.linear_head.unfreeze()
        
        match self.meta_mode:
            case "linear_eval":
                self.encoder.freeze()
            
            case "fine_tuning":
                self.encoder.unfreeze()
        
            case _:
                raise ValueError(f"Unsupported mode: {self.meta_mode}")
            
        self.encoder.remove_classifier_head()

        if self.continue_training:
            if os.path.exists(os.path.join(self.output_folder, "models")):
                self.encoder.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "encoder.pth"), map_location=self.device))
                self.linear_head.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "linear_head.pth"), map_location=self.device))
            else:
                raise ValueError(f"Checkpoint directory not found in {os.path.join(self.output_folder, 'models')}")
        
        self.encoder.to(self.device)
        self.linear_head.to(self.device)

        if self.world_size > 1:
            if self.meta_mode == "fine_tuning":
                self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
                self.encoder = DDP(self.encoder, device_ids=[self.rank], output_device=self.rank)
            self.linear_head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.linear_head)
            self.linear_head = DDP(self.linear_head, device_ids=[self.rank], output_device=self.rank)

    def _load_transform(self):
        self.train_transform = v2.Compose([
            v2.RandomApply(
                [v2.RandomResizedCrop(
                    size=self.data_crop_size,
                    scale=self.data_random_resized_crop_scale,
                    ratio=self.data_random_resized_crop_ratio
                )],
                p=self.data_random_resized_crop_p
            ) if self.data_random_resized_crop_use else v2.Resize((self.data_crop_size, self.data_crop_size)),

            v2.RandomHorizontalFlip(p=self.data_horizontal_flip_p) if self.data_horizontal_flip_use else v2.Identity(),

            v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True)
            ]),

            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

        test_resize_size = int(round(self.data_crop_size * 256 / 224))
        self.test_transform = v2.Compose([
            v2.Resize(test_resize_size),
            v2.CenterCrop(self.data_crop_size),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

    def _load_dataloader(self):
        self.train_dataset = datasets(
            operation="train",
            datasets_folder_path=self.data_datasets_path,
            dataset_name=self.data_train_dataset,
            separate_val_subset=self.data_separate_val_subset_use,
            val_size=self.data_separate_val_subset_size,
            transforms=[self.train_transform],
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
            drop_last=self.data_drop_last
        )

        if self.data_separate_val_subset_use:
            self.val_dataset = datasets(
                operation="val",
                datasets_folder_path=self.data_datasets_path,
                dataset_name=self.data_train_dataset,
                separate_val_subset=self.data_separate_val_subset_use,
                val_size=self.data_separate_val_subset_size,
                transforms=[self.test_transform],
                times=[1]
            )

            self.val_sampler = DistributedSampler(self.val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)

            self.val_dataloader = torch.utils.data.DataLoader(
                self.val_dataset,
                batch_size=self.data_batch_size,
                sampler=self.val_sampler,
                num_workers=self.data_num_workers,
                prefetch_factor=self.data_prefetch_factor,
                pin_memory=self.data_pin_memory,
                drop_last=False
            )
        
        self.test_dataset = datasets(
            operation="test",
            datasets_folder_path=self.data_datasets_path,
            dataset_name=self.data_train_dataset,
            separate_val_subset=self.data_separate_val_subset_use,
            val_size=self.data_separate_val_subset_size,
            transforms=[self.test_transform],
            times=[1]
        )

        self.test_sampler = DistributedSampler(self.test_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)

        self.test_dataloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.data_batch_size,
            sampler=self.test_sampler,
            num_workers=self.data_num_workers,
            prefetch_factor=self.data_prefetch_factor,
            pin_memory=self.data_pin_memory,
            drop_last=False
        )
    
    def has_val(self):
        return self.data_separate_val_subset_use

    def _load_criterion(self):
        match self.optimization_criterion:
            case "cross_entropy":
                self.criterion = torch.nn.CrossEntropyLoss()

            case _:
                raise ValueError(f"Unsupported criterion: {self.optimization_criterion}")
    
    def apply_criterion(self, output, labels):
        match self.optimization_criterion:
            case "cross_entropy":
                return self.criterion(output, labels)

            case _:
                raise ValueError(f"Unsupported criterion: {self.optimization_criterion}")

    def _load_optimizer(self):
        match self.optimization_optimizer:
            case "sgd":
                decay_params = []
                no_decay_params = []

                if self.meta_mode == "linear_eval":
                    modules = [self.linear_head] if self.world_size == 1 else [self.linear_head.module]
                else:
                    modules = [self.encoder, self.linear_head] if self.world_size == 1 else [self.encoder.module, self.linear_head.module]

                for module in modules:
                    for name, p in module.named_parameters():
                        if not p.requires_grad:
                            continue

                        if p.ndim > 1:
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

                self.optimizer = optim.SGD(
                    param_groups,
                    lr=self.optimization_lr[0],
                    momentum=0.9,
                )

            case _:
                raise ValueError(f"Unsupported optimizer: {self.optimization_optimizer}")

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

    def find_last_epoch(self):
        last_epoch_path = os.path.join(self.output_folder, "last_epoch.json")
        if not os.path.exists(last_epoch_path):
            return 0
        
        with open(last_epoch_path, "r") as f:
            last_epoch_data = json.load(f)
        
        return last_epoch_data.get("last_epoch", 0)

    def step_schedulers_to_epoch(self, epoch):
        if epoch == 0:
            return
        
        steps_per_epoch = len(self.train_dataloader)
        total_steps = epoch * steps_per_epoch

        for _ in range(total_steps):
            self.lr_scheduler.step()
            self.wd_scheduler.step()
    
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
        self.data_horizontal_flip_use = bool(self.config["data"]["horizontal_flip"]["use"])
        self.data_horizontal_flip_p = float(self.config["data"]["horizontal_flip"]["p"])
        self.data_random_resized_crop_use = bool(self.config["data"]["random_resized_crop"]["use"])
        self.data_random_resized_crop_p = float(self.config["data"]["random_resized_crop"]["p"])
        self.data_random_resized_crop_scale = list(map(float, self.config["data"]["random_resized_crop"]["scale"]))
        self.data_random_resized_crop_ratio = list(map(float, self.config["data"]["random_resized_crop"]["ratio"]))

        self.meta_checkpoint = bool(self.config["meta"]["checkpoint"])
        self.meta_mode = str(self.config["meta"]["mode"])
        self.meta_pretrained_weights = self.config["meta"]["pretrained_weights"]
        self.meta_save_every = int(self.config["meta"]["save_every"])
        self.meta_framework = self.config["meta"]["framework"]

        self.optimization_ipe_scale = float(self.config["optimization"]["ipe_scale"])
        self.optimization_lr = list(map(float, self.config["optimization"]["lr"]))
        self.optimization_weight_decay = list(map(float, self.config["optimization"]["weight_decay"]))
        self.optimization_epochs = int(self.config["optimization"]["epochs"])
        self.optimization_warmup_epochs = int(self.config["optimization"]["warmup_epochs"])
        self.optimization_optimizer = str(self.config["optimization"]["optimizer"])
        self.optimization_criterion = str(self.config["optimization"]["criterion"])

        self.data_datasets_path += "/" if not self.data_datasets_path.endswith("/") else ""
