from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from torchvision.transforms import v2
import torch.optim as optim
from PIL import Image
import torch
import copy
import os

from src.utils import write_on_log, plot_fig, write_on_csv, save_json, is_main_process, \
    recreate_csv_log, get_last_epoch, load_last_values
from src.schedulers import WarmupCosineSchedule, CosineWDSchedule, EMACosineSchedule
from src.datasets import datasets
from src.lars import LARS
from .models import Model_barlow_twins

class BarlowTwins():
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
        self.lr_values_weights = []
        self.wd_values = []

        if self.continue_training:
            self.last_epoch = get_last_epoch(self.output_folder)
            self.optimizer.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"optimizer.pth"), map_location=self.device))
            self.lr_scheduler_weights.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"lr_scheduler_weights.pth"), map_location=self.device))
            self.lr_scheduler_biases.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"lr_scheduler_biases.pth"), map_location=self.device))
            self.wd_scheduler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", f"wd_scheduler.pth"), map_location=self.device))
            self.scaler.load_state_dict(torch.load(os.path.join(self.output_folder, "models", "scaler.pth"), map_location=self.device))
            recreate_csv_log(self.output_folder, self.last_epoch)
            self.lr_values_weights, self.wd_values, _, self.train_loss = load_last_values(self.output_folder, self.last_epoch)

            write_on_log(f"Continuing training from epoch {self.last_epoch}...", self.output_folder)

    
    def train(self):
        write_on_log("Starting training...", self.output_folder)

        for epoch in range(1, self.optimization_num_epochs + 1):
            if self.continue_training and epoch <= self.last_epoch:
                continue

            write_on_log(f"Epoch {epoch}/{self.optimization_num_epochs}", self.output_folder)
            self.train_sampler.set_epoch(epoch)

            self.train_loss.append(0.0)
            num_samples = 0

            for iteration, (images, _) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                x1, x2 = images[0].to(self.device, non_blocking=True), images[1].to(self.device, non_blocking=True)
                self.adjust_learning_rate_biases()

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    loss = self.model(x1, x2)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                loss_value = loss.item()
                self.train_loss[-1] += loss_value * x1.size(0)
                num_samples += x1.size(0)

                self.lr_values_weights.append(self.lr_scheduler_weights.get_value())
                self.wd_values.append(self.wd_scheduler.get_value())

                write_on_csv(self.output_folder, epoch, iteration, loss_value, self.lr_values_weights[-1], self.wd_values[-1])

                self.lr_scheduler_weights.step()
                self.lr_scheduler_biases.step()
                self.wd_scheduler.step()
            
            self.train_loss[-1] /= num_samples

            self.save_models(epoch)

            write_on_log(f"Loss: {self.train_loss[-1]}", self.output_folder)

            plot_fig(range(len(self.train_loss)), "Epoch", self.train_loss, "Loss", f"loss", self.output_folder)
            plot_fig(range(len(self.lr_values_weights)), "Iteration", self.lr_values_weights, "Learning Rate", f"learning_rate", self.output_folder)
            plot_fig(range(len(self.wd_values)), "Iteration", self.wd_values, "Weight Decay", f"weight_decay", self.output_folder)

            save_json({"train_loss": self.train_loss}, self.output_folder, "training_info")

            save_json({"last_epoch": epoch}, self.output_folder, "last_epoch")

            write_on_log("", self.output_folder)

    def save_models(self, epoch):
        if not is_main_process():
            return

        os.makedirs(os.path.join(self.output_folder, "models"), exist_ok=True)

        model_state_dict = self.model.module.state_dict() if self.world_size > 1 else self.model.state_dict()
        optimizer_state_dict = self.optimizer.state_dict()
        lr_scheduler_weights_state_dict = self.lr_scheduler_weights.state_dict()
        lr_scheduler_biases_state_dict = self.lr_scheduler_biases.state_dict()
        wd_scheduler_state_dict = self.wd_scheduler.state_dict()
        scaler_state_dict = self.scaler.state_dict()

        torch.save({"model_state_dict": model_state_dict}, os.path.join(self.output_folder, "models", "model.pth"))
        torch.save(optimizer_state_dict, os.path.join(self.output_folder, "models", "optimizer.pth"))
        torch.save(lr_scheduler_weights_state_dict, os.path.join(self.output_folder, "models", "lr_scheduler_weights.pth"))
        torch.save(lr_scheduler_biases_state_dict, os.path.join(self.output_folder, "models", "lr_scheduler_biases.pth"))
        torch.save(wd_scheduler_state_dict, os.path.join(self.output_folder, "models", "wd_scheduler.pth"))
        torch.save(scaler_state_dict, os.path.join(self.output_folder, "models", "scaler.pth"))

        if self.meta_save_every > 0 and epoch % self.meta_save_every == 0:
            torch.save({"model_state_dict": model_state_dict}, os.path.join(self.output_folder, "models", f"model_epoch_{epoch}.pth"))
            torch.save(optimizer_state_dict, os.path.join(self.output_folder, "models", f"optimizer_epoch_{epoch}.pth"))
            torch.save(lr_scheduler_weights_state_dict, os.path.join(self.output_folder, "models", f"lr_scheduler_weights_epoch_{epoch}.pth"))
            torch.save(lr_scheduler_biases_state_dict, os.path.join(self.output_folder, "models", f"lr_scheduler_biases_epoch_{epoch}.pth"))
            torch.save(wd_scheduler_state_dict, os.path.join(self.output_folder, "models", f"wd_scheduler_epoch_{epoch}.pth"))
            torch.save(scaler_state_dict, os.path.join(self.output_folder, "models", f"scaler_epoch_{epoch}.pth"))

    def adjust_learning_rate_biases(self):
        for param_group in self.optimizer.param_groups:
            if "bias" in param_group and param_group["bias"] == True:
                 param_group["lr"] = self.lr_scheduler_biases.get_value()

    def _load_schedulers(self):
        self.lr_scheduler_weights = WarmupCosineSchedule(
            optimizer=self.optimizer,
            warmup_steps=self.optimization_warmup_epochs * len(self.train_dataloader),
            start_lr=self.optimization_lr_weights[0],
            middle_lr=self.optimization_lr_weights[1],
            final_lr=self.optimization_lr_weights[2],
            T_max=self.optimization_num_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
            param_group_filter=lambda group: not group.get("bias", False)
        )

        self.lr_scheduler_biases = WarmupCosineSchedule(
            optimizer=self.optimizer,
            warmup_steps=self.optimization_warmup_epochs * len(self.train_dataloader),
            start_lr=self.optimization_lr_biases[0],
            middle_lr=self.optimization_lr_biases[1],
            final_lr=self.optimization_lr_biases[2],
            T_max=self.optimization_num_epochs * len(self.train_dataloader) * self.optimization_ipe_scale,
            param_group_filter=lambda group: group.get("bias", False)
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
                param_weights = []
                param_biases = []
                for param in self.model.parameters():
                    if not param.requires_grad:
                        continue
                    if param.ndim == 1:
                        param_biases.append(param)
                    else:
                        param_weights.append(param)

                self.optimizer = LARS(
                    [
                        {"params": param_weights, "lr": self.optimization_lr_weights[0]},
                        {
                            "params": param_biases,
                            "lr": self.optimization_lr_biases[0],
                            "bias": True,
                            "WD_exclude": True,
                            "weight_decay": 0.0,
                        },
                    ],
                    lr=self.optimization_lr_weights[0],
                    weight_decay=self.optimization_weight_decay[0],
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
            transforms=[self.transform1, self.transform2],
            times=[1, 1]
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
        def __get_color_distortion(strength=1.0):
            collor_jitter = v2.ColorJitter(0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength)
            rnd_color_jitter = v2.RandomApply([collor_jitter], p=0.8)
            rnd_gray = v2.RandomGrayscale(p=0.2)

            return v2.Compose([rnd_color_jitter, rnd_gray])

        self.transform1 = v2.Compose([
            v2.RandomResizedCrop(self.data_crop_size, scale=self.data_crop_scale, ratio=self.data_crop_ratio, interpolation=Image.BICUBIC),
            v2.RandomHorizontalFlip(0.5) if self.data_horizontal_flip else v2.Identity(),
            __get_color_distortion(self.data_color_jitter) if self.data_color_jitter else v2.Identity(),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.9) if self.data_gaussian_blur else v2.Identity(),
            v2.RandomSolarize(threshold=128, p=0.0) if self.data_solarization else v2.Identity(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

        self.transform2 = v2.Compose([
            v2.RandomResizedCrop(self.data_crop_size, scale=self.data_crop_scale, ratio=self.data_crop_ratio, interpolation=Image.BICUBIC),
            v2.RandomHorizontalFlip(0.5) if self.data_horizontal_flip else v2.Identity(),
            __get_color_distortion(self.data_color_jitter) if self.data_color_jitter else v2.Identity(),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1) if self.data_gaussian_blur else v2.Identity(),
            v2.RandomSolarize(threshold=128, p=0.2) if self.data_solarization else v2.Identity(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

    def _load_models(self):
        match self.meta_model_name:
            case "resnet50":
                self.model = Model_barlow_twins(projection_head_dims=self.meta_projector, batch_size=self.data_batch_size, world_size=self.world_size, lambd=self.optimization_lambda)
            case _:
                raise ValueError(f"Unsupported model name: {self.meta_model_name}")
        
        if self.meta_pretrained_weights is not None:
            if os.path.exists(self.meta_pretrained_weights):
                self.model.load_weights(
                    weight_path=self.meta_pretrained_weights,
                    device=self.device
                )
            else:
                raise FileNotFoundError(f"Pretrained weights file not found at {self.meta_pretrained_weights}.")
        
        if self.continue_training:
            if os.path.exists(os.path.join(self.output_folder, "models")):
                self.model.load_weights(os.path.join(self.output_folder, "models", "model.pth"), device=self.device)
            else:
                raise FileNotFoundError("Checkpoint files not found for continuing training.")
        
        self.model.unfreeze_all()
        self.model.to(self.device)

        if self.world_size > 1:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.rank], output_device=self.rank)
        
        self.model.train()

    def _load_config(self):
        self.data_datasets_path = str(self.config["data"]["datasets_path"])
        self.data_train_dataset = str(self.config["data"]["train_dataset"])
        self.data_batch_size = int(self.config["data"]["batch_size"])
        self.data_crop_size = int(self.config["data"]["crop_size"])
        self.data_num_workers = int(self.config["data"]["num_workers"])
        self.data_prefetch_factor = int(self.config["data"]["prefetch_factor"])
        self.data_pin_memory = bool(self.config["data"]["pin_memory"])
        self.data_drop_last = bool(self.config["data"]["drop_last"])
        self.data_crop_scale = self.config["data"]["crop_scale"]
        self.data_crop_ratio = self.config["data"]["crop_ratio"]
        self.data_color_jitter = bool(self.config["data"]["color_jitter"])
        self.data_gaussian_blur = bool(self.config["data"]["gaussian_blur"])
        self.data_horizontal_flip = bool(self.config["data"]["horizontal_flip"])
        self.data_solarization = bool(self.config["data"]["solarization"])
        self.data_normalize_mean = list(map(float, self.config["data"]["normalize"]["mean"]))
        self.data_normalize_std = list(map(float, self.config["data"]["normalize"]["std"]))
        self.data_separate_val_subset_use = bool(self.config["data"]["separate_val_subset"]["use"])
        self.data_separate_val_subset_size = float(self.config["data"]["separate_val_subset"]["size"])

        self.meta_model_name = str(self.config["meta"]["model_name"])
        self.meta_checkpoint = bool(self.config["meta"]["checkpoint"])
        self.meta_pretrained_weights = self.config["meta"]["pretrained_weights"]
        self.meta_save_every = int(self.config["meta"]["save_every"])
        self.meta_projector = list(map(int, self.config["meta"]["projector"]))

        self.optimization_ipe_scale = float(self.config["optimization"]["ipe_scale"])
        self.optimization_lr_weights = list(map(float, self.config["optimization"]["lr_weights"]))
        self.optimization_lr_biases = list(map(float, self.config["optimization"]["lr_biases"]))
        self.optimization_weight_decay = list(map(float, self.config["optimization"]["weight_decay"]))
        self.optimization_num_epochs = int(self.config["optimization"]["num_epochs"])
        self.optimization_warmup_epochs = int(self.config["optimization"]["warmup_epochs"])
        self.optimization_optimizer = str(self.config["optimization"]["optimizer"])
        self.optimization_lambda = float(self.config["optimization"]["lambda"])

        self.data_datasets_path += "/" if not self.data_datasets_path.endswith("/") else ""
