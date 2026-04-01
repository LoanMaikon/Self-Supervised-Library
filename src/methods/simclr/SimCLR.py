from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import torch.optim as optim
import torch
import json
import os

from src.utils import write_on_log, plot_fig, write_on_csv, save_json, is_main_process, concat_all_gather
from src.schedulers import WarmupCosineSchedule, CosineWDSchedule
from .resnet import resnet50, projection_head
from src.imagenet import imagenet
from src.nt_xent import nt_xent
from src.lars import LARS

class SimCLR():
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

        if self.continue_training:
            self.last_epoch = self.find_last_epoch()
            self.step_schedulers_to_epoch(self.last_epoch)

            write_on_log(f"Continuing training from epoch {self.last_epoch}...", self.output_folder)

    def train(self):
        write_on_log("Starting training...", self.output_folder)
        scaler = torch.amp.GradScaler()

        train_loss = []
        lrs = []
        wds = []

        for epoch in range(1, self.optimization_num_epochs + 1):
            if self.continue_training and epoch <= self.last_epoch:
                continue

            write_on_log(f"Epoch {epoch}/{self.optimization_num_epochs}", self.output_folder)
            self.train_sampler.set_epoch(epoch)

            epoch_loss = 0.0

            for iteration, (x1, x2) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                x1, x2 = x1.to(self.device, non_blocking=True), x2.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    z1 = self.projection_head(self.encoder(x1))
                    z2 = self.projection_head(self.encoder(x2))

                    if self.world_size > 1:
                        z1 = concat_all_gather(z1)
                        z2 = concat_all_gather(z2)

                    loss = self.apply_criterion(z1, z2)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                loss_value = loss.item()
                epoch_loss += loss_value

                lrs.append(self.lr_scheduler.get_value())
                wds.append(self.wd_scheduler.get_value())

                write_on_csv(self.output_folder, epoch, iteration, loss_value, lrs[-1], wds[-1])

                self.lr_scheduler.step()
                self.wd_scheduler.step()
            
            epoch_loss /= len(self.train_dataloader)
            train_loss.append(epoch_loss)

            self.save_models()

            save_json({"last_epoch": epoch}, self.output_folder, "last_epoch")

            write_on_log(f"Epoch {epoch} loss: {epoch_loss:.4f}", self.output_folder)
            plot_fig(range(len(train_loss)), "Epoch", train_loss, "Loss", f"loss", self.output_folder)
            plot_fig(range(len(lrs)), "Iteration", lrs, "Learning Rate", f"learning_rate", self.output_folder)
            plot_fig(range(len(wds)), "Iteration", wds, "Weight Decay", f"weight_decay", self.output_folder)

    def save_models(self):
        if not is_main_process():
            return

        encoder_state_dict = self.encoder.module.state_dict() if self.world_size > 1 else self.encoder.state_dict()
        projection_head_state_dict = self.projection_head.module.state_dict() if self.world_size > 1 else self.projection_head.state_dict()

        os.makedirs(os.path.join(self.output_folder, "models"), exist_ok=True)

        torch.save(encoder_state_dict, os.path.join(self.output_folder, "models", "encoder.pth"))
        torch.save(projection_head_state_dict, os.path.join(self.output_folder, "models", "projection_head.pth"))

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
                    }
                ]
                
                self.base_optimizer = optim.SGD(param_groups, lr=self.optimization_lr[0], momentum=0.9)
                self.optimizer = LARS(
                    self.base_optimizer,
                    trust_coefficient=0.001
                )
            
            case _:
                raise ValueError(f"Unsupported optimizer: {self.optimization_optimizer}")

    def _load_criterion(self):
        match self.optimization_criterion:
            case "nt_xent":
                self.criterion = nt_xent(temperature=self.optimization_temperature)
            
            case _:
                raise ValueError(f"Unsupported criterion: {self.optimization_criterion}")
    
    def apply_criterion(self, z1, z2):
        match self.optimization_criterion:
            case "nt_xent":
                return self.criterion(z1, z2)
            
            case _:
                raise ValueError(f"Unsupported criterion: {self.optimization_criterion}")

    def _load_dataloader(self):
        self.train_dataset = imagenet(
            operation="train",
            datasets_folder_path=self.data_datasets_path,
            dataset_name=self.data_train_dataset,
            transform=self.transform,
            separate_val_subset=self.data_separate_val_subset_use,
            val_size=self.data_separate_val_subset_size,
            apply_data_augmentation=True,
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
        def __get_color_distortion(strength=1.0):
            collor_jitter = v2.ColorJitter(0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength)
            rnd_color_jitter = v2.RandomApply([collor_jitter], p=0.8)
            rnd_gray = v2.RandomGrayscale(p=0.2)

            return v2.Compose([rnd_color_jitter, rnd_gray])

        self.transform = v2.Compose([
            v2.RandomResizedCrop(self.data_crop_size, scale=self.data_crop_scale, ratio=self.data_crop_ratio),
            v2.RandomHorizontalFlip(0.5) if self.data_horizontal_flip else v2.Identity(),
            __get_color_distortion() if self.data_color_jitter else v2.Identity(),
            v2.RandomApply([v2.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5) if self.data_gaussian_blur else v2.Identity(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

    def _load_models(self):
        match self.meta_model_name:
            case "resnet50":
                self.encoder = resnet50(use_checkpoint=self.meta_checkpoint)
                self.projection_head = projection_head(encoder_out_features=self.encoder.fc.in_features, projection_dim=self.meta_projection_dim)

            case _:
                raise ValueError(f"Unsupported model name: {self.meta_model_name}")
        
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
            else:
                raise FileNotFoundError("Checkpoint files not found for continuing training.")

        self.encoder = self.encoder.to(self.device)
        self.projection_head = self.projection_head.to(self.device)

        # Removing classifier head from ResNet if it exists
        self.encoder.remove_classifier_head()

        self.encoder.unfreeze_encoder()
        self.projection_head.unfreeze()

        if self.world_size > 1:
            self.encoder = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.encoder)
            self.projection_head = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.projection_head)

            self.encoder = DDP(self.encoder, device_ids=[self.rank], output_device=self.rank)
            self.projection_head = DDP(self.projection_head, device_ids=[self.rank], output_device=self.rank)
        
        self.encoder.train()
        self.projection_head.train()

    def _load_config(self):
        self.data_datasets_path = str(self.config["data"]["datasets_path"])
        self.data_train_dataset = str(self.config["data"]["train_dataset"])
        self.data_batch_size = int(self.config["data"]["batch_size"])
        self.data_crop_size = int(self.config["data"]["crop_size"])
        self.data_num_workers = int(self.config["data"]["num_workers"])
        self.data_prefetch_factor = int(self.config["data"]["prefetch_factor"])
        self.data_pin_memory = bool(self.config["data"]["pin_memory"])
        self.data_drop_last = bool(self.config["data"]["drop_last"])
        self.data_crop_scale = list(map(float, self.config["data"]["crop_scale"]))
        self.data_crop_ratio = list(map(float, self.config["data"]["crop_ratio"]))
        self.data_color_jitter = bool(self.config["data"]["color_jitter"])
        self.data_gaussian_blur = bool(self.config["data"]["gaussian_blur"])
        self.data_horizontal_flip = bool(self.config["data"]["horizontal_flip"])
        self.data_normalize_mean = list(map(float, self.config["data"]["normalize"]["mean"]))
        self.data_normalize_std = list(map(float, self.config["data"]["normalize"]["std"]))
        self.data_separate_val_subset_use = bool(self.config["data"]["separate_val_subset"]["use"])
        self.data_separate_val_subset_size = float(self.config["data"]["separate_val_subset"]["size"])

        self.meta_model_name = str(self.config["meta"]["model_name"])
        self.meta_checkpoint = bool(self.config["meta"]["checkpoint"])
        self.meta_projection_dim = int(self.config["meta"]["projection_dim"])
        self.meta_pretrained_weights = self.config["meta"]["pretrained_weights"]

        self.optimization_num_epochs = int(self.config["optimization"]["num_epochs"])
        self.optimization_lr = list(map(float, self.config["optimization"]["lr"]))
        self.optimization_weight_decay = list(map(float, self.config["optimization"]["weight_decay"]))
        self.optimization_warmup_epochs = int(self.config["optimization"]["warmup_epochs"])
        self.optimization_optimizer = str(self.config["optimization"]["optimizer"])
        self.optimization_temperature = float(self.config["optimization"]["temperature"])
        self.optimization_criterion = str(self.config["optimization"]["criterion"])
        self.optimization_ipe_scale = float(self.config["optimization"]["ipe_scale"])

        self.data_datasets_path += "/" if not self.data_datasets_path.endswith("/") else ""
