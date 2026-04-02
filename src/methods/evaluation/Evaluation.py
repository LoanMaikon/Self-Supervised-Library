from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import torch.optim as optim
import torch
import json
import os

from src.methods.byol.resnet import resnet50 as byol_resnet50, resnet200 as byol_resnet200
from src.methods.simclr.resnet import resnet50 as simclr_resnet50
from src.methods.ijepa.models import vit_tiny as ijepa_vit_tiny, vit_small as ijepa_vit_small, vit_base as ijepa_vit_base, \
    vit_large as ijepa_vit_large, vit_huge as ijepa_vit_huge, vit_giant as ijepa_vit_giant

from src.utils import write_on_log, plot_fig, write_on_csv, save_json, is_main_process
from src.schedulers import WarmupCosineSchedule, CosineWDSchedule
from .linear_head import LinearHead
from src.imagenet import imagenet

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
        self._load_models()
        self._load_transform()
        self._load_dataloader()
        self._load_criterion()
        self._load_optimizer()
        self._load_schedulers()

        if self.continue_training:
            self.last_epoch = self._find_last_epoch()
            self._step_schedulers_to_epoch(self.last_epoch)
            self.train_loss_values = self._get_train_loss_values()
            self.val_loss_values = [] if not self.has_val() else self._get_val_loss_values()
            self.val_accuracy_values = [] if not self.has_val() else self._get_val_accuracy_values()
            self._recreate_csv_log()

            write_on_log(f"Continuing training from epoch {self.last_epoch}...", self.output_folder)
    
    def train(self):
        write_on_log("Starting training...", self.output_folder)
        scaler = torch.amp.GradScaler()

        train_loss = [] if not self.continue_training else self.train_loss_values
        val_loss = [] if not self.continue_training else self.val_loss_values
        val_accuracy = [] if not self.continue_training else self.val_accuracy_values
        lrs = [] if not self.continue_training else self.lr_values
        wds = [] if not self.continue_training else self.wd_values

        for epoch in range(1, self.optimization_epochs + 1):
            if self.continue_training and epoch <= self.last_epoch:
                continue

            write_on_log(f"Epoch {epoch}/{self.optimization_epochs}", self.output_folder)
            self.train_sampler.set_epoch(epoch)

            epoch_train_loss = 0.0

            self.encoder.train()
            self.linear_head.train()
            for iteration, (images, labels) in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()

                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    features = self.encoder(images)
                    features = self.encoder.get_features(features)
                    output = self.linear_head(features)
                    loss = self.apply_criterion(output, labels)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

                loss_value = loss.item()
                epoch_train_loss += loss_value

                lrs.append(self.lr_scheduler.get_value())
                wds.append(self.wd_scheduler.get_value())

                write_on_csv(self.output_folder, epoch, iteration, loss_value, lrs[-1], wds[-1])

                self.lr_scheduler.step()
                self.wd_scheduler.step()

            epoch_train_loss /= len(self.train_dataloader)
            train_loss.append(epoch_train_loss)

            self.save_models(epoch)

            save_json({"last_epoch": epoch}, self.output_folder, "last_epoch")

            write_on_log(f"Train loss: {epoch_train_loss}", self.output_folder)

            plot_fig(range(len(train_loss)), "Epoch", train_loss, "Loss", f"loss", self.output_folder)
            plot_fig(range(len(lrs)), "Iteration", lrs, "Learning Rate", f"learning_rate", self.output_folder)
            plot_fig(range(len(wds)), "Iteration", wds, "Weight Decay", f"weight_decay", self.output_folder)

            if self.has_val():
                epoch_val_loss = 0.0
                epoch_val_accuracy = 0.0
                self.encoder.eval()
                self.linear_head.eval()

                with torch.no_grad():
                    for images, labels in self.val_dataloader:
                        images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                        features = self.encoder(images)
                        features = self.encoder.get_features(features)
                        output = self.linear_head(features)

                        loss = self.apply_criterion(output, labels)
                        accuracy = (output.argmax(dim=1) == labels).float().mean().item()

                        epoch_val_loss += loss.item()
                        epoch_val_accuracy += accuracy

                epoch_val_loss /= len(self.val_dataloader)
                epoch_val_accuracy /= len(self.val_dataloader)
                val_loss.append(epoch_val_loss)
                val_accuracy.append(epoch_val_accuracy)

                write_on_log(f"Validation loss: {epoch_val_loss}", self.output_folder)
                write_on_log(f"Validation accuracy: {epoch_val_accuracy}", self.output_folder)

                plot_fig(range(len(val_loss)), "Epoch", val_loss, "Loss", f"val_loss", self.output_folder)
                plot_fig(range(len(val_accuracy)), "Epoch", val_accuracy, "Accuracy", f"val_accuracy", self.output_folder)

                write_on_csv(self.output_folder, epoch, iteration, epoch_train_loss, lrs[-1], wds[-1], val_loss[-1], val_accuracy[-1])
            else:
                write_on_csv(self.output_folder, epoch, iteration, epoch_train_loss, lrs[-1], wds[-1])

            write_on_log("", self.output_folder)
    
    def test(self):
        write_on_log("Starting testing...", self.output_folder)

        self.encoder.eval()
        self.linear_head.eval()

        test_loss = 0.0
        test_accuracy = 0.0

        with torch.no_grad():
            for images, labels in self.test_dataloader:
                images, labels = images.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)

                features = self.encoder(images)
                features = self.encoder.get_features(features)
                output = self.linear_head(features)

                loss = self.apply_criterion(output, labels)
                accuracy = (output.argmax(dim=1) == labels).float().mean().item()

                test_loss += loss.item()
                test_accuracy += accuracy
        
        test_loss /= len(self.test_dataloader)
        test_accuracy /= len(self.test_dataloader)

        write_on_log(f"Test loss: {test_loss}", self.output_folder)
        write_on_log(f"Test accuracy: {test_accuracy}", self.output_folder)

        save_json({"test_loss": test_loss, "test_accuracy": test_accuracy}, self.output_folder, "test_results")

    def save_models(self, epoch):
        if not is_main_process():
            return
    
        os.makedirs(os.path.join(self.output_folder, "models"), exist_ok=True)

        torch.save(self.encoder.module.state_dict() if self.world_size > 1 else self.encoder.state_dict(), os.path.join(self.output_folder, "models", "encoder.pth"))
        torch.save(self.linear_head.module.state_dict() if self.world_size > 1 else self.linear_head.state_dict(), os.path.join(self.output_folder, "models", "linear_head.pth"))

        if self.meta_save_every > 0 and epoch % self.meta_save_every == 0:
            torch.save(self.encoder.module.state_dict() if self.world_size > 1 else self.encoder.state_dict(), os.path.join(self.output_folder, "models", f"encoder_epoch_{epoch}.pth"))
            torch.save(self.linear_head.module.state_dict() if self.world_size > 1 else self.linear_head.state_dict(), os.path.join(self.output_folder, "models", f"linear_head_epoch_{epoch}.pth"))
    
    def _recreate_csv_log(self):
        if not is_main_process():
            return

        csv_path = os.path.join(self.output_folder, "log.csv")
        with open(csv_path, "r") as f:
            lines = f.readlines()

        new_lines = lines[:1] + [line for line in lines[1:] if int(line.split(",")[1]) <= self.last_epoch]

        with open(csv_path, "w") as f:
            f.writelines(new_lines)

    def _find_last_epoch(self):
        last_epoch_path = os.path.join(self.output_folder, "last_epoch.json")
        if not os.path.exists(last_epoch_path):
            return 0
        
        with open(last_epoch_path, "r") as f:
            last_epoch_data = json.load(f)
        
        return last_epoch_data.get("last_epoch", 0)

    def _step_schedulers_to_epoch(self, epoch):
        if epoch == 0:
            return
        
        steps_per_epoch = len(self.train_dataloader)
        total_steps = epoch * steps_per_epoch

        self.lr_values = []
        self.wd_values = []
        self.ema_values = []
        for _ in range(total_steps):
            self.lr_values.append(self.lr_scheduler.get_value())
            self.wd_values.append(self.wd_scheduler.get_value())
            self.ema_values.append(self.ema_scheduler.get_value())
            self.lr_scheduler.step()
            self.wd_scheduler.step()
            self.ema_scheduler.step()
    
    def _get_train_loss_values(self):
        csv_path = os.path.join(self.output_folder, "log.csv")
        with open(csv_path, "r") as f:
            lines = f.readlines()[1:]
        train_loss_values = []
        for line in lines:
            epoch = line.split(",")[1]
            train_loss = line.split(",")[3]
            if int(epoch) <= self.last_epoch:
                train_loss_values.append(float(train_loss))

        return train_loss_values

    def _get_val_loss_values(self):
        csv_path = os.path.join(self.output_folder, "log.csv")
        with open(csv_path, "r") as f:
            lines = f.readlines()[1:]
        val_loss_values = []
        for line in lines:
            epoch = line.split(",")[1]
            val_loss = line.split(",")[7]
            if int(epoch) <= self.last_epoch:
                val_loss_values.append(float(val_loss))

        return val_loss_values
    
    def _get_val_accuracy_values(self):
        csv_path = os.path.join(self.output_folder, "log.csv")
        with open(csv_path, "r") as f:
            lines = f.readlines()[1:]
        val_accuracy_values = []
        for line in lines:
            epoch = line.split(",")[1]
            val_accuracy = line.split(",")[8]
            if int(epoch) <= self.last_epoch:
                val_accuracy_values.append(float(val_accuracy))

        return val_accuracy_values

    def _load_models(self):
        def __try_load_models():
            errors = []

            try:
                self.encoder = simclr_resnet50(self.meta_checkpoint)
                self.encoder.load_weights(self.evaluate_weights, device=self.device)
                return
            except Exception as e:
                errors.append(("simclr_resnet50", str(e)))

            try:
                self.encoder = byol_resnet50(self.meta_checkpoint)
                self.encoder.load_weights(self.evaluate_weights, device=self.device)
                return
            except Exception as e:
                errors.append(("byol_resnet50", str(e)))
            
            try:
                self.encoder = byol_resnet200(self.meta_checkpoint)
                self.encoder.load_weights(self.evaluate_weights, device=self.device)
                return
            except Exception as e:
                errors.append(("byol_resnet200", str(e)))

            try:
                self.encoder = ijepa_vit_tiny(checkpoint=self.meta_checkpoint, patch_size=16)
                self.encoder.load_weights(self.evaluate_weights, device=self.device)
                return
            except Exception as e:
                errors.append(("ijepa_vit_tiny_16", str(e)))
            
            try:
                self.encoder = ijepa_vit_tiny(checkpoint=self.meta_checkpoint, patch_size=14)
                self.encoder.load_weights(self.evaluate_weights, device=self.device)
                return
            except Exception as e:
                errors.append(("ijepa_vit_tiny_14", str(e)))

            try:
                self.encoder = ijepa_vit_small(checkpoint=self.meta_checkpoint, patch_size=16)
                self.encoder.load_weights(self.evaluate_weights, device=self.device)
                return
            except Exception as e:
                errors.append(("ijepa_vit_small_16", str(e)))
            
            try:
                self.encoder = ijepa_vit_small(checkpoint=self.meta_checkpoint, patch_size=14)
                self.encoder.load_weights(self.evaluate_weights, device=self.device)
                return
            except Exception as e:
                errors.append(("ijepa_vit_small_14", str(e)))


            try:
                self.encoder = ijepa_vit_base(checkpoint=self.meta_checkpoint, patch_size=16)
                self.encoder.load_weights(self.evaluate_weights, device=self.device)
                return
            except Exception as e:
                errors.append(("ijepa_vit_base_16", str(e)))
            
            try:
                self.encoder = ijepa_vit_base(checkpoint=self.meta_checkpoint, patch_size=14)
                self.encoder.load_weights(self.evaluate_weights, device=self.device)
                return
            except Exception as e:
                errors.append(("ijepa_vit_base_14", str(e)))

            try:
                self.encoder = ijepa_vit_large(checkpoint=self.meta_checkpoint, patch_size=16)
                self.encoder.load_weights(self.evaluate_weights, device=self.device)
                return
            except Exception as e:
                errors.append(("ijepa_vit_large_16", str(e)))

            try:
                self.encoder = ijepa_vit_large(checkpoint=self.meta_checkpoint, patch_size=14)
                self.encoder.load_weights(self.evaluate_weights, device=self.device)
                return
            except Exception as e:
                errors.append(("ijepa_vit_large_14", str(e)))

            try:
                self.encoder = ijepa_vit_huge(checkpoint=self.meta_checkpoint, patch_size=16)
                self.encoder.load_weights(self.evaluate_weights, device=self.device)
                return
            except Exception as e:
                errors.append(("ijepa_vit_huge_16", str(e)))
            
            try:
                self.encoder = ijepa_vit_huge(checkpoint=self.meta_checkpoint, patch_size=14)
                self.encoder.load_weights(self.evaluate_weights, device=self.device)
                return
            except Exception as e:
                errors.append(("ijepa_vit_huge_14", str(e)))

            try:
                self.encoder = ijepa_vit_giant(checkpoint=self.meta_checkpoint, patch_size=16)
                self.encoder.load_weights(self.evaluate_weights, device=self.device)
                return
            except Exception as e:
                errors.append(("ijepa_vit_giant_16", str(e)))
            
            try:
                self.encoder = ijepa_vit_giant(checkpoint=self.meta_checkpoint, patch_size=14)
                self.encoder.load_weights(self.evaluate_weights, device=self.device)
                return
            except Exception as e:
                errors.append(("ijepa_vit_giant_14", str(e)))

            raise ValueError(
                f"Failed to load weights from {self.evaluate_weights}. Errors: {errors}."
            )

        __try_load_models()

        self.linear_head = LinearHead(self.encoder.get_output_dim(), 1000).to(self.device)
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
                self.encoder = DDP(self.encoder, device_ids=[self.rank], output_device=self.rank)
            self.linear_head = DDP(self.linear_head, device_ids=[self.rank], output_device=self.rank)

    def _load_transform(self):
        self.transform = v2.Compose([
            v2.Resize((self.data_crop_size, self.data_crop_size)),
            v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]),
            v2.Normalize(mean=self.data_normalize_mean, std=self.data_normalize_std),
        ])

    def _load_dataloader(self):
        self.train_dataset = imagenet(
            operation="train",
            datasets_folder_path=self.data_datasets_path,
            dataset_name=self.data_train_dataset,
            transform=self.transform,
            separate_val_subset=self.data_separate_val_subset_use,
            val_size=self.data_separate_val_subset_size,
            apply_data_augmentation=False,
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
            self.val_dataset = imagenet(
                operation="val",
                datasets_folder_path=self.data_datasets_path,
                dataset_name=self.data_train_dataset,
                transform=self.transform,
                separate_val_subset=True,
                val_size=self.data_separate_val_subset_size,
                apply_data_augmentation=False,
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
        
        self.test_dataset = imagenet(
            operation="test",
            datasets_folder_path=self.data_datasets_path,
            dataset_name=self.data_train_dataset,
            transform=self.transform,
            separate_val_subset=False,
            val_size=0,
            apply_data_augmentation=False,
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
                param_groups = [
                    {
                        'params': (p for n, p in self.encoder.named_parameters()
                                if ('bias' not in n) and (len(p.shape) != 1)),
                        'layer_adaptation': True,
                        'weight_decay': self.optimization_weight_decay[0],
                    },
                    {
                        'params': (p for n, p in self.linear_head.named_parameters()
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
                        'params': (p for n, p in self.linear_head.named_parameters()
                                if ('bias' in n) or (len(p.shape) == 1)),
                        'WD_exclude': True,
                        'weight_decay': 0,
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

        self.meta_checkpoint = bool(self.config["meta"]["checkpoint"])
        self.meta_mode = str(self.config["meta"]["mode"])
        self.meta_pretrained_weights = self.config["meta"]["pretrained_weights"]
        self.meta_save_every = int(self.config["meta"]["save_every"])

        self.optimization_ipe_scale = float(self.config["optimization"]["ipe_scale"])
        self.optimization_lr = list(map(float, self.config["optimization"]["lr"]))
        self.optimization_weight_decay = list(map(float, self.config["optimization"]["weight_decay"]))
        self.optimization_epochs = int(self.config["optimization"]["epochs"])
        self.optimization_warmup_epochs = int(self.config["optimization"]["warmup_epochs"])
        self.optimization_optimizer = str(self.config["optimization"]["optimizer"])
        self.optimization_criterion = str(self.config["optimization"]["criterion"])

        self.data_datasets_path += "/" if not self.data_datasets_path.endswith("/") else ""
