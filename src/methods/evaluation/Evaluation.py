from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import torch.optim as optim
import torch
import json
import os

from src.utils import write_on_log, plot_fig, write_on_csv, save_json, is_main_process

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
            self.last_epoch = self.find_last_epoch()
            self.step_schedulers_to_epoch(self.last_epoch)

            write_on_log(f"Continuing training from epoch {self.last_epoch}...", self.output_folder)
    
    def train(self):
        pass

    def test(self):
        pass

    def _load_models(self):
        pass

    def _load_transform(self):
        pass

    def _load_dataloader(self):
        pass

    def _load_criterion(self):
        pass

    def _load_optimizer(self):
        pass

    def _load_schedulers(self):
        pass

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

        self.optimization_ipe_scale = float(self.config["optimization"]["ipe_scale"])
        self.optimization_lr = list(map(float, self.config["optimization"]["lr"]))
        self.optimization_weight_decay = list(map(float, self.config["optimization"]["weight_decay"]))
        self.optimization_epochs = int(self.config["optimization"]["epochs"])
        self.optimization_warmup_epochs = int(self.config["optimization"]["warmup_epochs"])
        self.optimization_optimizer = str(self.config["optimization"]["optimizer"])
        self.optimization_criterion = str(self.config["optimization"]["criterion"])
