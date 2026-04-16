from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import v2
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch
import copy
import os

from src.utils import write_on_log, plot_fig, write_on_csv, save_json, is_main_process, \
    recreate_csv_log, get_last_epoch, step_schedulers_to_epoch, load_last_values
from src.schedulers import WarmupCosineSchedule, CosineWDSchedule, EMACosineSchedule, \
    LinearTemperatureSchedule
from src.datasets import datasets

class DINOv1():
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

        self.train_loss = []

        self.lr_values = []
        self.wd_values = []
        self.ema_values = []

        if self.continue_training:
            self.last_epoch = get_last_epoch(self.output_folder)
            step_schedulers_to_epoch(self.last_epoch, len(self.train_dataloader), self.lr_scheduler, self.wd_scheduler, self.ema_scheduler)
            recreate_csv_log(self.output_folder, self.last_epoch)
            self.lr_values, self.wd_values, self.ema_values, self.train_loss = load_last_values(self.output_folder, self.last_epoch)

            write_on_log(f"Continuing training from epoch {self.last_epoch}...", self.output_folder)

    def train(self):
        pass

    def save_models(self, epoch):
        pass

    def update_target_network(self, ema):
        pass

    def _load_schedulers(self):
        pass

    def _load_optimizer(self):
        pass

    def _load_criterion(self):
        pass

    def apply_criterion(self, z1, z2):
        pass

    def _load_dataloader(self):
        pass

    def _load_transform(self):
        pass

    def _load_models(self):
        pass

    def _load_config(self):
        pass
