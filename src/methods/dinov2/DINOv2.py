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
        self._load_center()

        # . . .

    def train(self):
        pass        

    def save_models(self, epoch):
        pass

    def update_center(self, target_outputs):
        pass

    def _load_center(self):
        pass

    def update_target_network(self, ema):
        pass

    def _load_schedulers(self):
        pass

    def _load_optimizer(self):
        pass

    def _load_dataloader(self):
        pass

    def _load_transform(self):
        pass

    def _load_models(self):
        pass

    def _load_config(self):
        pass
