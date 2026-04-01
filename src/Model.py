from torchvision.transforms import v2
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
import shutil
import os
from time import strftime, localtime
import json
import copy
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from src.utils import is_main_process
from src.methods.simclr.SimCLR import SimCLR
from src.methods.ijepa.IJEPA import IJEPA
from src.methods.byol.BYOL import BYOL

class Model():
    def __init__(self,
                 config,
                 output_folder,
                 rank,
                 world_size,
                 continue_training,
                ):
        
        self.config = config
        self.output_folder = output_folder
        self.rank = rank
        self.world_size = world_size
        self.continue_training = continue_training

        self._load_device()
        self._create_output_folder()
        self._load_config()
    
    def train(self):
        self.method.train()

    def _create_output_folder(self):
        if is_main_process():
            os.makedirs(self.output_folder, exist_ok=False)

    def _load_device(self):
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
    
    def _load_config(self):
        with open(self.config, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.mode = self.config["mode"]

        match self.mode:
            case "linear_evaluation":
                pass
            
            case "fine_tuning":
                pass

            case "simclr":
                self.method = SimCLR(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )

            case "byol":
                self.method = BYOL(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )

            case "ijepa":
                self.method = IJEPA(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )

            case _:
                raise ValueError(f"Unsupported mode '{self.mode}'. Supported modes are: linear_evaluation, fine_tuning, simclr, byol, ijepa.")
