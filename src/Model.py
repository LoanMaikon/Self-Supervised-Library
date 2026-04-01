import shutil
import torch
import yaml
import os


from src.methods.simclr.SimCLR import SimCLR
from src.methods.ijepa.IJEPA import IJEPA
from src.methods.byol.BYOL import BYOL
from src.utils import is_main_process

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
            if self.continue_training:
                os.makedirs(self.output_folder, exist_ok=True)
            else:
                os.makedirs(self.output_folder, exist_ok=False)
            
            shutil.copy(self.config, os.path.join(self.output_folder, "config.yaml"))

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
