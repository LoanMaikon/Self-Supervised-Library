import shutil
import torch
import yaml
import os

from src.utils import is_main_process

class Model():
    def __init__(self,
                 config,
                 output_folder,
                 rank,
                 world_size,
                 evaluate_weights,
                 continue_training,
                ):
        
        self.config = config
        self.output_folder = output_folder
        self.rank = rank
        self.world_size = world_size
        self.evaluate_weights = evaluate_weights
        self.continue_training = continue_training

        self._load_device()
        self._create_output_folder()
        self._load_config()
    
    def train(self):
        self.method.train()
    
    def test(self):
        self.method.test()

    def is_evaluating(self):
        return self.mode == "evaluate"

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

        if self.mode == "evaluate":
            if self.evaluate_weights is None:
                raise ValueError(f"--evaluate_weights should be passed when mode is 'evaluate'.")
            
            if not self.continue_training:
                if not os.path.exists(self.evaluate_weights) and self.evaluate_weights not in ["supervised_resnet50", "random_resnet50"]:
                    raise ValueError(f"'{self.evaluate_weights}' does not exist for evaluation.")

        elif self.mode != "evaluate" and self.evaluate_weights is not None:
            raise ValueError(f"--evaluate_weights should not be passed when mode is not 'evaluate'.")

        match self.mode:
            case "evaluate":
                from src.methods.evaluation.Evaluation import Evaluation

                self.method = Evaluation(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    evaluate_weights=self.evaluate_weights,
                    continue_training=self.continue_training,
                )

            case "simclr":
                from src.methods.simclr.SimCLR import SimCLR

                self.method = SimCLR(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )

            case "byol":
                from src.methods.byol.BYOL import BYOL

                self.method = BYOL(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )

            case "ijepa":
                from src.methods.ijepa.IJEPA import IJEPA

                self.method = IJEPA(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )
            
            case "swav":
                from src.methods.swav.SwAV import SwAV

                self.method = SwAV(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )

            case "mae":
                from src.methods.mae.MAE import MAE

                self.method = MAE(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )
            
            case "dinov1":
                from src.methods.dinov1.DINOv1 import DINOv1

                self.method = DINOv1(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )

            case "ibot":
                from src.methods.ibot.iBOT import iBOT

                self.method = iBOT(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )
            
            case "dinov2":
                from src.methods.dinov2.DINOv2 import DINOv2

                self.method = DINOv2(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )
            
            case "barlow_twins":
                from src.methods.barlow_twins.BarlowTwins import BarlowTwins

                self.method = BarlowTwins(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )
            
            case "vicreg":
                from src.methods.vicreg.VICReg import VICReg

                self.method = VICReg(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )
            
            case "simsiam":
                from src.methods.simsiam.SimSiam import SimSiam

                self.method = SimSiam(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )
            
            case "msn":
                from src.methods.msn.MSN import MSN

                self.method = MSN(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )
            
            case "mocov1":
                from src.methods.mocov1.MoCov1 import MoCov1

                self.method = MoCov1(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )
            
            case "mocov2":
                from src.methods.mocov2.MoCov2 import MoCov2

                self.method = MoCov2(
                    opened_config=self.config,
                    output_folder=self.output_folder,
                    device=self.device,
                    rank=self.rank,
                    world_size=self.world_size,
                    continue_training=self.continue_training,
                )

            case _:
                raise ValueError(f"Unsupported mode '{self.mode}'.")
