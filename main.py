import torch.distributed as dist
import argparse
import torch
import os

from src.utils import is_distributed
from src.Model import Model

def main():
    args = get_args()
    local_rank, world_size = setup_distributed()

    model = Model(
        config=args.config,
        output_folder=args.output_folder,
        rank=local_rank,
        world_size=world_size,
        continue_training=args.continue_training,
    )

    model.train()
    
    cleanup_distributed()

def get_args():
    parser = argparse.ArgumentParser(description="SSL Library")

    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file in configs/.")
    parser.add_argument("--devices", type=str, required=True, nargs="+", help="Devices (e.g., cuda:0, cuda:1).")
    parser.add_argument("--output_folder", type=str, required=True, help="Directory to save outputs.")
    parser.add_argument("--continue_training", action="store_true", help="Whether to continue training from the last checkpoint in the output folder.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        raise ValueError(f"Config file '{args.config}' does not exist.")
    
    if args.continue_training and not os.path.exists(args.output_folder):
        raise ValueError(f"Output folder '{args.output_folder}' does not exist for continuing training.")
    
    if not args.continue_training and os.path.exists(args.output_folder):
        raise ValueError(f"Output folder '{args.output_folder}' already exists. Use --continue_training to continue training or choose a different output folder.")
    
    if not is_distributed():
        device_str = str(args.devices[0])
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str.replace("cuda:", "")

    args.output_folder += "/" if not args.output_folder.endswith("/") else ""

    return args

def setup_distributed():
    if not is_distributed():
        return 0, 1

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://"
    )

    return local_rank, world_size

def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

'''
nohup torchrun --nproc_per_node=2 main.py --config configs/pretraining_simclr.yaml --devices cuda:0 cuda:1 --output_path ../test_output &
'''
