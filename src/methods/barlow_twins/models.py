# Code from https://github.com/facebookresearch/barlowtwins/blob/main/main.py

import torch.nn as nn
import torchvision
import torch

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class Model_barlow_twins(nn.Module):
    def __init__(self, projection_head_dims=None, batch_size=None, world_size=None, lambd=None, use_checkpoint=False):
        super().__init__()

        self.batch_size = batch_size
        self.world_size = world_size
        self.lambd = lambd
        self.use_checkpoint = use_checkpoint

        self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048] + projection_head_dims if projection_head_dims is not None else [2048, 8192, 8192, 8192] # Adjust this in evaluation if you change the default
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

        self.backbone_out_features = 2048

    def forward(self, y1, y2):
        if not self.use_checkpoint:
            z1 = self.projector(self.backbone(y1))
            z2 = self.projector(self.backbone(y2))
        else:
            z1 = torch.utils.checkpoint.checkpoint(self.backbone, y1)
            z1 = torch.utils.checkpoint.checkpoint(self.projector, z1)
            z2 = torch.utils.checkpoint.checkpoint(self.backbone, y2)
            z2 = torch.utils.checkpoint.checkpoint(self.projector, z2)

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        batch_size_per_gpu = self.batch_size
        global_batch_size = batch_size_per_gpu * self.world_size
        c.div_(global_batch_size)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss
    
    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.projector.parameters():
            param.requires_grad = True

    def eval_forward(self, x):
        return self.backbone(x)

    def get_eval_output_dim(self):
        return self.backbone_out_features

    def load_weights(self, weight_path, device):
        checkpoint = torch.load(weight_path, map_location=device)

        print(checkpoint.keys())

        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        errors = []

        try:
            self.load_state_dict(clean_state_dict)
            return
        except Exception as e:
            errors.append(("barlow_twins", str(e)))

        try:
            self.backbone.load_state_dict(clean_state_dict)
            return
        except Exception as e:
            errors.append(("backbone", str(e)))
        
        raise ValueError(
            f"Failed to load weights from {weight_path}. "
            f"Tried: {clean_state_dict.keys()}"
        )

    def remove_decoder(self):
        del self.decoder_embed
        del self.mask_token
        del self.decoder_pos_embed
        del self.decoder_blocks
        del self.decoder_norm
        del self.decoder_pred
        self.decoder_embed = None
        self.mask_token = None
        self.decoder_pos_embed = None
        self.decoder_blocks = None
        self.decoder_norm = None
        self.decoder_pred = None
    
    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.projector.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.projector.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        for param in self.projector.parameters():
            param.requires_grad = True
    
    def get_output_dim(self):
        return self.backbone_out_features

    def remove_classifier_head(self):
        return

    def get_features(self, features):
        return features
