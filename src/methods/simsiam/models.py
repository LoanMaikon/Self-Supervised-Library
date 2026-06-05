# Code from https://github.com/facebookresearch/simsiam/blob/main/simsiam/builder.py

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torchvision.models as models


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, use_checkpoint=False):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

        self.use_checkpoint = use_checkpoint
        self.out_dim = dim

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """
        
        if not self.use_checkpoint:
            # compute features for one view
            z1 = self.encoder(x1) # NxC
            z2 = self.encoder(x2) # NxC

            p1 = self.predictor(z1) # NxC
            p2 = self.predictor(z2) # NxC
        
        else:
            # compute features for one view
            z1 = torch.utils.checkpoint.checkpoint(self.encoder, x1) # NxC
            z2 = torch.utils.checkpoint.checkpoint(self.encoder, x2) # NxC

            p1 = torch.utils.checkpoint.checkpoint(self.predictor, z1) # NxC
            p2 = torch.utils.checkpoint.checkpoint(self.predictor, z2) # NxC

        return p1, p2, z1.detach(), z2.detach()

    def remove_classifier_head(self):
        self.encoder.fc = nn.Identity()
        del self.predictor
    
    def get_output_dim(self):
        return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def get_features(self, features):
        return features
    
    def load_weights(self, weight_path, device):
        checkpoint = torch.load(weight_path, map_location=device)

        state_dict = checkpoint.get("state_dict", checkpoint)
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        errors = []

        try:
            self.load_state_dict(clean_state_dict)
            return
        except Exception as e:
            errors.append(("simsiam", str(e)))
        
        try:
            self.remove_classifier_head()
            self.load_state_dict(clean_state_dict, strict=False)
            return
        except Exception as e:
            errors.append(("simsiam_no_head", str(e)))
        
        raise ValueError(
            f"Failed to load weights from {weight_path}. "
            f"Tried: {errors}"
        )

    def eval_forward(self, x):
        assert self.encoder.fc == nn.Identity(), "You must call remove_classifier_head() before using eval_forward()"
        return self.encoder(x)

    def get_eval_output_dim(self):
        return self.out_dim
    
def resnet50(dim=2048, pred_dim=512, use_checkpoint=False):
    return SimSiam(models.resnet50, dim=dim, pred_dim=pred_dim, use_checkpoint=use_checkpoint)
