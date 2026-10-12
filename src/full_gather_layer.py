# Incorporated third-party source; local adaptations for Self-Supervised Library.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Source scope, attribution and modifications: LICENSES/THIRD_PARTY_NOTICES.
# Incorporated-source terms: MIT.
# Full upstream license copy: LICENSES/VICReg-MIT.txt

# Code from https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

import torch.distributed as dist
import torch

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]
