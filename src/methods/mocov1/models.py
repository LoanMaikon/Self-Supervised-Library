# Code adapted from https://gitee.com/facebookresearch/moco
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torchvision.models import resnet50 as resnet50_original

def is_dist_avail_and_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return torch.distributed.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()

class MoCo(nn.Module):
    def __init__(
        self,
        base_encoder,
        dim=128,
        K=65536,
        m=0.999,
        T=0.07,
        mlp=False,
        use_checkpoint=False,
    ):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.use_checkpoint = use_checkpoint

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        self.eval_out_dim = self.encoder_q.fc.in_features

        if mlp:  # hack: brute-force replacement, as in official MoCo v2
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                self.encoder_q.fc,
            )
            self.encoder_k.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.ReLU(),
                self.encoder_k.fc,
            )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not updated by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # The MoCo forward output dimension is always `dim`, even when mlp=True.
        self.out_dim = dim

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder.
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # Gather keys before updating queue. In single-GPU/non-distributed mode,
        # concat_all_gather(keys) simply returns keys.
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0, (
            f"Queue size K={self.K} must be divisible by the gathered batch size "
            f"{batch_size}. Use drop_last=True or choose a compatible K/batch_size."
        )

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        world_size = get_world_size()
        if world_size == 1:
            return x, None

        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index on the same device as x
        idx_shuffle = torch.randperm(batch_size_all, device=x.device)

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        world_size = get_world_size()
        if world_size == 1 or idx_unshuffle is None:
            return x

        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def _encode_q(self, im_q):
        if not self.use_checkpoint:
            return self.encoder_q(im_q)
        return checkpoint.checkpoint(self.encoder_q, im_q, use_reentrant=False)

    def forward(self, im_q, im_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q = self._encode_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN in DDP; identity in single-GPU mode
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle; identity in single-GPU mode
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        # compute logits
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)

        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels

    def eval_forward(self, x):
        if not self.use_checkpoint:
            return self.encoder_q(x)
        return checkpoint.checkpoint(self.encoder_q, x, use_reentrant=False)

    def get_output_dim(self):
        return self.out_dim

    def freeze(self):
        for param in self.encoder_q.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.encoder_q.parameters():
            param.requires_grad = True

    def get_features(self, features):
        return features

    def get_eval_output_dim(self):
        return self.eval_out_dim

    def remove_classifier_head(self):
        self.encoder_q.fc = nn.Identity()
        if hasattr(self, "encoder_k"):
            del self.encoder_k
        if hasattr(self, "queue"):
            del self.queue
        if hasattr(self, "queue_ptr"):
            del self.queue_ptr

    def del_components(self):
        if hasattr(self, "encoder_k"):
            del self.encoder_k
        if hasattr(self, "queue"):
            del self.queue
        if hasattr(self, "queue_ptr"):
            del self.queue_ptr

    def load_weights(self, weight_path, device):
        checkpoint_dict = torch.load(weight_path, map_location=device)

        state_dict = checkpoint_dict.get("state_dict", checkpoint_dict)
        clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        errors = []
        try:
            self.load_state_dict(clean_state_dict)
            return
        except Exception as e:
            errors.append(("normal", str(e)))

        try:
            self.del_components()
            self.load_state_dict(clean_state_dict)
            return
        except Exception as e:
            errors.append(("del_components", str(e)))

        try:
            self.remove_classifier_head()
            self.load_state_dict(clean_state_dict)
            return
        except Exception as e:
            errors.append(("remove_classifier", str(e)))

        raise ValueError(
            f"Failed to load weights from {weight_path}. "
            f"Tried: {errors}"
            f"state_dict keys: {list(clean_state_dict.keys())}"
        )

@torch.no_grad()
def concat_all_gather(tensor):
    world_size = get_world_size()
    if world_size == 1:
        return tensor

    tensors_gather = [torch.ones_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def resnet50(dim=128, queue_size=65536, ema=0.999, temp=0.07, mlp=False, use_checkpoint=False):
    return MoCo(
        resnet50_original,
        dim=dim,
        K=queue_size,
        m=ema,
        T=temp,
        mlp=mlp,
        use_checkpoint=use_checkpoint,
    )
