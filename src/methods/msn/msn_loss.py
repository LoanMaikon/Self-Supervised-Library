# Code from https://github.com/facebookresearch/msn/blob/main/src/losses.py

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch
import math
from src.utils import AllReduce


logger = getLogger()

class msn_loss():
    def __init__(self, num_views=1, me_max=True, return_preds=False):
        self.num_views = num_views
        self.me_max = me_max
        self.return_preds = return_preds

        self.softmax = torch.nn.Softmax(dim=1)

    def sharpen(self, p, T):
        sharp_p = p**(1./T)
        sharp_p /= torch.sum(sharp_p, dim=1, keepdim=True)
        return sharp_p

    def snn(self, query, supports, support_labels, temp=0.1):
        """ Soft Nearest Neighbours similarity classifier """
        query = torch.nn.functional.normalize(query)
        supports = torch.nn.functional.normalize(supports)
        return self.softmax(query @ supports.T / temp) @ support_labels

    def compute_loss(
        self,
        anchor_views,
        target_views,
        prototypes,
        proto_labels,
        student_temperature,
        T,
        use_entropy=False,
        use_sinkhorn=False,
    ):
        # Step 1: compute anchor predictions
        probs = self.snn(anchor_views, prototypes, proto_labels, temp=student_temperature)

        # Step 2: compute targets for anchor predictions
        with torch.no_grad():
            targets = self.sharpen(self.snn(target_views, prototypes, proto_labels), T=T)
            if use_sinkhorn:
                targets = distributed_sinkhorn(targets)
            targets = torch.cat([targets for _ in range(self.num_views)], dim=0)

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = torch.mean(torch.sum(torch.log(probs**(-targets)), dim=1))

        # Step 4: compute me-max regularizer
        rloss = 0.
        if self.me_max:
            avg_probs = AllReduce.apply(torch.mean(probs, dim=0))
            rloss = - torch.sum(torch.log(avg_probs**(-avg_probs))) + math.log(float(len(avg_probs)))

        sloss = 0.
        if use_entropy:
            sloss = torch.mean(torch.sum(torch.log(probs**(-probs)), dim=1))

        if self.return_preds:
            return loss, rloss, sloss, targets

        return loss, rloss, sloss

@torch.no_grad()
def distributed_sinkhorn(Q, num_itr=3, use_dist=True):
    _got_dist = use_dist and torch.distributed.is_available() \
        and torch.distributed.is_initialized() \
        and (torch.distributed.get_world_size() > 1)

    if _got_dist:
        world_size = torch.distributed.get_world_size()
    else:
        world_size = 1

    Q = Q.T
    B = Q.shape[1] * world_size  # number of samples to assign
    K = Q.shape[0]  # how many prototypes

    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    if _got_dist:
        torch.distributed.all_reduce(sum_Q)
    Q /= sum_Q

    for it in range(num_itr):
        # normalize each row: total weight per prototype must be 1/K
        sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
        if _got_dist:
            torch.distributed.all_reduce(sum_of_rows)
        Q /= sum_of_rows
        Q /= K

        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B

    Q *= B  # the colomns must sum to 1 so that Q is an assignment
    return Q.T