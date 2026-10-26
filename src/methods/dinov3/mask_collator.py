# Incorporated third-party source; local adaptations for Self-Supervised Library.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Source scope, attribution and modifications: LICENSES/README.md.
# Incorporated-source terms: DINOv3 License Agreement, last updated August 19, 2025.
# Full upstream license copy: LICENSES/THIRD_PARTY_LICENSES.txt [section: dinov3]

# Adapted from facebookresearch/dinov3 data masking/collate utilities.
# Keeps backward compatibility with datasets that return a flat list of crops.

import random
import math
import numpy as np
import torch


class MaskingGenerator:
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        return "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height,
            self.width,
            self.min_num_patches,
            self.max_num_patches,
            self.num_masking_patches,
            self.log_aspect_ratio[0],
            self.log_aspect_ratio[1],
        )

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)
                num_masked = mask[top : top + h, left : left + w].sum()
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1
                if delta > 0:
                    break
        return delta

    def complete_mask_randomly(self, mask, num_masking_patches):
        m = mask.reshape(-1)
        missing = int(num_masking_patches - m.sum())
        if missing > 0:
            candidates = np.where(~m)[0]
            if missing > candidates.size:
                missing = candidates.size
            to_add = np.random.choice(candidates, size=missing, replace=False)
            m[to_add] = True
        return m.reshape(mask.shape)

    def __call__(self, num_masking_patches=0):
        num_masking_patches = int(num_masking_patches)
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        if num_masking_patches <= 0:
            return mask
        num_masking_patches = min(num_masking_patches, self.num_patches)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)
            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            mask_count += delta
        return self.complete_mask_randomly(mask, num_masking_patches)

def _unwrap_sample(sample):
    return sample[0] if isinstance(sample, (tuple, list)) and len(sample) >= 1 else sample

def _views_are_dict(views):
    return isinstance(views, dict) and "global_crops" in views and "local_crops" in views

def collate_data_and_cast(
    samples_list,
    mask_ratio_tuple,
    mask_probability,
    dtype,
    n_global_crops=None,
    n_local_crops=None,
    n_tokens=None,
    mask_generator=None,
    random_circular_shift=False,
    local_batch_size=None,
    n_gram_crops=None,
):
    if len(samples_list) == 0:
        raise ValueError("samples_list is empty")
    if mask_generator is None:
        raise ValueError("mask_generator must be provided")
    if n_tokens is None:
        raise ValueError("n_tokens must be provided")

    first_views = _unwrap_sample(samples_list[0])
    if _views_are_dict(first_views):
        n_global_crops = len(first_views["global_crops"])
        n_local_crops = len(first_views["local_crops"])
        collated_global_crops = torch.stack(
            [_unwrap_sample(s)["global_crops"][i] for i in range(n_global_crops) for s in samples_list]
        )
        collated_local_crops = torch.stack(
            [_unwrap_sample(s)["local_crops"][i] for i in range(n_local_crops) for s in samples_list]
        )
        has_gram = "gram_teacher_crops" in first_views and first_views["gram_teacher_crops"] is not None
        if has_gram:
            collated_gram_teacher_crops = torch.stack(
                [_unwrap_sample(s)["gram_teacher_crops"][i] for i in range(n_global_crops) for s in samples_list]
            )
        else:
            collated_gram_teacher_crops = None
    else:
        if n_global_crops is None or n_local_crops is None:
            raise ValueError("n_global_crops and n_local_crops are required for legacy list samples")
        views0 = list(first_views)
        paired_global_crops = isinstance(views0[0], (tuple, list))
        if n_gram_crops is None:
            n_gram_crops = max(0, len(views0) - n_global_crops - n_local_crops)
        collated_global_crops = torch.stack(
            [(_unwrap_sample(s)[i][0] if paired_global_crops else _unwrap_sample(s)[i])
             for i in range(n_global_crops) for s in samples_list]
        )
        collated_local_crops = torch.stack(
            [_unwrap_sample(s)[n_global_crops + i] for i in range(n_local_crops) for s in samples_list]
        )
        if paired_global_crops:
            collated_gram_teacher_crops = torch.stack(
                [_unwrap_sample(s)[i][1] for i in range(n_global_crops) for s in samples_list]
            )
        elif n_gram_crops and len(views0) >= n_global_crops + n_local_crops + n_gram_crops:
            start = n_global_crops + n_local_crops
            collated_gram_teacher_crops = torch.stack(
                [_unwrap_sample(s)[start + i] for i in range(n_gram_crops) for s in samples_list]
            )
        else:
            collated_gram_teacher_crops = None

    if local_batch_size is not None:
        B = n_global_crops * local_batch_size
    else:
        B = len(collated_global_crops)

    N = int(n_tokens)
    n_samples_masked = int(B * mask_probability)
    probs = torch.linspace(*mask_ratio_tuple, n_samples_masked + 1)
    upperbound = 0
    masks_list = []
    for i in range(n_samples_masked):
        prob_max = probs[i + 1]
        mask = torch.BoolTensor(mask_generator(int(N * prob_max)))
        if random_circular_shift and mask.numel() > 0:
            shift_x = random.randint(0, mask.shape[0] - 1)
            shift_y = random.randint(0, mask.shape[1] - 1)
            mask = torch.roll(mask, (shift_x, shift_y), (0, 1))
        masks_list.append(mask)
        upperbound += int(N * prob_max)
    for _ in range(n_samples_masked, B):
        masks_list.append(torch.BoolTensor(mask_generator(0)))

    random.shuffle(masks_list)
    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()
    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    out = {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_local_crops": collated_local_crops.to(dtype),
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        "global_batch_size": torch.tensor(len(samples_list), dtype=torch.long),
    }
    if collated_gram_teacher_crops is not None:
        out["collated_gram_teacher_crops"] = collated_gram_teacher_crops.to(dtype)
    return out
