# https://github.com/bytedance/ibot/blob/main/loader.py

from multiprocessing import Value
import numpy as np
import random
import torch
import math

class MaskCollator(object):
    def __init__(self,
                 patch_size,
                 global_crop_size,
                 local_crop_size,
                 pred_ratio,
                 pred_ratio_var,
                 pred_aspect_ratio,
                 num_global_crops,
                 num_local_crops,
                 num_attempts=10,
                 pred_shape="block",
                 pred_start_epoch=0,
        ):
        super(MaskCollator, self).__init__()

        self.patch_size = patch_size
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size

        self.pred_ratio = pred_ratio[0] if isinstance(pred_ratio, list) and len(pred_ratio) == 1 else pred_ratio
        self.pred_ratio_var = pred_ratio_var[0] if isinstance(pred_ratio_var, list) and len(pred_ratio_var) == 1 else pred_ratio_var
        if isinstance(self.pred_ratio, list) and not isinstance(self.pred_ratio_var, list):
            self.pred_ratio_var = [self.pred_ratio_var] * len(self.pred_ratio)

        self.pred_aspect_ratio = pred_aspect_ratio
        self.log_aspect_ratio = tuple(math.log(x) for x in pred_aspect_ratio)

        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.num_attempts = num_attempts
        self.pred_shape = pred_shape
        self.pred_start_epoch = pred_start_epoch

    def get_pred_ratio(self):
        if hasattr(self, "epoch") and self.epoch < self.pred_start_epoch:
            return 0

        if isinstance(self.pred_ratio, list):
            pred_ratio = []
            for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
                assert prm >= prv
                pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
                pred_ratio.append(pr)
            pred_ratio = random.choice(pred_ratio)
        else:
            assert self.pred_ratio >= self.pred_ratio_var
            pred_ratio = random.uniform(self.pred_ratio - self.pred_ratio_var,
                                        self.pred_ratio + self.pred_ratio_var) if self.pred_ratio_var > 0 else self.pred_ratio
        return pred_ratio

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __call__(self, batch):
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        masks_by_sample = []
        for i in range(B):
            sample_masks = []
            for img in batch[i][0]:
                try:
                    H, W = img.shape[1] // self.patch_size, img.shape[2] // self.patch_size
                except Exception:
                    continue

                high = self.get_pred_ratio() * H * W

                if self.pred_shape == "block":
                    mask = np.zeros((H, W), dtype=bool)
                    mask_count = 0
                    while mask_count < high:
                        max_mask_patches = high - mask_count
                        delta = 0
                        for _ in range(self.num_attempts):
                            low = (min(H, W) // 3) ** 2
                            target_area = random.uniform(low, max_mask_patches)
                            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                            h = int(round(math.sqrt(target_area * aspect_ratio)))
                            w = int(round(math.sqrt(target_area / aspect_ratio)))
                            if w < W and h < H:
                                top = random.randint(0, H - h)
                                left = random.randint(0, W - w)

                                num_masked = mask[top: top + h, left: left + w].sum()
                                if 0 < h * w - num_masked <= max_mask_patches:
                                    for ii in range(top, top + h):
                                        for jj in range(left, left + w):
                                            if mask[ii, jj] == 0:
                                                mask[ii, jj] = 1
                                                delta += 1

                            if delta > 0:
                                break

                        if delta == 0:
                            break
                        mask_count += delta

                elif self.pred_shape == "rand":
                    mask = np.hstack([
                        np.zeros(H * W - int(high)),
                        np.ones(int(high)),
                    ]).astype(bool)
                    np.random.shuffle(mask)
                    mask = mask.reshape(H, W)
                else:
                    raise ValueError(f"Unsupported pred_shape: {self.pred_shape}")

                sample_masks.append(mask)

            masks_by_sample.append(sample_masks)

        if not masks_by_sample:
            return collated_batch, []

        num_crops = len(masks_by_sample[0])
        masks = []
        for crop_idx in range(num_crops):
            crop_masks = [torch.from_numpy(masks_by_sample[b][crop_idx]).bool() for b in range(B)]
            masks.append(torch.stack(crop_masks, dim=0))

        return collated_batch, masks
