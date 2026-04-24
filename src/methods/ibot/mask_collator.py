# https://github.com/bytedance/ibot/blob/main/loader.py

from multiprocessing import Value
import numpy as np
import random
import torch
import math

'''
Return (batch, masks)
'''
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

    ):
        super(MaskCollator, self).__init__()

        self.patch_size = patch_size
        self.global_crop_size = global_crop_size
        self.local_crop_size = local_crop_size
        self.pred_ratio = pred_ratio
        self.pred_ratio_var = pred_ratio_var
        self.pred_aspect_ratio = pred_aspect_ratio
        self.num_global_crops = num_global_crops
        self.num_local_crops = num_local_crops
        self.num_attempts = num_attempts

    def get_pred_ratio(self):
        pred_ratio = []
        for prm, prv in zip(self.pred_ratio, self.pred_ratio_var):
            if prm == 0:
                pred_ratio.append(0)
                continue

            assert prm >= prv
            pr = random.uniform(prm - prv, prm + prv) if prv > 0 else prm
            pred_ratio.append(pr)
        pred_ratio = random.choice(pred_ratio)

        return pred_ratio

    def __call__(self, batch):
        B = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        masks = []
        for i in range(B):
            masks.append([])
            for img in batch[i][0]:
                H, W = img.shape[1] // self.patch_size, img.shape[2] // self.patch_size

                pred_ratio = self.get_pred_ratio()
                num_patches = int(H * W * pred_ratio)

                mask = np.zeros((H, W), dtype=bool)
                mask_count = 0
                while mask_count < num_patches:
                    max_mask_patches = num_patches - mask_count

                    delta = 0
                    for attempt in range(self.num_attempts):
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
                                for i in range(top, top + h):
                                    for j in range(left, left + w):
                                        if mask[i, j] == 0:
                                            mask[i, j] = 1
                                            delta += 1

                        if delta > 0:
                            break

                    if delta == 0:
                        break
                    else:
                        mask_count += delta
                
                masks[-1].append(mask)
            
            masks[-1] = torch.from_numpy(np.stack(masks[-1], axis=0)).float() # num_crops, H, W
            
            masks = torch.c

        return collated_batch, masks
