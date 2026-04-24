from multiprocessing import Value
import torch
import math

'''
Return (images, masks)
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

        self._itr_counter = Value('i', -1)

        def __call__(self, batch):
            # . . .

            pass