from multiprocessing import Value
import torch
import math

# From https://github.com/facebookresearch/ijepa/blob/main/src/masks/multiblock.py

'''
Return (images, encoder_masks, predictor_masks)
'''
class MaskCollator(object):
    def __init__(self, 
                 crop_size,
                 patch_size,
                 n_targets, # Number of target patches to predict
                 min_keep, # Minimum number of context patches
                 context_mask_scale,
                 pred_aspect_ratio,
                 pred_mask_scale,
                ):
        super(MaskCollator, self).__init__()

        self.crop_size = crop_size
        self.patch_size = patch_size
        self.n_targets = n_targets
        self.min_keep = min_keep
        self.pred_aspect_ratio = pred_aspect_ratio
        self.context_mask_scale = context_mask_scale
        self.pred_mask_scale = pred_mask_scale

        self.height, self.width = crop_size // patch_size, crop_size // patch_size
        self._itr_counter = Value('i', -1)
    
    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    '''
    Return the block size (height, width) for a mask given the input image size, the desired mask scale and aspect ratio.
    '''
    def _sample_block_size(self, generator, scale, aspect_ratio):
        _rand = torch.rand(1, generator=generator).item()

        # Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)

        # Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)

        # Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return (h, w)

    '''
    Return a block mask (indexes) and its complement of the given size, sampled from the input image.
    If acceptable_regions is given, the block mask is constrained to be within the acceptable regions.
    '''
    def _sample_block_mask(self, b_size, acceptable_regions=None):
        '''
        Restrict a given mask to a set of acceptable regions by iteratively masking out unacceptable regions until we find a valid mask.
        '''
        def _constrain_mask(mask, acceptable_regions, tries=0):
            N = max(int(len(acceptable_regions) - tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]

        h, w = b_size

        # Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:

            # Sample block top-left corner
            top = torch.randint(0, self.height - h, (1,))
            left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1

            # Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                _constrain_mask(mask, acceptable_regions, tries)
            mask = torch.nonzero(mask.flatten())

            # If mask is too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
        mask = mask.squeeze()

        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0

        return mask, mask_complement
    
    '''
    Create encoder and predictor masks when collating images into a batch
    # 1. sample encoder block (size + location) using seed
    # 2. sample pred block (size) using seed
    # 3. sample several encoder block locations for each image (w/o seed)
    # 4. sample several pred block locations for each image (w/o seed)
    # 5. return encoder mask and pred mask
    '''
    def __call__(self, batch):
        B = len(batch)

        collated_batch = torch.utils.data.default_collate(batch)

        seed = self.step()
        generator = torch.Generator()
        generator.manual_seed(seed)

        pred_size = self._sample_block_size(
            generator=generator,
            scale=self.pred_mask_scale,
            aspect_ratio=self.pred_aspect_ratio
        )

        context_size = self._sample_block_size(
            generator=generator,
            scale=self.context_mask_scale,
            aspect_ratio=(1., 1.)
        )

        collated_masks_preds, collated_masks_context = [], []
        min_keep_pred = self.height * self.width
        min_keep_context = self.height * self.width

        for _ in range(B):

            batch_masks_preds, batch_masks_preds_C = [], []
            for _ in range(self.n_targets):
                mask, mask_C = self._sample_block_mask(pred_size)

                batch_masks_preds.append(mask)
                batch_masks_preds_C.append(mask_C)

                # If a batch has a small pred mask, we will trim all pred masks in the batch to that size
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_preds.append(batch_masks_preds)

            # Acceptable regions are the complement of the pred masks, so that encoder masks do not overlap with pred masks
            mask, _ = self._sample_block_mask(context_size, acceptable_regions=batch_masks_preds_C)
            min_keep_context = min(min_keep_context, len(mask))
            collated_masks_context.append([mask])

        # Cutting out extra context and pred patches so that all masks in the batch have the same number of patches
        collated_masks_preds = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_preds]
        collated_masks_context = [[cm[:min_keep_context] for cm in cm_list] for cm_list in collated_masks_context]

        collated_masks_preds = torch.utils.data.default_collate(collated_masks_preds)
        collated_masks_context = torch.utils.data.default_collate(collated_masks_context)

        return collated_batch, collated_masks_context, collated_masks_preds
