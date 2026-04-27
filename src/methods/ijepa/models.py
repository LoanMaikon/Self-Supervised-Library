# Using as base https://github.com/facebookresearch/ijepa/blob/main/src/models/vision_transformer.py

import torch.nn.functional as F
from functools import partial
import torch.nn as nn
import torch
import math

from src.pos_embed import get_2d_sincos_pos_embed
from src.utils import trunc_normal_, repeat_interleave_batch

"""
:param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
:param masks: list of tensors containing indices of patches in [N] to keep
"""
def apply_masks(x, masks):
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)

"""
Transform an image into a tensor of shape [B, N, D] with all the patches embedded (e.g. tokens)
[B, N, D] = Batch, Tokens, Embedding
"""
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embedding_dimension):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embedding_dimension = embedding_dimension

        self.num_patches = (self.image_size // self.patch_size) * (self.image_size // self.patch_size)

        self.proj = nn.Conv2d(self.in_channels, self.embedding_dimension, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # x.shape = [B, C, H, W]
        # self.proj(x).shape = [B, D, H/patch_size, W/patch_size]
        # self.proj(x).flatten(2).shape = [B, D, N]
        # self.proj(x).flatten(2).transpose(1, 2).shape = [B, N, D] = Batch, Tokens (patches), Embedding
        return self.proj(x).flatten(2).transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop):
        super().__init__()

        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # q, k, v -> (B, num_heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x) # x.shape = (B, N, C)

        return x, attn

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer, drop):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        return x

"""
Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
"""
class DropPath(nn.Module):
    def __init__(self, drop_prob):
        super(DropPath, self).__init__()

        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def drop_path(x, drop_prob, training):
        if drop_prob == 0. or not training:
            return x

        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor

        return output

'''
Each block has:
- LayerNorm
- Multi-Head Self Attention
- DropPath
- LayerNorm
- MLP
'''
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, qkv_bias, qk_scale, 
                 drop, attn_drop, drop_path, norm_layer, act_layer=nn.GELU):
        super().__init__()

        self.norm1 = norm_layer(dim)

        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

"""
Encoder main class
1. Tokenize the image into patches and embed them
2. Add positional embedding to the tokens
3. Pass the tokens through the transformer blocks
...
"""
class VisionTransformer(nn.Module):
    def __init__(self,
                embed_dim,
                depth,
                num_heads,
                mlp_ratio,
                patch_size,
                checkpoint,
                in_channels=3,
                image_size=224,
                qkv_bias=True,
                qk_scale=None,
                drop_rate=0.0,
                attention_drop_rate=0.0,
                drop_path_rate=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_std=0.02,
                ):
        super(VisionTransformer, self).__init__()

        self.checkpoint = checkpoint
        self.embed_dim = embed_dim

        # Divide the image into patches and embed them
        self.patch_embed = PatchEmbedding(image_size=image_size, patch_size=patch_size, in_channels=in_channels, embedding_dimension=embed_dim)

        # Positional embedding (not learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim), requires_grad=False)
        values = get_2d_sincos_pos_embed(embed_dim, int(self.patch_embed.num_patches**0.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(values).float().unsqueeze(0))

        # Create the transformer blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # Stochastic depth decay - more drop for deeper blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attention_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)

        # Initialize weights
        self.init_std = init_std
        self.apply(self._init_weights)
        self._fix_init_weight()
    
    def get_embed_dim(self):
        return self.embed_dim

    def get_num_patches(self):
        return self.patch_embed.num_patches

    def get_features(self, features):
        return features.mean(dim=1)

    def _fix_init_weight(self):
        def __rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            __rescale(layer.attn.proj.weight.data, layer_id + 1)
            __rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    '''
    If the number of patches in the input x is different from the number of patches in the positional embedding, 
    we need to interpolate the positional embedding to match the number of patches in x.
    '''
    def _interpolate_pos_encoding(self, x, pos_embed):
        npatch = x.shape[1]
        N = pos_embed.shape[1]

        if npatch == N:
            return pos_embed

        dim = x.shape[-1]

        h = w = int(math.sqrt(npatch))
        h0 = w0 = int(math.sqrt(N))

        if h * w != npatch or h0 * w0 != N:
            raise ValueError(
                f"Positional embedding requires square patch grids."
                f"Got npatch={npatch} (h*w={h*w}) and N={N} (h0*w0={h0*w0})."
            )

        # [1, N, D] -> [1, D, h0, w0]
        pos = pos_embed.reshape(1, h0, w0, dim).permute(0, 3, 1, 2)

        pos = F.interpolate(pos, size=(h, w), mode='bicubic', align_corners=False)

        # [1, D, h, w] -> [1, npatch, D]
        pos = pos.permute(0, 2, 3, 1).reshape(1, npatch, dim)

        return pos

    def get_output_dim(self):
        return self.embed_dim

    def remove_classifier_head(self):
        pass

    def load_weights(self, weight_path, device):
        checkpoint = torch.load(weight_path, map_location=device)

        if "encoder" in checkpoint:
            state_dict = checkpoint["encoder"]
        else:
            state_dict = checkpoint.get("state_dict", checkpoint)
        
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        errors = []
        try:
            self.load_state_dict(clean_state_dict)
            return
        except Exception as e:
            errors.append(("state_dict", str(e)))
        
        raise ValueError(
            f"Failed to load weights from {weight_path}. "
            f"Tried: {clean_state_dict.keys()}"
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x, masks=None):
        if masks is not None:
            if not isinstance(masks, list):
                masks = [masks]

        # Patchify x
        x = self.patch_embed(x)
        B, N, D = x.shape

        # Add positional embedding to x
        pos_embed = self._interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        # Mask x
        if masks is not None:
            x = apply_masks(x, masks)

        # Forward propagation
        if not self.checkpoint:
            for block in self.blocks:
                x = block(x)
        else:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)

        if self.norm is not None:
            x = self.norm(x)

        return x # x.shape = (B, N, D)

    def eval_forward(self, x): # Concat last 4 layers
        x = self.patch_embed(x)
        B, N, D = x.shape

        pos_embed = self._interpolate_pos_encoding(x, self.pos_embed)
        x = x + pos_embed

        features = []
        if not self.checkpoint:
            for block in self.blocks:
                x = block(x)
                features.append(x)
        else:
            for block in self.blocks:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
                features.append(x)
        
        if self.norm is not None:
            features = [self.norm(feature) for feature in features]
        
        avg_features = [self.get_features(feature) for feature in features[-4:]]
        avg_features = torch.cat(avg_features, dim=-1)

        return avg_features

    def get_eval_output_dim(self):
        return self.embed_dim * 4

"""
Predictor main class
1. Take the output of the encoder and pass it through a few transformer blocks
2. Return the predicted tokens (patches) of the masked patches
"""
class VisionTransformerPredictor(nn.Module):
    def __init__(
        self,
        num_patches,
        embed_dim,
        depth,
        predictor_embed_dim,
        num_heads,
        checkpoint,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_std=0.02,
    ):
        super().__init__()

        self.checkpoint = checkpoint

        self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_embed_dim))

        # Create positional embedding for the predictor (not learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, predictor_embed_dim), requires_grad=False)
        values = get_2d_sincos_pos_embed(predictor_embed_dim, int(num_patches ** 0.5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(values).float().unsqueeze(0))

        # Create the transformer blocks for the predictor
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # Stochastic depth decay - more drop for deeper blocks
        self.predictor_blocks = nn.ModuleList([
            Block(
                dim=predictor_embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)
        ])
        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, embed_dim, bias=True)

        # Initialize weights
        self.init_std = init_std
        trunc_normal_(self.mask_token, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def load_weights(self, weight_path, device):
        checkpoint = torch.load(weight_path, map_location=device)

        state_dict = checkpoint.get("state_dict", checkpoint)

        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        errors = []
        try:
            self.load_state_dict(clean_state_dict)
            return
        except Exception as e:
            errors.append(("ijepa", str(e)))
        
        raise ValueError(
            f"Failed to load weights from {weight_path}. "
            f"Tried: {clean_state_dict.keys()}"
        )
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    """
    x = output of the encoder for the context tokens (shape: [B, n_keep, D])
    masks_x = list of tensors containing indices of context tokens in [N] to keep (shape: [B, n_keep])
    masks = list of tensors containing indices of masked tokens in [N] to predict (shape: [B, n_mask])
    """
    def forward(self, x, masks_x, masks):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        if not isinstance(masks_x, list):
            masks_x = [masks_x]

        if not isinstance(masks, list):
            masks = [masks]

        # Batch Size
        B = len(x) // len(masks_x)

        # Map from encoder-dim to pedictor-dim
        x = self.predictor_embed(x)

        # Add positional embedding to x tokens
        x_pos_embed = self.pos_embed.repeat(B, 1, 1)
        x += apply_masks(x_pos_embed, masks_x)

        _, N_ctxt, D = x.shape

        # Concat mask tokens to x
        pos_embs = self.pos_embed.repeat(B, 1, 1)
        pos_embs = apply_masks(pos_embs, masks)
        pos_embs = repeat_interleave_batch(pos_embs, B, repeat=len(masks_x))
        pred_tokens = self.mask_token.repeat(pos_embs.size(0), pos_embs.size(1), 1)
        pred_tokens += pos_embs
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # Forward propagation
        if not self.checkpoint:
            for blk in self.predictor_blocks:
                x = blk(x)
        else:
            for blk in self.predictor_blocks:
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
        x = self.predictor_norm(x)

        # Return preds for mask tokens
        x = x[:, N_ctxt:]
        x = self.predictor_proj(x)

        return x

def vit_predictor(num_patches, embed_dim, depth, predictor_embed_dim, num_heads, checkpoint):
    return VisionTransformerPredictor(num_patches=num_patches, embed_dim=embed_dim, depth=depth, 
                                      predictor_embed_dim=predictor_embed_dim, num_heads=num_heads, 
                                      checkpoint=checkpoint)

def vit_tiny(patch_size, checkpoint):
    return VisionTransformer(embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, patch_size=patch_size, checkpoint=checkpoint)

def vit_small(patch_size, checkpoint):
    return VisionTransformer(embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, patch_size=patch_size, checkpoint=checkpoint)

def vit_base(patch_size, checkpoint):
    return VisionTransformer(embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, patch_size=patch_size, checkpoint=checkpoint)

def vit_large(patch_size, checkpoint):
    return VisionTransformer(embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, patch_size=patch_size, checkpoint=checkpoint)

def vit_huge(patch_size, checkpoint):
    return VisionTransformer(embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, patch_size=patch_size, checkpoint=checkpoint)

def vit_giant(patch_size, checkpoint):
    return VisionTransformer(embed_dim=1408, depth=40, num_heads=16, mlp_ratio=48/11, patch_size=patch_size, checkpoint=checkpoint)
