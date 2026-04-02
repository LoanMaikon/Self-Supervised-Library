'''
Code from https://github.com/ajtejankar/byol-convert/blob/main/resnet.py
'''

import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
import torch.nn as nn
from math import ceil
import torch
import os

from src.utils import is_main_process

def calc_padding_same(in_height, in_width, strides, filter_height, filter_width):
    out_height = ceil(float(in_height) / float(strides[0]))
    out_width = ceil(float(in_width) / float(strides[1]))
    pad_along_height = max((out_height - 1) * strides[0] + filter_height - in_height, 0)
    pad_along_width = max((out_width - 1) * strides[1] + filter_width - in_width, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return (pad_left, pad_right, pad_top, pad_bottom)

def pad_same(inp, filt):
    if isinstance(filt, nn.MaxPool2d):
        filt_kernel = [filt.kernel_size]*2
        filt_stride = [filt.stride]*2
    else:
        filt_kernel = filt.kernel_size
        filt_stride = filt.stride
    padding = calc_padding_same(*inp.shape[2:], filt_stride, *filt_kernel)
    return F.pad(inp, padding)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=0, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = pad_same(x, self.conv1)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = pad_same(out, self.conv2)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = pad_same(out, self.conv3)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            x = pad_same(x, self.downsample[0])
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, use_checkpoint, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, width_multiplier=1):
        super(ResNet, self).__init__()
        self.use_checkpoint = use_checkpoint
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64 * width_multiplier
        dim_inner = 64 * width_multiplier
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=0,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.layer1 = self._make_layer(block, dim_inner, layers[0])
        self.layer2 = self._make_layer(block, dim_inner * 2, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, dim_inner * 4, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, dim_inner * 8, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width_multiplier* 512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)
        
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        for param in self.fc.parameters():
            param.requires_grad = True
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def get_features(self, features):
        return features
    
    def get_output_dim(self):
        return self.fc.in_features
    
    def remove_classifier_head(self):
        self.fc = nn.Identity()

    def fit_classifier_head(self, num_classes):
        self.fc = nn.Linear(self.fc.in_features, num_classes)

    def load_weights(self, weight_path, device):
        checkpoint = torch.load(weight_path, map_location=device)

        state_dict = checkpoint.get("state_dict", checkpoint)
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        errors = []
        try:
            self.load_state_dict(clean_state_dict)
            return
        except Exception as e:
            errors.append(("projection_head", str(e)))

        try:
            self.remove_classifier_head()
            self.load_state_dict(clean_state_dict)
            return
        except Exception as e:
            errors.append(("remove_classifier_head", str(e)))

        try:
            self.fit_classifier_head(1000)
            self.load_state_dict(clean_state_dict)
            return
        except Exception as e:
            errors.append(("fit_classifier_head", str(e)))
        
        raise ValueError(
            f"Failed to load weights from {weight_path}. "
            f"Tried: {errors}"
        )
    
    def _forward_impl_checkpoint(self, x):
        # See note [TorchScript super()]
        x = pad_same(x, self.conv1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = pad_same(x, self.maxpool)
        x = self.maxpool(x)

        x = checkpoint.checkpoint(self.layer1, x)
        x = checkpoint.checkpoint(self.layer2, x)
        x = checkpoint.checkpoint(self.layer3, x)
        x = checkpoint.checkpoint(self.layer4, x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = pad_same(x, self.conv1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = pad_same(x, self.maxpool)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x) if not self.use_checkpoint else self._forward_impl_checkpoint(x)

class MLPHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=4096, out_dim=256):
        super(MLPHead, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.projection_head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.projection_head(x)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def load_weights(self, weight_path, device):
        checkpoint = torch.load(weight_path, map_location=device)

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
            f"Tried: {errors}"
        )

def _resnet(arch, block, layers, pretrained, progress, use_checkpoint, **kwargs):
    model = ResNet(block, layers, use_checkpoint, **kwargs)
    return model

def resnet50(pretrained=False, progress=True, use_checkpoint=False, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress, use_checkpoint,
                   **kwargs)

def resnet200(pretrained=False, progress=True, use_checkpoint=False, **kwargs):
    r"""ResNet-200 2x model from BYOL

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_multiplier'] = 2
    return _resnet('resnet200', Bottleneck, [3, 24, 36, 3], pretrained, progress, use_checkpoint,
                   **kwargs)

def mlp_head(in_dim, hidden_dim=4096, out_dim=256):
    return MLPHead(in_dim, hidden_dim, out_dim)
