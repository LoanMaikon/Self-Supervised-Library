# Code from https://github.com/facebookresearch/swav/blob/main/src/resnet50.py

import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
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
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
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

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(
            self,
            block,
            layers,
            use_checkpoint,
            zero_init_residual=False,
            groups=1,
            widen=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=None,
            normalize=False,
            output_dim=0,
            hidden_mlp=0,
            nmb_prototypes=0,
            eval_mode=False,
    ):
        super(ResNet, self).__init__()

        self.use_checkpoint = use_checkpoint

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.eval_mode = eval_mode
        self.padding = nn.ConstantPad2d(1, 0.0)

        self.inplanes = width_per_group * widen
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # change padding 3 -> 2 compared to original torchvision code because added a padding layer
        num_out_filters = width_per_group * widen
        self.conv1 = nn.Conv2d(
            3, num_out_filters, kernel_size=7, stride=2, padding=2, bias=False
        )
        self.bn1 = norm_layer(num_out_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, num_out_filters, layers[0])
        num_out_filters *= 2
        self.layer2 = self._make_layer(
            block, num_out_filters, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        num_out_filters *= 2
        self.layer3 = self._make_layer(
            block, num_out_filters, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        num_out_filters *= 2
        self.layer4 = self._make_layer(
            block, num_out_filters, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # normalize output features
        self.l2norm = normalize

        self.encoder_out_features = num_out_filters * block.expansion

        # projection head
        if output_dim == 0:
            self.projection_head = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(num_out_filters * block.expansion, output_dim)
        else:
            self.projection_head = nn.Sequential(
                nn.Linear(num_out_filters * block.expansion, hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_mlp, output_dim),
            )

        # prototype layer
        if output_dim > 0 and nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)
        else:
            self.prototypes = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
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
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

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
            self.remove_unnecessary_modules()
            self.remove_projection_head()
            self.load_state_dict(clean_state_dict)
            return
        except Exception as e:
            errors.append(("remove_unnecessary_modules", str(e)))
        
        raise ValueError(
            f"Failed to load weights from {weight_path}. "
            f"Tried: {errors}"
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
    
    def remove_projection_head(self):
        self.projection_head = None
    
    def remove_classifier_head(self):
        return
    
    def get_output_dim(self):
        return self.encoder_out_features

    def forward_normal(self, x):
        x = self.padding(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x

    def forward_checkpoint(self, x):
        x = self.padding(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = torch.utils.checkpoint.checkpoint(self.layer1, x)
        x = torch.utils.checkpoint.checkpoint(self.layer2, x)
        x = torch.utils.checkpoint.checkpoint(self.layer3, x)
        x = torch.utils.checkpoint.checkpoint(self.layer4, x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x
    
    def remove_unnecessary_modules(self):
        self.projection_head = None
        self.prototypes = None

    def forward(self, inputs):
        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in inputs]),
            return_counts=True,
        )[1], 0)
        start_idx = 0
        for end_idx in idx_crops:
            if self.use_checkpoint:
                _out = self.forward_checkpoint(torch.cat(inputs[start_idx: end_idx]).to(inputs[0].device))
            else:
                _out = self.forward_normal(torch.cat(inputs[start_idx: end_idx]).to(inputs[0].device))
            if start_idx == 0:
                output = _out
            else:
                output = torch.cat((output, _out))
            start_idx = end_idx
        
        return output
    
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProjectionHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )
    
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
        
        raise ValueError(
            f"Failed to load weights from {weight_path}. "
            f"Tried: {errors}"
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.mlp(x)

class Prototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super(Prototypes, self).__init__()
        self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

    def load_weights(self, weight_path, device):
        checkpoint = torch.load(weight_path, map_location=device)

        state_dict = checkpoint.get("state_dict", checkpoint)
        clean_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        errors = []
        try:
            self.load_state_dict(clean_state_dict)
            return
        except Exception as e:
            errors.append(("prototypes", str(e)))
        
        raise ValueError(
            f"Failed to load weights from {weight_path}. "
            f"Tried: {errors}"
        )

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.prototypes(x)
    
def projection_head(input_dim, hidden_dim, output_dim):
    return ProjectionHead(input_dim, hidden_dim, output_dim)

def resnet50(use_checkpoint, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], use_checkpoint=use_checkpoint, **kwargs)

def resnet50w2(use_checkpoint, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=2, use_checkpoint=use_checkpoint, **kwargs)

def resnet50w4(use_checkpoint, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=4, use_checkpoint=use_checkpoint, **kwargs)

def resnet50w5(use_checkpoint, **kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], widen=5, use_checkpoint=use_checkpoint, **kwargs)

def prototypes(output_dim, nmb_prototypes):
    return Prototypes(output_dim, nmb_prototypes)
