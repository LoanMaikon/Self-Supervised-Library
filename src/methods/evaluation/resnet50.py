import torchvision.models as models
import torch.utils.checkpoint
import torch.nn as nn
import torch


class ResNet50(nn.Module):
    def __init__(self, use_checkpoint, pretrained):
        super(ResNet50, self).__init__()

        self.use_checkpoint = use_checkpoint
        self.pretrained = pretrained
        
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2) if self.pretrained else models.resnet50(weights=None)
    
    def remove_classifier_head(self):
        self.backbone.fc = nn.Identity()
    
    def freeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _forward_impl(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        return x
    
    def _forward_impl_checkpoint(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = torch.utils.checkpoint.checkpoint(self.backbone.layer1, x, use_reentrant=False)
        x = torch.utils.checkpoint.checkpoint(self.backbone.layer2, x, use_reentrant=False)
        x = torch.utils.checkpoint.checkpoint(self.backbone.layer3, x, use_reentrant=False)
        x = torch.utils.checkpoint.checkpoint(self.backbone.layer4, x, use_reentrant=False)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.backbone.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl_checkpoint(x) if self.use_checkpoint else self._forward_impl(x)

def resnet50(use_checkpoint, pretrained):
    return ResNet50(use_checkpoint, pretrained)
