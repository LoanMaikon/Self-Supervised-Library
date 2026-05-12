import torch.nn.functional as f
import torch.nn as nn
import torch

INPUT_SHAPE = (3, 32, 32)
NUM_CLASSES = 2

class Hochuli(nn.Module):
    def __init__(self, in_channels):
        super(Hochuli, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1600, 64)
        self.fc2 = nn.Identity()

        self.out_features = 64

    def forward(self, x):
        x = f.relu(self.conv1(x))
        x = self.pool1(x)
        x = f.relu(self.conv2(x))
        x = self.pool2(x)
        x = f.relu(self.conv3(x))
        x = self.flatten(x)
        x = f.relu(self.fc1(x))
        x = self.fc2(x)

        return x
    
    def load_weights(self, weight_path, device):
        checkpoint = torch.load(weight_path, map_location=device)

        state_dict = checkpoint.get("state_dict", checkpoint)

        errors = []

        try:
            self.load_state_dict(state_dict)
            return
        except Exception as e:
            errors.append(("direct_load", str(e)))

        raise RuntimeError(f"Failed to load weights from {weight_path} with the following errors: {errors}")
    
    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

    def get_out_features(self):
        return self.out_features

    def remove_classifier_head(self):
        self.fc2 = nn.Identity()
    
    def eval_forward(self, x):
        return self.forward(x)

    def get_eval_output_dim(self):
        return self.out_features
