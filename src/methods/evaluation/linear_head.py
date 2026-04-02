import torch


class LinearHead(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearHead, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=True)

    def unfreeze(self):
        for param in self.linear.parameters():
            param.requires_grad = True

    def freeze(self):
        for param in self.linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.linear(x)
