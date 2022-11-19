import torch
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, input_dim=3, dim=128):
        super().__init__()
        self.first_conv = nn.Conv2d(input_dim, dim, 3, 1, 1)
        self.avgpool2d = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.avgpool2d(x).flatten(1)
        x = self.linear(x)
        return x