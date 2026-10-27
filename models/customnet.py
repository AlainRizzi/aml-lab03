"""The three-convolution CNN from Lab 02."""
import torch
from torch import nn


class CustomNet(nn.Module):
    """200-class classifier for Tiny ImageNet.

    Adaptive average pooling permits both 64x64 and 224x224 inputs.
    """
    def __init__(self, num_classes=200):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.conv1(x).relu())
        x = self.pool(self.conv2(x).relu())
        x = self.pool(self.conv3(x).relu())
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc1(x)
