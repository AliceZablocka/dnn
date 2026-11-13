"""Architektura CustomPlantCNN (przeniesiona 1:1 z notebooka)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleResBlock(nn.Module):
    """Blok rezydualny: Conv3x3-BN-ReLU-Conv3x3-BN + skip connection."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class CustomPlantCNN(nn.Module):
    """Stem 7x7 + 4 bloki rezydualne (64-128-256-512) + GAP + Dropout + FC."""

    def __init__(self, num_classes: int = 6, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = SimpleResBlock(64, 64, stride=1)     # 56x56
        self.layer2 = SimpleResBlock(64, 128, stride=2)    # 28x28
        self.layer3 = SimpleResBlock(128, 256, stride=2)   # 14x14
        self.layer4 = SimpleResBlock(256, 512, stride=2)   # 7x7

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = torch.flatten(self.avgpool(x), 1)
        x = self.dropout(x)
        return self.fc(x)  # logity (sigmoid dopiero przy predykcji)