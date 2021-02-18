import torch.nn as nn
import torch


class Block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, padding=kernel_size // 2 + 1,
                              padding_mode="zeros")
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._blocks = nn.Sequential(
            Block(3, 64),
            Block(64, 128),
            Block(128, 256),
            Block(256, 512),
            Block(512, 512),
            Block(512, 512),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self._blocks(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
