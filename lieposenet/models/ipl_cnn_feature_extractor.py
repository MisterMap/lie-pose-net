import torch.nn as nn
import torch


class Block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, padding=kernel_size // 2,
                              padding_mode="zeros")
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        before_max_pool = x
        x = self.maxpool(x)
        return x, before_max_pool


class IPL(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3):
        super().__init__()
        self._repeats = 2 if output_channel > input_channel else 1
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sum_conv = nn.Conv2d(input_channel, input_channel, bias=False,
                                  kernel_size=2, stride=2, padding=0, padding_mode="reflect")
        nn.init.constant_(self.sum_conv.weight.data, 0)
        self.sum_conv.weight.data[torch.arange(0, input_channel), torch.arange(0, input_channel)] = 1
        self.sum_conv.weight.requires_grad = False
        # self.px_conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size,
        #                                padding=kernel_size // 2, padding_mode="reflect")
        # self.py_conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size,
        #                                padding=kernel_size // 2, padding_mode="reflect")
        self.upsample = nn.Upsample(scale_factor=2)

    def forward(self, px, py, x):
        max_x = self.maxpool(x)
        dx = x - self.upsample(max_x)
        exp_x = torch.exp(dx)
        sum_exp_x = self.sum_conv(exp_x)
        assert torch.all(sum_exp_x >= 1.)
        px = self.sum_conv(exp_x * px) / sum_exp_x
        py = self.sum_conv(exp_x * py) / sum_exp_x
        assert torch.all(px == px)
        # return self.px_conv(px), self.py_conv(py)
        return px.repeat_interleave(self._repeats, dim=1), py.repeat_interleave(self._repeats, dim=1)

class IPLCNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self._blocks = nn.ModuleList([
            Block(3, 64),
            Block(64, 128),
            Block(128, 256),
            Block(256, 512),
            Block(512, 512),
            Block(512, 512)
        ])
        self._ipl_blocks = nn.ModuleList([
            IPL(64, 128),
            IPL(128, 256),
            IPL(256, 512),
            IPL(512, 512),
            IPL(512, 512),
            IPL(512, 512)
        ])
        self.last_conv = nn.Conv2d(1024, 512, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        px = torch.linspace(-1, 1, x.shape[2], device=x.device)
        px = px[:, None].expand(x.shape[0], 64, x.shape[2], x.shape[3])
        py = torch.linspace(-1, 1, x.shape[3], device=x.device)
        py = py[None, :].expand(x.shape[0], 64, x.shape[2], x.shape[3])
        for block, ipl_block in zip(self._blocks, self._ipl_blocks):
            x, before_max_pool = block(x)
            px, py = ipl_block(px, py, before_max_pool)
        x = self.last_conv(torch.cat([px, py], dim=1))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
