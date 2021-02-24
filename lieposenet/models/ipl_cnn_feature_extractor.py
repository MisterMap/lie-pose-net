import torch.nn as nn
import torch


class Block(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size, padding=kernel_size // 2,
                              padding_mode="reflect")
        self.bn = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        before_max_pool = x
        x = self.maxpool(x)
        return x, before_max_pool


class IPL(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, stride=2, padding=None):
        super().__init__()
        self._repeats = 2 if output_channel > input_channel else 1
        self.maxpool = nn.MaxPool2d(kernel_size=stride, stride=stride)
        self.sum_conv = nn.Conv2d(input_channel, input_channel, bias=False,
                                  kernel_size=stride, stride=stride, padding=0, padding_mode="reflect")
        nn.init.constant_(self.sum_conv.weight.data, 0)
        self.sum_conv.weight.data[torch.arange(0, input_channel), torch.arange(0, input_channel)] = 1
        self.sum_conv.weight.requires_grad = False
        if padding is None:
            padding = kernel_size // 2
        self.px_conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size,
                                       padding=padding, padding_mode="reflect")
        self.py_conv = nn.Conv2d(input_channel, output_channel, kernel_size=kernel_size,
                                       padding=padding, padding_mode="reflect")
        self.upsample = nn.Upsample(scale_factor=stride)

    def forward(self, px, py, x):
        max_x = self.maxpool(x)
        dx = x - self.upsample(max_x)
        exp_x = torch.exp(dx)
        sum_exp_x = self.sum_conv(exp_x)
        assert torch.all(sum_exp_x >= 1.)
        px = self.sum_conv(exp_x * px) / sum_exp_x
        py = self.sum_conv(exp_x * py) / sum_exp_x
        assert torch.all(px == px)
        return self.px_conv(px), self.py_conv(py)
        # return px.repeat_interleave(self._repeats, dim=1), py.repeat_interleave(self._repeats, dim=1)


class IPLCNNFeatureExtractor(nn.Module):
    def __init__(self, dimension_size=16):
        super().__init__()
        self._dimension_size = dimension_size
        self._blocks = nn.ModuleList([
            Block(3, dimension_size, stride=4),
            Block(dimension_size, dimension_size, stride=4),
            Block(dimension_size, dimension_size, stride=4),
            Block(dimension_size, dimension_size, stride=2),
            # Block(dimension_size, dimension_size),
            # Block(dimension_size, dimension_size),
            # Block(dimension_size, dimension_size)
        ])
        self._ipl_blocks = nn.ModuleList([
            IPL(dimension_size, dimension_size, stride=4),
            IPL(dimension_size, dimension_size, stride=4),
            IPL(dimension_size, dimension_size, stride=4),
            IPL(dimension_size, dimension_size, stride=2, kernel_size=1),
            # IPL(dimension_size, dimension_size),
            # IPL(dimension_size, dimension_size),
            # IPL(dimension_size, dimension_size)
        ])
        # self.last_conv = nn.Conv2d(3 * dimension_size, 3 * dimension_size, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(3 * dimension_size, 1)
        self.fx = None
        self.fy = None

    def forward(self, x):
        px = torch.linspace(-1, 1, x.shape[2], device=x.device)
        px = px[:, None].expand(x.shape[0], self._dimension_size, x.shape[2], x.shape[3])
        py = torch.linspace(-1, 1, x.shape[3], device=x.device)
        py = py[None, :].expand(x.shape[0], self._dimension_size, x.shape[2], x.shape[3])
        for block, ipl_block in zip(self._blocks, self._ipl_blocks):
            x, before_max_pool = block(x)
            px, py = ipl_block(px, py, before_max_pool)
        self.px = px
        self.py = py
        x = torch.cat([px, py, x], dim=1)
        # x = self.last_conv()
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
