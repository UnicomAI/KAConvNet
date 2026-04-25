import einops
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Union
import math

from timm.layers import DropPath
from torch import nn

import torch



class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """

    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = torch.sigmoid(x + 0.001)
        return inputs * x.view(-1, self.input_channels, 1, 1)


class ConvFFN(nn.Module):

    def __init__(self, in_channels, internal_channels, out_channels, drop_path):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.preffn_bn = nn.BatchNorm2d(in_channels)
        self.pw1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1, padding=0,
                      groups=1),
            nn.BatchNorm2d(internal_channels)
        )
        self.pw2 = nn.Sequential(
            nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0,
                      groups=1),
            nn.BatchNorm2d(out_channels)
        )

        self.nonlinear = nn.GELU()

    def forward(self, x):
        out = self.preffn_bn(x)
        out = self.pw1(out)
        out = self.nonlinear(out)
        out = self.pw2(out)
        return x + self.drop_path(out)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, drop_path):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, 1, padding=kernel_size // 2)

        self.se = SEBlock(in_channels, in_channels // 4)

        self.norm = nn.BatchNorm2d(in_channels)

        self.convffn = ConvFFN(in_channels, in_channels * 4, in_channels, drop_path)

    def forward(self, x):
        x = self.se(self.norm(self.conv(x))) + x

        return self.convffn(x)
        # return x


class ConvNetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, drop_path, depth):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(in_channels, kernel_size, drop_path) for _ in range(depth)
        ])

    def forward(self, x):
        y = x
        for blk in self.blocks:
            y = blk(y)
        return x + y


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)


class DownSample(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, 3, 2, 1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )


class ConvNet(nn.Module):
    def __init__(self, in_channels, class_nums, channels=None, block_nums=None,
                 drop_path=0.1):
        super().__init__()
        if block_nums is None:
            block_nums = [1, 1, 3, 1]
        if channels is None:
            channels = [32, 64, 128, 256]
        self.stem = Stem(in_channels, channels[0])
        self.stage1 = ConvNetBlock(channels[0], 3, drop_path, block_nums[0])
        self.downsample2 = DownSample(channels[0], channels[1])
        self.stage2 = ConvNetBlock(channels[1], 3, drop_path, block_nums[1])
        self.downsample3 = DownSample(channels[1], channels[2])
        self.stage3 = ConvNetBlock(channels[2], 3, drop_path, block_nums[2])
        self.downsample4 = DownSample(channels[2], channels[3])
        self.stage4 = ConvNetBlock(channels[3], 3, drop_path, block_nums[3])

        self.dropout4 = nn.Dropout(0.25)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()

        self.linear1 = nn.Linear(channels[3], 1024)
        self.linear2 = nn.Linear(1024, class_nums)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(self.downsample2(x))
        x = self.stage3(self.downsample3(x))
        x = self.stage4(self.downsample4(x))
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.dropout4(x)
        x = self.linear2(x)
        return x


if __name__ == "__main__":
    import time

    # input = torch.rand(1, 3, 224, 224)
    model = KAConvNet(3, 1000)
    # output = model(input)
    # print(output.shape)
    total = sum([param.nelement() for param in model.parameters()])
    # 精确地计算：1MB=1024KB=1048576字节
    print('Number of parameter: % .4fM' % (total / 1048576))
