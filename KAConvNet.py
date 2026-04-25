import einops
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Union
import math

from timm.layers import DropPath
from torch import nn

import torch


def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size, n_channels, n, m = matrix.shape
    h = np.floor((n + 2 * padding - kernel_side - (kernel_side - 1) * (dilation - 1)) / stride).astype(int) + 1
    return h, h, batch_size


class PSiLU(torch.nn.Module):
    def __init__(self, normal_shape):
        super(PSiLU, self).__init__()
        self.normal_shape = normal_shape
        self.alpha = nn.Parameter(torch.ones(normal_shape), requires_grad=True)
        self.base_act = nn.SiLU()

    def forward(self, x):
        return self.alpha * self.base_act(x)

class PTanh(torch.nn.Module):
    def __init__(self, normal_shape):
        super(PTanh, self).__init__()
        self.normal_shape = normal_shape
        self.alpha1 = nn.Parameter(torch.ones(normal_shape) / 2, requires_grad=True)
        self.alpha2 = nn.Parameter(torch.ones(normal_shape) / 2, requires_grad=True)
        self.alpha3 = nn.Parameter(torch.randn(normal_shape) / 10, requires_grad=True)
        self.base_act = nn.Tanh()

    def forward(self, x):
        return torch.where(x < 0, self.alpha1 * self.base_act(x), self.alpha2 * self.base_act(x)) + self.alpha3
        
        
class PLinear(torch.nn.Module):
    def __init__(self, normal_shape):
        super(PLinear, self).__init__()
        self.normal_shape = normal_shape
        self.alpha1 = nn.Parameter(torch.randn(normal_shape) / 2 , requires_grad=True)
        self.alpha2 = nn.Parameter(torch.randn(normal_shape) / 2, requires_grad=True)
        self.alpha3 = nn.Parameter(torch.rand(normal_shape) / 10, requires_grad=True)
        # self.base_act = nn.Tanh()

    def forward(self, x):
        return torch.where(x < 0, self.alpha1 * x, self.alpha2 * x) + self.alpha3


class KAConvolution(torch.nn.Module):
    def __init__(
            self,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1
    ):
        """
        Args
        """
        super(KAConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                               padding=padding, dilation=dilation, groups=in_channels)

        self.act1 = nn.SiLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, kernel_size * kernel_size), stride=1,
                              padding=0, dilation=1, groups=in_channels)

        self.act2 = PLinear((1, in_channels, 1, kernel_size * kernel_size))
        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x: torch.Tensor):
        kaconv1_feature = self.conv1(self.act1(x))
        h, w, batch_size = calc_out_dims(x, self.kernel_size, self.stride, self.dilation, self.padding)
        unfold_feature = self.unfold(x)
        unfold_feature = einops.rearrange(unfold_feature, 'b (cin k2) c -> b cin c k2', cin=self.in_channels,
                                          k2=self.kernel_size * self.kernel_size)
        kaconv2_feature = self.conv2(self.act2(unfold_feature))
        kaconv2_feature = einops.rearrange(kaconv2_feature, 'b cout n m -> b cout (n m)')
        kaconv2_feature = kaconv2_feature.reshape(batch_size, self.out_channels, h, w)
        return kaconv1_feature + kaconv2_feature


class KAConvolutionLayer(torch.nn.Module):
    def __init__(
            self,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1
    ):
        """
        Args
        """
        super(KAConvolutionLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.convkan = KAConvolution(in_channels, in_channels * (2 * kernel_size * kernel_size + 1), kernel_size,
                                     stride, padding, dilation)
        self.norm = nn.BatchNorm2d(in_channels * (2 * kernel_size * kernel_size + 1))
        self.mlp1 = nn.Conv2d(in_channels * (2 * kernel_size * kernel_size + 1), out_channels, 1, 1)
        self.act_mlp1 = nn.PReLU(in_channels * (2 * kernel_size * kernel_size + 1), 0.2)
        # self.mlp2 = nn.Conv2d(in_channels * (2 * kernel_size * kernel_size + 1), out_channels, 1, 1)
        # self.act_mlp2 = PLinear((1, in_channels * (2 * kernel_size * kernel_size + 1), 1, 1))

    def forward(self, x: torch.Tensor):
        kaconv_feature = self.convkan(x)
        kaconv_feature = self.norm(kaconv_feature)
        # kaconv_feature = self.mlp1(self.act_mlp1(kaconv_feature)) * self.mlp2(self.act_mlp2(kaconv_feature))
        kaconv_feature = self.mlp1(self.act_mlp1(kaconv_feature))

        return kaconv_feature


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
        x = torch.sigmoid(x + 0.0001)
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


class KAConvBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, drop_path):
        super().__init__()
        self.conv = KAConvolutionLayer(in_channels, in_channels, kernel_size, padding=kernel_size // 2)

        self.se = SEBlock(in_channels, in_channels // 4)

        self.norm = nn.BatchNorm2d(in_channels)

        self.convffn = ConvFFN(in_channels, in_channels * 4, in_channels, drop_path)

    def forward(self, x):
        x = self.se(self.norm(self.conv(x))) + x

        return self.convffn(x)
        # return x


class KAConvNetBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, drop_path, depth):
        super().__init__()
        self.blocks = nn.ModuleList([
            KAConvBlock(in_channels, kernel_size, drop_path) for _ in range(depth)
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
            KAConvolutionLayer(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
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


class KAConvNet(nn.Module):
    def __init__(self, in_channels, class_nums, channels=None, block_nums=None,
                 drop_path=0.1):
        super().__init__()
        if block_nums is None:
            block_nums = [2, 2, 6, 2]
        if channels is None:
            channels = [16, 32, 64, 128]
        self.stem = Stem(in_channels, channels[0])
        self.stage1 = KAConvNetBlock(channels[0], 5, drop_path, block_nums[0])
        self.downsample2 = DownSample(channels[0], channels[1])
        self.stage2 = KAConvNetBlock(channels[1], 3, drop_path, block_nums[1])
        self.downsample3 = DownSample(channels[1], channels[2])
        self.stage3 = KAConvNetBlock(channels[2], 3, drop_path, block_nums[2])
        self.downsample4 = DownSample(channels[2], channels[3])
        self.stage4 = KAConvNetBlock(channels[3], 3, drop_path, block_nums[3])

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
    from torchvision import models

    temp_input = torch.rand(4, 3, 224, 224)
    temp_model = models.resnet18(pretrained=False)
    for i in range(20):
        output = temp_model(temp_input)
        
    
    global start_time 
    start_time = time.time()
    input_data = torch.randn(1, 256, 32, 32)
    model = KAConvolutionLayer(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1)
    
    end_time = time.time()
    
    print(end_time - start_time)
