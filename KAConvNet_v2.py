import einops
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Union
import math

from timm.layers import DropPath
from torch import nn
import time

# from torchstat import stat

import torch


def calc_out_dims(matrix, kernel_side, stride, dilation, padding):
    batch_size, n_channels, n, m = matrix.shape
    h = np.floor((n + 2 * padding - kernel_side - (kernel_side - 1) * (dilation - 1)) / stride).astype(int) + 1
    return h, h, batch_size


class PLinear(torch.nn.Module):
    def __init__(self, normal_shape):
        super(PLinear, self).__init__()
        self.normal_shape = normal_shape
        self.alpha1 = nn.Parameter(torch.randn(normal_shape) / 2, requires_grad=True)
        self.alpha2 = nn.Parameter(torch.randn(normal_shape) / 2, requires_grad=True)
        self.alpha3 = nn.Parameter(torch.zeros(normal_shape), requires_grad=True)
        # self.base_act = nn.Tanh()

    def forward(self, x):
        return torch.where(x < 0, self.alpha1 * x, self.alpha2 * x) + self.alpha3


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=[2, 3], keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


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
        tmp_time0 = time.time()
        act_1 = self.act1(x)
        tmp_time1 = time.time()
        print("act1", tmp_time1 - tmp_time0)
        kaconv1_feature = self.conv1(act_1)
        tmp_time2 = time.time()
        print("dwconv1", tmp_time2 - tmp_time1)
        print("dwconv1+act1", tmp_time2 - tmp_time0)
        h, w, batch_size = calc_out_dims(x, self.kernel_size, self.stride, self.dilation, self.padding)
        unfold_feature = self.unfold(x)
        unfold_feature = einops.rearrange(unfold_feature, 'b (cin k2) c -> b cin c k2', cin=self.in_channels,
                                          k2=self.kernel_size * self.kernel_size)
        
        # tmp_time6 = time.time()
        # temp_kaconv2_feature = self.conv2(self.act2(unfold_feature))
        # tmp_time7 = time.time()
        # print("self.conv2(self.act2(unfold_feature))", tmp_time7 - tmp_time6)
        
        tmp_time3 = time.time()
        act_2 = self.act2(unfold_feature)
        tmp_time4 = time.time()
        kaconv2_feature = self.conv2(act_2)
        tmp_time5 = time.time()
        print("act2", tmp_time4 - tmp_time3)
        print("dwconv2", tmp_time5 - tmp_time4)
        print("dwconv2+act2", tmp_time5 - tmp_time3)
        kaconv2_feature = einops.rearrange(kaconv2_feature, 'b cout n m -> b cout (n m)')
        kaconv2_feature = kaconv2_feature.reshape(batch_size, self.out_channels, h, w)
        result = kaconv1_feature * kaconv2_feature
        return result


class KAConvolutionLayer(torch.nn.Module):
    def __init__(
            self,
            in_channels=64,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1
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
        self.mlp1 = nn.Conv2d(in_channels * (2 * kernel_size * kernel_size + 1), out_channels, 1, 1, groups=groups)
        self.act_mlp1 = nn.PReLU(in_channels * (2 * kernel_size * kernel_size + 1), 0.2)
        # self.mlp2 = nn.Conv2d(in_channels * (2 * kernel_size * kernel_size + 1), out_channels, 1, 1)
        # self.act_mlp2 = PLinear((1, in_channels * (2 * kernel_size * kernel_size + 1), 1, 1))

    def forward(self, x):
        temp_time1 = time.time()
        kaconv_feature = self.convkan(x)
        temp_time2 = time.time()
        # print("self.convkan", temp_time2 - temp_time1)
        
        kaconv_feature = self.norm(kaconv_feature)
        temp_time3 = time.time()
        kaconv_feature = self.mlp1(self.act_mlp1(kaconv_feature))
        temp_time4 = time.time()
        print("dwconv3", temp_time4 - temp_time3)

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
        x = torch.sigmoid(x + 0.001)
        return inputs * x.view(-1, self.input_channels, 1, 1)


class ConvFFN(nn.Module):

    def __init__(self, in_channels, internal_channels, out_channels):
        super().__init__()
        self.pw1 = nn.Conv2d(in_channels=in_channels, out_channels=internal_channels, kernel_size=1, stride=1,
                             padding=0, groups=1)

        self.pw2 = nn.Conv2d(in_channels=internal_channels, out_channels=out_channels, kernel_size=1, stride=1,
                             padding=0, groups=1)

        self.grn = GRN(internal_channels)
        self.nonlinear = nn.GELU()
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.pw1(x)
        out = self.grn(self.nonlinear(out))
        out = self.pw2(out)
        return out


class KAConvBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, drop_path):
        super().__init__()
        self.conv = KAConvolutionLayer(in_channels, in_channels, kernel_size, padding=kernel_size // 2,
                                       groups=in_channels)

        self.se = SEBlock(in_channels, in_channels // 4)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm = nn.BatchNorm2d(in_channels)

        self.convffn = ConvFFN(in_channels, in_channels * 4, in_channels)

    def forward(self, x):
        out = self.convffn(self.se(self.norm(self.conv(x))))
        return x + self.drop_path(out)


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
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, 3, 2, 1, groups=out_channels),
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
            block_nums = [2, 2, 3, 1]
        if channels is None:
            channels = [32, 64, 128, 256]
        self.stem = Stem(in_channels, channels[0])
        self.stage1 = KAConvNetBlock(channels[0], 5, drop_path, block_nums[0])
        self.downsample2 = DownSample(channels[0], channels[1])
        self.stage2 = KAConvNetBlock(channels[1], 3, drop_path, block_nums[1])
        self.downsample3 = DownSample(channels[1], channels[2])
        self.stage3 = KAConvNetBlock(channels[2], 3, drop_path, block_nums[2])
        self.downsample4 = DownSample(channels[2], channels[3])
        self.stage4 = KAConvNetBlock(channels[3], 3, drop_path, block_nums[3])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()

        self.linear1 = nn.Linear(channels[3], class_nums)
        # self.linear2 = nn.Linear(1024, class_nums)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(self.downsample2(x))
        x = self.stage3(self.downsample3(x))
        x = self.stage4(self.downsample4(x))
        x = self.avgpool(x)
        x = self.flat(x)
        x = self.linear1(x)
        return x


if __name__ == "__main__":
    import time
    from torchvision import models
    
    device = 'cuda:0'

    temp_input = torch.rand(4, 3, 224, 224).to(device)
    temp_model = models.resnet18(pretrained=False).to(device)
    for i in range(50):
        output = temp_model(temp_input)
        
    
    input_data = torch.randn(32, 256, 64, 64).to(device)
    model = KAConvolutionLayer(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1).to(device)
    model2 = nn.Conv2d(256, 256, 3, 1, 1).to(device)
    model3 = nn.SiLU()
    start_time = time.time()
    
    output = model(input_data)
    
    end_time = time.time()
    
    print("KAConvLayer", end_time - start_time)
    
    start_time = time.time()
    output = model3(model2(input_data))
    end_time = time.time()
    
    print("ConvLayer", end_time - start_time)