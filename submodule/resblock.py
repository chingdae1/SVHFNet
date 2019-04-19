import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, k_size=3, pad=1, activation=F.relu, downsample=False):
        super(Block, self).__init__()
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.activation = activation
        self.downsample = downsample
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, k_size, padding=pad)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, k_size, padding=pad)
        nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2))
        self.conv1 = nn.utils.spectral_norm(self.conv1)
        self.conv2 = nn.utils.spectral_norm(self.conv2)
        self.learnable_sc = (in_channels != out_channels) or downsample
        if self.learnable_sc:
            self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            nn.init.xavier_uniform_(self.conv_sc.weight.data, 1.)
            self.conv_sc = nn.utils.spectral_norm(self.conv_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.conv_sc(x)
            if self.downsample:
                return F.avg_pool2d(x, 2)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, pad=1, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, k_size, padding=pad)
        self.conv2 = nn.Conv2d(out_channels, out_channels, k_size, padding=pad)
        nn.init.xavier_uniform_(self.conv1.weight.data, math.sqrt(2))
        nn.init.xavier_uniform_(self.conv2.weight.data, math.sqrt(2))
        self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        nn.init.xavier_uniform_(self.conv_sc.weight.data, 1.)
        self.conv_sc = nn.utils.spectral_norm(self.conv_sc)

    def residual(self, x):
        h = x
        h = self.conv1(h)
        h = self.activation(h)
        h = self.conv2(h)
        h = F.avg_pool2d(h, 2)
        return h

    def shortcut(self, x):
        return self.conv_sc(F.avg_pool2d(x, 2))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)