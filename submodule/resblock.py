import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, pad=1, activation=F.relu, downsample=False):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d