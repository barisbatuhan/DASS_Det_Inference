import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .network_blocks import BaseConv

class YOLOXHeadStem(nn.Module):

    def __init__(self,
                 width=1.0,
                 in_channels=[256, 512, 1024],
                 act="silu"):
        
        super().__init__()
        
        self.stems = nn.ModuleList()
        
        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )

    def forward(self, xin):
        outputs = []
        for k, x in enumerate(xin):
            outputs.append(self.stems[k](x))
        return outputs
