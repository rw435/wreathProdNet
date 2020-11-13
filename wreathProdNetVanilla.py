import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from Vanilla import VOX_mean, activation_func, mySequential

EPS = 1e-8
GPU = 1
PARTITION = 9

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.ModuleList([
            nn.Identity(),
        ]) 
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()
    
    def forward(self, x, voxel_indices, num_points):
        residual = x

        if self.should_apply_shortcut:
            residual = self.shortcut(x)

        for block in self.blocks:
            inputs = (x, voxel_indices, num_points)
            x, _, _ = block(inputs)

        x += residual

        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class ResNetResidualBlock(ResidualBlock):
    def __init__(self, in_channels, out_channels, expansion=1, activation='relu',
                *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.expansion, self.act = expansion, activation
        # self.shortcut = nn.Sequential(
        #     nn.Conv1d(self.in_channels, self.expanded_channels, kernel_size=1,
        #               stride=1, bias=False),
        #     # nn.Linear(self.in_channels, self.out_channels, bias=False)
        #     )
        
        
    @property
    def expanded_channels(self):
        return self.out_channels * self.expansion
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.expanded_channels

def bn_weight_act(in_channels, out_channels, vox_op, activation):
    return mySequential(nn.BatchNorm1d(in_channels),
                        activation_func(activation),
                        vox_op,
                        )

class ResNetBasicBlock(ResNetResidualBlock):
    """
    Basic pre-activation ResNet block
    """
    expansion = 1
    def __init__(self, in_channels, out_channels, *args, **kwargs):
        super().__init__(in_channels, out_channels, *args, **kwargs)
        self.vox_op1 = VOX_mean(in_channels, out_channels)
        self.vox_op2 = VOX_mean(out_channels * self.expansion, out_channels)
        self.blocks = nn.ModuleList([
            bn_weight_act(self.in_channels, self.out_channels, 
                            vox_op=self.vox_op1, activation=self.act),
            bn_weight_act(self.out_channels, self.expanded_channels,
                            vox_op=self.vox_op2, activation=self.act)
        ])

class ResNetLayer(nn.Module):
    """
    A ResNet layer composed by `n` blocks stacked one after the other
    """
    def __init__(self, in_channels, out_channels,
                block=ResNetBasicBlock, n=1, *args, **kwargs):
        super().__init__()
        self.res_blocks = nn.ModuleList([
            block(in_channels, out_channels, *args, **kwargs),
            *[block(out_channels * block.expansion, 
                    out_channels, *args, **kwargs) for _ in range(n - 1)]
        ])

    def forward(self, x, voxel_indices, num_points):
        for res_block in self.res_blocks:
            x = res_block(x, voxel_indices, num_points)
        return x

class ResNetEncoder(nn.Module):
    """
    ResNet encoder composed by layers with increasing features.
    """
    def __init__(self, input_dim=3, output_dim=8, 
                block_sizes=[64, 128, 256, 512], 
                depths=[2,2,2,2],
                 activation='relu', block=ResNetBasicBlock, *args, **kwargs):
        super().__init__()
        self.block_sizes = block_sizes

        # Avoid batch-norming the first layer
        self.init_layer = mySequential(
            VOX_mean(input_dim, self.block_sizes[0],),
            activation_func(activation),
        )
                
        self.in_out_block_sizes = list(zip(block_sizes, block_sizes[1:]))
        self.layers = nn.ModuleList([ 
            ResNetLayer(block_sizes[0], block_sizes[0], 
                        n=depths[0], activation=activation, 
                        block=block,*args, **kwargs),
            *[ResNetLayer(in_channels * block.expansion, out_channels,
                        n=n, activation=activation, 
                        block=block, *args, **kwargs) 
              for (in_channels, out_channels), n, in zip(self.in_out_block_sizes, depths[1:])],
            VOX_mean(self.in_out_block_sizes[-1][-1], output_dim,),       
        ])
        
        
    def forward(self, x, voxel_indices, num_points):
        inputs = (x, voxel_indices, num_points)
        x, _, _ = self.init_layer(inputs)

        for i, layer in enumerate(self.layers):
            x = layer(x, voxel_indices, num_points)
            
        return x

class geomPoolResNet(nn.Module):
    def __init__(self, input_dim, output_dim, *args, **kwargs):
        super().__init__()
        self.encoder = ResNetEncoder(input_dim, output_dim, *args, **kwargs)
        
    def forward(self, x, voxel_indices, num_points):
        out = self.encoder(x, voxel_indices, num_points)

        return out

        
