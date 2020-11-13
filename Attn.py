import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

EPS = 1e-8
GPU = 0
PARTITION = 9

def activation_func(activation):
    return  nn.ModuleDict([
        ['relu', nn.ReLU()],
        ['leaky_relu', nn.LeakyReLU(negative_slope=0.01)],
        ['selu', nn.SELU()],
        ['none', nn.Identity()]
    ])[activation]

class CircularPad3d(nn.Module):
    def __init__(self, padding=(0, 0, 0, 0, 0, 0)):
        super(CircularPad3d, self).__init__()
        self.p = padding

    def forward(self, x):
        x = F.pad(x, self.p, mode="circular")

        return x

class Conv_Block(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=(3, 3, 3), stride=(1, 1, 1)):
        super(Conv_Block, self).__init__()
        
        pad_dim = kernel[0] // 2
        self.padding = (pad_dim, pad_dim, pad_dim, pad_dim, pad_dim, pad_dim)

        # print(kernel)
        # print(stride)
        
        self.conv_block = nn.Sequential(
            CircularPad3d(self.padding),
            nn.Conv3d(in_dim, out_dim, kernel_size=kernel, stride=stride),
        )
    
    def forward(self, x):
        x = self.conv_block(x)

        return x

class VOX_mean(nn.Module):
    def __init__(self, in_dim, out_dim, num_classes):
        super(VOX_mean, self).__init__()
        self.conv = Conv_Block(in_dim, out_dim)
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

        self.class_weights = nn.Parameter(torch.randn(in_dim, out_dim, num_classes, num_classes))
        self.attn_weights = nn.Parameter(torch.randn(in_dim, num_classes))

        self.in_dim = in_dim
        self.out_dim = out_dim
    
    def forward(self, x, voxel_indices, num_points):
        # Reshape x to channels last
        self.batch_size = x.shape[0]
        x = x.view(self.batch_size, -1, self.in_dim)

        w_prime = torch.einsum('bnk,kl->bnl', [x, self.attn_weights])
        A = F.softmax(w_prime, dim=2)
        pooled_attn_x = torch.einsum('bnl,bnk -> blk', [A, x])
        pooled_attn_x = torch.einsum('blk, kolh->bho', [pooled_attn_x, self.class_weights])
        attn_out = torch.einsum('bnl, blc->bnc', [A, pooled_attn_x])

        x = x.view(-1, self.in_dim)

        zeros = torch.zeros(self.batch_size * PARTITION ** 3, self.in_dim)
        if GPU is not None:
            zeros = zeros.cuda(GPU)
        conv_ready = zeros.index_add(0, voxel_indices, x)

        x = x.view(self.batch_size, -1, self.in_dim)
        everything = self.Lambda(x.mean(1, keepdim=True))
        nothing = self.Gamma(x)

        per_voxel_count = num_points.unsqueeze(dim=1)
        per_voxel_count = per_voxel_count.expand(-1, self.in_dim)
        mean_conv_ready = conv_ready.div(per_voxel_count)
        mean_conv_ready = mean_conv_ready.view(self.batch_size, self.in_dim, 
                                                PARTITION, PARTITION, PARTITION)
        out = self.conv(mean_conv_ready)
        out = out.view(self.batch_size, 
                        PARTITION, PARTITION, PARTITION, self.out_dim)

        gather_out = out.view(self.batch_size * PARTITION ** 3, self.out_dim)
        voxel_indices_tesellate = voxel_indices.unsqueeze(dim=1)
        voxel_indices_tesellate = voxel_indices_tesellate.expand(-1, self.out_dim)

        gather_out = torch.gather(gather_out, 0, voxel_indices_tesellate)

        layer_out = (gather_out + 
                        (everything + nothing).view(-1, self.out_dim) + 
                        attn_out.view(-1, self.out_dim)
                    )

        layer_out = layer_out.view(self.batch_size, self.out_dim, -1)

        return layer_out

class mySequential(nn.Sequential):
    def forward(self, inputs):
        x, voxel_indices, num_points = inputs
        for module in self._modules.values():
            # Check that module type is VOX_mean
            if isinstance(module, VOX_mean):
                x = module(x, voxel_indices, num_points)
            else:
                x = module(x)
        # Tuple-ize the output
        inputs = (x, voxel_indices, num_points)
        return inputs

def clip_grad(model, max_norm):
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm ** 2

    total_norm = total_norm ** (0.5)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in model.parameters():
            p.grad.data.mul_(clip_coef)
            
    return total_norm

