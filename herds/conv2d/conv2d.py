#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:47:16 2021

@author: juanu
"""

import numpy as np
np.random.seed(0)

import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

import torch_mlir

# define convolution parameters based on TinyYolo v2 layer
# layer 1 
batch_size = 1
in_height = 416 #52 #26 #104 #416
in_width = 416 #52 #26 #104 #416
in_depth = 48 #4 
conv_height = 3
conv_width = 3
conv_stride_height = 1
conv_stride_width = 1
out_height = int(in_height/conv_stride_height)
out_width = int(in_width/conv_stride_width)
out_depth = 64 #16

l2_tiles = 4 #2


i0 = torch.zeros((1, in_depth, in_width, in_height), dtype=torch.float32)

builder = torch_mlir.ModuleBuilder()

# define simple neural network
class CONV2D_NN(nn.Module):
    def __init__(self):
        super(CONV2D_NN, self).__init__()

        #self.conv1 = nn.Conv2d(1, 8, kernel_size=3)
        self.conv1 = nn.Conv2d(in_depth, out_depth, kernel_size=(conv_height,conv_width), padding=(1,1))

    def forward(self, x):
        x = self.conv1(x)
        return x

graph = CONV2D_NN()

with builder.capture_function("graph", [i0]) as f:
    t2 = graph(i0)
    f.returns([t2])

t2_mlir = builder.module
#print(t2_mlir)

from mlir.ir import *
from mlir.passmanager import *
import mlir.transforms
from npcomp.compiler.generic.backend import refjit as refjit_backend

# pm = PassManager.parse("func(aten-recognize-kernels)",
# pm = PassManager.parse("func(aten-recognize-kernels),numpy-public-functions-to-tensor",
pm = PassManager.parse("func(aten-recognize-kernels),numpy-public-functions-to-tensor,canonicalize", context=builder.module.context)
pm.run(t2_mlir)
print(t2_mlir)
