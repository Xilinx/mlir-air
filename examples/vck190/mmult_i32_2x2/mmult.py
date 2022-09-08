# (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

import torch
import torch_mlir
import numpy

from air.backend import linalg_on_tensors as backend

class mmult(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.mm(a,b)

program = mmult()

shape = [64,64]
dtype = torch.int32

module = torch_mlir.compile(
    program,
    (torch.ones(shape, dtype=dtype), torch.ones(shape, dtype=dtype)),
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
)

print(module)