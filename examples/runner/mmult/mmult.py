# (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

import torch
import torch_mlir

class mmult(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.mm(a,b)

program = mmult()
mlir = torch_mlir.compile(program, (torch.ones(64,64), torch.ones(64,64)), output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
print(mlir)