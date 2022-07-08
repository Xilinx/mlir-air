# (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASS

import torch
import torch_mlir
import numpy

from air.backend import linalg_on_tensors as backend

shape = [128,128]
dtype = torch.int32

class mmult(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c):
        x = torch.mm(b,c)
        y = a*x
        return y

program = mmult()
module = torch_mlir.compile(
        program,
        (torch.ones(shape, dtype=dtype), torch.ones(shape, dtype=dtype), torch.ones(shape, dtype=dtype)),
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
    )

print(module)

airbackend = backend.LinalgOnTensorsAirBackend()
compiled = airbackend.compile(module)
jit_module = airbackend.load(compiled)

a = torch.randint(100, shape, dtype=dtype)
b = torch.randint(100, shape, dtype=dtype)
c = torch.randint(100, shape, dtype=dtype)
d = torch.tensor(
    jit_module.forward(a.numpy(),b.numpy(),c.numpy()))

print(f"input:\n{a}\n{b}\n{c}\noutput:\n{d}")

if torch.equal(a*torch.mm(b,c),d):
    print("PASS!")
else:
    print("failed.")