# (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASS

import torch
import numpy


import torch
import torch_mlir
import numpy

from air.backend import linalg_on_tensors as backend

shape = [64,64]
dtype = torch.int32

class mmult(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.mm(a,b)

program = mmult()
module = torch_mlir.compile(
    program,
    (torch.ones(shape, dtype=dtype), torch.ones(shape, dtype=dtype)),
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
)

print(module)

airbackend = backend.LinalgOnTensorsAirBackend()
compiled = airbackend.compile(module)
jit_module = airbackend.load(compiled)

a = torch.randint(100, shape, dtype=dtype)
b = torch.randint(100, shape, dtype=dtype)
c = torch.tensor(
    jit_module.forward(a.numpy(),b.numpy()))

print(f"input:\n{a}\n{b}\noutput:\n{c}")

errs = (torch.mm(a,b) == c)
unique, counts = numpy.unique(errs, return_counts=True)
d = dict(zip(unique, counts))
errs = d.get(False,0)
count = d.get(True,0)
if errs>0:
    print(f"{count}/{errs+count} Correct\n")
if torch.equal(torch.mm(a,b),c):
    print("PASS!")
else:
    print("failed.")