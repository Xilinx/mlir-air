# (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASS

import torch
import torch_mlir
import numpy

from air.backend import linalg_on_tensors as backend

shape = [10240]
dtype = torch.float

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        x = torch.relu(a)
        return x

program = model()
module = torch_mlir.compile(
        program,
        (torch.ones(shape, dtype=dtype)),
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
)

print(module)

airbackend = backend.LinalgOnTensorsAirBackend()
compiled = airbackend.compile(module)
jit_module = airbackend.load(compiled)

a = torch.randint(-100, 100, shape, dtype=dtype)
b = torch.tensor(
    jit_module.forward(a.numpy()))

print(f"input:\n{a}\noutput:\n{b}")

if torch.equal(torch.relu(a),b):
    print("PASS!")
else:
    print("failed.")