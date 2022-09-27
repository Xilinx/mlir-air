# ./python/test/torch_mlir_e2e/matmul_mul_i32.py -*- Python -*-

# Copyright (C) 2021-2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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