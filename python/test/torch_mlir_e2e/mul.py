# ./python/test/torch_mlir_e2e/mul.py -*- Python -*-

# Copyright (C) 2022, Xilinx Inc.
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
# CHECK: PASSED

import torch
import torch_mlir
import numpy

from air.backend import linalg_on_tensors as backend

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        x = a * b
        return x

def run_test(dtype, shape):
    program = model()
    module = torch_mlir.compile(
        program,
        (torch.ones(shape, dtype=dtype), torch.ones(shape, dtype=dtype)),
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
    )

    print(module)

    airbackend = backend.LinalgOnTensorsAirBackend()
    compiled = airbackend.compile(module)
    jit_module = airbackend.load(compiled)

    a = torch.randint(size = shape, low=1, high=100, dtype=dtype)
    b = torch.randint(size = shape, low=1, high=100, dtype=dtype)
    c = torch.tensor(
        jit_module.forward(a.numpy(),b.numpy()))

    print(f"input:\n{a}\n{b}\noutput:\n{c}")

    if torch.equal(a*b,c):
        print("PASS!")
        return 1
    else:
        errs = (a*b == c)
        print(numpy.unique(errs.numpy(), return_counts=True))
        print("failed.")
    return 0

sizes = [
    [64,64,32],
    [16,32,8,64],
    [4096],
    [128,128]
]

dtypes = [
    torch.int32,
    torch.float
]

passed = 0
for t in dtypes:
    for s in sizes:
        passed = passed + run_test(t,s)

num_tests = len(sizes)*len(dtypes)
if passed != num_tests:
    print (f"failed. {passed}/{num_tests}")
else:
    print (f"PASSED! {passed}/{num_tests}")
