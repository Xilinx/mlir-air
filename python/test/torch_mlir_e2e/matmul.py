# ./python/test/torch_mlir_e2e/matmul.py -*- Python -*-

# Copyright (C) 2021-2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASS

import torch
import torch._dynamo as dynamo
import numpy
from air.backend import linalg_on_tensors as backend

air_backend = backend.make_dynamo_backend()

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.mm(a,b)

def run_test(dtype, shape):
    program = model()
    dynamo_program = dynamo.optimize(air_backend)(program)

    a = torch.randint(100, [shape[0],shape[1]], dtype=dtype)
    b = torch.randint(100, [shape[1],shape[2]], dtype=dtype)
    c = dynamo_program(a, b)
    c_ref = program(a, b)

    print(f"input:\n{a}\n{b}\noutput:\n{c}")

    if torch.allclose(c_ref,c):
        print("PASS!")
        return 1
    else:
        print(numpy.unique(errs.numpy(), return_counts=True))
        print("failed.")
        return 0

sizes = [
    [32,128,64],
    [64,64,64],
    [128,32,128],
]
dtypes = [
    torch.float,
    torch.int32
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
