# ./python/test/torch_mlir_e2e/add.py -*- Python -*-
#
# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# REQUIRES: torch_mlir

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASSED

import torch
import torch._dynamo as dynamo
import numpy

import air.backend.linalg_on_tensors as air_backend
from torch_mlir import fx


class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        x = a + b
        return x


def run_test(dtype, shape):
    torch_program = model()

    a = torch.randint(size=shape, low=1, high=100, dtype=dtype)
    b = torch.randint(size=shape, low=1, high=100, dtype=dtype)
    m = fx.export_and_import(torch_program, a, b, func_name="forward")

    backend = air_backend.LinalgOnTensorsAirBackend()
    air_program = backend.load(backend.compile(m))

    c_ref = torch_program(a, b)
    c = torch.tensor(air_program.forward(a.numpy(), b.numpy()))

    print(f"input:\n{a}\n{b}\noutput:\n{c}")

    if torch.allclose(c_ref, c):
        print("PASS!")
        return 1
    else:
        errs = c_ref == c
        print(numpy.unique(errs.numpy(), return_counts=True))
        print("failed.")
    return 0


sizes = [[16, 32, 32, 16], [4096], [8, 32, 64], [128, 128]]

dtypes = [torch.int32, torch.float]

passed = 0
for t in dtypes:
    for s in sizes:
        passed = passed + run_test(t, s)

num_tests = len(sizes) * len(dtypes)
if passed != num_tests:
    print(f"failed. {passed}/{num_tests}")
else:
    print(f"PASSED! {passed}/{num_tests}")
