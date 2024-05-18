# ./python/test/torch_mlir_e2e/mul_cpu.py -*- Python -*-
#
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# REQUIRES: torch_mlir, needs_update

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASSED

import torch
import torch._dynamo as dynamo

from air.backend import cpu_backend as backend

verbose = False


class model_mul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        x = a * b
        return x


air_backend = backend.make_dynamo_backend(verbose=verbose)


def run_test(model, dtype, shape):
    torch_model = model()
    dynamo_model = dynamo.optimize(air_backend)(torch_model)

    a = torch.randint(size=shape, low=1, high=100, dtype=dtype)
    b = torch.randint(size=shape, low=1, high=100, dtype=dtype)
    c = dynamo_model(a, b)
    c_ref = torch_model(a, b)

    if verbose:
        print(f"input:\n{a}\n{b}\noutput:\n{c}")

    if torch.allclose(c_ref, c):
        print("PASS!")
        return 1
    else:
        import numpy

        errs = c_ref == c
        print(numpy.unique(errs.numpy(), return_counts=True))
        print("failed.")
    return 0


sizes = [[4, 4, 16, 16], [4, 32, 32], [1024 * 10], [128, 128]]

dtypes = [torch.int32, torch.float]

passed = 0
for t in dtypes:
    for s in sizes:
        passed = passed + run_test(model_mul, t, s)

num_tests = len(sizes) * len(dtypes)
if passed != num_tests:
    print(f"failed. {passed}/{num_tests}")
else:
    print(f"PASSED! {passed}/{num_tests}")
