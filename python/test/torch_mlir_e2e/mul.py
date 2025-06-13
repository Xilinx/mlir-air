# ./python/test/torch_mlir_e2e/mul_cpu.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# REQUIRES: torch_mlir, ryzen_ai

# RUN: mkdir -p mul && cd mul
# RUN: %run_on_npu1% %PYTHON %s | FileCheck %s
# CHECK: PASSED!

import torch
from torch_mlir import fx

from air.backend.xrt import XRTBackend

verbose = False


class model_mul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        x = a * b
        return x


def run_test(model, dtype, shape):
    print("building...")
    torch_model = model()

    a = torch.randint(size=shape, low=1, high=100, dtype=dtype)
    b = torch.randint(size=shape, low=1, high=100, dtype=dtype)
    m = fx.export_and_import(
        torch_model, a, b, output_type="linalg-on-tensors", func_name="forward"
    )

    backend = XRTBackend(verbose=verbose)
    air_program = backend.load(backend.compile_from_torch_mlir(m, verbose=verbose))

    print("running...")
    c_ref = torch_model(a, b)
    c = torch.ones_like(c_ref)
    [_, _, c_out] = air_program(a.numpy(), b.numpy(), c.numpy())
    c_out = c_out.reshape(c_ref.shape)
    if verbose:
        print(f"input:\n{a}\n{b}\noutput:\n{c_out}")

    if torch.allclose(c_ref, torch.tensor(c_out)):
        print("PASS!")
        return 1
    else:
        import numpy

        errs = c_ref == torch.tensor(c_out)
        print(numpy.unique(errs.numpy(), return_counts=True))
        print("failed.")
    return 0


sizes = [[128, 128], [32, 32, 32], [1024 * 32]]

dtypes = [torch.int32, torch.float]

passed = 0
num_tests = 0
for t in dtypes:
    for s in sizes:
        print(f"running test for {t} and {s}")
        num_tests = num_tests + 1
        passed = passed + run_test(model_mul, t, s)

if passed != num_tests:
    print(f"failed. {passed}/{num_tests}")
else:
    print(f"PASSED! {passed}/{num_tests}")
