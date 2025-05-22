# ./python/test/torch_mlir_e2e/relu.py -*- Python -*-

# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# REQUIRES: torch_mlir, ryzen_ai

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASSED!

import torch
from torch_mlir import fx

from air.backend.xrt import XRTBackend

verbose = False


class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a):
        x = torch.relu(a)
        return x


def run_test(dtype, shape):
    print("building...")
    program = model()

    a = torch.randint(size=shape, low=-100, high=100, dtype=dtype)
    m = fx.export_and_import(
        program, a, output_type="linalg-on-tensors", func_name="forward"
    )

    backend = XRTBackend(verbose=verbose)
    air_program = backend.load(backend.compile_from_torch_mlir(m, verbose=verbose))

    c_ref = program(a)
    c = torch.ones_like(c_ref)
    [_, c_out] = air_program(a.numpy(), c.numpy())
    c_out = c_out.reshape(c_ref.shape)

    print(f"input:\n{a}\noutput:\n{c_out}")

    if torch.allclose(c_ref, torch.tensor(c_out)):
        print("PASS!")
        return 1
    else:
        errs = c_ref == torch.tensor(c_out)
        print(numpy.unique(errs.numpy(), return_counts=True))
        print("failed.")
    return 0


sizes = [[10 * 1024], [128, 64]]

dtypes = [torch.float]

passed = 0
for t in dtypes:
    for s in sizes:
        passed = passed + run_test(t, s)

num_tests = len(sizes) * len(dtypes)
if passed != num_tests:
    print(f"failed. {passed}/{num_tests}")
else:
    print(f"PASSED! {passed}/{num_tests}")
