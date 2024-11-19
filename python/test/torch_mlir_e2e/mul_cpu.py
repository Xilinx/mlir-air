# ./python/test/torch_mlir_e2e/mul_cpu.py -*- Python -*-
#
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# REQUIRES: torch_mlir

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASSED! 8/8

import torch
from torch_mlir import fx

from air.backend import cpu_backend
from air.passmanager import PassManager

verbose = False


class model_mul(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        x = a * b
        return x


def pipeline(module):
    with module.operation.context as ctx:
        pipeline = (
            "builtin.module("
            + ",".join(
                [
                    "canonicalize",
                    "cse",
                    "air-linalg-codegen",
                    "air-par-to-herd{depth=0}",
                    "air-copy-to-dma",
                    "canonicalize",
                    "cse",
                ]
            )
            + ")"
        )
        pm = PassManager.parse(pipeline)
        pm.run(module.operation)
        pm = PassManager.parse(cpu_backend.DEFAULT_PIPELINE)
        pm.run(module.operation)
    return module


def run_test(model, dtype, shape):
    torch_model = model()

    a = torch.randint(size=shape, low=1, high=100, dtype=dtype)
    b = torch.randint(size=shape, low=1, high=100, dtype=dtype)
    m = fx.export_and_import(torch_model, a, b, func_name="forward")

    backend = cpu_backend.AirCpuBackend()
    air_program = backend.load(
        backend.compile_from_torch_mlir(m, pipeline=pipeline, verbose=verbose)
    )

    c_ref = torch_model(a, b)
    c = torch.tensor(air_program.forward(a.numpy(), b.numpy()))

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
