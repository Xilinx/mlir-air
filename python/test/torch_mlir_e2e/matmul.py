# ./python/test/torch_mlir_e2e/matmul.py -*- Python -*-

# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# REQUIRES: torch_mlir, ryzen_ai

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASS

import torch
from torch_mlir import fx

from air.backend.xrt import XRTBackend
from air.passmanager import PassManager
from air.compiler.util import run_transform
from air.ir import Module

verbose = False


class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.mm(a, b)


def pipeline(module):
    with module.operation.context as ctx:
        pipeline = (
            "builtin.module("
            + ",".join(["air-linalg-codegen{test-patterns=true}"])
            + ")"
        )
        pm = PassManager.parse(pipeline)
        pm.run(module.operation)
        if verbose:
            print("Optimized linalg Module")
            print(module)
    transform_ir_string = """
    transform.with_pdl_patterns {
    ^bb0(%arg0: !pdl.operation):
        transform.sequence %arg0 : !pdl.operation failures(propagate) {
        ^bb1(%arg1: !pdl.operation):
            %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
            %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1  : (!pdl.operation) -> !pdl.operation
            %matmul_1, %loops:2 = transform.air.linalg_tile %matmul [64, 64, 0]
            %fill_1 = transform.air.fuse_into_containing_op %fill into %loops#1
            transform.air.linalg_promote %fill_1 {"operands_to_promote"=[1], "memory_space"="L2"}
            transform.air.linalg_promote %matmul_1 {"operands_to_promote"=[2], "memory_space"="L2"}
            transform.air.linalg_promote %matmul_1 {"operands_to_promote"=[0,1], "memory_space"="L2"}
            %matmul_2, %loops_2:2 = transform.air.linalg_tile %matmul_1 [32, 32, 0]
            %fill_2 = transform.air.fuse_into_containing_op %fill_1 into %loops_2#1
            transform.air.linalg_promote %fill_2 {"operands_to_promote"=[1], "memory_space"="L1"}
            transform.air.linalg_promote %matmul_2 {"operands_to_promote"=[2], "memory_space"="L1"}
            %matmul_3, %reduction_loop = transform.air.linalg_tile %matmul_2 [0, 0, 32]
            transform.air.linalg_promote %matmul_3 {"operands_to_promote"=[0,1], "memory_space"="L1"}
        }
    }
    """
    transform_ir = Module.parse(transform_ir_string, context=module.context)
    run_transform(transform_ir, module)
    pipeline = (
        "builtin.module("
        + ",".join(
            [
                "canonicalize",
                "cse",
                "air-par-to-herd{depth=-1}",
                "air-par-to-launch{has-air-segment=true}",
                "air-copy-to-dma",
                "canonicalize",
                "cse",
            ]
        )
        + ")"
    )
    pm = PassManager.parse(pipeline)
    pm.run(module.operation)
    print(module)
    return module


def run_test(dtype, shape):
    print("building...")
    torch_model = model()

    a = torch.randint(100, [shape[0], shape[1]], dtype=dtype)
    b = torch.randint(100, [shape[1], shape[2]], dtype=dtype)
    m = fx.export_and_import(
        torch_model, a, b, output_type="linalg-on-tensors", func_name="forward"
    )

    backend = XRTBackend(verbose=verbose)
    air_program = backend.load(
        backend.compile_from_torch_mlir(m, pipeline=pipeline, verbose=verbose)
    )

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

        print(numpy.unique(errs.numpy(), return_counts=True))
        print("failed.")
        return 0


sizes = [
    [512, 64, 128],
    [512, 256, 512],
]
dtypes = [torch.float32]

passed = 0
num_tests = 0
for t in dtypes:
    for s in sizes:
        print(f"running test for {t} and {s}")
        num_tests = num_tests + 1
        try:
            passed = passed + run_test(t, s)
        except Exception as e:
            print("test failed:", e)
            pass

if passed != num_tests:
    print(f"failed. {passed}/{num_tests}")
else:
    print(f"PASSED! {passed}/{num_tests}")
