# ./python/test/torch_mlir_e2e/matmul_cpu.py -*- Python -*-

# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# REQUIRES: torch_mlir, dont_run

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASSED

import torch
from torch_mlir import fx

# this import has side-effect of registering the dialect
import air.dialects.air
from air.ir import *
import air.backend.cpu_backend as cpu_backend
from air.compiler.util import run_transform
from air.passmanager import PassManager

verbose = False


def transform_to_air_0(module):
    with module.context as ctx:
        pipeline = (
            "builtin.module("
            + ",".join(
                [
                    "air-linalg-codegen",
                    "air-par-to-herd{depth=-1}",
                    "air-copy-to-dma",
                    "air-return-elimination",
                    "canonicalize",
                    "cse",
                ]
            )
            + ")"
        )
        pm = PassManager.parse(pipeline)
        pm.run(module.operation)
        if verbose:
            print("AIR Module")
            print(module)
        pm = PassManager.parse(cpu_backend.DEFAULT_PIPELINE)
        pm.run(module.operation)
    return module


def transform_to_air_1(module):
    with module.context as ctx:
        pipeline = (
            "builtin.module("
            + ",".join(["air-linalg-codegen{test-patterns=true}"])
            + ")"
        )
        pm = PassManager.parse(pipeline)
        pm.run(module.operation)
        transform_ir_string = """
        transform.with_pdl_patterns {
        ^bb0(%arg0: !pdl.operation):
        pdl.pattern @match_copy : benefit(1) {
            %args = pdl.operands
            %results = pdl.types
            %op = pdl.operation "memref.copy"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
            pdl.rewrite %op with "transform.dialect"
        }

        transform.sequence %arg0 : !pdl.operation failures(propagate) {
        ^bb1(%arg1: !pdl.operation):
            %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
            %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
            %matmul_1, %outer_tile_loops:2 = transform.air.linalg_tile %matmul [16, 16, 0]
            %fill_1 = transform.air.fuse_into_containing_op %fill into %outer_tile_loops#1

            %2 = transform.merge_handles %fill_1, %matmul_1 : !pdl.operation
            transform.air.linalg_promote %2 {"group_size"=2, "operands_to_promote"=[1,4], "memory_space"="L1"}

            %herd_matmuls = transform.foreach %outer_tile_loops#0 : !pdl.operation -> !pdl.operation {
            ^bb2(%herd: !pdl.operation):
                %matmul_herd = transform.structured.match ops{["linalg.matmul"]} in %herd : (!pdl.operation) -> !pdl.operation
                transform.yield %matmul_herd : !pdl.operation
            }
            %inner_matmul, %reduction_loop = transform.air.linalg_tile %herd_matmuls [0, 0, 16]
            transform.air.linalg_promote %inner_matmul {"operands_to_promote"=[0,1], "memory_space"="L1"}
        }
        }
        """
        transform_ir = Module.parse(transform_ir_string)
        run_transform(transform_ir, module)
        pipeline = (
            "builtin.module("
            + ",".join(
                [
                    "canonicalize",
                    "cse",
                    "air-linalg-codegen",
                    "air-par-to-herd{depth=0}",
                    "air-copy-to-dma",
                    "air-return-elimination",
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

    if verbose:
        print(module)
    return module


# module -> module
def transform_to_air_2(module):
    grid_size = [2, 2]
    with module.context as ctx:
        pipeline = (
            "builtin.module("
            + ",".join(["air-linalg-codegen{test-patterns=true}"])
            + ")"
        )
        pm = PassManager.parse(pipeline)
        pm.run(module.operation)
        transform_ir_string = """
        transform.with_pdl_patterns {
        ^bb0(%arg0: !pdl.operation):
        pdl.pattern @match_copy : benefit(1) {
            %args = pdl.operands
            %results = pdl.types
            %op = pdl.operation "memref.copy"(%args : !pdl.range<value>) -> (%results : !pdl.range<type>)
            pdl.rewrite %op with "transform.dialect"
        }

        transform.sequence %arg0 : !pdl.operation failures(propagate) {
        ^bb1(%arg1: !pdl.operation):
            %fill = transform.structured.match ops{["linalg.fill"]} in %arg1 : (!pdl.operation) -> !pdl.operation
            %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1 : (!pdl.operation) -> !pdl.operation
            %matmul_1, %outer_tile_loops:2 = transform.air.linalg_tile %matmul [32, 32, 0]
            //%fill_1 = transform.air.fuse_into_containing_op %fill into %outer_tile_loops#1

            %matmul_2, %inner_tile_loops:2 = transform.air.linalg_tile %matmul_1 [16, 16, 0]
            //%fill_2 = transform.air.fuse_into_containing_op %fill_1 into %outer_tile_loops#1

            //%2 = transform.merge_handles %fill_2, %matmul_2 : !pdl.operation
            //transform.air.linalg_promote %2 {"group_size"=2, "operands_to_promote"=[1,4], "memory_space"="L1"}

            %herd_matmuls = transform.foreach %outer_tile_loops#0 : !pdl.operation -> !pdl.operation {
            ^bb2(%herd: !pdl.operation):
                %matmul_herd = transform.structured.match ops{["linalg.matmul"]} in %herd : (!pdl.operation) -> !pdl.operation
                transform.yield %matmul_herd : !pdl.operation
            }
            %inner_matmul, %reduction_loop = transform.air.linalg_tile %herd_matmuls [0, 0, 16]
            transform.air.linalg_promote %inner_matmul {"operands_to_promote"=[0,1,2], "memory_space"="L1"}
        }
        }
        """
        transform_ir = Module.parse(transform_ir_string)
        run_transform(transform_ir, module)
        pipeline = (
            "builtin.module("
            + ",".join(
                [
                    "canonicalize",
                    "cse",
                    "air-par-to-herd{depth=-1}",
                    "air-par-to-launch{depth=0}",
                    "air-copy-to-dma",
                    "air-dma-to-channel",
                    "air-return-elimination",
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

    if verbose:
        print(module)
    return module


class model_mm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.mm(a, b)


def run_test(dtype, shape):
    torch_model = model_mm()

    # shape = [s*4 for s in shape]
    a = torch.randint(10, [shape[0], shape[1]], dtype=dtype) + 1
    b = torch.randint(10, [shape[1], shape[2]], dtype=dtype) + 1
    m = fx.export_and_import(
        torch_model, a, b, output_type="linalg-on-tensors", func_name="forward"
    )

    backend = cpu_backend.AirCpuBackend()
    air_program = backend.load(
        backend.compile_from_torch_mlir(m, pipeline=transform_to_air_1, verbose=verbose)
    )

    c_ref = torch_model(a, b)
    c = torch.ones_like(c_ref)
    air_program(a.numpy(), b.numpy(), c.numpy())

    if verbose:
        print(f"input:\n{a}\n{b}\noutput:\n{c}\nref:\n{c_ref}")

    if torch.allclose(c_ref, c):
        print(dtype, shape, "PASS!")
        return 1
    else:
        print("failed.")
        return 0


import random

sizes = [[64, 64, 64]]
# for i in range(0, 4):
#     m = [random.randint(2, 8), random.randint(2, 8), random.randint(2, 8)]
#     s = [i * 32 for i in m]
#     sizes.append(s)
print(sizes)
dtypes = [
    torch.float,
    # torch.int32,
    # torch.int8,
]

passed = 0
num_tests = 0
for t in dtypes:
    for s in sizes:
        try:
            num_tests = num_tests + 1
            passed = passed + run_test(t, s)
        except Exception as e:
            print(e)
            pass

if passed != num_tests:
    print(f"failed. {passed}/{num_tests}")
else:
    print(f"PASSED! {passed}/{num_tests}")
