# ./python/test/torch_mlir_e2e/matmul_cpu.py -*- Python -*-

# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# REQUIRES: torch_mlir, dont_run

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASSED

from aie.dialects._gpu_ops_gen import module
import torch
from torch_mlir import fx

# this import has side-effect of registering the dialect
import air.dialects.air
from air.ir import *
import air.backend.cpu_backend as cpu_backend
from air.compiler.util import run_transform
from air.passmanager import PassManager

verbose = False


def transform_to_air(module):
    with module.context, Location.unknown():

        # Run the buffer-results-to-out-params pass to convert result buffers into out params.
        pm_br2op = PassManager.parse(
            "builtin.module(air-linalg-codegen{test-patterns=true},buffer-results-to-out-params{hoist-static-allocs=true})"
        )
        pm_br2op.run(module.operation)
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
                %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
                %matmul = transform.structured.match ops{["linalg.matmul"]} in %arg1  : (!pdl.operation) -> !pdl.operation
                %matmul_1, %loop = transform.air.linalg_tile %matmul [64, 64, 0]
                %fill_1 = transform.air.fuse_into_containing_op %fill into %loop
                transform.air.linalg_promote %fill_1 {"operands_to_promote"=[1], "memory_space"="L2"}
                transform.air.linalg_promote %matmul_1 {"operands_to_promote"=[2], "memory_space"="L2"}
                transform.air.linalg_promote %matmul_1 {"operands_to_promote"=[0,1], "memory_space"="L2"}
                %matmul_2, %loop_2 = transform.air.linalg_tile %matmul_1 [32, 32, 0]
                %fill_2 = transform.air.fuse_into_containing_op %fill_1 into %loop_2
                transform.air.linalg_promote %fill_2 {"operands_to_promote"=[1], "memory_space"="L1"}
                transform.air.linalg_promote %matmul_2 {"operands_to_promote"=[2], "memory_space"="L1"}
                %matmul_3, %reduction_loop = transform.air.linalg_tile %matmul_2 [0, 0, 32]
                transform.air.linalg_promote %matmul_3 {"operands_to_promote"=[0,1], "memory_space"="L1"}

                %herd_tile_par = transform.loop.forall_to_parallel %loop_2  : (!pdl.operation) -> !pdl.operation
                %herd = transform.air.par_to_herd %herd_tile_par
                %launch_par = transform.loop.forall_to_parallel %loop  : (!pdl.operation) -> !pdl.operation
                %launch = transform.air.par_to_launch %launch_par
                %copies = transform.pdl_match @match_copy in %arg0 : (!pdl.operation) -> !pdl.operation
                %h = transform.air.copy_to_dma %copies
                %f = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
       
            }
        }
        """
        transform_ir = Module.parse(transform_ir_string)
        run_transform(transform_ir, module)
        pm = PassManager.parse("builtin.module(lower-affine, canonicalize, cse)")
        pm.run(module.operation)
        with open("air_transform.mlir", "w", encoding="utf-8") as f:
            f.write(str(module))
        if verbose:
            print(f"Transformed module: {module}")

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
        backend.compile_from_torch_mlir(m, pipeline=transform_to_air, verbose=verbose)
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

sizes = [[128, 128, 128], [128, 128, 128]]
# for i in range(0, 4):
#     m = [random.randint(2, 8), random.randint(2, 8), random.randint(2, 8)]
#     s = [i * 64 for i in m]
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
