# ./python/test/torch_mlir_e2e/matmul.py -*- Python -*-

# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# REQUIRES: torch_mlir, ryzen_ai

# RUN: mkdir -p matmul && cd matmul
# RUN: %run_on_npu1% %PYTHON %s | FileCheck %s
# CHECK: PASS

import torch
from torch_mlir import fx

from air.backend.xrt import XRTBackend
from air.passmanager import PassManager
from air.compiler.util import run_transform
from air.ir import Module, Location

verbose = False


class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.mm(a, b)


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
                %launch = transform.air.par_to_launch %launch_par {"has_air_segment"=true}
                %copies = transform.pdl_match @match_copy in %arg0 : (!pdl.operation) -> !pdl.operation
                %h = transform.air.copy_to_dma %copies
                %f = transform.structured.match ops{["func.func"]} in %arg1 : (!pdl.operation) -> !pdl.operation
       
            }
        }
        """
        transform_ir = Module.parse(transform_ir_string)
        run_transform(transform_ir, module)
        with open("air_transform.mlir", "w", encoding="utf-8") as f:
            f.write(str(module))
        if verbose:
            print(f"Transformed module: {module}")

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
        backend.compile_from_torch_mlir(m, pipeline=transform_to_air, verbose=verbose)
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
        print("failed.")
        return 0


sizes = [
    [512, 64, 128],
    [128, 64, 512],
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
