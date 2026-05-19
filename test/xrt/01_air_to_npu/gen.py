# gen.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from air.dialects import linalg, arith, func, memref
from air.dialects.air import module_builder
from air.dialects.linalg.opdsl.lang import *
from air.compiler.util import run_transform
import argparse

from air.backend.xrt import XRTBackend
from air.backend.xrt_runner import XRTRunner
from air.ir import *
import air.passmanager

import numpy as np

np.random.seed(42)


@linalg_structured_op
def matmul_mono(
    A=TensorDef(T, S.M, S.K),
    B=TensorDef(T, S.K, S.N),
    C=TensorDef(T, S.M, S.N, output=True),
):
    domain(D.m, D.n, D.k)
    C[D.m, D.n] += A[D.m, D.k] * B[D.k, D.n]


@module_builder
def matmul_on_tensors(m, n, k):
    dtype = IntegerType.get_signless(width=32)

    @func.FuncOp.from_py_func(
        MemRefType.get((m, k), dtype), MemRefType.get((k, n), dtype)
    )
    def forward(lhs, rhs):
        out = memref.AllocOp(MemRefType.get((m, n), dtype), [], [])
        zero = arith.ConstantOp(dtype, 0)
        zero_fill = linalg.fill(zero, outs=[out])
        matmul_mono(lhs, rhs, outs=[out])
        return out


parser = argparse.ArgumentParser(prog="aie.py")
parser.add_argument(
    "--trace-size",
    dest="trace_size",
    default=131072,
    type=int,
    help="Create packet routed traces for cores and memtiles",
)
parser.add_argument(
    "--trace-offset",
    dest="trace_offset",
    default=65536,
    type=int,
    help="Trace buffer offset appended to output",
)
parser.add_argument(
    "--trace-file-name",
    dest="trace_file",
    default="trace.txt",
    type=str,
    help="Name of the trace file generated",
)

opts = parser.parse_args()

air_module = matmul_on_tensors(128, 128, 256)
context = air_module.context

################################################
## Tiling
################################################

pm = air.passmanager.PassManager.parse(
    air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE,
    context=context,
)
pm.run(air_module.operation)
with open("air_input.mlir", "w") as f:
    f.write(str(air_module))

transform_ir_string = """
    module attributes {transform.with_named_sequence} {
      transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        %matmul = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!transform.any_op) -> !transform.any_op
        // First level tiling: air.launch
        %matmul_1, %loop = transform.air.linalg_tile %matmul [64, 64, 0]
        %parallal = transform.loop.forall_to_parallel %loop  : (!transform.any_op) -> !transform.any_op
        %fill_1 = transform.air.fuse_into_containing_op %fill into %parallal : (!transform.any_op, !transform.any_op) -> !transform.any_op
        %matmul_2 = transform.structured.match ops{["linalg.generic"]} in %parallal  : (!transform.any_op) -> !transform.any_op
        // Second level tiling: air.segment
        %matmul_3, %loop_1 = transform.air.linalg_tile %matmul_2 [64, 64, 0]
        %parallal_1 = transform.loop.forall_to_parallel %loop_1  : (!transform.any_op) -> !transform.any_op
        %fill_2 = transform.air.fuse_into_containing_op %fill_1 into %parallal_1 : (!transform.any_op, !transform.any_op) -> !transform.any_op
        %matmul_3_1 = transform.structured.match ops{["linalg.generic"]} in %parallal_1  : (!transform.any_op) -> !transform.any_op
        transform.air.linalg_promote %fill_2 {"operands_to_promote"=[1], "memory_space"="L2"} : (!transform.any_op) -> !transform.any_op
        transform.air.linalg_promote %matmul_3_1 {"operands_to_promote"=[2], "memory_space"="L2"} : (!transform.any_op) -> !transform.any_op
        transform.air.linalg_promote %matmul_3_1 {"operands_to_promote"=[0,1], "memory_space"="L2"} : (!transform.any_op) -> !transform.any_op
        // Third level tiling: air.herd
        %matmul_4, %loop_2 = transform.air.linalg_tile %matmul_3_1 [32, 32, 0]
        %parallal_2 = transform.loop.forall_to_parallel %loop_2  : (!transform.any_op) -> !transform.any_op
        %fill_3 = transform.air.fuse_into_containing_op %fill_2 into %parallal_2 : (!transform.any_op, !transform.any_op) -> !transform.any_op
        %matmul_5 = transform.structured.match ops{["linalg.generic"]} in %parallal_2  : (!transform.any_op) -> !transform.any_op
        transform.air.linalg_promote %fill_3 {"operands_to_promote"=[1], "memory_space"="L1"} : (!transform.any_op) -> !transform.any_op
        transform.air.linalg_promote %matmul_5 {"operands_to_promote"=[2], "memory_space"="L1"} : (!transform.any_op) -> !transform.any_op
        // Fourth level tiling: scf.for (reduction)
        %matmul_6, %reduction_loop = transform.air.linalg_tile %matmul_5 [0, 0, 32]
        transform.air.linalg_promote %matmul_6 {"operands_to_promote"=[0,1], "memory_space"="L1"} : (!transform.any_op) -> !transform.any_op
        %scffor = transform.loop.forall_to_for %reduction_loop  : (!transform.any_op) -> !transform.any_op
        %herd = transform.air.par_to_herd %parallal_2 : (!transform.any_op) -> !transform.any_op
        %segment = transform.air.par_to_segment %parallal_1 : (!transform.any_op) -> !transform.any_op
        %launch = transform.air.par_to_launch %parallal : (!transform.any_op) -> !transform.any_op
        %copies = transform.structured.match ops{["memref.copy"]} in %arg1 : (!transform.any_op) -> !transform.any_op
        %h = transform.air.copy_to_dma %copies : (!transform.any_op) -> !transform.any_op
      transform.yield
    }
    }
"""
transform_ir = Module.parse(transform_ir_string, context=context)
run_transform(transform_ir, air_module)

with open("air_tiled.mlir", "w") as f:
    f.write(str(air_module))

################################################
## Binding scf.paralell to air hierarchies
################################################

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "buffer-results-to-out-params",
            "canonicalize",
            "cse",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)

###############################################
# Run compile and load
###############################################

# Matrix A: (128, 256)
A = np.random.randint(-10, 10, size=(128, 256), dtype=np.int32)

# Matrix B: (256, 128)
B = np.random.randint(-10, 10, size=(256, 128), dtype=np.int32)
C = np.matmul(A, B)
runner = XRTRunner(
    air_loop_fusion=True,
    omit_while_true_loop=False,
    use_lock_race_condition_fix=True,
    trace_offset=opts.trace_offset,
    trace_size=opts.trace_size,
    trace_file=opts.trace_file,
)
exit(
    runner.run_test(
        air_module,
        inputs=[A, B],
        expected_outputs=[C],
    )
)
