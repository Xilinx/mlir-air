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
from air.ir import *
import air.passmanager


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
        %matmul = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        // First level tiling: air.launch
        %matmul_1, %loop = transform.air.linalg_tile %matmul [64, 64, 0]
        %parallal = transform.loop.forall_to_parallel %loop  : (!pdl.operation) -> !pdl.operation
        %fill_1 = transform.air.fuse_into_containing_op %fill into %parallal
        %matmul_2 = transform.structured.match ops{["linalg.generic"]} in %parallal  : (!pdl.operation) -> !pdl.operation
        // Second level tiling: air.segment
        %matmul_3, %loop_1 = transform.air.linalg_tile %matmul_2 [64, 64, 0]
        %parallal_1 = transform.loop.forall_to_parallel %loop_1  : (!pdl.operation) -> !pdl.operation
        %fill_2 = transform.air.fuse_into_containing_op %fill_1 into %parallal_1
        %matmul_3_1 = transform.structured.match ops{["linalg.generic"]} in %parallal_1  : (!pdl.operation) -> !pdl.operation
        transform.air.linalg_promote %fill_2 {"operands_to_promote"=[1], "memory_space"="L2"}
        transform.air.linalg_promote %matmul_3_1 {"operands_to_promote"=[2], "memory_space"="L2"}
        transform.air.linalg_promote %matmul_3_1 {"operands_to_promote"=[0,1], "memory_space"="L2"}
        // Third level tiling: air.herd
        %matmul_4, %loop_2 = transform.air.linalg_tile %matmul_3_1 [32, 32, 0]
        %parallal_2 = transform.loop.forall_to_parallel %loop_2  : (!pdl.operation) -> !pdl.operation
        %fill_3 = transform.air.fuse_into_containing_op %fill_2 into %parallal_2
        %matmul_5 = transform.structured.match ops{["linalg.generic"]} in %parallal_2  : (!pdl.operation) -> !pdl.operation
        transform.air.linalg_promote %fill_3 {"operands_to_promote"=[1], "memory_space"="L1"}
        transform.air.linalg_promote %matmul_5 {"operands_to_promote"=[2], "memory_space"="L1"}
        // Fourth level tiling: scf.for (reduction)
        %matmul_6, %reduction_loop = transform.air.linalg_tile %matmul_5 [0, 0, 32]
        transform.air.linalg_promote %matmul_6 {"operands_to_promote"=[0,1], "memory_space"="L1"}
        %scffor = transform.loop.forall_to_for %reduction_loop  : (!pdl.operation) -> !pdl.operation
        %herd = transform.air.par_to_herd %parallal_2
        %segment = transform.air.par_to_segment %parallal_1
        %launch = transform.air.par_to_launch %parallal
        %copies = transform.pdl_match @match_copy in %arg0 : (!pdl.operation) -> !pdl.operation
        %h = transform.air.copy_to_dma %copies
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

backend = XRTBackend(
    air_loop_fusion=True,
    trace_offset=opts.trace_offset,
    trace_size=opts.trace_size,
)
backend.compile(air_module)
