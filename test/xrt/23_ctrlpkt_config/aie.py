# aie.py -*- Python -*-
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import air
import air.compiler.util
from air.dialects import linalg, arith, func, memref
from air.ir import *
import air.passmanager
from air.dialects.air import module_builder
from air.dialects.linalg.opdsl.lang import *
from air.compiler.util import run_transform
import argparse
import sys


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
    default=0,
    type=int,
    help="Create packet routed traces for cores and memtiles",
)
parser.add_argument(
    "--trace-offset",
    dest="trace_offset",
    default=0,
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
    transform.sequence %arg0 : !pdl.operation failures(propagate) {
    ^bb1(%arg1: !pdl.operation):
        %fill = transform.structured.match ops{["linalg.fill"]} in %arg1  : (!pdl.operation) -> !pdl.operation
        %matmul = transform.structured.match ops{["linalg.generic"]} in %arg1  : (!pdl.operation) -> !pdl.operation
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
            "air-par-to-launch{depth=0 has-air-segment=true}",
            "air-par-to-herd{depth=0}",
            "scf-forall-to-for",
            "air-copy-to-dma",
            "canonicalize",
            "cse",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)

with open("air_sync.mlir", "w") as f:
    f.write(str(air_module))

################################################
## Extract event dependency and optimize schedule
################################################

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "air-dependency",
            "air-hoist-dma-in-accum-pattern",
            "air-broadcast-detection",
            "air-specialize-dma-broadcast",
            "air-dma-to-channel",
            "canonicalize",
            "cse",
            "air-dependency-canonicalize",
            "canonicalize",
            "cse",
            "air-isolate-async-dma-loop-nests",
            "canonicalize",
            "cse",
            "air-fuse-channels",
            "canonicalize",
            "cse",
            ### Scaling to 4 AIE columns
            "func.func(air-split-l2-memref)",
            "canonicalize",
            "cse",
            "air-isolate-async-dma-loop-nests",
            ###
            "canonicalize",
            "cse",
            "func.func(air-fuse-alloc-dealloc)",
            "func.func(air-shrink-memref-sizes-by-access)",
            "air-label-scf-for-to-ping-pong",
            "air-ping-pong-transform",
            "canonicalize",
            "cse",
            "func.func(air-opt-memtile-dma-bds{device=npu1})",
            "canonicalize",
            "cse",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)
with open("aircc_input.mlir", "w") as f:
    f.write(str(air_module))

################################################
## Place herd to segment
################################################

air_async_module = Module.parse(str(air_module), context=context)
col_anchor = 1 if opts.trace_size > 0 else 0
pipeline = (
    "builtin.module("
    + ",".join(
        [
            "func.func(air-collapse-herd)",
            "canonicalize",
            "cse",
            "air-place-herds{num-rows=4 num-cols=1 row-anchor=2 col-anchor="
            + str(col_anchor)
            + "}",
            "canonicalize",
            "cse",
            "func.func(air-renumber-dma)",
            "func.func(convert-linalg-to-loops)",
        ]
    )
    + ")"
)

pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)
with open("air_placed.mlir", "w") as f:
    f.write(str(air_module))

# ################################################
# ## MLIR-AIR to MLIR-AIE
# ################################################

air_to_aie_pass = "air-to-aie{row-offset=2 col-offset=0 device=npu1 emit-while-loop=true use-pkt-flow-at-shim-dma=true use-lock-race-condition-fix=true"
if opts.trace_size > 0:
    air_to_aie_pass = air_to_aie_pass + " insert-trace-packet-flow=true"
air_to_aie_pass = air_to_aie_pass + "}"
pipeline = (
    "builtin.module("
    + ",".join(
        [
            air_to_aie_pass,
            "canonicalize",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)
with open("aircc_decomp_aiecc.mlir", "w") as f:
    f.write(str(air_module))

################################################
## MLIR-AIR runtime lowering
################################################

pipeline = (
    "builtin.module("
    + ",".join(
        [
            "func.func(air-opt-shim-dma-bds{device=npu1})",
            "air-to-std",
            "airrt-to-npu{"
            + f"trace-offset={opts.trace_offset} trace-size={opts.trace_size}"
            + "}",
            "canonicalize",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)
with open("aie.mlir", "w") as f:
    f.write(str(air_module))
