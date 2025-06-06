# mmult_aie2.py -*- Python -*-
#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import air.compiler.util

from air.dialects import func, linalg, tensor, arith, memref
from air.dialects.air import module_builder
from air.dialects.linalg.opdsl.lang import *
from air.ir import *
from air.compiler.util import run_transform
import air.passmanager

import sys
import argparse

# Default values.
M = 512
N = 512
K = 512
TILE_L1_M = 64
TILE_L1_N = 64
TILE_L1_K = 64
TILE_L2_M = 128
TILE_L2_N = 128


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
    dtype = BF16Type.get()

    @func.FuncOp.from_py_func(
        MemRefType.get((m, k), dtype), MemRefType.get((k, n), dtype)
    )
    def forward(lhs, rhs):
        out = memref.AllocOp(MemRefType.get((m, n), dtype), [], [])
        zero = arith.ConstantOp(dtype, 0.0)
        zero_fill = linalg.fill(zero, outs=[out])
        matmul_mono(lhs, rhs, outs=[out])
        return out


parser = argparse.ArgumentParser(prog="mmult_aie2.py")
parser.add_argument(
    "--m",
    default=M,
    type=int,
    help="M dimension size in a (MxK) * (KxN) matmul",
)
parser.add_argument(
    "--n",
    default=N,
    type=int,
    help="N dimension size in a (MxK) * (KxN) matmul",
)
parser.add_argument(
    "--k",
    default=K,
    type=int,
    help="K dimension size in a (MxK) * (KxN) matmul",
)
parser.add_argument(
    "--tile-l1-m",
    default=TILE_L1_M,
    type=int,
    help="M dimension size of each L1 tile",
)
parser.add_argument(
    "--tile-l1-n",
    default=TILE_L1_N,
    type=int,
    help="N dimension size of each L1 tile",
)
parser.add_argument(
    "--tile-l1-k",
    default=TILE_L1_K,
    type=int,
    help="K dimension size of each L1 tile",
)
parser.add_argument(
    "--tile-l2-m",
    default=TILE_L2_M,
    type=int,
    help="M dimension size of each L2 tile",
)
parser.add_argument(
    "--tile-l2-n",
    default=TILE_L2_N,
    type=int,
    help="N dimension size of each L2 tile",
)
opts = parser.parse_args()

# Inferred herd sizes from tiling sizes.
herd_x = int(opts.tile_l2_m / opts.tile_l1_m)
herd_y = int(opts.tile_l2_n / opts.tile_l1_n)

air_module = matmul_on_tensors(opts.m, opts.n, opts.k)
context = air_module.context

# convert linalg on tensors to linalg on memrefs
pm = air.passmanager.PassManager.parse(
    air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE,
    context=context,
)
pm.run(air_module.operation)

transform_ir_string = f"""
transform.with_pdl_patterns {{
^bb0(%arg0: !pdl.operation):
    transform.sequence %arg0 : !pdl.operation failures(propagate) {{
    ^bb1(%arg1: !pdl.operation):
        %fill = transform.structured.match ops{{["linalg.fill"]}} in %arg1  : (!pdl.operation) -> !pdl.operation
        %matmul = transform.structured.match ops{{["linalg.generic"]}} in %arg1  : (!pdl.operation) -> !pdl.operation
        %matmul_1, %loops:2 = transform.air.linalg_tile %matmul [{opts.tile_l2_m}, {opts.tile_l2_n}, 0]
        %fill_1 = transform.air.fuse_into_containing_op %fill into %loops#1
        transform.air.linalg_promote %fill_1 {{"operands_to_promote"=[1], "memory_space"="L2"}}
        transform.air.linalg_promote %matmul_1 {{"operands_to_promote"=[2], "memory_space"="L2"}}
        transform.air.linalg_promote %matmul_1 {{"operands_to_promote"=[0,1], "memory_space"="L2"}}
        %matmul_2, %loops_2:2 = transform.air.linalg_tile %matmul_1 [{opts.tile_l1_m}, {opts.tile_l1_n}, 0]
        %fill_2 = transform.air.fuse_into_containing_op %fill_1 into %loops_2#1
        transform.air.linalg_promote %fill_2 {{"operands_to_promote"=[1], "memory_space"="L1"}}
        transform.air.linalg_promote %matmul_2 {{"operands_to_promote"=[2], "memory_space"="L1"}}
        %matmul_3, %reduction_loop = transform.air.linalg_tile %matmul_2 [0, 0, {opts.tile_l1_k}]
        transform.air.linalg_promote %matmul_3 {{"operands_to_promote"=[0,1], "memory_space"="L1"}}
    }}
}}
"""
transform_ir = Module.parse(transform_ir_string, context=context)
run_transform(transform_ir, air_module)

# tile and map to air
pipeline = (
    "builtin.module("
    + ",".join(
        [
            "buffer-results-to-out-params",
            "air-par-to-herd{depth=-1}",
            "air-par-to-launch{has-air-segment=true}",
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

# generate dependency information for runner
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
            "func.func(air-fuse-alloc-dealloc)",
            "func.func(air-shrink-memref-sizes-by-access)",
            "air-label-scf-for-to-ping-pong",
            "air-ping-pong-transform",
            "canonicalize", "cse",
            "air-place-herds{num-rows="
            + str(herd_x)
            + " num-cols="
            + str(herd_y)
            + " row-anchor=0 col-anchor=0}",
        ]
    )
    + ")"
)
pm = air.passmanager.PassManager.parse(pipeline, context=context)
pm.run(air_module.operation)

with open("air_ir_debug.mlir", "w") as f:
    f.write(str(air_module))

arch = {
    "clock": 1000000000,
    "cores": 1,
    "datatypes": [
        {"bytes": 1, "name": "i8"},
        {"bytes": 2, "name": "bf16"},
        {"bytes": 4, "name": "i32"},
    ],
    "devicename": "testdevice",
    "kernels": {
        "linalg.copy": {
            "datatypes": {
                "i8": {"ops_per_core_per_cycle": 32, "efficiency": 1},
                "bf16": {"ops_per_core_per_cycle": 32, "efficiency": 1},
                "i32": {"ops_per_core_per_cycle": 16, "efficiency": 1},
            },
            "name": "linalg.copy",
        },
        "linalg.fill": {
            "datatypes": {
                "i8": {"ops_per_core_per_cycle": 32, "efficiency": 1},
                "bf16": {"ops_per_core_per_cycle": 32, "efficiency": 1},
                "i32": {"ops_per_core_per_cycle": 16, "efficiency": 1},
            },
            "name": "linalg.fill",
        },
        "linalg.generic": {
            "datatypes": {
                "i8": {"macs_per_core_per_cycle": 256, "efficiency": 1},
                "bf16": {"macs_per_core_per_cycle": 128, "efficiency": 1},
                "i32": {"macs_per_core_per_cycle": 1, "efficiency": 1},
            },
            "name": "linalg.generic",
        },
        "linalg.matmul": {
            "datatypes": {
                "i8": {"macs_per_core_per_cycle": 256, "efficiency": 1},
                "bf16": {"macs_per_core_per_cycle": 128, "efficiency": 1},
                "i32": {"macs_per_core_per_cycle": 1, "efficiency": 1},
            },
            "name": "linalg.matmul",
        },
    },
    "dus": {
        "count": [4, 4],
        "memory": {"memory_space": "L2", "bytes": 524288},
        "ports": {
            "outbound": {"count": 6, "bytes_per_second": 4000000000},
            "inbound": {"count": 6, "bytes_per_second": 4000000000},
        },
        "tiles": {
            "count": [1, 4],
            "memory": {"memory_space": "L1", "bytes": 65536},
            "ports": {
                "outbound": {"count": 2, "bytes_per_second": 4000000000},
                "inbound": {"count": 2, "bytes_per_second": 4000000000},
            },
        },
    },
    "noc": {
        "outbound": {"count": 8, "bytes_per_second": 4000000000},
        "inbound": {"count": 8, "bytes_per_second": 4000000000},
    },
}

runner = air.compiler.util.Runner(arch, "trace.out", "core")
trace = runner.run(air_module, "forward")
