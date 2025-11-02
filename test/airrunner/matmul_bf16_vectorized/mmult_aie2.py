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
HERD_M = 4
HERD_N = 4

parser = argparse.ArgumentParser(prog="mmult_aie2.py")
parser.add_argument(
    "--input-file",
    default="input.mlir",
    type=str,
    help="Input file containing input IR in AIR dialect",
)
parser.add_argument(
    "--herd-m",
    type=int,
    default=HERD_M,
    help="Number of L1 tiles along the M dimension",
)
parser.add_argument(
    "--herd-n",
    type=int,
    default=HERD_N,
    help="Number of L1 tiles along the N dimension",
)
opts = parser.parse_args()

context = air.ir.Context()
with open(opts.input_file, "r") as f:
    air_ir_string = f.read()
air_module = Module.parse(air_ir_string, context=context)

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
            "air-place-herds{num-rows="
            + str(opts.herd_m)
            + " num-cols="
            + str(opts.herd_n)
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

runner = air.compiler.util.Runner(arch, "trace.out", "core", "all")
trace = runner.run(air_module, "matmul_bf16")
