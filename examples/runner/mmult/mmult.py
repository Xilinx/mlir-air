# mmult.py -*- Python -*-
#
# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import air.compiler.util

from air.mlir.dialects import func
from air.mlir.dialects import linalg
from air.mlir.ir import *
import air.mlir.passmanager

import sys

def matmul_on_tensors(m, n, k, dtype):
    module = Module.create()
    with InsertionPoint(module.body):
        @func.FuncOp.from_py_func(
            RankedTensorType.get((m, k), dtype), RankedTensorType.get((k, n), dtype),
            RankedTensorType.get((m, n), dtype))
        def matmul(lhs, rhs, out):
            linalg.matmul(lhs, rhs, outs=[out])
    return module


with air.mlir.ir.Context(), Location.unknown():

    air_module = matmul_on_tensors(512, 512, 512, BF16Type.get())
    
    # convert linalg on tensors to linalg on memrefs
    pm = air.mlir.passmanager.PassManager.parse(air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE)
    pm.run(air_module)

    args = sys.argv[1:]
    if len(args) and args[0] == '-dump-linalg':
        print (air_module)
        exit(0)

    # tile and map to air
    pipeline = "builtin.module("+",".join([
        "air-linalg-codegen{l2-tile-size=64,64,64 l2-promote=true l1-tile-size=32,32,32 l1-promote=true}",
        "canonicalize", "cse",
        "air-par-to-herd{depth=1}",
        "air-copy-to-dma",
        "air-par-to-launch{has-air-segment=true}",
        "canonicalize", "cse",
    ])+')'
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module)
    
    print ("\nAIR Dialect Module\n")
    print (air_module)

    # generate dependency information for runner
    pipeline = "builtin.module("+",".join([
        "air-dependency",
        "air-dependency-schedule-opt",
        "air-specialize-dma-broadcast",
        "air-dma-to-channel",
        "canonicalize", "cse",
        "air-dependency-canonicalize",
        "air-dependency-parse-graph{output-dir=dot_graphs/}",
        "canonicalize", "cse",
        "air-place-herds{num-rows=2 num-cols=2 row-anchor=0 col-anchor=0}"
    ])+')'
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module)

    print ("\nAIR Dialect Module (async)\n")
    print (air_module)

    arch = {
    "clock": 1000000000,
    "cores": 1,
    "datatypes": [
        {
        "bytes": 2,
        "name": "bf16"
        },
        {
        "bytes": 4,
        "name": "f32"
        }
    ],
    "devicename": "testdevice",
    "kernels": {
        "linalg.copy": {
            "datatypes": {
                "bf16": {
                    "ops_per_core_per_cycle": 8,
                    "efficiency": 1
                },
                "f32": {
                    "ops_per_core_per_cycle": 8,
                    "efficiency": 1
                }
            },
            "name": "linalg.copy"
        },
        "linalg.fill": {
            "datatypes": {
                "bf16": {
                    "ops_per_core_per_cycle": 8,
                    "efficiency": 1
                },
                "f32": {
                    "ops_per_core_per_cycle": 8,
                    "efficiency": 1
                }
            },
            "name": "linalg.fill"
        },
        "linalg.matmul": {
            "datatypes": {
                "bf16": {
                    "ops_per_core_per_cycle": 8,
                    "efficiency": 1
                },
                "f32": {
                    "ops_per_core_per_cycle": 8,
                    "efficiency": 1
                }
            },
            "name": "linalg.matmul"
        }
    },
    "ops_per_core_per_cycle": 512,
    "num_herd_slots": 4,
    "num_dispatch_queues": 8,
    "num_dispatch_dma_queues" : 2,
    "num_core_dma_queues" : 2,
    "dus": {
        "count": [4, 1],
        "memory": {
            "memory_space": "L2",
            "bytes": 524288
        },
        "ports": {
            "outbound": {
                "count": 6,
                "bytes_per_second": 100000000000
            },
            "inbound": {
                "count": 6,
                "bytes_per_second": 100000000000
            }
        },
        "tiles": {
            "count": [1, 4],
            "memory": {
                "memory_space": "L1",
                "bytes": 65536
            },
            "ports": {
                "outbound": {
                    "count": 2,
                    "bytes_per_second": 100000000000
                },
                "inbound": {
                    "count": 2,
                    "bytes_per_second": 100000000000
                }
            }
        }
    },
    "noc": {
        "outbound": {
            "count": 4,
            "bytes_per_second": 100000000000
        },
        "inbound": {
            "count": 4,
            "bytes_per_second": 100000000000
        }
    }
  }

runner = air.compiler.util.Runner(arch)
trace = runner.run(air_module, "matmul")

with open("trace.out", "w") as f:
   f.write(trace)
