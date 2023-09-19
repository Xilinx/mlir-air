# mmult.py -*- Python -*-
#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import air.compiler.util

from air.mlir.dialects import func, linalg, tensor, arith
from air.mlir.ir import *
import air.mlir.passmanager

import sys

M = int(sys.argv[1])
N = int(sys.argv[2])
K = int(sys.argv[3])
Tiling_L1_m = int(sys.argv[4])
Tiling_L1_n = int(sys.argv[5])
Tiling_L1_k = int(sys.argv[6])

herd_x = int(M / Tiling_L1_m)
herd_y = int(N / Tiling_L1_n)

def matmul_on_tensors(m, n, k, dtype):
    module = Module.create()
    with InsertionPoint(module.body):
        @func.FuncOp.from_py_func(
            RankedTensorType.get((m, k), dtype), RankedTensorType.get((k, n), dtype))
        def matmul(lhs, rhs):
            out = tensor.EmptyOp([m, n], dtype)
            zero = arith.ConstantOp(dtype, 0.0)
            zero_fill = linalg.fill(zero, outs=[out])
            return linalg.matmul(lhs, rhs, outs=[zero_fill])
    return module


with air.mlir.ir.Context(), Location.unknown():

    air_module = matmul_on_tensors(M, N, K, F32Type.get())
    
    # convert linalg on tensors to linalg on memrefs
    pm = air.mlir.passmanager.PassManager.parse(air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE)
    pm.run(air_module.operation)

    args = sys.argv[1:]
    if len(args) and args[0] == '-dump-linalg':
        print (air_module)
        exit(0)

    # tile and map to air
    pipeline = "builtin.module("+",".join([
        "buffer-results-to-out-params",
        "air-linalg-codegen{l1-tile-size=" + str(Tiling_L1_m) + "," + str(Tiling_L1_n) + "," + str(Tiling_L1_k) + " l1-promote=true}",
        "air-par-to-herd",
        "air-copy-to-dma",
        "canonicalize", "cse",
        "air-insert-launch-and-segment-around-herd",
    ])+')'
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)
    
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
        # "air-dependency-parse-graph{output-dir=dot_graphs/}",
        "air-place-herds{num-rows=" + str(herd_x) + " num-cols=" + str(herd_y) + " row-anchor=0 col-anchor=0}",
        "air-label-scf-for-to-ping-pong",
        "air-ping-pong-transform"
    ])+')'
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    print ("\nAIR Dialect Module (async)\n")
    print (air_module)

    arch = {
    "clock": 1000000000,
    "cores": 1,
    "datatypes": [
        {
        "bytes": 1,
        "name": "i8"
        },
        {
        "bytes": 4,
        "name": "f32"
        },
        {
        "bytes": 4,
        "name": "i32"
        }
    ],
    "devicename": "testdevice",
    "kernels": {
        "linalg.copy": {
            "datatypes": {
                "i8": {
                    "ops_per_core_per_cycle": 32,
                    "efficiency": 1
                },
                "f32": {
                    "ops_per_core_per_cycle": 8,
                    "efficiency": 1
                },
                "i32": {
                    "ops_per_core_per_cycle": 8,
                    "efficiency": 1
                }
            },
            "name": "linalg.copy"
        },
        "linalg.fill": {
            "datatypes": {
                "i8": {
                    "ops_per_core_per_cycle": 32,
                    "efficiency": 1
                },
                "f32": {
                    "ops_per_core_per_cycle": 8,
                    "efficiency": 1
                },
                "i32": {
                    "ops_per_core_per_cycle": 8,
                    "efficiency": 1
                }
            },
            "name": "linalg.fill"
        },
        "linalg.generic": {
            "datatypes": {
                "i8": {
                    "ops_per_core_per_cycle": 1,
                    "efficiency": 1
                },
                "f32": {
                    "ops_per_core_per_cycle": 1,
                    "efficiency": 1
                },
                "i32": {
                    "ops_per_core_per_cycle": 1,
                    "efficiency": 1
                }
            },
            "name": "linalg.generic"
        },
        "linalg.matmul": {
            "datatypes": {
                "i8": {
                    "macs_per_core_per_cycle": 128,
                    "efficiency": 1
                },
                "f32": {
                    "macs_per_core_per_cycle": 8,
                    "efficiency": 1
                },
                "i32": {
                    "macs_per_core_per_cycle": 8,
                    "efficiency": 1
                }
            },
            "name": "linalg.matmul"
        }
    },
    "dus": {
        "count": [1, 1],
        "memory": {
            "memory_space": "L2",
            "bytes": 1
        },
        "ports": {
            "outbound": {
                "count": 1,
                "bytes_per_second": 4000000000
            },
            "inbound": {
                "count": 1,
                "bytes_per_second": 4000000000
            }
        },
        "tiles": {
            "count": [50, 8],
            "memory": {
                "memory_space": "L1",
                "bytes": 32768
            },
            "ports": {
                "outbound": {
                    "count": 2,
                    "bytes_per_second": 4000000000
                },
                "inbound": {
                    "count": 2,
                    "bytes_per_second": 4000000000
                }
            }
        }
    },
    "noc": {
        "outbound": {
            "count": 6,
            "bytes_per_second": 4000000000
        },
        "inbound": {
            "count": 6,
            "bytes_per_second": 4000000000
        }
    }
  }

runner = air.compiler.util.Runner(arch, "trace.out", "core")
trace = runner.run(air_module, "matmul")
