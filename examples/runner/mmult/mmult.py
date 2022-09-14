# (c) Copyright 2022 Advanced Micro Devices Inc. All Rights Reserved.

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
    pipeline = ",".join([
        "air-linalg-codegen{l1-tile-size=32,32,32 l2-tile-size=64,64,64 l2-promote=true}",
        "air-par-to-herd{depth=1}",
        "air-par-to-launch{depth=0}",
        "air-copy-to-dma",
        "canonicalize", "cse",
    ])
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module)
    
    print ("\nAIR Dialect Module\n")
    print (air_module)

    # generate dependency information for runner
    pm = air.mlir.passmanager.PassManager.parse("air-dependency,canonicalize,cse")
    pm.run(air_module)

    print ("\nAIR Dialect Module (async)\n")
    print (air_module)

    arch = {
    "clock": 1000000000,
    "cores": 1,
    "datatype": {
        "bytes": 2,
        "name": "fp16"
    },
    "devicename": "testdevice",
    "interfaces": [
        {
        "bytes_per_second": 100000000000,
        "dst": 1,
        "src": 0
        },
        {
        "bytes_per_second": 100000000000,
        "dst": 0,
        "src": 1
        },
        {
        "bytes_per_second": 100000000000,
        "dst": 2,
        "src": 0
        },
        {
        "bytes_per_second": 100000000000,
        "dst": 0,
        "src": 2
        },
        {
        "bytes_per_second": 100000000000,
        "dst": 2,
        "src": 1
        },
        {
        "bytes_per_second": 100000000000,
        "dst": 1,
        "src": 2
        }
    ],
    "kernels": {
        "linalg.copy": {
        "efficiency": 1,
        "name": "linalg.copy"
        },
        "linalg.fill": {
        "efficiency": 1,
        "name": "linalg.fill"
        },
        "linalg.matmul": {
        "efficiency": 1,
        "name": "linalg.matmul"
        }
    },
    "ops_per_core_per_cycle": 512,
    "num_herd_slots": 4,
    "num_dispatch_queues": 8,
    "num_dispatch_dma_queues" : 2,
    "num_core_dma_queues" : 2
    }

runner = air.compiler.util.Runner(arch)
trace = runner.run(air_module, "matmul")

with open("trace.out", "w") as f:
   f.write(trace)
