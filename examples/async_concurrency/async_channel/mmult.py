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
    pm.run(air_module.operation)

    args = sys.argv[1:]
    if len(args) and args[0] == '-dump-linalg':
        print (air_module)
        exit(0)

    # tile and map to air
    pipeline = "builtin.module("+",".join([
        "air-linalg-codegen{l2-tile-size=64,64,64 l2-promote=true l1-tile-size=16,16,16 l1-promote=true}",
        "air-par-to-herd{depth=1}",
        "air-par-to-launch{depth=0 has-air-segment=true}",
        "air-copy-to-dma",
        "canonicalize", "cse",
    ])+')'
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    with open('output1.mlir', 'w') as f:
        f.write(str(air_module))

    # async dep
    pipeline = "builtin.module("+",".join([
        "air-dependency",
        "air-dependency-schedule-opt",
        # "air-specialize-dma-broadcast", # Uncomment to lower to specialized channels
        "air-dma-to-channel",
        "canonicalize", "cse",
        "air-dependency-canonicalize",
        "air-dependency-parse-graph{output-dir=dot_graphs/}"
    ])+')'
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    with open('output2.mlir', 'w') as f:
        f.write(str(air_module))
