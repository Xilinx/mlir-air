# mmult.py -*- Python -*-
#
# Copyright (C) 2022, Advanced Micro Devices. All rights reserved.
# SPDX-License-Identifier: MIT

import air.compiler.util

from air.dialects import func
from air.dialects import linalg
from air.ir import *
import air.passmanager

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


with air.ir.Context(), Location.unknown():

    air_module = matmul_on_tensors(512, 512, 512, BF16Type.get())
    
    # convert linalg on tensors to linalg on memrefs
    pm = air.passmanager.PassManager.parse(air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE)
    pm.run(air_module)

    pipeline = ",".join([
        "air-linalg-codegen{l1-tile-size=32,32,32 l1-tile-permute=2,0,1 l2-tile-size=64,64,32 l2-promote=false}",
        "affine-to-air{herd-assign-depth=1}",
        "canonicalize", "cse",
        "air-specialize-dma",
        "air-promote-dma",
        "canonicalize", "cse",
        "air-pipeline-to-affine",
        "canonicalize", "cse",
    ])
    pm = air.passmanager.PassManager.parse(pipeline)
    pm.run(air_module)

    print (air_module)
