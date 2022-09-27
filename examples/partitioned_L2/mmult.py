# ./examples/partitioned_L2/mmult.py -*- Python -*-

# Copyright (C) 2022, Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module)

    print (air_module)
