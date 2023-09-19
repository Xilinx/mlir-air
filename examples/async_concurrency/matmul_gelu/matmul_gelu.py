# matmul_gelu.py -*- Python -*-
#
# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
from torch.nn import functional as F
import torch_mlir

import air.mlir.ir
import air.mlir.passmanager
import air.compiler.util

import sys

M = 384*64
N = 1024
K = 1024
dtype=torch.bfloat16

class mmult(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return F.gelu(torch.mm(a,b))


program = mmult()
mlir = torch_mlir.compile(
    program,
    (torch.ones((M,K), dtype=dtype), torch.ones((K,N), dtype=dtype)),
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
)

args = sys.argv[1:]
if len(args) and args[0] == '-dump-linalg':
    print(mlir)
    exit(0)

with air.mlir.ir.Context():
    # convert torch_mlir.ir.Module to air.mlir.ir.Module
    air_module = air.mlir.ir.Module.parse(str(mlir))

    # convert linalg on tensors to linalg on memrefs
    pm = air.mlir.passmanager.PassManager.parse(
        air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE)
    pm.run(air_module.operation)

    pipeline = "builtin.module("+",".join([
        "air-linalg-codegen{l2-tile-size=64,64,64 l2-promote=true l1-tile-size=32,32,32 l1-promote=true}",
        "canonicalize", "cse",
        "air-par-to-herd{depth=1}", # matmul
        "air-par-to-herd{depth=0}", # gelu
        "air-copy-to-dma",
        "air-par-to-launch{has-air-segment=true}",
        "canonicalize", "cse",
    ])+')'
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    with open('output1.mlir', 'w') as f:
        f.write(str(air_module))

    pipeline = "builtin.module("+",".join([
        "air-dependency",
        "air-dependency-schedule-opt",
        "air-specialize-dma-broadcast",
        "air-dependency-canonicalize",
        "air-dependency-parse-graph{output-dir=dot_graphs/}",
    ])+')'
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module.operation)

    with open('output2.mlir', 'w') as f:
        f.write(str(air_module))
