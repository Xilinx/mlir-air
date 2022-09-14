# (c) Copyright 2022 AMD Inc. All Rights Reserved.

import torch
from torch.nn import functional as F
import torch_mlir

import air.mlir.ir
import air.mlir.passmanager
import air.compiler.util

import sys

M = 64
N = 64
K = 256
dtype=torch.bfloat16

class mmult(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, q, k, v):
        qk = torch.mm(torch.mm(x,q),torch.mm(x,k))
        qkv = torch.mm(F.softmax(qk,dim=-1), torch.mm(x,v))
        return qkv

program = mmult()
mlir = torch_mlir.compile(
    program,
    (torch.ones((M,K), dtype=dtype), torch.ones((K,N), dtype=dtype), 
    torch.ones((K,N), dtype=dtype), torch.ones((K,N), dtype=dtype)),
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
    pm.run(air_module)

##  Removed from pipeline for now:
    pipeline = ",".join([
        "air-linalg-codegen{l2-tile-size=64,64,64 l2-promote=true l1-tile-size=32,32,32 l1-promote=true}",
        "canonicalize", "cse",
        "air-copy-to-dma",
        "air-par-to-herd{depth=1}", # matmul
        "air-par-to-herd{depth=0}", # softmax
        "air-par-to-launch{has-air-partition=true}",
        "canonicalize", "cse",
    ])
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module)

    with open('output1.mlir', 'w') as f:
        f.write(str(air_module))

    pipeline = ",".join([
        "air-dependency",
        "air-dependency-schedule-opt",
        "air-specialize-dma-broadcast",
    ])
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module)

    with open('output2.mlir', 'w') as f:
        f.write(str(air_module))
