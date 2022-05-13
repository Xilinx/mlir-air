# (c) Copyright 2022 Advanced Micro Devices Inc. All Rights Reserved.

import torch
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
        return torch.mm(a,b)


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
    pm.run(air_module)

    pipeline = ",".join([
        # "buffer-results-to-out-params",
        # "air-linalg-codegen{l2-tile-size=0,0,256 l2-promote=false l1-tile-size=64,64,64 l1-promote=false}",
        "air-linalg-codegen{l1-tile-size=192,256,0 l1-promote=false}",
        "air-linalg-codegen{l2-tile-size=0,0,256 l2-promote=true l1-tile-size=64,64,256 l1-promote=false}",
        "canonicalize", "cse",
        # "func.func(scf-parallel-loop-tiling{parallel-loop-tile-sizes=1,4})",
        # "canonicalize", "cse",
        # "func.func(scf-parallel-loop-tiling{parallel-loop-tile-sizes=3,1})",
        # "canonicalize", "cse",
        # "scf-parallel-loop-collapsing{collapsed-indices-0=0 collapsed-indices-1=1 collapsed-indices-2=2,3}",
        "canonicalize", "cse",
        "func.func(air-pipeline-reduce)",
        "canonicalize", "cse",
        "scf-parallel-loop-collapsing{collapsed-indices-0=0,1}",
        "canonicalize", "cse",
        
    ])
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module)

print(air_module)
