# airbert.py -*- Python -*-
#
# Copyright (C) 2022, Advanced Micro Devices. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
from torch.nn import functional as F
import torch_mlir

import air.mlir.ir
from air.backend import linalg_on_tensors as backend
import air.compiler.aircc.main as aircc
import air.mlir.passmanager
import air.compiler.util

import sys

M = 64
N = 64
K = 256
dtype=torch.float

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

lowering_pipeline = "builtin.module("+",".join([
    "air-linalg-name",
    "air-linalg-codegen{input-filter=linalg.matmul1 herd-size=2,2 l1-tile-size=32,32,32}",
    "air-linalg-codegen{input-filter=linalg.matmul2 herd-size=2,2 l1-tile-size=32,32,32}",
    "air-linalg-codegen{input-filter=linalg.matmul3 herd-size=2,2 l1-tile-size=32,32,32}",
    # "air-linalg-codegen{input-filter=linalg.generic7 herd-size=1,1 l1-tile-size=64,64,64}",
    # "air-linalg-codegen{input-filter=linalg.generic8 herd-size=1,1 l1-tile-size=64,64,64}",
    # "air-linalg-codegen{input-filter=linalg.generic9 herd-size=1,1 l1-tile-size=64,64,64}",
    # "air-linalg-codegen{input-filter=linalg.generic11 herd-size=1,1 l1-tile-size=64,64,64}",
    # "air-linalg-codegen{input-filter=linalg.generic12 herd-size=1,1 l1-tile-size=64,64,64}",
    "air-linalg-codegen{input-filter=linalg.matmul12 herd-size=2,2 l1-tile-size=32,32,32}",
    "air-linalg-codegen{input-filter=linalg.matmul13 herd-size=2,2 l1-tile-size=32,32,32}",
    "air-rm-linalg-name",
    "canonicalize",
    "cse",
    "air-par-to-herd",
    "air-copy-to-dma",
    "canonicalize",
    "cse"
])+')'

airbackend = backend.LinalgOnTensorsAirBackend()
compiled = airbackend.compile(mlir, lowering_pipeline)
jit_module = airbackend.load(compiled)

x = torch.rand((M,K), dtype=dtype)
q = torch.rand((K,N), dtype=dtype)
k = torch.rand((K,N), dtype=dtype)
v = torch.rand((K,N), dtype=dtype)

# compute golden reference
qkv_ref = program(x, q, k, v)
print(qkv_ref)

# compute using AIR
qkv = torch.tensor(
    jit_module.forward(x.numpy(), q.numpy(), k.numpy(), v.numpy()))
print(qkv)

if torch.allclose(qkv_ref, qkv):
    print("PASS!")
else:
    print("failed.")