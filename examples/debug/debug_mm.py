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

M = 32
N = 32
K = 32
dtype=torch.float

class mmult(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mm(x, y)

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

airbackend = backend.LinalgOnTensorsAirBackend()
compiled = airbackend.compile(mlir, backend.LINALG_MEMREF_TO_AIR_PIPELINE)
jit_module = airbackend.load(compiled)

x = torch.rand((M,K), dtype=dtype)
y = torch.rand((K,N), dtype=dtype)

# compute golden reference
mm_ref = program(x, y)
print(mm_ref)

# compute using AIR
mm = torch.tensor(
    jit_module.forward(x.numpy(), y.numpy()))
print(mm)

if torch.allclose(mm_ref, mm):
    print("PASS!")
else:
    print("failed.")
