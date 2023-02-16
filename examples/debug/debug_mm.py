# debug_mm.py -*- Python -*-
#
# Copyright (C) 2023, Advanced Micro Devices. All rights reserved.
# SPDX-License-Identifier: MIT

import torch
from torch.nn import functional as F
import torch_mlir

import air.mlir.ir
from air.backend import linalg_on_tensors as backend
import air.compiler.aircc.main as aircc
import air.mlir.passmanager
import air.compiler.util

import air.mlir._mlir_libs._airRt as airrt

import sys

M = 64
N = 64
K = 64
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
compiled = airbackend.compile(mlir, backend.LINALG_MEMREF_TO_AIR_PIPELINE, False, [9,2])
airrt.host.init()
a = airrt.host.get_agents()
q = airrt.host.queue_create(a[0])
airbackend.handle = airrt.host.module_load_from_file("./torch.mlir.so", q)
jit_module = airbackend.refbackend.load(compiled)

################################
# Do some profiling setup here #
################################

core92 = airrt.host.get_tile_addr(9,2)
pc_reg = 0x00030280
val = airrt.host.read32(core92 + pc_reg)
print('Core 9,2 PC @ ',hex(core92+pc_reg),' = ',hex(val))

################################
################################

x = torch.rand((M,K), dtype=dtype)
y = torch.rand((K,N), dtype=dtype)

# compute golden reference
mm_ref = program(x, y)
print(mm_ref)

# compute using AIR
mm = torch.tensor(
    jit_module.forward(x.numpy(), y.numpy()))
print(mm)

##############################
# Read profiling values here #
##############################

val = airrt.host.read32(core92 + pc_reg)
print('Core 9,2 PC @ ',hex(core92+pc_reg),' = ',hex(val))

##############################
##############################

if torch.allclose(mm_ref, mm):
    print("PASS!")
else:
    print("failed.")
