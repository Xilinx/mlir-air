# (c) Copyright 2022 Advanced Micro Devices Inc. All Rights Reserved.

import torch
import torch_mlir

import air.mlir.ir
import air.mlir.passmanager
import air.compiler.util

import sys

class mmult(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.mm(a,b)

program = mmult()
mlir = torch_mlir.compile(program, (torch.ones(256,256), torch.ones(256,256)), output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)

args = sys.argv[1:]
if len(args) and args[0] == '-dump-linalg':
    print(mlir)
    exit(0)

with air.mlir.ir.Context():
    # convert torch_mlir.ir.Module to air.mlir.ir.Module
    air_module = air.mlir.ir.Module.parse(str(mlir))

    # convert linalg on tensors to linalg on memrefs
    pm = air.mlir.passmanager.PassManager.parse(air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE)
    pm.run(air_module)

    # tile, convert to air, generate dependencies
    pipeline = ",".join([
        "buffer-results-to-out-params",
        "air-linalg-codegen{l1-tile-size=32,32,32}",
        "affine-to-air",
        "canonicalize", "cse",
        "air-dependency"
    ])
    pm = air.mlir.passmanager.PassManager.parse(pipeline)
    pm.run(air_module)

print (air_module)

runner = air.compiler.util.Runner("arch.json")
trace = runner.run(air_module, "forward")

with open("trace.out", "w") as f:
    f.write(trace)
