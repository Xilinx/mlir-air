# mlp.py -*- Python -*-
#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# This example converts a simple MLP model to MLIR-AIR
import torch
import torch.nn as nn
import torch_mlir

import air.mlir.ir
import air.mlir.passmanager
import air.compiler.util

# Model specifications
class MLP(torch.nn.Module):
    # Toy quantized FC model for digit recognition on MNIST

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(28 * 28, 28 * 28)
        self.fc2 = nn.Linear(28 * 28, 28 * 28 * 4)
        self.fc3 = nn.Linear(28 * 28 * 4, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

mlp = MLP()

# Inputs to the model
batch_size = 56
x = torch.randn((batch_size, 28, 28))

# Emit the model to Linalg MLIR
module = torch_mlir.compile(mlp, x, output_type="linalg-on-tensors")

with air.mlir.ir.Context():
    # convert torch_mlir.ir.Module to air.mlir.ir.Module
    air_module = air.mlir.ir.Module.parse(str(module))

    with open('output0.mlir', 'w') as f:
        f.write(str(air_module))

    # convert linalg on tensors to linalg on memrefs
    pm = air.mlir.passmanager.PassManager.parse(
        air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE)
    pm.run(air_module.operation)

    # AIR pipeline
    lowering_pipeline = "builtin.module("+",".join([
        "air-linalg-name",
        "air-linalg-codegen{input-filter=linalg.matmul2 l2-tile-size=56,56,56 l2-promote=true l1-tile-size=14,14,14 l1-promote=true}",
        "air-linalg-codegen{input-filter=linalg.matmul7 l2-tile-size=56,56,56 l2-promote=true l1-tile-size=14,14,14 l1-promote=true}",
        "air-linalg-codegen{input-filter=linalg.matmul12 l2-tile-size=56,10,56 l2-promote=true l1-tile-size=14,10,14 l1-promote=true}",
        "air-rm-linalg-name", 
        "air-par-to-herd{depth=1}",
        "air-par-to-launch{has-air-partition=true}",
        "air-copy-to-dma",
        "canonicalize", "cse",
    ])+')'

    pm = air.mlir.passmanager.PassManager.parse(lowering_pipeline)
    pm.run(air_module.operation)

    with open('output1.mlir', 'w') as f:
        f.write(str(air_module))

    lowering_pipeline = "builtin.module("+",".join([
        "air-dependency",
        "air-dependency-schedule-opt",
        "air-specialize-dma-broadcast",
        "air-dma-to-channel",
        "canonicalize", "cse",
        "air-dependency-canonicalize",
        "air-dependency-parse-graph{output-dir=dot_graphs/}",
    ])+')'
    pm = air.mlir.passmanager.PassManager.parse(lowering_pipeline)
    pm.run(air_module.operation)

    with open('output2.mlir', 'w') as f:
        f.write(str(air_module))