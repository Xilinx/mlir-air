# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# Also available under a BSD-style license. See LICENSE.

import torch_mlir.ir
import torch_mlir.passmanager

import air.mlir.ir
import air.mlir.passmanager

# Imported for side effects.
import torch_mlir.all_passes_registration
import air.mlir.all_passes_registration

from torch_mlir_e2e_test.utils import run_pipeline_with_repro_report
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

from .abc import AirBackend

import air.mlir._mlir_libs._airMlir
import air.compiler.aircc.main as aircc

__all__ = [
    "LinalgOnTensorsAirBackend",
]

ATEN_TO_LINALG_TENSORS_PIPELINE = ",".join([
    "torch-verify-invariants-before-backend-lowering",
    "builtin.func(convert-torch-to-linalg)",
    "builtin.func(convert-torch-to-std)",
    "builtin.func(convert-torch-to-scf)",
    "builtin.func(std-expand)",
    "canonicalize",
    "resolve-shaped-type-result-dims",
    "cse",
    "torch-func-backend-type-conversion",
    "builtin.func(torch-finalizing-backend-type-conversion)"
])

LINALG_TENSOR_TO_MEMREF_PIPELINE = ",".join([
    # Bufferize.
    "tensor-constant-bufferize",
    "builtin.func(scf-bufferize)",
    "builtin.func(linalg-bufferize)",
    "builtin.func(std-bufferize)",
    "builtin.func(tensor-bufferize)",
    "func-bufferize",
    "builtin.func(finalizing-bufferize)",
    "canonicalize",
    "cse"
])

LINALG_MEMREF_TO_AIRRT_PIPELINE = ",".join([
    "air-linalg-codegen",
    "canonicalize",
    "cse",
    "affine-to-air",
    "canonicalize",
    "cse"
])

LOWERING_PIPELINE = ",".join([
    # Bufferize.
    "tensor-constant-bufferize",
    "builtin.func(scf-bufferize)",
    "builtin.func(linalg-bufferize)",
    "builtin.func(std-bufferize)",
    "builtin.func(tensor-bufferize)",
    "func-bufferize",
    "builtin.func(finalizing-bufferize)",
    # Munge to make it ExecutionEngine compatible.
    # Specifically, we rewrite calling convention boundaries to be in terms
    # of unranked memref, and we rewrite the return to actually be a
    # callback that consumes the return (the final munged function always
    # returns void at the C level -- we get the return value by providing the
    # callback).
    "refback-munge-calling-conventions",
    # Lower to LLVM
    "builtin.func(convert-linalg-to-loops)",
    "builtin.func(lower-affine)",
    "builtin.func(convert-scf-to-std)",
    "builtin.func(refback-expand-ops-for-llvm)",
    "builtin.func(convert-math-to-llvm)",
    "convert-memref-to-llvm",
    "convert-std-to-llvm",
    "reconcile-unrealized-casts",
])

class LinalgOnTensorsAirBackend(AirBackend):
    """Main entry-point for the linalg-on-tensors based AIR backend.

    This currently uses the linalg-on-tensors RefBackend for actual execution.
    """
    def __init__(self):
        super().__init__()
        self.refbackend = RefBackendLinalgOnTensorsBackend()

    def compile(self, imported_module: torch_mlir.ir.Module):
        """Compiles an imported module, with a flat list of functions.
        The module is expected to be in linalg-on-tensors + scalar code form.
        TODO: More clearly define the backend contract. Generally this will
        extend to support globals, lists, and other stuff.

        Args:
          imported_module: The MLIR module consisting of funcs in the torch
            dialect.
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """

        run_pipeline_with_repro_report(
            imported_module, ATEN_TO_LINALG_TENSORS_PIPELINE,
            "Lowering ATen IR to linalg on tensors")

        with air.mlir.ir.Context():
            air_module = air.mlir.ir.Module.parse(str(imported_module))
            pm = air.mlir.passmanager.PassManager.parse(LINALG_TENSOR_TO_MEMREF_PIPELINE)
            pm.run(air_module)
            pm = air.mlir.passmanager.PassManager.parse(LINALG_MEMREF_TO_AIRRT_PIPELINE)
            pm.run(air_module)
            aircc.run(air_module,['-o', 'mlir.air.a', '--sysroot=/work/aarch64/mnt', '-row-offset=2', '-col-offset=7', 'torch.mlir'])
            with open('air_project/llvm.torch.mlir') as f:
                imported_module = torch_mlir.ir.Module.parse(f.read(),imported_module.context)
        return imported_module

    def load(self, module):
        """Loads a compiled artifact into the runtime."""
        return self.refbackend.load(module)
