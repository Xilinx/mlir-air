# ./python/air/backend/cpu_backend.py -*- Python -*-
#
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import torch
import torch_mlir.ir
import torch_mlir.passmanager
from torch_mlir import torchscript

import air.ir
import air.passmanager

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import (
    RefBackendLinalgOnTensorsBackend,
)

from .abc import AirBackend

import air.compiler.util
from air.backend import linalg_on_tensors

import ctypes
from pathlib import Path

from typing import List

path = Path(air.backend.__file__).resolve().parent
ctypes.CDLL(
    f"{path}/../../../runtime_lib/x86_64/aircpu/libaircpu.so", mode=ctypes.RTLD_GLOBAL
)
ctypes.CDLL(
    f"/FIXME/PATH/TO/llvm/lib/libmlir_async_runtime.so.20.0git", mode=ctypes.RTLD_GLOBAL
)

__all__ = ["AirCpuBackend", "DEFAULT_PIPELINE"]

DEFAULT_PIPELINE = (
    "builtin.module(" + ",".join(["air-to-async", "canonicalize", "cse"]) + ")"
)

ASYNC_TO_LLVM_PIPELINE = (
    "builtin.module("
    + ",".join(
        [
            "func.func(buffer-deallocation)",
            "async-to-async-runtime",
            "async-runtime-ref-counting",
            "async-runtime-ref-counting-opt",
            "convert-async-to-llvm",
            "canonicalize",
            "cse",
        ]
    )
    + ")"
)

# copied from torch-mlir
REF_BACKEND_LOWERING_PIPELINE = (
    "builtin.module("
    + ",".join(
        [
            "func.func(refback-generalize-tensor-pad)",
            # Apply some optimizations. It would be great if MLIR had more useful
            # optimizations that worked out of the box here.
            # Note: When measured, this doesn't seem to actually help that much
            # for the linalg-on-tensors backend.
            # This is likely because if things are naturally fusable we usually already
            # emit things in that form from the high level (e.g. single linalg-generic).
            # Other backends are likely to benefit more.
            "func.func(linalg-fuse-elementwise-ops)",
            # Bufferize.
            "func.func(tm-tensor-bufferize)",
            "one-shot-bufferize{copy-before-write bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
            "refback-mlprogram-bufferize",
            "func.func(finalizing-bufferize)",
            # "func.func(buffer-deallocation)",
            # Munge to make it ExecutionEngine compatible.
            # Specifically, we rewrite calling convention boundaries to be in terms
            # of unranked memref, and we rewrite the return to actually be a
            # callback that consumes the return (the final munged function always
            # returns void at the C level -- we get the return value by providing the
            # callback).
            "refback-munge-calling-conventions",
            # Insert global variable and instruction sequence for getting the next
            # global seed used in stateful rng.
            # Lower to LLVM
            "func.func(tm-tensor-to-loops)",
            "func.func(refback-munge-memref-copy)",
            "func.func(convert-linalg-to-loops)",
            "func.func(lower-affine)",
            "convert-scf-to-cf",
            "func.func(refback-expand-ops-for-llvm)",
            "func.func(arith-expand)",
            "func.func(convert-math-to-llvm)",
            # Handle some complex mlir::math ops (e.g. atan2)
            "convert-math-to-libm",
            "expand-strided-metadata",
            "finalize-memref-to-llvm",
            "lower-affine",
            "func.func(convert-arith-to-llvm)",
            "convert-func-to-llvm",
            "convert-cf-to-llvm",
            "reconcile-unrealized-casts",
        ]
    )
    + ")"
)


class AirCpuBackend(AirBackend):
    """Main entry-point for the AIR CPU backend.

    This currently uses the torch-mlir linalg-on-tensors RefBackend
    for JIT execution.

    """

    def __init__(self):
        super().__init__()
        self.handle = None
        self.refbackend = RefBackendLinalgOnTensorsBackend()

    def __del__(self):
        self.unload()

    def compile(
        self,
        air_module: air.ir.Module,
        pipeline=None,
        verbose=False,
        segment_offset=None,
        segment_size=None,
    ):
        """Compiles an imported module, with a flat list of functions.

        The module is expected to be AIR dialect.
        Args:
          air_module: The MLIR module consisting of functions containing
            AIR dialect.
          pipeline: The custom lowering pipeline to use for lowering.
            The default is `air.backend.cpu_backend.DEFAULT_PIPELINE`
          segment_offset: default location for generated segments as [colOffset, rowOffset]
          segment_size: default size for generated segments as [numCols, numRows]
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """

        if pipeline is None:
            pipeline = DEFAULT_PIPELINE

        s = str(air_module)
        with air.ir.Context() as ctx:
            ctx.allow_unregistered_dialects = True
            # make a copy of the input MLIR
            air_module = air.ir.Module.parse(s)

            if verbose:
                print("Running MLIR pass pipeline: ", pipeline)

            if callable(pipeline):
                air_module = pipeline(air_module)
            else:
                pm = air.passmanager.PassManager.parse(pipeline)
                pm.run(air_module.operation)

            if verbose:
                print("Async Module:")
                print(air_module)

            pm = air.passmanager.PassManager.parse(ASYNC_TO_LLVM_PIPELINE)
            pm.run(air_module.operation)

            if verbose:
                print("LLVM Module:")
                print(air_module)

        with torch_mlir.ir.Context():
            torch_mlir_module = torch_mlir.ir.Module.parse(str(air_module))
            pm = torch_mlir.passmanager.PassManager.parse(REF_BACKEND_LOWERING_PIPELINE)
            pm.run(torch_mlir_module.operation)
        return torch_mlir_module

    def compile_from_torch_mlir(
        self,
        imported_module: torch_mlir.ir.Module,
        pipeline=None,
        verbose=False,
        segment_offset=None,
        segment_size=None,
    ):
        if type(imported_module) is torch_mlir.ir.Module:
            with imported_module.operation.context:
                imported_module = torchscript.lower_mlir_module(
                    False, torchscript.OutputType.LINALG_ON_TENSORS, imported_module
                )

                pm = torch_mlir.passmanager.PassManager.parse(
                    "builtin.module(refback-mlprogram-bufferize)"
                )
                pm.run(imported_module.operation)

        if verbose:
            print("Torch Module:")
            print(imported_module)

        with air.ir.Context():
            air_module = air.ir.Module.parse(str(imported_module))
            pm = air.passmanager.PassManager.parse(
                air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE
            )

            if verbose:
                print(
                    "Running MLIR pass pipeline: ",
                    air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE,
                )

            pm.run(air_module.operation)

        return self.compile(air_module, pipeline, verbose, segment_offset, segment_size)

    def load(self, module):
        """Load a compiled artifact."""
        return self.refbackend.load(module)

    def unload(self):
        """Unload any loaded module and release resources."""
        pass
