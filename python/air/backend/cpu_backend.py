# ./python/air/backend/cpu_backend.py -*- Python -*-
#
# Copyright (C) 2023-2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import torch_mlir.ir
import torch_mlir.passmanager

from air.compiler.aircc.configure import install_path
import air.ir
import air.passmanager

import os

# override the default library paths for the mlir extras refbackend
os.environ["ASYNC_RUNTIME_LIB_PATH"] = (
    f"{install_path()}/python/air/_mlir_libs/libmlir_async_runtime.so"
)
os.environ["C_RUNNER_UTILS_LIB_PATH"] = (
    f"{install_path()}/python/air/_mlir_libs/libmlir_c_runner_utils.so"
)
os.environ["RUNNER_UTILS_LIB_PATH"] = (
    f"{install_path()}/python/air/_mlir_libs/libmlir_runner_utils.so"
)
from aie.extras.runtime.refbackend import LLVMJITBackend
import aie.ir as aieir

from .abc import AirBackend

import air.compiler.util

import ctypes

ctypes.CDLL(
    f"{install_path()}/runtime_lib/x86_64/aircpu/libaircpu.so", mode=ctypes.RTLD_GLOBAL
)
ctypes.CDLL(
    f"{install_path()}/python/air/_mlir_libs/libmlir_async_runtime.so",
    mode=ctypes.RTLD_GLOBAL,
)

__all__ = ["AirCpuBackend", "DEFAULT_PIPELINE"]

DEFAULT_PIPELINE = (
    "builtin.module(" + ",".join(["air-to-async", "canonicalize", "cse"]) + ")"
)

ASYNC_TO_LLVM_PIPELINE = (
    "builtin.module("
    + ",".join(
        [
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
            "one-shot-bufferize{copy-before-write bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map}",
            "func.func(convert-linalg-to-loops)",
            "func.func(lower-affine)",
            "convert-scf-to-cf",
            "func.func(arith-expand)",
            "func.func(convert-math-to-llvm)",
            "convert-math-to-libm",
            "expand-strided-metadata",
            "finalize-memref-to-llvm",
            "lower-affine",
            "func.func(convert-arith-to-llvm)",
            "convert-func-to-llvm",
            "convert-cf-to-llvm",
            "reconcile-unrealized-casts",
            "canonicalize",
            "cse",
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
        self.backend = LLVMJITBackend()

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
        with aieir.Context(), aieir.Location.unknown():
            compiled_module = self.backend.compile(
                aieir.Module.parse(str(air_module)),
                pipeline=REF_BACKEND_LOWERING_PIPELINE,
                kernel_name="forward",
            )
        return compiled_module

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
                pm = torch_mlir.passmanager.PassManager.parse(
                    "builtin.module(refback-mlprogram-bufferize)"
                )
                pm.run(imported_module.operation)

        with air.ir.Context():
            linalg_module = air.ir.Module.parse(str(imported_module))
            pm = air.passmanager.PassManager.parse(
                air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE
            )
            if verbose:
                print(
                    "Running MLIR pass pipeline: ",
                    air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE,
                )
            pm.run(linalg_module.operation)

            if verbose:
                print("Linalg Module:")
                print(linalg_module)

        return self.compile(
            linalg_module, pipeline, verbose, segment_offset, segment_size
        )

    def load(self, module):
        """Load a compiled artifact."""

        def wrapped_function(*args):
            """Wrap the function"""
            try:
                with aieir.Context(), aieir.Location.unknown():
                    loaded = self.backend.load(module)
                    f = getattr(loaded, "forward")
                    return f(*args)
            except Exception as e:
                print(f"Error in wrapped function: {e}")
                pass
            return None

        return wrapped_function

    def unload(self):
        """Unload any loaded module and release resources."""
        # self.backend = None
        pass
