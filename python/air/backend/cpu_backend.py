# ./python/air/backend/cpu_backend.py -*- Python -*-
#
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import torch
import torch_mlir.ir
from torch_mlir.dynamo import make_simple_dynamo_backend

import air.mlir.ir
import air.mlir.passmanager

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

from .abc import AirBackend

import air.compiler.util
from air.backend import linalg_on_tensors

import ctypes
from pathlib import Path

from typing import List

path = Path(air.backend.__file__).resolve().parent
ctypes.CDLL(f"{path}/../../../runtime_lib/aircpu/libaircpu.so", mode=ctypes.RTLD_GLOBAL)
ctypes.CDLL(f"/work/acdc/build/llvm/lib/libmlir_async_runtime.so.16git", mode=ctypes.RTLD_GLOBAL)

__all__ = [
    "AirCpuBackend",
    "make_dynamo_backend",
    "DEFAULT_PIPELINE"
]

DEFAULT_PIPELINE = "builtin.module("+",".join([
    "air-to-async",
    "canonicalize",
    "cse"
])+")"

ASYNC_TO_LLVM_PIPELINE = "builtin.module("+",".join([
    "async-to-async-runtime",
    "async-runtime-ref-counting",
    "async-runtime-ref-counting-opt",
    "func.func(convert-linalg-to-affine-loops)",
    "lower-affine",
    "convert-async-to-llvm",
    "canonicalize",
    "cse"
])+")"

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

    def compile(self, air_module: air.mlir.ir.Module, pipeline=None,
                verbose=False, segment_offset=None, segment_size=None):
        """Compiles an imported module, with a flat list of functions.

        The module is expected to be AIR dialect.
        Args:
          imported_module: The MLIR module consisting of functions containing
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
        with air_module.context:
            # make a copy of the input MLIR
            air_module = air.mlir.ir.Module.parse(s)

            if verbose:
                print("Running MLIR pass pipeline: ", pipeline)

            pm = air.mlir.passmanager.PassManager.parse(pipeline)
            pm.run(air_module)

            if verbose:
                print("Async Module:")
                print(air_module)

            pm = air.mlir.passmanager.PassManager.parse(ASYNC_TO_LLVM_PIPELINE)
            pm.run(air_module)

            # if verbose:
            #     print("LLVM Module:")
            #     print(air_module)

        with torch_mlir.ir.Context():
            torch_mlir_module = torch_mlir.ir.Module.parse(str(air_module))
        return self.refbackend.compile(torch_mlir_module)

    def load(self, module):
        """Load a compiled artifact."""
        return self.refbackend.load(module)

    def unload(self):
        """Unload any loaded module and release resources."""
        pass

def make_dynamo_backend(pipeline=None, verbose=False):
    """Make a PyTorch dynamo backend using AirCpuBackend.

    Args:
        pipeline: The custom lowering pipeline to use for lowering. First
            `air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE` is applied,
            then `pipeline`.
            The default is `air.backend.linalg_on_tensors.LINALG_MEMREF_TO_AIR_PIPELINE`
        verbose: enable verbose output
        segment_offset: default location for generated segments as [colOffset, rowOffset]
        segment_size: default size for generated segments as [numCols, numRows]
    Returns:
        A PyTorch dynamo backend
    """
    backend = AirCpuBackend()
    @make_simple_dynamo_backend
    def air_backend(fx_graph: torch.fx.GraphModule,
                    example_inputs: List[torch.Tensor]):
        
        # get the linalg mlir of the model from torch_mlir
        mlir_module = torch_mlir.compile(
            fx_graph, example_inputs,
            output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)

        with air.mlir.ir.Context():
            air_module = air.mlir.ir.Module.parse(str(mlir_module))
            pm = air.mlir.passmanager.PassManager.parse(air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE)
            pm.run(air_module)
            if pipeline is None:
                pm = air.mlir.passmanager.PassManager.parse(linalg_on_tensors.LINALG_MEMREF_TO_AIR_PIPELINE)
                pm.run(air_module)
            else:
                pm = air.mlir.passmanager.PassManager.parse(pipeline)
                pm.run(air_module)

            if verbose:
                print("AIR Module:")
                print(air_module)

        compiled = backend.compile(air_module, verbose=verbose)

        # return a function for invoking the compiled model
        def compiled_callable(*inputs):
            inputs = [x.numpy() for x in inputs]
            loaded = backend.load(compiled)
            result = loaded.forward(*inputs)
            return torch.from_numpy(result)
        return compiled_callable
    return air_backend