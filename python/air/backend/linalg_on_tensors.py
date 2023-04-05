# ./python/air/backend/linalg_on_tensors.py -*- Python -*-
#
# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import torch
import torch_mlir.ir
import torch_mlir.passmanager
from torch_mlir.dynamo import make_simple_dynamo_backend

import air.mlir.ir
import air.mlir.passmanager

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

from .abc import AirBackend

import air.compiler.util
import air.compiler.aircc.main as aircc

import ctypes
from pathlib import Path
from typing import List

path = Path(air.backend.__file__).resolve().parent
try:
    ctypes.CDLL(f"{path}/../../../runtime_lib/airhost/libairhost_shared.so", mode=ctypes.RTLD_GLOBAL)
except:
    pass
import air.mlir._mlir_libs._airRt as airrt

__all__ = [
    "LinalgOnTensorsAirBackend",
    "make_dynamo_backend",
    "LINALG_MEMREF_TO_AIR_PIPELINE"
]

LINALG_MEMREF_TO_AIR_PIPELINE = "builtin.module("+",".join([
    "air-linalg-codegen",
    "canonicalize",
    "cse",
    "air-par-to-herd",
    "air-copy-to-dma",
    "canonicalize",
    "cse"
])+")"

class LinalgOnTensorsAirBackend(AirBackend):
    """Main entry-point for the linalg-on-tensors based AIR backend.

    This currently uses the torch-mlir linalg-on-tensors RefBackend
    for JIT execution. aircc produces a RefBackend compatible wrapper
    function for AIR generated host code. The wrapper is compiled and
    executed by RefBackend when invoked from python. The load method
    ensures that the AIR runtime is initialized and that the AIR binary
    is loaded into memory before any compiled functions are invoked.
    The unload method should be called to unload the binary and release
    runtime resources.

    """
    def __init__(self):
        super().__init__()
        self.handle = None
        self.refbackend = RefBackendLinalgOnTensorsBackend()

    def __del__(self):
        self.unload()

    def compile(self, imported_module: torch_mlir.ir.Module, pipeline=None,
                verbose=False, segment_offset=None, segment_size=None):
        """Compiles an imported module, with a flat list of functions.

        The module is expected to be in linalg-on-tensors + scalar code form.
        Args:
          imported_module: The MLIR module consisting of funcs in the torch
            dialect.
          pipeline: The custom lowering pipeline to use for lowering. First
            `air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE` is applied,
            then `pipeline`.
            The default is `air.backend.linalg_on_tensors.LINALG_MEMREF_TO_AIR_PIPELINE`
          segment_offset: default location for generated segments as [colOffset, rowOffset]
          segment_size: default size for generated segments as [numCols, numRows]
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """

        if segment_offset is None:
            segment_offset = [7, 2]

        if segment_size is None:
            segment_size = [10, 6]

        if pipeline is None:
            pipeline = LINALG_MEMREF_TO_AIR_PIPELINE

        with imported_module.context:
            pm = torch_mlir.passmanager.PassManager.parse('builtin.module(refback-mlprogram-bufferize)')
            pm.run(imported_module)

        with air.mlir.ir.Context():
            air_module = air.mlir.ir.Module.parse(str(imported_module))
            pm = air.mlir.passmanager.PassManager.parse(
                air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE)

            if verbose:
                print("Running MLIR pass pipeline: ",
                      air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE)

            pm.run(air_module)

            if verbose:
                print("Running MLIR pass pipeline: ", pipeline)

            pm = air.mlir.passmanager.PassManager.parse(pipeline)
            pm.run(air_module)

            if verbose:
                print("AIR Module:")
                print(air_module)

            aircc_options = ['torch.mlir', '--shared', '-o', 'torch.mlir.so']
            aircc_options = aircc_options + \
                             [f"-row-offset={segment_offset[1]}",
                              f"-col-offset={segment_offset[0]}"]
            aircc_options = aircc_options + \
                             [f"-num-rows={segment_size[1]}",
                              f"-num-cols={segment_size[0]}"]

            if verbose:
                aircc_options = aircc_options + ['-v']

            aircc.run(air_module,aircc_options)

            with open("air_project/refback.torch.mlir") as f:
                imported_module = torch_mlir.ir.Module.parse(f.read(),imported_module.context)

        return self.refbackend.compile(imported_module)

    def load(self, module):
        """Load a compiled artifact into the air runtime."""
        airrt.host.init()
        a = airrt.host.get_agents()
        q = airrt.host.queue_create(a[0])
        self.handle = airrt.host.module_load_from_file("./torch.mlir.so", q)
        return self.refbackend.load(module)

    def unload(self):
        """Unload any loaded module and shutdown the air runtime."""
        if self.handle:
            airrt.host.module_unload(self.handle)
        self.handle = None
        airrt.host.shut_down()

def make_dynamo_backend(pipeline=None, verbose=False,
                        segment_offset=None, segment_size=None):
    """Make a PyTorch dynamo backend using LinalgOnTensorsAirBackend.

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
    backend = LinalgOnTensorsAirBackend()
    @make_simple_dynamo_backend
    def air_backend(fx_graph: torch.fx.GraphModule,
                    example_inputs: List[torch.Tensor]):
        
        # get the linalg mlir of the model from torch_mlir
        mlir_module = torch_mlir.compile(
            fx_graph, example_inputs,
            output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)

        # compile the mlir model with aircc
        compiled = backend.compile(mlir_module, pipeline=pipeline,
            verbose=verbose, segment_offset=segment_offset,
            segment_size=segment_size)

        # return a function for invoking the compiled model
        def compiled_callable(*inputs):
            inputs = [x.numpy() for x in inputs]
            loaded = backend.load(compiled)
            result = loaded.forward(*inputs)
            backend.unload()
            return torch.from_numpy(result)
        return compiled_callable
    return air_backend