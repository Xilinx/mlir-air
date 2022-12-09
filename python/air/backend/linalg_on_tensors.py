# ./python/air/backend/linalg_on_tensors.py -*- Python -*-
#
# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import torch_mlir.ir
import torch_mlir.passmanager

import air.mlir.ir
import air.mlir.passmanager

from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

from .abc import AirBackend

import air.compiler.util
import air.compiler.aircc.main as aircc

import ctypes
from pathlib import Path

path = Path(air.backend.__file__).resolve().parent
try:
    ctypes.CDLL(f"{path}/../../../runtime_lib/airhost/libairhost_shared.so", mode=ctypes.RTLD_GLOBAL)
except:
    pass
import air.mlir._mlir_libs._airRt as airrt

__all__ = [
    "LinalgOnTensorsAirBackend",
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

    This currently uses the linalg-on-tensors RefBackend for actual execution.
    """
    def __init__(self):
        super().__init__()
        self.handle = None
        self.refbackend = RefBackendLinalgOnTensorsBackend()

    def __del__(self):
        self.unload()

    def compile(self, imported_module: torch_mlir.ir.Module, pipeline=None,
                verbose=False, partition_offset=None, partition_size=None):
        """Compiles an imported module, with a flat list of functions.
        The module is expected to be in linalg-on-tensors + scalar code form.
        Args:
          imported_module: The MLIR module consisting of funcs in the torch
            dialect.
          pipeline: The custom lowering pipeline to use for lowering. First
            `air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE` is applied,
            then `pipeline`.
            The default is `air.backend.linalg_on_tensors.LINALG_MEMREF_TO_AIR_PIPELINE`
          partition_offset: default location for generated partitions as [colOffset, rowOffset]
          partition_size: default size for generated partitions as [numCols, numRows]
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """

        if partition_offset is None:
            partition_offset = [7, 2]

        if partition_size is None:
            partition_size = [10, 6]

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
                             [f"-row-offset={partition_offset[1]}",
                              f"-col-offset={partition_offset[0]}"]
            aircc_options = aircc_options + \
                             [f"-num-rows={partition_size[1]}",
                              f"-num-cols={partition_size[0]}"]

            if verbose:
                aircc_options = aircc_options + ['-v']

            aircc.run(air_module,aircc_options)

            with open("air_project/refback.torch.mlir") as f:
                imported_module = torch_mlir.ir.Module.parse(f.read(),imported_module.context)

        return self.refbackend.compile(imported_module)

    def load(self, module):
        """Loads a compiled artifact into the runtime."""
        airrt.host.init()
        a = airrt.host.get_agents()
        q = airrt.host.queue_create(a[0])
        self.handle = airrt.host.module_load_from_file("./torch.mlir.so", q)
        return self.refbackend.load(module)

    def unload(self):
        if self.handle:
            airrt.host.module_unload(self.handle)
        self.handle = None
        airrt.host.shut_down()
