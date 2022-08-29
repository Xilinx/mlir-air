# (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

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

LINALG_MEMREF_TO_AIRRT_PIPELINE = ",".join([
    "air-linalg-codegen",
    "canonicalize",
    "cse",
    "air-par-to-herd",
    "air-copy-to-dma",
    "canonicalize",
    "cse"
])

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

    def compile(self, imported_module: torch_mlir.ir.Module, verbose=False):
        """Compiles an imported module, with a flat list of functions.
        The module is expected to be in linalg-on-tensors + scalar code form.
        Args:
          imported_module: The MLIR module consisting of funcs in the torch
            dialect.
        Returns:
          An opaque, backend specific compiled artifact object that can be
          passed to `load`.
        """

        with air.mlir.ir.Context():
            air_module = air.mlir.ir.Module.parse(str(imported_module))
            pm = air.mlir.passmanager.PassManager.parse(air.compiler.util.LINALG_TENSOR_TO_MEMREF_PIPELINE)
            pm.run(air_module)
            pm = air.mlir.passmanager.PassManager.parse(LINALG_MEMREF_TO_AIRRT_PIPELINE)
            pm.run(air_module)
            aircc.run(air_module,['--shared', '-o', 'torch.mlir.so', '--sysroot=/', '-row-offset=2', '-col-offset=7', 'torch.mlir'] + (['-v'] if verbose else []))
            with open('air_project/refback.torch.mlir') as f:
                imported_module = torch_mlir.ir.Module.parse(f.read(),imported_module.context)

        return self.refbackend.compile(imported_module)

    def load(self, module):
        """Loads a compiled artifact into the runtime."""
        airrt.host.init_libxaie()
        q = airrt.host.queue_create()
        self.handle = airrt.host.module_load_from_file("./torch.mlir.so", q)
        return self.refbackend.load(module)

    def unload(self):
        if self.handle:
            airrt.host.module_unload(self.handle)
        self.handle = None