# (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
import torch
import torch_mlir

#
# Capture some MLIR
#

t0 = torch.zeros(512, dtype=torch.float32)
t1 = torch.zeros(512, dtype=torch.float32)

builder = torch_mlir.ModuleBuilder()
with builder.capture_function("task", [t0,t1]) as f:
    t2 = t0 * t1
    f.returns([t2])

t2_mlir = builder.module

#
# transform captured MLIR to ATen
#

from mlir.ir import *
from mlir.passmanager import *
import mlir.transforms
from npcomp.compiler.generic.backend import refjit as refjit_backend

pm = PassManager.parse("func(aten-recognize-kernels),numpy-public-functions-to-tensor,canonicalize",
                        context=builder.module.context)
pm.run(t2_mlir)
print(t2_mlir)
