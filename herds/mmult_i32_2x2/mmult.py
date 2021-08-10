import torch
import torch_mlir

import air
from air.compiler.jit import Compiler

t0 = torch.randint(high=10, size=(128,128), dtype=torch.int32)
t1 = torch.randint(high=10, size=(128,128), dtype=torch.int32)

builder = torch_mlir.ModuleBuilder()
with builder.capture_function("task", [t0,t1]) as f:
    t2 = torch.mm(t0, t1)
    f.returns([t2])
t2_mlir = builder.module
#print(builder.module)

c = Compiler()
m = c.torch_to_aten(t2_mlir)
print(m)
