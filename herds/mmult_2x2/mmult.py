import torch
import torch_mlir

import air
from air.compiler.jit import Compiler

t0 = torch.randn((64,64))
t1 = torch.randn((64,64))

builder = torch_mlir.ModuleBuilder()
with builder.capture_function("task", [t0,t1]) as f:
    t2 = torch.mm(t0, t1)
    f.returns([t2])
t2_mlir = builder.module
#print(builder.module)

c = Compiler()
m = c.torch_to_aten(t2_mlir)
#m = c.aten_to_linalg(m)
print(m)

#c.air_linalg_codegen(m)
#print(m)

#c.affine_to_air(m)
#print(m)

#c.air_to_aie(m, row_offset=2, col_offset=7)
#print(m)
