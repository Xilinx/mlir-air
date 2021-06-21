import torch
import torch_mlir

import air
from air.compiler.jit import Compiler

t0 = torch.randint(high=10, size=(128,128), dtype=torch.int32)
t1 = torch.randint(high=10, size=(128,128), dtype=torch.int32)
t3 = torch.randint(high=10, size=(128,128), dtype=torch.int32)

builder = torch_mlir.ModuleBuilder()
with builder.capture_function("task", [t0,t1,t3]) as f:
    t2 = torch.mm(t0, t1)
    t4 = torch.mm(t2, t3)
    f.returns([t4])
t2_mlir = builder.module
#print(builder.module)

c = Compiler()
m = c.torch_to_aten(t2_mlir)

#m = c.aten_to_airrt(m)
#m = c.airrt_to_llvm(m)
print(m)

#c.air_linalg_codegen(m)
#print(m)

#c.affine_to_air(m)
#print(m)

#c.air_to_aie(m, row_offset=2, col_offset=7)
#print(m)
