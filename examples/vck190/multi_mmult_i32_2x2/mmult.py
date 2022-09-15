
import torch
import torch_mlir

shape = [128,128]
dtype = torch.int32
class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, t0, t1, t2):
        t3 = torch.mm(t0, t1)
        t4 = torch.mm(t3, t2)
        return t4

program = model()

module = torch_mlir.compile(
    program,
    (torch.ones(shape, dtype=dtype), torch.ones(shape, dtype=dtype), torch.ones(shape, dtype=dtype)),
    output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
)

print(module)
