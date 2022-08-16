# (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASS

import torch
import numpy


import torch
import torch_mlir
import numpy

from air.backend import linalg_on_tensors as backend

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b):
        return torch.mm(a,b)

def run_test(dtype, shape):
    program = model()
    module = torch_mlir.compile(
        program,
        (torch.ones([shape[0],shape[1]], dtype=dtype), torch.ones([shape[1],shape[2]], dtype=dtype)),
        output_type=torch_mlir.OutputType.LINALG_ON_TENSORS
    )

    print(module)

    airbackend = backend.LinalgOnTensorsAirBackend()
    compiled = airbackend.compile(module)
    jit_module = airbackend.load(compiled)

    a = torch.randint(100, [shape[0],shape[1]], dtype=dtype)
    b = torch.randint(100, [shape[1],shape[2]], dtype=dtype)
    c = torch.tensor(
        jit_module.forward(a.numpy(),b.numpy()))

    print(f"input:\n{a}\n{b}\noutput:\n{c}")

    errs = (torch.mm(a,b) == c)
    unique, counts = numpy.unique(errs, return_counts=True)
    d = dict(zip(unique, counts))
    errs = d.get(False,0)
    count = d.get(True,0)
    if errs>0:
        print(f"{count}/{errs+count} Correct\n")
    if torch.equal(torch.mm(a,b),c):
        print("PASS!")
        return 1
    else:
        print("failed.")
        return 0

sizes = [
    [32,128,64],
    [64,64,64],
    [128,32,128],
]
dtypes = [
    torch.float,
    torch.int32
]

passed = 0
for t in dtypes:
    for s in sizes:
        passed = passed + run_test(t,s)

num_tests = len(sizes)*len(dtypes)
if passed != num_tests:
    print (f"failed. {passed}/{num_tests}")
else:
    print (f"PASSED! {passed}/{num_tests}")
