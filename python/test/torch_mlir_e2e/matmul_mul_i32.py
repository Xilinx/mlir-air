# ./python/test/torch_mlir_e2e/matmul_mul_i32.py -*- Python -*-

# Copyright (C) 2021-2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASS

import torch
import torch._dynamo as dynamo

from air.backend import linalg_on_tensors as backend

air_backend = backend.make_dynamo_backend()

shape = [128, 128]
dtype = torch.int32


class mmult(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, a, b, c):
        x = torch.mm(b, c)
        y = a * x
        return y


model = mmult()
dynamo_model = dynamo.optimize(air_backend)(model)

a = torch.randint(100, shape, dtype=dtype)
b = torch.randint(100, shape, dtype=dtype)
c = torch.randint(100, shape, dtype=dtype)
d = dynamo_model(a, b, c)

print(f"input:\n{a}\n{b}\n{c}\noutput:\n{d}")

if torch.equal(a * torch.mm(b, c), d):
    print("PASS!")
else:
    print("failed.")
