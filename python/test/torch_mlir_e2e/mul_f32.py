# (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASS

import torch
from torch import nn
import numpy

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import extract_annotations
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

from torch_mlir.passmanager import PassManager
from air.backend import linalg_on_tensors as backend

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([64,64], torch.float, True),
        ([64,64], torch.float, True),
    ])
    def forward(self, a,b):
        x = a * b
        return x

program = model()
scripted = torch.jit.script(program)

class_annotator = ClassAnnotator()
extract_annotations(program, scripted, class_annotator)

mb = ModuleBuilder()
mb.import_module(scripted._c, class_annotator)

with mb.module.context:
    pm = PassManager.parse('torchscript-module-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline')
    pm.run(mb.module)

airbackend = backend.LinalgOnTensorsAirBackend()
compiled = airbackend.compile(mb.module)
jit_module = airbackend.load(compiled)

a = torch.rand(size=[64,64], dtype=torch.float)
b = torch.rand(size=[64,64], dtype=torch.float)
c = torch.tensor(
    jit_module.forward(a.numpy(),b.numpy()))

print(f"input:\n{a}\n{b}\noutput:\n{c}")

errs = (a*b == c)
print(numpy.unique(errs.numpy(), return_counts=True))
if torch.equal(a*b,c):
    print("PASS!")
else:
    print("failed.")