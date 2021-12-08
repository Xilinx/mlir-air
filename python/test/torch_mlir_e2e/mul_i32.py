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

SIZE = [128,128]
#SIZE = [1024]

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        (SIZE, torch.int32, True),
        (SIZE, torch.int32, True),
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

a = torch.randint(size = SIZE, low=1, high=100, dtype=torch.int32)
b = torch.randint(size =SIZE, low=1, high=100, dtype=torch.int32)
c = torch.tensor(
    jit_module.forward(a.numpy(),b.numpy()))

print(f"input:\n{a}\n{b}\noutput:\n{c}")

errs = (a*b == c)
unique, counts = numpy.unique(errs, return_counts=True)
d = dict(zip(unique, counts))
errs = d.get(False,0)
count = d.get(True,0)
if errs>0:
    print(f"{count}/{errs+count} Correct\n")

if torch.equal(a*b,c):
    print("PASS!")
else:
    print("failed.")