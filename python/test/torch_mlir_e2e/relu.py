# (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASS

import torch
from torch import nn

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
        ([1024], torch.float, True)
    ])
    def forward(self, a):
        x = torch.relu(a)
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

a = torch.randint(-100, 100, [1024], dtype=torch.float)
b = torch.tensor(
    jit_module.forward(a.numpy()))

print(f"input:\n{a}\noutput:\n{b}")

if torch.equal(torch.relu(a),b):
    print("PASS!")
else:
    print("failed.")