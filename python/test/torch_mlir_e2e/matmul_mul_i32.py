# (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

# RUN: %PYTHON %s | FileCheck %s
# CHECK: PASS

import torch

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import extract_annotations
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

from torch_mlir.passmanager import PassManager
from air.backend import linalg_on_tensors as backend

class mmult(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([128,128], torch.int32, True),
        ([128,128], torch.int32, True),
        ([128,128], torch.int32, True)
    ])
    def forward(self, a, b, c):
        x = torch.mm(b,c)
        y = a*x
        return y

program = mmult()
scripted = torch.jit.script(program)

class_annotator = ClassAnnotator()
extract_annotations(program, scripted, class_annotator)

mb = ModuleBuilder()
mb.import_module(scripted._c, class_annotator)

pass_pipeline = 'torchscript-module-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline'
pm = PassManager.parse(pass_pipeline, mb.module.context)
pm.run(mb.module)

airbackend = backend.LinalgOnTensorsAirBackend()
compiled = airbackend.compile(mb.module)
jit_module = airbackend.load(compiled)

a = torch.randint(100, [128,128], dtype=torch.int32)
b = torch.randint(100, [128,128], dtype=torch.int32)
c = torch.randint(100, [128,128], dtype=torch.int32)
d = torch.tensor(
    jit_module.forward(a.numpy(),b.numpy(),c.numpy()))

print(f"input:\n{a}\n{b}\n{c}\noutput:\n{d}")

if torch.equal(a*torch.mm(b,c),d):
    print("PASS!")
else:
    print("failed.")