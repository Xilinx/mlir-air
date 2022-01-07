# (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

import torch
from torch import nn

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import extract_annotations
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

class model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @export
    @annotate_args([
        None,
        ([16*1024], torch.int32, True),
        ([16*1024], torch.int32, True)
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

pm = PassManager.parse('torchscript-module-to-torch-backend-pipeline,torch-backend-to-linalg-on-tensors-backend-pipeline', mb.module.context)
pm.run(mb.module)
print(mb.module)

