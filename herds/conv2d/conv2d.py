#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:47:16 2021

@author: juanu
"""
#!/usr/bin/env python3

import torch

from torch_mlir.dialects.torch.importer.jit_ir import ClassAnnotator, ModuleBuilder
from torch_mlir.dialects.torch.importer.jit_ir.torchscript_annotations import extract_annotations
from torch_mlir_e2e_test.torchscript.annotations import annotate_args, export

from torch_mlir.passmanager import PassManager
from torch_mlir_e2e_test.linalg_on_tensors_backends.refbackend import RefBackendLinalgOnTensorsBackend

def nn_conv2d_with_empty_bias_and_weights(*args, **kargs):
    """Returns a Conv2d with bias+weights set to dense tensors for easier reading"""
    conv = torch.nn.Conv2d(*args, **kargs)
    if conv.bias is not None:
        conv.bias = torch.nn.Parameter(torch.ones_like(conv.bias), requires_grad=False)
    conv.weight = torch.nn.Parameter(torch.ones_like(conv.weight), requires_grad=False)
    return conv


class Conv2D(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv4 = nn_conv2d_with_empty_bias_and_weights(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
        )

    @export
    @annotate_args([
        None,
        ([1,64,64,64], torch.float32, True)
    ])
    def forward(self, x):
        return self.conv4(x)

program = Conv2D()
scripted = torch.jit.script(program)

class_annotator = ClassAnnotator()
extract_annotations(program, scripted, class_annotator)

mb = ModuleBuilder()
mb.import_module(scripted._c, class_annotator)

pm = PassManager.parse('torchscript-module-to-torch-backend-pipeline', mb.module.context)
pm.run(mb.module)
print(mb.module)
