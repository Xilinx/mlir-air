import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_mlir

i0 = torch.zeros((1, 8, 32, 32), dtype=torch.float32)

builder = torch_mlir.ModuleBuilder()

class CNV2(nn.Module):
    def __init__(self):
        super(CNV2, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.conv6 = nn.Conv2d(64, 64, kernel_size=3)
        #self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        #x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        #x = F.relu(self.conv6(F.relu(self.conv5(x))))
        #x = x.view(-1, 64)
        #print(x.shape)
        #x = F.softmax(self.fc1(x))
        #x = torch.cat((x, x))
        return x


graph = CNV2()
with builder.capture_function("graph", [i0]) as f:
    t2 = graph(i0)
    f.returns([t2])

t2_mlir = builder.module
#print(t2_mlir)

from mlir.ir import *
from mlir.passmanager import *
import mlir.transforms
from npcomp.compiler.generic.backend import refjit as refjit_backend

pm = PassManager.parse("func(aten-recognize-kernels),numpy-public-functions-to-tensor,canonicalize",
                       context=builder.module.context)
pm.run(t2_mlir)
print(t2_mlir)


