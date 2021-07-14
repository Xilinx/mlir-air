###############################################################################
#  Copyright (c) 2021, Xilinx, Inc.
#  All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without 
#  modification, are permitted provided that the following conditions are met:
#
#  1.  Redistributions of source code must retain the above copyright notice, 
#     this list of conditions and the following disclaimer.
#
#  2.  Redistributions in binary form must reproduce the above copyright 
#      notice, this list of conditions and the following disclaimer in the 
#      documentation and/or other materials provided with the distribution.
#
#  3.  Neither the name of the copyright holder nor the names of its 
#      contributors may be used to endorse or promote products derived from 
#      this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, 
#  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR 
#  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR 
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
#  OR BUSINESS INTERRUPTION). HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
#  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
#  OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF 
#  ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

###############################################################################
#
#
#     Author: Kristof Denolf <kristof@xilinx.com>
#     Date:   2021/07/08
#
###############################################################################


# Ramon's conv2d example: https://gitenterprise.xilinx.com/XRLabs/acdc/blob/main/air/herds/conv2d/conv2d.py

# TODO: add code run MLIR code and compare to the pytorch and opencv result

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.functional as F
#import torch.optim as optim
#from torchvision import datasets, transforms
#from torch.optim.lr_scheduler import StepLR
import cv2
import OpenCVUtils as cvUtils

import threading, queue
import numpy as np

class CNV22D(nn.Module):
    def __init__(self):
        super(CNV22D,self).__init__()
        self.filter2D_1 = nn.Conv2d(1,1,kernel_size=(3,3),padding=(1,1))
        self.filter2D_2 = nn.Conv2d(1,1,kernel_size=(3,3),padding=(1,1))
        self.setWeights()
    
    def setWeights(self):
        self.filter2D_1.weight.data = torch.tensor([[[[1.0/9.0,1.0/9.0 ,1.0/9.0],[1.0/9.0,1.0/9.0 ,1.0/9.0],[1.0/9.0,1.0/9.0 ,1.0/9.0]]]])
        self.filter2D_2.weight.data = torch.tensor([[[[1.0/9.0,1.0/9.0 ,1.0/9.0],[1.0/9.0,1.0/9.0 ,1.0/9.0],[1.0/9.0,1.0/9.0 ,1.0/9.0]]]])

    def forward(self,x):
        out = self.filter2D_2(self.filter2D_1(x))
        return out


device = "cpu" #    device = torch.device("cuda" if args.cuda else "cpu")

model = CNV22D().to(device)
img = cv2.imread("/group/xrlabs/imagesAndVideos/images/bigBunny_1080.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#cv2.imshow("input image", gray)
#gray = np.expand_dims(gray, axis=(0,1))
grayTensor = torch.from_numpy(np.expand_dims(gray, axis=(0,1))).type(torch.float)
filtered = model.forward(grayTensor)
filteredPyTorchConvertedToCV = np.array(filtered[0,0,:,:].detach())
#filteredCV.converTo(filteredInt8, cv2.CV_8UC1)
#run similar test in opencv
height, width  = gray.shape
dstTmp = np.ones((height,width),np.float32)
dstCV = np.ones((height,width),np.float32)
kernelCV = np.ones((3,3),np.float32)/9.0
cv2.filter2D(gray,cv2.CV_32F,kernelCV,dstTmp,borderType=cv2.BORDER_CONSTANT)
cv2.filter2D(dstTmp,cv2.CV_32F,kernelCV,dstCV, borderType=cv2.BORDER_CONSTANT)
numberOfDifferences,error = cvUtils.imageCompare(filteredPyTorchConvertedToCV,dstCV)
if numberOfDifferences>0:
    print("Test FAILED: number of differences in image: %d, average error per pixel: %f" %(numberOfDifferences,error))
    #raise Exception('failedTest')
    #difference = cv2.absdiff(test,golden)
else:
    print("Test PASSED")
#cv2.imshow("pytorch image", filteredPyTorchConvertedToCV.astype(np.uint8))
#cv2.imshow("opencv image", dstCV.astype(np.uint8))
#cv2.waitKey()
#cv2.destroyAllWindows()


# capture MLIR
import torch_mlir
builder = torch_mlir.ModuleBuilder()
graph = CNV22D()
with builder.capture_function("graph", [grayTensor]) as MLIRfunction:
    filteredMLIR= graph(grayTensor)
    MLIRfunction.returns([filteredMLIR])

MLIRfunction_mlir = builder.module
print("BEFORE CLEAN UP")
print(MLIRfunction_mlir)

# Convert capture MLIR to ATEN

from mlir.ir import *
from mlir.passmanager import *
import mlir.transforms
from npcomp.compiler.generic.backend import refjit as refjit_backend
pm = PassManager.parse("func(aten-recognize-kernels),numpy-public-functions-to-tensor,canonicalize", context=builder.module.context)
pm.run(MLIRfunction_mlir)

print("AFTER CLEAN UP")
print(MLIRfunction_mlir)

import os
print(os.path.basename(__file__)) 

outputFileName = os.path.splitext(os.path.basename(__file__))[0] + '.mlir'

with open(outputFileName,"w") as f:
    f.write(str(MLIRfunction_mlir))

