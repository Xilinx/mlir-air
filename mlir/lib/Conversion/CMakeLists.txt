# (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

add_air_library(AIRConversionPasses
ConvertToAIRPass.cpp
AIRLoweringPass.cpp
AIRRtToLLVMPass.cpp
AIRToAIEPass.cpp
AIRToCpuPass.cpp
AIRPipeline.cpp
Passes.cpp

DEPENDS
AIRConversionPassIncGen
AIRDialect
AIRRtDialect

LINK_COMPONENTS
Core

LINK_LIBS
AIRDialect
AIRRtDialect
AIRUtil
AIE
MLIRIR
MLIRLinalgUtils
MLIRLinalgTransforms
MLIRSupport
MLIRTransforms
)