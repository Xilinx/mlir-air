# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set(LLVM_OPTIONAL_SOURCES AIRToAIEPass.cpp)

set(_sources
  AIRLoweringPass.cpp
  AIRPipeline.cpp
  AIRRtToLLVMPass.cpp
  AIRToAsyncPass.cpp
  ConvertToAIRPass.cpp
  Passes.cpp
)

set(_libs
  AIRDialect
  AIRRtDialect
  AIRUtil
  MLIRIR
  MLIRLinalgTransforms
  MLIRLinalgUtils
  MLIRSupport
  MLIRTransforms
)

set(_depends
  AIRConversionPassIncGen
  AIRDialect
  AIRRtDialect
)

if(BUILD_WITH_AIE)
  list(APPEND _sources AIRToAIEPass.cpp)
  list(APPEND _depends AIRToAIEConversionPassIncGen)
  list(APPEND _libs AIE)
  add_definitions(-DBUILD_WITH_AIE)
endif()

add_air_library(AIRConversionPasses
  ${_sources}

  DEPENDS
  ${_depends}

  LINK_COMPONENTS
  Core

  LINK_LIBS
  ${_libs}
)
