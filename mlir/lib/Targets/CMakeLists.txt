# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

if(BUILD_WITH_AIE)
  add_definitions(-DBUILD_WITH_AIE)
endif()

add_air_library(AIRTargets
  AIRTargets.cpp
  AIRHerdToJSON.cpp

  LINK_LIBS

  PUBLIC
  AIRRtDialect
  AIRDialect
  MLIRIR
  MLIRSupport
  MLIRTransforms
)
