# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_mlir_library(
  AIRInterface
  AIRInterfaces.cpp
  AIRBufferizationInterfaces.cpp

  DEPENDS
  AIRDialect

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
)

# Make headers visible to dependents
target_include_directories(AIRInterface PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/mlir/include>
  $<INSTALL_INTERFACE:include>
)
