# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_mlir_public_c_api_library(
  AIRCAPI
  Dialects.cpp
  Registration.cpp
  Runner.cpp
  Transform.cpp
  LINK_LIBS
  PUBLIC
  AIRDialect
  AIRInitAll)
