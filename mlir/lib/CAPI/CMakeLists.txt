# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

add_mlir_library(AIRCAPI
Dialects.cpp
Registration.cpp
Runner.cpp

DEPENDS

ENABLE_AGGREGATION
LINK_COMPONENTS
Core

LINK_LIBS PUBLIC
AIRDialect
AIRInitAll
MLIRIR
MLIRSupport
)