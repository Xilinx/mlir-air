# (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

add_mlir_library(AIRCAPI
Dialects.cpp
Registration.cpp

DEPENDS

ENABLE_AGGREGATION
LINK_COMPONENTS
Core

LINK_LIBS PUBLIC
AIRInitAll
)