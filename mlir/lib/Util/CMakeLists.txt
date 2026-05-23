# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set(AIRUTIL_LINK_LIBS
  MLIRIR
  MLIRTransforms
)

# AIE target model is queried for the shim address-gen granularity when
# computing DMA inner-element alignment. Conditional on AIR_ENABLE_AIE; the
# fallback path in Util.cpp is used when AIE is disabled or no DeviceOp is
# reachable from the op.
if(AIR_ENABLE_AIE)
  list(APPEND AIRUTIL_LINK_LIBS AIE)
endif()

add_mlir_library(AIRUtil
  Util.cpp
  Outliner.cpp
  CostModel.cpp
  Runner.cpp
  Dependency.cpp
  DependencyDot.cpp
  DirectedAdjacencyMap.cpp
  MatmulCodegenConfig.cpp

  DEPENDS
  AIRDialect

  LINK_LIBS PUBLIC
  ${AIRUTIL_LINK_LIBS}
)
