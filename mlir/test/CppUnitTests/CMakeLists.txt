# Copyright (C) 2023, Xilinx Inc. All rights reserved.
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

include(CTest)

add_executable(directed_adjacency_map  directed_adjacency_map.cpp)
add_test(NAME DirectedAdjacencyMap COMMAND directed_adjacency_map)
target_link_libraries(directed_adjacency_map PRIVATE AIRUtil)
set(AIR_CPP_UNIT_TESTS directed_adjacency_map)

if(AIR_ENABLE_AIE)
  add_executable(dma_bd_limits dma_bd_limits.cpp)
  add_test(NAME DmaBdLimits COMMAND dma_bd_limits)
  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  target_link_libraries(dma_bd_limits PRIVATE AIRUtil AIRDialect AIE
    MLIRParser ${dialect_libs})
  list(APPEND AIR_CPP_UNIT_TESTS dma_bd_limits)

  add_executable(circular_chain_bds circular_chain_bds.cpp)
  add_test(NAME CircularChainBDs COMMAND circular_chain_bds)
  target_link_libraries(circular_chain_bds PRIVATE AIRConversionPasses
    AIRUtil AIRDialect AIE MLIRParser ${dialect_libs})
  list(APPEND AIR_CPP_UNIT_TESTS circular_chain_bds)
endif()

add_custom_target(check-air-cpp COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS ${AIR_CPP_UNIT_TESTS})

add_dependencies(check-all check-air-cpp)
