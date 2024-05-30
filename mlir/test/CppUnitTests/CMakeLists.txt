# Copyright (C) 2023, Xilinx Inc. All rights reserved.
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

include(CTest)

add_executable(directed_adjacency_map  directed_adjacency_map.cpp)
add_test(NAME DirectedAdjacencyMap COMMAND directed_adjacency_map)
add_custom_target(check-air-cpp COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS directed_adjacency_map)
target_link_libraries(directed_adjacency_map PRIVATE AIRUtil)

add_dependencies(check-all check-air-cpp)
