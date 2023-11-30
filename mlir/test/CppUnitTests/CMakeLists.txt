# Copyright (C) 2023, Xilinx Inc. All rights reserved.
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

include(CTest)

add_executable(dependency_graph_base.cpp  dependency_graph_base.cpp)
add_test(NAME DependencyGraphBase COMMAND dependency_graph_base.cpp)
add_custom_target(check-air-cpp COMMAND ${CMAKE_CTEST_COMMAND} DEPENDS dependency_graph_base.cpp)


add_dependencies(check-all check-air-cpp)

