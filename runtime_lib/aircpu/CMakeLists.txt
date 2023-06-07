# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../airhost/include
)

add_library(aircpu SHARED
    memory.cpp
    channel.cpp
   )
set_property(TARGET aircpu PROPERTY POSITION_INDEPENDENT_CODE ON)

set_target_properties(aircpu PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/runtime_lib)
install(TARGETS aircpu DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/aircpu)
