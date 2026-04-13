# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Find ROCm/HIP
if(DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH $ENV{ROCM_PATH})
else()
  set(ROCM_PATH "/opt/rocm")
endif()

if(EXISTS "${ROCM_PATH}/include/hip/hip_runtime.h")
  message(STATUS "Found ROCm at: ${ROCM_PATH}")

  add_library(airgpu SHARED
    gpu_runtime.cpp
    gpu_test_utils.cpp
    vmem_allocator.cpp
    symmetric_heap.cpp
    fd_passing.cpp
  )

  set_property(TARGET airgpu PROPERTY POSITION_INDEPENDENT_CODE ON)

  target_include_directories(airgpu PRIVATE
    ${ROCM_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

  target_link_libraries(airgpu PRIVATE
    ${ROCM_PATH}/lib/libamdhip64.so
  )

  set_target_properties(airgpu PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${AIR_RUNTIME_TARGET}/airgpu
  )

  install(TARGETS airgpu
    DESTINATION lib
  )
else()
  message(STATUS "ROCm not found at ${ROCM_PATH}, skipping airgpu build")
endif()
