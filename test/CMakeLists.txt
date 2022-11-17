# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0068)
  cmake_policy(SET CMP0068 NEW)
  set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
endif()

if(POLICY CMP0075)
  cmake_policy(SET CMP0075 NEW)
endif()

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

project(air LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)

find_package(LibXAIE)
get_filename_component(XILINX_XAIE_LIB_DIR ${XILINX_XAIE_LIBS} DIRECTORY)
get_filename_component(XILINX_XAIE_DIR ${XILINX_XAIE_LIB_DIR} DIRECTORY)

if(NOT AIE_BINARY_DIR)
find_package(AIE REQUIRED CONFIG)
endif()

if(NOT AIR_BINARY_DIR)
find_package(AIR REQUIRED CONFIG)
endif()

find_package(Vitis)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using AIEConfig.cmake in: ${AIE_DIR}")
message(STATUS "Using AIRConfig.cmake in: ${AIR_DIR}")

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL aarch64)
  set(DEFAULT_ENABLE_BOARD_TESTS ON)
else()
  set(DEFAULT_ENABLE_BOARD_TESTS OFF)
endif()
option(ENABLE_BOARD_TESTS "Enable board tests" ${DEFAULT_ENABLE_BOARD_TESTS})

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
add_definitions(${LLVM_DEFINITIONS})

set(ENABLE_CHESS_TESTS ON CACHE BOOL "Enable backend tests using xchesscc")
if(ENABLE_CHESS_TESTS)
set(CONFIG_ENABLE_CHESS_TESTS 1)
else()
set(CONFIG_ENABLE_CHESS_TESTS 0)
endif()

if(ENABLE_BOARD_TESTS)
set(CONFIG_ENABLE_BOARD_TESTS 1)
set(AIR_TEST_LIT_ARGS "-sv -j 1 --timeout 300 --time-tests --order=random")
else()
set(CONFIG_ENABLE_BOARD_TESTS 0)
set(AIR_TEST_LIT_ARGS "-sv -j 24 --timeout 300 --time-tests --order=random")
endif()
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(TEST_DEPENDS
  FileCheck count not
  )

add_lit_testsuite(check-air-board "Running the air board regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS}
  )
set_target_properties(check-air-board PROPERTIES FOLDER "Tests")

add_custom_target(build-check-air)
add_dependencies(check-air-board build-check-air)

if(NOT TARGET check-all)
  add_custom_target(check-all)
endif()
add_dependencies(check-all check-air-board)
