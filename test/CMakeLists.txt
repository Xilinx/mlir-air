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
find_package(Python3 COMPONENTS Interpreter)

find_package(LibXAIE)

if(NOT AIE_BINARY_DIR)
find_package(AIE REQUIRED CONFIG)
endif()

if(NOT AIR_BINARY_DIR)
find_package(AIR REQUIRED CONFIG)
endif()

find_package(Vitis 2023.2 COMPONENTS AIE AIE2)
find_package(XRT)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using AIEConfig.cmake in: ${AIE_DIR}")
message(STATUS "Using AIRConfig.cmake in: ${AIR_DIR}")

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

# ON: build and run end-to-end tests
# OFF: build only
option(ENABLE_RUN_AIRHOST_TESTS "Run end-to-end tests on vck5000 with libairhost" OFF)
option(ENABLE_RUN_XRT_TESTS "Run end-to-end tests on Ryzen AI with XRT" OFF)

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

set(AIR_TEST_LIT_ARGS "-sv --timeout 300 --time-tests --order=random")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(TEST_DEPENDS
  FileCheck count not
  )

# Peano-only test suite
add_lit_testsuite(check-air-e2e-peano "Running AIR E2E tests (Peano backend only)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} --filter "peano"
)
set_target_properties(check-air-e2e-peano PROPERTIES FOLDER "Tests")

# Chess-only test suite
add_lit_testsuite(check-air-e2e-chess "Running AIR E2E tests (Chess backend only)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} --filter "chess"
)
set_target_properties(check-air-e2e-chess PROPERTIES FOLDER "Tests")

# AIR-Runner-only test suite
add_lit_testsuite(check-air-runner "Running AIR-Runner tests (simulation only)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} --filter "airrunner"
)
set_target_properties(check-air-e2e-peano PROPERTIES FOLDER "Tests")

add_dependencies(check-all check-air-runner)
# Unfiltered full test suite
add_lit_testsuite(check-air-e2e "Running AIR E2E tests (all backends)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS}
)
set_target_properties(check-air-e2e PROPERTIES FOLDER "Tests")

add_custom_target(build-check-air)
add_dependencies(check-air-e2e build-check-air)
add_dependencies(check-air-e2e-peano build-check-air)
add_dependencies(check-air-e2e-chess build-check-air)

if(NOT TARGET check-all)
  add_custom_target(check-all)
endif()
add_dependencies(check-all check-air-e2e)
