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

if(AIR_ENABLE_AIE)
  find_package(LibXAIE)
  if(NOT AIE_BINARY_DIR)
    find_package(AIE REQUIRED CONFIG)
  endif()
  find_package(Vitis 2023.2 COMPONENTS AIE AIE2)
endif()

if(NOT AIR_BINARY_DIR)
  find_package(AIR REQUIRED CONFIG)
endif()

find_package(XRT)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
if(AIR_ENABLE_AIE)
  message(STATUS "Using AIEConfig.cmake in: ${AIE_DIR}")
endif()
message(STATUS "Using AIRConfig.cmake in: ${AIR_DIR}")

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

# ON: build and run end-to-end tests
# OFF: build only
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

set(AIR_TEST_LIT_ARGS "-sv --timeout 1000 --time-tests --order=random")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(TEST_DEPENDS
  FileCheck count not
  )

# Peano-only test suite (non-LLM peano tests; filtered by filename suffix).
# LLM tests are handled by the dedicated targets below so they can use -j1
# (sequential) to avoid concurrent NPU inference OOM crashes.
add_lit_testsuite(check-programming-examples-peano "Running AIR programming examples (Peano backend only)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} --filter "peano" --filter-out "llms/"
)
set_target_properties(check-programming-examples-peano PROPERTIES FOLDER "Tests")

# Chess-only test suite
add_lit_testsuite(check-programming-examples-chess "Running AIR programming examples (Chess backend only)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} --filter "chess"
)
set_target_properties(check-programming-examples-chess PROPERTIES FOLDER "Tests")

# Unfiltered full test suite
add_lit_testsuite(check-programming-examples "Running AIR programming examples (all backends)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS}
)
set_target_properties(check-programming-examples PROPERTIES FOLDER "Tests")

# LLM compile-only suite: all models, no HF download, no NPU dispatch.
# Matched by path (llms/.*/run_npu2_compile) so lit filename need NOT carry
# any backend suffix. Sequential (-j1) because concurrent LLM compilations
# can exhaust host memory on CI runners.
add_lit_testsuite(check-programming-examples-llms-compile
  "Running LLM compile-only tests (all models, sequential)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} -j1 --filter "llms/.*/run_npu2_compile"
)
set_target_properties(check-programming-examples-llms-compile PROPERTIES FOLDER "Tests")

# LLM compile-only suite: llama32_1b only (fast PR gate, always runs in CI).
add_lit_testsuite(check-programming-examples-llms-compile-llama32_1b
  "Running LLM compile test: llama32_1b only (PR gate)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} -j1 --filter "llms/llama32_1b/run_npu2_compile"
)
set_target_properties(check-programming-examples-llms-compile-llama32_1b PROPERTIES FOLDER "Tests")

# LLM verify suite: all models, requires hf_token + NPU.
# Sequential (-j1) mandatory — concurrent LLM inference causes NPU OOM.
# Not invoked in PR CI (hf_token unavailable in fork PRs); run by repo
# owners via workflow_dispatch or post-merge CI with secrets available.
add_lit_testsuite(check-programming-examples-llms-verify
  "Running LLM end-to-end verify tests (all models, hf_token required, sequential)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} -j1 --filter "llms/.*/run_npu2_verify"
)
set_target_properties(check-programming-examples-llms-verify PROPERTIES FOLDER "Tests")

# LLM profiling suite: all models, requires hf_token + NPU. Compiles then
# profiles each model, emitting <model>.perf.json (TTFT + decode tok/s) next to
# the run. Sequential (-j1) — concurrent LLM inference causes NPU OOM. Not a PR
# gate (needs secrets + the dedicated perf runner); driven by the nightly perf
# workflow. Correctness is gated separately by the verify suite above.
add_lit_testsuite(check-programming-examples-llms-profile
  "Running LLM profiling (all models, hf_token required, sequential)"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS ${AIR_TEST_LIT_ARGS} -j1 --filter "llms/.*/run_npu2_profile"
)
set_target_properties(check-programming-examples-llms-profile PROPERTIES FOLDER "Tests")

add_dependencies(check-all check-programming-examples)
