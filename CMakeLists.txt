# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

cmake_minimum_required(VERSION 3.12)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

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

if(POLICY CMP0091)
  cmake_policy(SET CMP0091 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(AIR LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

set(PEANO_INSTALL_DIR
    "<unset>"
    CACHE STRING "Location of Peano compiler")

include(ExternalProject)

find_package(MLIR REQUIRED CONFIG)
find_package(Boost REQUIRED)
find_package(AIE REQUIRED)
find_package(LibXAIE)
find_package(Boost REQUIRED)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Using AIEConfig.cmake in: ${AIE_DIR}")

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS
       "Build the LLVM tools. If OFF, just generate build targets." ON)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv --timeout=30")
if(MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS
    "${LIT_ARGS_DEFAULT}"
    CACHE STRING "Default options for lit")
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" FORCE)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# setup python
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

# python build directory
if(NOT AIR_PYTHON_PACKAGES_DIR)
  set(AIR_PYTHON_PACKAGES_DIR "${CMAKE_CURRENT_BINARY_DIR}/python")
endif()

# python install directory
if(NOT AIR_PYTHON_INSTALL_DIR)
  set(AIR_PYTHON_INSTALL_DIR "python")
endif()

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

link_directories(${AIE_BINARY_DIR}/lib)
include_directories(${AIE_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/mlir/include)
include_directories(${PROJECT_BINARY_DIR}/mlir/include)
include_directories(${Boost_INCLUDE_DIRS})

add_definitions(${LLVM_DEFINITIONS})

add_custom_target(check-all)

# Make sure we build the docs
add_custom_target(docs ALL)
add_dependencies(docs mlir-doc)

set(AIR_RUNTIME_TARGETS
    ""
    CACHE STRING "Architectures to compile the runtime libraries for.")
set(AIR_RUNTIME_TEST_TARGET
    ""
    CACHE STRING "Runtime architecture to test with.")
set(AIR_RUNTIME_TEST_TARGET_VAL ${AIR_RUNTIME_TEST_TARGET})

foreach(target ${AIR_RUNTIME_TARGETS})
  # By default, we test the first architecture in AIR_RUNTIME_TARGETS.
  # Alternatively, this can be defined to force testing with a particular
  # architecture.
  if(NOT AIR_RUNTIME_TEST_TARGET_VAL)
    set(AIR_RUNTIME_TEST_TARGET_VAL ${target})
    message("Testing with AIR runtime target: ${AIR_RUNTIME_TEST_TARGET_VAL}")
  endif()

  if(NOT EXISTS ${${target}_TOOLCHAIN_FILE})
    message(
      FATAL_ERROR
        "Toolchain file ${${target}_TOOLCHAIN_FILE} not found! Cannot build target ${target}."
    )
  endif()
  message(
    "Building AIR runtime for ${target} with TOOLCHAIN=${${target}_TOOLCHAIN_FILE}"
  )
  ExternalProject_Add(
    air_runtime_lib_${target}
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}/runtime_libTmp/${target}
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/runtime_lib
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/runtime_lib/${target}
    CMAKE_CACHE_ARGS -DCMAKE_MODULE_PATH:STRING=${CMAKE_MODULE_PATH}
    CMAKE_ARGS -DCMAKE_TOOLCHAIN_FILE=${${target}_TOOLCHAIN_FILE}
               -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
               -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
               -DCMAKE_ASM_COMPILER=${CMAKE_ASM_COMPILER}
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
               -DLibXAIE_ROOT=${LibXAIE_ROOT}
               -DAIE_DIR=${AIE_DIR}
    BUILD_ALWAYS true
    STEP_TARGETS clean build install test
    USES_TERMINAL_CONFIGURE true
    USES_TERMINAL_BUILD true
    USES_TERMINAL_TEST true
    USES_TERMINAL_INSTALL true
    TEST_BEFORE_INSTALL true
    TEST_EXCLUDE_FROM_MAIN true)
endforeach()

if(NOT LLVM_INSTALL_TOOLCHAIN_ONLY)
  install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/runtime_lib/airhost/include
    DESTINATION include/..
    COMPONENT runtime_lib-headers
    FILES_MATCHING
    PATTERN "*.h"
    PATTERN "*.hpp")

  if(NOT LLVM_ENABLE_IDE)
    add_llvm_install_targets(install-runtime_lib-headers DEPENDS
                             runtime_lib-headers COMPONENT runtime_lib-headers)
  endif()
endif()

add_subdirectory(python)
if(NOT AIR_RUNTIME_TEST_TARGET_VAL)
  message(
    "Skipping tests: No Runtime architecture found. Please configure AIR_RUNTIME_TARGETS."
  )
elseif(NOT LibXAIE_FOUND)
  message("Skipping tests: LibXAIE not found.")
else()
  add_subdirectory(test)
endif()
add_subdirectory(tools)
add_subdirectory(mlir)
add_subdirectory(cmake/modules)
