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

option(AIR_LIBXAIE_ENABLE "Enables libxaie functionality" OFF)


project(air LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(MLIR REQUIRED CONFIG)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

function(npcomp_detect_pybind11_install)
  if(pybind11_DIR)
    message(STATUS "Using explicit pybind11 cmake directory: ${pybind11_DIR} (-Dpybind11_DIR to change)")
  else()
    message(STATUS "Checking for pybind11 in python path...")
    execute_process(
      COMMAND "${Python3_EXECUTABLE}"
      -c "import pybind11;print(pybind11.get_cmake_dir(), end='')"
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      RESULT_VARIABLE STATUS
      OUTPUT_VARIABLE PACKAGE_DIR
      ERROR_QUIET)
    if(NOT STATUS EQUAL "0")
      message(CHECK_FAIL "not found (install via 'pip install pybind11' or set pybind11_DIR)")
      return()
    endif()
    set(pybind11_DIR "${PACKAGE_DIR}" PARENT_SCOPE)
  endif()
endfunction()
npcomp_detect_pybind11_install()
find_package(pybind11 REQUIRED)
set(pybind11_INCLUDE_DIR ${pybind11_INCLUDE_DIR})

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

option(LLVM_INCLUDE_TOOLS "Generate build targets for the LLVM tools." ON)
option(LLVM_BUILD_TOOLS "Build the LLVM tools. If OFF, just generate build targets." ON)

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

# Define the default arguments to use with 'lit', and an option for the user to
# override.
set(LIT_ARGS_DEFAULT "-sv")
if (MSVC_IDE OR XCODE)
  set(LIT_ARGS_DEFAULT "${LIT_ARGS_DEFAULT} --no-progress-bar")
endif()
set(LLVM_LIT_ARGS "${LIT_ARGS_DEFAULT}" CACHE STRING "Default options for lit")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

if (AIR_LIBXAIE_ENABLE)
  add_compile_definitions(AIR_LIBXAIE_ENABLE)
  if (AIR_LIBXAIE_DIR)
    set(AIR_LIBXAIE_INCDIR "${AIR_LIBXAIE_DIR}/include")
    set(AIR_LIBXAIE_LIBDIR "${AIR_LIBXAIE_DIR}/lib")
  else()
    set(AIR_LIBXAIE_DIR "/opt/xaiengine")
    set(AIR_LIBXAIE_INCDIR "${AIR_LIBXAIE_DIR}/include")
    set(AIR_LIBXAIE_LIBDIR "${AIR_LIBXAIE_DIR}/lib")
  endif()
  set(AIR_LIBXAIE_LIBS "${AIR_LIBXAIE_LIBDIR}/libxaiengine.so")
  include_directories(${AIR_LIBXAIE_INCDIR})
endif()

#include_directories(${LLVM_INCLUDE_DIRS})
#include_directories(${MLIR_INCLUDE_DIRS})
#include_directories(${PROJECT_SOURCE_DIR}/dialect/include)
#include_directories(${PROJECT_BINARY_DIR}/dialect/include)
#add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(python)