# Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

include(AddMLIRPython)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(AIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/air")

# The AIR copy of the MLIR bindings is in the `air.` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=air.")

add_definitions(-DLIBXAIENGINEV2)

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(AirPythonSources
  ROOT_DIR "${AIR_PYTHON_ROOT_DIR}"
)

declare_mlir_python_sources(AirPythonExtensions)

declare_mlir_python_sources(AirPythonSources.Dialects
  ADD_TO_PARENT AirPythonSources
)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT AirPythonSources.Dialects
  ROOT_DIR "${AIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/AirBinding.td
  SOURCES
    dialects/air.py
    dialects/_air_ops_ext.py
  DIALECT_NAME air
  GEN_ENUM_BINDINGS_TD_FILE "dialects/AirBinding.td"
)

declare_mlir_dialect_extension_python_bindings(
  ADD_TO_PARENT AirPythonSources.Dialects
  ROOT_DIR "${AIR_PYTHON_ROOT_DIR}"
  TD_FILE dialects/AirExtensionBinding.td
  SOURCES
    dialects/_air_transform_ops_ext.py
  DIALECT_NAME transform
  EXTENSION_NAME air_transform)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(AirPythonExtensions.MLIR
  MODULE_NAME _air
  ADD_TO_PARENT AirPythonExtensions
  ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}
  SOURCES
    AIRMLIRModule.cpp
  EMBED_CAPI_LINK_LIBS
    AIRCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

# Only building this if we are building the hsa runtime, as it requires hsa.h
if(hsa-runtime64_FOUND)
  message(STATUS "Building python bindings because we are building the runtime")
  declare_mlir_python_extension(AirPythonExtensions.AIRRt
    MODULE_NAME _airRt
    ADD_TO_PARENT AirPythonExtensions
    ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR}
    SOURCES
      AirHostModule.cpp
    EMBED_CAPI_LINK_LIBS
      AIRCAPI
    PRIVATE_LINK_LIBS
      LLVMSupport
  )
endif()

add_mlir_python_common_capi_library(AirAggregateCAPI
  INSTALL_COMPONENT AirPythonModules
  INSTALL_DESTINATION python/air/_mlir_libs
  OUTPUT_DIRECTORY "${AIR_PYTHON_PACKAGES_DIR}/air/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.Core
    MLIRPythonExtension.RegisterEverything
    MLIRPythonExtension.ExecutionEngine
    AirPythonSources
    AirPythonExtensions
)

add_mlir_python_modules(AirMLIRPythonModules
  ROOT_PREFIX "${AIR_PYTHON_PACKAGES_DIR}/air"
  INSTALL_PREFIX "python/air"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.Core
    MLIRPythonExtension.RegisterEverything
    MLIRPythonExtension.ExecutionEngine
    AirPythonExtensions
  COMMON_CAPI_LINK_LIBS
    AirAggregateCAPI
    AIRCAPI
  )

add_mlir_python_modules(AirPythonModules
  ROOT_PREFIX "${AIR_PYTHON_PACKAGES_DIR}/air"
  INSTALL_PREFIX "python/air"
  DECLARED_SOURCES
    AirPythonSources
  COMMON_CAPI_LINK_LIBS
    AirAggregateCAPI
)

include_directories(
  ${AIE_INCLUDE_DIRS}/../runtime_lib
  ${CMAKE_CURRENT_SOURCE_DIR}/../runtime_lib/airhost/include
)

# Only include this if we are building the hsa runtime
if(hsa-runtime64_FOUND)
  include_directories(
    ${hsa-runtime64_DIR}/../../../include
  )
endif()

add_dependencies(AirPythonModules AirBackendPythonModules)
add_subdirectory(air/backend)

add_dependencies(AirPythonModules AirCompilerPythonModules)
add_subdirectory(air/compiler)

add_subdirectory(test)
