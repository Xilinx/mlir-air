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
  PYTHON_BINDINGS_LIBRARY
    nanobind
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
    PYTHON_BINDINGS_LIBRARY
      nanobind
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

################################################################################
# MLIR has switched to using nanobind 2.4 which enforces a check on the domain, 
# but MLIR's add_mlir_python_extension and add_mlir_python_modules methods are
# fixing the NB_DOMAIN as mlir, leading to domain conflict between air and aie.
# Below is a workaround which re-implements the above methods by exposing the
# NB_DOMAIN for customization. 
# TODO: Remove these when upstream LLVM fixes this issue.
function(add_mlir_python_extension_domain libname extname)
cmake_parse_arguments(ARG
""
"INSTALL_COMPONENT;INSTALL_DIR;OUTPUT_DIRECTORY;PYTHON_BINDINGS_LIBRARY"
"SOURCES;LINK_LIBS;NB_DOMAIN"
${ARGN})
set(eh_rtti_enable)
if (MSVC)
  set(eh_rtti_enable /EHsc /GR)
elseif(LLVM_COMPILER_IS_GCC_COMPATIBLE OR CLANG_CL)
  set(eh_rtti_enable -frtti -fexceptions)
endif ()
nanobind_add_module(${libname}
  NB_DOMAIN ${ARG_NB_DOMAIN}
  ${ARG_SOURCES}
)
if (LLVM_COMPILER_IS_GCC_COMPATIBLE OR CLANG_CL)
  target_compile_options(nanobind-static
    PRIVATE
      -Wno-cast-qual
      -Wno-zero-length-array
      -Wno-nested-anon-types
      -Wno-c++98-compat-extra-semi
      -Wno-covered-switch-default
      ${eh_rtti_enable}
  )
endif()
target_compile_options(${libname} PRIVATE ${eh_rtti_enable})
set_target_properties(
  ${libname} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${ARG_OUTPUT_DIRECTORY}
  OUTPUT_NAME "${extname}"
  NO_SONAME ON
)
target_link_libraries(${libname}
  PRIVATE
  ${ARG_LINK_LIBS}
)
target_link_options(${libname}
  PRIVATE
    $<$<PLATFORM_ID:Linux>:LINKER:--exclude-libs,ALL>
)
if(ARG_INSTALL_DIR)
  install(TARGETS ${libname}
    COMPONENT ${ARG_INSTALL_COMPONENT}
    LIBRARY DESTINATION ${ARG_INSTALL_DIR}
    ARCHIVE DESTINATION ${ARG_INSTALL_DIR}
    RUNTIME DESTINATION ${ARG_INSTALL_DIR}
  )
endif()
endfunction()

function(add_mlir_python_modules_domain name)
cmake_parse_arguments(ARG
  ""
  "ROOT_PREFIX;INSTALL_PREFIX"
  "COMMON_CAPI_LINK_LIBS;DECLARED_SOURCES;NB_DOMAIN"
  ${ARGN})
function(_process_target modules_target sources_target)
  get_target_property(_source_type ${sources_target} mlir_python_SOURCES_TYPE)
  if(_source_type STREQUAL "pure")
    # Pure python sources to link into the tree.
    set(_pure_sources_target "${modules_target}.sources.${sources_target}")
    add_mlir_python_sources_target(${_pure_sources_target}
      INSTALL_COMPONENT ${modules_target}
      INSTALL_DIR ${ARG_INSTALL_PREFIX}
      OUTPUT_DIRECTORY ${ARG_ROOT_PREFIX}
      SOURCES_TARGETS ${sources_target}
    )
    add_dependencies(${modules_target} ${_pure_sources_target})
  elseif(_source_type STREQUAL "extension")
    get_target_property(_module_name ${sources_target} mlir_python_EXTENSION_MODULE_NAME)
    get_target_property(_bindings_library ${sources_target} mlir_python_BINDINGS_LIBRARY)
    set(_extension_target "${modules_target}.extension.${_module_name}.dso")
    add_mlir_python_extension_domain(${_extension_target} "${_module_name}"
      INSTALL_COMPONENT ${modules_target}
      INSTALL_DIR "${ARG_INSTALL_PREFIX}/_mlir_libs"
      OUTPUT_DIRECTORY "${ARG_ROOT_PREFIX}/_mlir_libs"
      PYTHON_BINDINGS_LIBRARY ${_bindings_library}
      NB_DOMAIN ${ARG_NB_DOMAIN}
      LINK_LIBS PRIVATE
        ${sources_target}
        ${ARG_COMMON_CAPI_LINK_LIBS}
    )
    add_dependencies(${modules_target} ${_extension_target})
    mlir_python_setup_extension_rpath(${_extension_target})
  endif()
endfunction()

add_custom_target(${name} ALL)
_flatten_mlir_python_targets(_flat_targets ${ARG_DECLARED_SOURCES})
foreach(sources_target ${_flat_targets})
  _process_target(${name} ${sources_target})
endforeach()

if(NOT LLVM_ENABLE_IDE)
  add_llvm_install_targets(
    install-${name}
    DEPENDS ${name}
    COMPONENT ${name})
endif()
endfunction()
################################################################################

add_mlir_python_modules_domain(AirMLIRPythonModules
  ROOT_PREFIX "${AIR_PYTHON_PACKAGES_DIR}/air"
  INSTALL_PREFIX "python/air"
  NB_DOMAIN "_air"
  DECLARED_SOURCES
    MLIRPythonSources
    MLIRPythonExtension.Core
    MLIRPythonExtension.RegisterEverything
    MLIRPythonExtension.ExecutionEngine
    AirPythonExtensions
  COMMON_CAPI_LINK_LIBS
    AirAggregateCAPI
  )

add_mlir_python_modules_domain(AirPythonModules
  ROOT_PREFIX "${AIR_PYTHON_PACKAGES_DIR}/air"
  INSTALL_PREFIX "python/air"
  NB_DOMAIN "_air"
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
