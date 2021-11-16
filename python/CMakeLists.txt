include(AddMLIRPython)

# The directory at which the Python import tree begins.
# See documentation for `declare_mlir_python_sources`'s ROOT_DIR
# argument.
set(AIR_PYTHON_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/air")

# We vendor our own MLIR instance in the `air` namespace.
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=air.")

################################################################################
# Sources
################################################################################

declare_mlir_python_sources(AirPythonSources)
declare_mlir_python_sources(AirPythonExtensions)

declare_mlir_python_sources(AirPythonSources.Dialects
  ROOT_DIR "${AIR_PYTHON_ROOT_DIR}"
  ADD_TO_PARENT AirPythonSources
)

# declare_mlir_dialect_python_bindings(
#   ADD_TO_PARENT AirPythonSources.Dialects
#   ROOT_DIR "${AIR_PYTHON_ROOT_DIR}"
#   TD_FILE dialects/TorchBinding.td
#   SOURCES dialects/torch/__init__.py
#   DIALECT_NAME torch
# )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../runtime_lib/airhost/include)
link_directories(${CMAKE_CURRENT_BINARY_DIR}/../runtime_lib/airhost)

################################################################################
# Extensions
################################################################################

declare_mlir_python_extension(AirPythonExtensions.MLIR
  MODULE_NAME _airMlir
  ADD_TO_PARENT AirPythonExtensions
  ROOT_DIR ../mlir/python
  SOURCES
    AIRMLIRModule.cpp
  EMBED_CAPI_LINK_LIBS
    AIRCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

declare_mlir_python_extension(AirPythonExtensions.AIRRt
  MODULE_NAME _airRt
  ADD_TO_PARENT AirPythonExtensions
  ROOT_DIR ../runtime_lib/python
  SOURCES
    Module.cpp
    LibAirHostModule.cpp
  EMBED_CAPI_LINK_LIBS
    AIRCAPI
  PRIVATE_LINK_LIBS
    LLVMSupport
)

################################################################################
# Generate packages and shared library
# Downstreams typically will not use these, but they are useful for local
# testing.
################################################################################

set(_source_components
  MLIRPythonSources
  MLIRPythonExtension.Core
  MLIRPythonExtension.AllPassesRegistration
  MLIRPythonExtension.ExecutionEngine
  AirPythonSources
  AirPythonExtensions
)

add_mlir_python_common_capi_library(AirAggregateCAPI
  INSTALL_COMPONENT AirPythonModules
  INSTALL_DESTINATION ${AIR_PYTHON_INSTALL_DIR}/air/mlir/_mlir_libs
  OUTPUT_DIRECTORY "${AIR_PYTHON_PACKAGES_DIR}/air/mlir/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_SOURCES ${_source_components}
)

add_mlir_python_modules(AirPythonModules
  ROOT_PREFIX "${AIR_PYTHON_PACKAGES_DIR}/air/mlir"
  INSTALL_PREFIX "${AIR_PYTHON_INSTALL_DIR}/air/mlir"
  DECLARED_SOURCES ${_source_components}
  COMMON_CAPI_LINK_LIBS
    AirAggregateCAPI
    AIRCAPI
  )

add_dependencies(AirPythonModules AirBackendPythonModules)
add_subdirectory(air/backend)

add_dependencies(AirPythonModules AirCompilerPythonModules)
add_subdirectory(air/compiler)
