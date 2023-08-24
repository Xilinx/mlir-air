# Copyright (C) 2022, Xilinx Inc. All rights reserved. Copyright (C) 2022,
# Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT

declare_mlir_python_sources(AirBackendPythonSources)

declare_mlir_python_sources(
  AirBackendPythonSources.Core
  ROOT_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT
  AirBackendPythonSources
  SOURCES_GLOB
  *.py)

add_mlir_python_modules(
  AirBackendPythonModules
  ROOT_PREFIX
  "${AIR_PYTHON_PACKAGES_DIR}/air/backend"
  INSTALL_PREFIX
  "${AIR_PYTHON_INSTALL_DIR}/air/backend"
  DECLARED_SOURCES
  AirBackendPythonSources)
