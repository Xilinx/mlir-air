# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

declare_mlir_python_sources(AirCompilerPythonSources)

declare_mlir_python_sources(AirCompilerPythonSources.Core
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  ADD_TO_PARENT AirCompilerPythonSources
  SOURCES_GLOB
    aircc/*.py
    *.py
)

add_mlir_python_modules(AirCompilerPythonModules
  ROOT_PREFIX "${AIR_PYTHON_PACKAGES_DIR}/air/compiler"
  INSTALL_PREFIX "${AIR_PYTHON_INSTALL_DIR}/air/compiler"
  DECLARED_SOURCES AirCompilerPythonSources
  )

set(AIRCC_BINARY_DIR ${PROJECT_BINARY_DIR}/python/air/compiler/aircc)
configure_file(aircc/configure.py.in ${AIRCC_BINARY_DIR}/configure.py)
install(PROGRAMS ${AIRCC_BINARY_DIR}/configure.py DESTINATION ${AIR_PYTHON_INSTALL_DIR}/air/compiler/aircc)
