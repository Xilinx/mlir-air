# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
if(NOT AIE_BINARY_DIR)
  find_package(AIE REQUIRED CONFIG)
endif()

if(NOT AIR_BINARY_DIR)
  find_package(AIR REQUIRED CONFIG)
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TEST_DEPENDS
  FileCheck count not
  air-opt
  AirBackendPythonModules
  AirCompilerPythonModules
  )

add_lit_testsuite(check-air-python "Running the air python regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  ARGS "-sv"
  )

set_target_properties(check-air-python PROPERTIES FOLDER "Tests")

add_lit_testsuites(AIRPYTHON ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TEST_DEPENDS})
add_dependencies(check-all check-air-python)

