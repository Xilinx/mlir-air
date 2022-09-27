
# Copyright (C) 2022, Xilinx Inc.
# Copyright (C) 2022, Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

if(NOT AIE_BINARY_DIR)
  find_package(AIE REQUIRED CONFIG)
endif()

if(NOT AIR_BINARY_DIR)
  find_package(AIR REQUIRED CONFIG)
endif()

find_package(Vitis)
find_package(Torch 1.11)

if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL aarch64)
  set(DEFAULT_ENABLE_BOARD_TESTS ON)
else()
  set(DEFAULT_ENABLE_BOARD_TESTS OFF)
endif()
option(ENABLE_BOARD_TESTS "Enable board tests" ${DEFAULT_ENABLE_BOARD_TESTS})

if(ENABLE_BOARD_TESTS)
  set(CONFIG_ENABLE_BOARD_TESTS 1)
  message("enable board tests")
else()
  set(CONFIG_ENABLE_BOARD_TESTS 0)
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
  ARGS "-sv -j 1"
  )

set_target_properties(check-air-python PROPERTIES FOLDER "Tests")

add_lit_testsuites(AIRPYTHON ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TEST_DEPENDS})
add_dependencies(check-all check-air-python)

