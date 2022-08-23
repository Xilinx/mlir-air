# (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

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

