# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set(PYTHON_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin)

set(AIRCC_FILES
  aircc.py)

set(AIRCC_TARGETS ${AIRCC_FILES})
list(TRANSFORM AIRCC_TARGETS PREPEND ${PROJECT_BINARY_DIR}/bin/)

# Stuff our python into the build area:
add_custom_target(aircc.py ALL DEPENDS ${AIRCC_TARGETS})
foreach(file ${AIRCC_FILES})
add_custom_command(OUTPUT ${PROJECT_BINARY_DIR}/bin/${file}
                   COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
                   ${PROJECT_BINARY_DIR}/bin/${file}
                   DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

# Install our python stuff too.
install(PROGRAMS aircc.py DESTINATION ${PYTHON_INSTALL_PATH})
