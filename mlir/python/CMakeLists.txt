# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

include_directories(
  ${Python3_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR}
)

add_library(AIRMLIRPyExt SHARED
  AIRMLIRModule.cpp
  AIRRunnerModule.cpp
)

set_target_properties(AIRMLIRPyExt PROPERTIES OUTPUT_NAME _airmlir)
set_target_properties(AIRMLIRPyExt PROPERTIES PREFIX
                      "${PYTHON_MODULE_PREFIX}")
set_target_properties(AIRMLIRPyExt PROPERTIES SUFFIX
                      "${PYTHON_MODULE_EXTENSION}")

set(_libs
  AIRCAPI
  AIRDialect
  AIRRtDialect
  AIRUtil
  AIRConversionPasses
  AIRTransformPasses
)

if(BUILD_WITH_AIE)
  list(APPEND _libs AIE)
endif()

target_link_libraries(AIRMLIRPyExt
  PRIVATE
  ${_libs}
  ${Python3_LIBRARIES}
)

set_target_properties(AIRMLIRPyExt PROPERTIES CXX_VISIBILITY_PRESET "hidden")

install(TARGETS AIRMLIRPyExt
LIBRARY DESTINATION python COMPONENT python
ARCHIVE DESTINATION python COMPONENT python
RUNTIME DESTINATION bin COMPONENT python)
