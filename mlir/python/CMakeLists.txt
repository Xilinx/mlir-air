
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

include_directories(
  # ${CMAKE_CURRENT_SOURCE_DIR}
  # ${CMAKE_CURRENT_BINARY_DIR}
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

target_link_libraries(AIRMLIRPyExt
  PRIVATE
  AIRCAPI
  AIRDialect
  AIRRtDialect
  AIRUtil
  AIRConversionPasses
  AIRTransformPasses
  ${Python3_LIBRARIES}
  AIE
)

set_target_properties(AIRMLIRPyExt PROPERTIES CXX_VISIBILITY_PRESET "hidden")

# target_compile_options(AIRMLIRPyExt PRIVATE
#   # Enable RTTI and exceptions.
#   -frtti -fexceptions
#   # Noisy pybind warnings
#   -Wno-unused-value
#   -Wno-covered-switch-default
# )
# mlir_check_all_link_libraries(AIRMLIRPyExt)

install(TARGETS AIRMLIRPyExt
LIBRARY DESTINATION python COMPONENT python
ARCHIVE DESTINATION python COMPONENT python
RUNTIME DESTINATION bin COMPONENT python)
