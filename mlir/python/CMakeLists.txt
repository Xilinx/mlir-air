

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
