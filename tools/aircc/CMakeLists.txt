# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# --- C++ aircc binary ---

llvm_map_components_to_libnames(llvm_libs support core irreader)

add_llvm_tool(aircc aircc.cpp)
llvm_update_compile_flags(aircc)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

set(LIBS
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  AIRDialect
  AIRRtDialect
  AIRUtil
  AIRConversionPasses
  AIRTransformPasses
  AIRInitAll
  MLIRAffineAnalysis
  MLIRAnalysis
  MLIRDialect
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  ${llvm_libs}
)

if(AIR_ENABLE_AIE)
  list(APPEND LIBS AIE)
endif()

target_link_libraries(aircc PRIVATE ${LIBS})

# --- Python aircc.py (kept for backward compatibility) ---

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
