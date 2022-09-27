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

# Declare an air library which can be compiled in libAIR.so.
# This is adapted from npcomp and add_mlir_library.
function(add_air_library name)
  cmake_parse_arguments(ARG
    "SHARED;EXCLUDE_FROM_LIBAIR"
    ""
    "ADDITIONAL_HEADERS;DEPENDS;LINK_COMPONENTS;LINK_LIBS"
  ${ARGN})
  set(srcs)
  # TODO: Port the source description logic for IDEs from add_mlir_library.


  if(ARG_SHARED)
    # Rule explicitly requested a shared library.
    set(LIBTYPE SHARED)
  else()
    if(NOT ARG_EXCLUDE_FROM_LIBAIR)
      set_property(GLOBAL APPEND PROPERTY AIR_STATIC_LIBS ${name})
    endif()
  endif()

  # TODO: Enable air header export.
  # list(APPEND ARG_DEPENDS air-generic-headers)
  llvm_add_library(
    ${name} ${LIBTYPE} ${ARG_UNPARSED_ARGUMENTS} ${srcs}
    OBJECT
    DEPENDS ${ARG_DEPENDS}
    LINK_COMPONENTS ${ARG_LINK_COMPONENTS}
    LINK_LIBS ${ARG_LINK_LIBS})

  set_target_properties(${name} PROPERTIES FOLDER "AIR libraries")

  install(TARGETS ${name}
    LIBRARY DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT ${name}
    ARCHIVE DESTINATION lib${LLVM_LIBDIR_SUFFIX} COMPONENT ${name}
    RUNTIME DESTINATION bin COMPONENT ${name})
    
endfunction()

# Declare the library associated with a dialect.
function(add_air_dialect_library name)
  set_property(GLOBAL APPEND PROPERTY AIR_DIALECT_LIBS ${name})
  # TODO: Add DEPENDS air-headers
  add_air_library(${ARGV})
endfunction()

# Declare the library associated with a conversion.
function(add_air_conversion_library name)
  set_property(GLOBAL APPEND PROPERTY AIR_CONVERSION_LIBS ${name})
  # TODO: Add DEPENDS air-headers
  add_air_library(${ARGV})
endfunction()

function(add_air_executable name)
  add_executable(${ARGV})
  llvm_update_compile_flags(${name})
  add_link_opts( ${name} )
  set_output_directory(${name}
    BINARY_DIR ${PROJECT_BINARY_DIR}/bin
    LIBRARY_DIR ${PROJECT_BINARY_DIR}/lib)
  if (LLVM_PTHREAD_LIB)
    # libpthreads overrides some standard library symbols, so main
    # executable must be linked with it in order to provide consistent
    # API for all shared libaries loaded by this executable.
    target_link_libraries(${name} PRIVATE ${LLVM_PTHREAD_LIB})
  endif()

  install(TARGETS ${name}
    RUNTIME DESTINATION ${LLVM_UTILS_INSTALL_DIR}
    COMPONENT ${name})

endfunction()

function(air_enable_exceptions name)
  target_compile_options(${name} PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -fexceptions
  >
  $<$<CXX_COMPILER_ID:MSVC>:
  /EHsc>
  )
endfunction()
