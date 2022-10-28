
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
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${AIE_INCLUDE_DIRS}/../runtime_lib
)

add_definitions(-DLIBXAIENGINEV2)

if(BUILD_AIR_PCIE)
  message("BUILD_AIR_PCIE=${BUILD_AIR_PCIE}")
  add_definitions(-DAIR_PCIE)
endif()

if(LIBXAIE_FOUND)
add_library(airhost STATIC
    memory.cpp
    queue.cpp
    host.cpp
    utility.cpp   
)
set_property(TARGET airhost PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(airhost_shared SHARED
    memory.cpp
    queue.cpp
    host.cpp
    utility.cpp   
)
set_property(TARGET airhost_shared PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries(airhost
  ${AIR_LIBXAIE_LIBS}
  dl
)

set_target_properties(airhost PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/runtime_lib)
install(TARGETS airhost DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/airhost)

set_target_properties(airhost_shared PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/runtime_lib)
install(TARGETS airhost_shared DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/airhost)


add_subdirectory(include)

# Stuff into the build area:
add_custom_target(copy-runtime-libs ALL)
foreach(file ${INSTALLS})
    add_custom_target(copy-runtime-libs-${file} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${file})
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
                    ${CMAKE_CURRENT_BINARY_DIR}/${file}
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    add_dependencies(copy-runtime-libs copy-runtime-libs-${file} )
endforeach()

# Install files
set(INSTALLS memory.cpp queue.cpp host.cpp utility.cpp)
install(FILES ${INSTALLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/airhost)

else()
message("LIBXAIE_FOUND not set.  Skipping building airhost library")
endif()
