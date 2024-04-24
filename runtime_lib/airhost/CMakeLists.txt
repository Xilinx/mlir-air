# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${AIE_INCLUDE_DIRS}/../runtime_lib/x86_64/test_lib/include
    ${AIE_INCLUDE_DIRS}/../runtime_lib
    ${CMAKE_CURRENT_SOURCE_DIR}/../../driver
)

add_definitions(-DLIBXAIENGINEV2)

# Only build the runtime if hsa was found
if (hsa-runtime64_FOUND)
  include_directories(${XILINX_XAIE_INCLUDE_DIR})
  link_libraries(${XILINX_XAIE_LIBS})

  add_library(airhost STATIC
      memory.cpp
      queue.cpp
      runtime.cpp
      host.cpp
      pcie-ernic.cpp
      pcie-ernic-dev-mem-allocator.cpp
      network.cpp
  )
  set_property(TARGET airhost PROPERTY POSITION_INDEPENDENT_CODE ON)

  add_library(airhost_shared SHARED
      memory.cpp
      queue.cpp
      runtime.cpp
      host.cpp
      pcie-ernic.cpp
      pcie-ernic-dev-mem-allocator.cpp
      network.cpp
    )
  set_property(TARGET airhost_shared PROPERTY POSITION_INDEPENDENT_CODE ON)

  add_library(libelf_pic STATIC IMPORTED)
  set_target_properties(libelf_pic PROPERTIES
    IMPORTED_LOCATION "/lib/x86_64-linux-gnu/libelf.so"
  )

  target_link_libraries(airhost
    ${AIR_LIBXAIE_LIBS}
    dl
    hsa-runtime64::hsa-runtime64
  )

  target_link_libraries(airhost_shared
    hsa-runtime64::hsa-runtime64
    libelf_pic
  )

  set_target_properties(airhost PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${AIR_RUNTIME_TARGET}/airhost)
  install(TARGETS airhost DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIR_RUNTIME_TARGET}/airhost)

  set_target_properties(airhost_shared PROPERTIES
          LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${AIR_RUNTIME_TARGET}/airhost)
  install(TARGETS airhost_shared DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/${AIR_RUNTIME_TARGET}/airhost)


  # Stuff into the build area:
  add_custom_target(copy-runtime-libs ALL)
  foreach(file ${INSTALLS})
  add_custom_target(copy-runtime-libs-${file} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${file})
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}
                      COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
                      ${PROJECT_BINARY_DIR}/${AIR_RUNTIME_TARGET}/airhost/${file}
                      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
                      add_dependencies(copy-runtime-libs copy-runtime-libs-${file} )
                      endforeach()

endif()

add_subdirectory(include)
