# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

set(INSTALLS
    air.hpp
    air_channel.h
    air_host.h
    air_host_impl.h
    air_network.h
    air_queue.h
    air_tensor.h
    hsa_defs.h
    pcie-ernic.h
    pcie-ernic-defines.h
    pcie-ernic-dev-mem-allocator.h
    utility.hpp)

# Stuff into the build area:
add_custom_target(copy-runtime-includes ALL)
foreach(file ${INSTALLS})
  add_custom_target(copy-runtime-includes-${file} ALL
                    DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${file})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
            ${CMAKE_CURRENT_BINARY_DIR}/${file}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
  add_dependencies(copy-runtime-includes copy-runtime-includes-${file})
endforeach()

# Install too
install(FILES ${INSTALLS}
        DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/airhost/include)
