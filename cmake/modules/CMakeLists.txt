# Copyright (C) 2022, Xilinx Inc. All rights reserved.
# Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Generate a list of CMake library targets so that other CMake projects can link
# against them. LLVM calls its version of this file LLVMExports.cmake, but the
# usual CMake convention seems to be ${Project}Targets.cmake.
set(AIR_INSTALL_PACKAGE_DIR lib${LLVM_LIBDIR_SUFFIX}/cmake/air)
set(air_cmake_builddir "${CMAKE_BINARY_DIR}/${AIR_INSTALL_PACKAGE_DIR}")

# Export targets for the build directory
export(EXPORT AIRTargets FILE ${air_cmake_builddir}/AIRTargets.cmake)

# Generate AIRConfig.cmake for the build tree.
set(AIR_CONFIG_CMAKE_DIR "${air_cmake_builddir}")
set(AIR_CONFIG_BINARY_DIR "${PROJECT_BINARY_DIR}")
set(AIR_CONFIG_TOOLS_BINARY_DIR "${PROJECT_BINARY_DIR}/bin")
set(AIR_CONFIG_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include"
                            "${PROJECT_BINARY_DIR}/include")
set(AIR_CONFIG_EXPORTS_FILE "\${AIR_CMAKE_DIR}/AIRTargets.cmake")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AIRConfig.cmake.in
               ${air_cmake_builddir}/AIRConfig.cmake @ONLY)
set(AIR_CONFIG_CMAKE_DIR)
set(AIR_CONFIG_BINARY_DIR)
set(AIR_CONFIG_TOOLS_BINARY_DIR)
set(AIR_CONFIG_INCLUDE_DIRS)
set(AIR_CONFIG_EXPORTS_FILE)

# Generate AIRConfig.cmake for the install tree.
set(AIR_CONFIG_CODE
    "
# Compute the installation prefix from this file location.
get_filename_component(AIR_INSTALL_PREFIX \"\${CMAKE_CURRENT_LIST_FILE}\" PATH)"
)
# Construct the proper number of get_filename_component(... PATH) calls to
# compute the installation prefix.
string(REGEX REPLACE "/" ";" _count "${AIR_INSTALL_PACKAGE_DIR}")
foreach(p ${_count})
  set(AIR_CONFIG_CODE "${AIR_CONFIG_CODE}
get_filename_component(AIR_INSTALL_PREFIX \"\${AIR_INSTALL_PREFIX}\" PATH)")
endforeach(p)

set(AIR_CONFIG_CMAKE_DIR "\${AIR_INSTALL_PREFIX}/${AIR_INSTALL_PACKAGE_DIR}")
set(AIR_CONFIG_BINARY_DIR "\${AIR_INSTALL_PREFIX}")
set(AIR_CONFIG_TOOLS_BINARY_DIR "\${AIR_INSTALL_PREFIX}/bin")
set(AIR_CONFIG_INCLUDE_DIRS "\${AIR_INSTALL_PREFIX}/include")
set(AIR_CONFIG_EXPORTS_FILE "\${AIR_CMAKE_DIR}/AIRTargets.cmake")
set(MLIR_CONFIG_INCLUDE_EXPORTS_FILE "\${AIR_CMAKE_DIR}/MLIRTargets.cmake")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/AIRConfig.cmake.in
               ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/AIRConfig.cmake @ONLY)
set(AIR_CONFIG_CODE)
set(AIR_CONFIG_CMAKE_DIR)
set(AIR_CONFIG_BINARY_DIR)
set(AIR_CONFIG_TOOLS_BINARY_DIR)
set(AIR_CONFIG_INCLUDE_DIRS)
set(AIR_CONFIG_EXPORTS_FILE)

# export targets for the install directory
install(
  EXPORT AIRTargets
  DESTINATION ${AIR_INSTALL_PACKAGE_DIR}
  COMPONENT air-cmake-exports)
install(
  EXPORT MLIRTargets
  DESTINATION ${AIR_INSTALL_PACKAGE_DIR}
  COMPONENT air-cmake-exports)
install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/AIRConfig.cmake
  DESTINATION ${AIR_INSTALL_PACKAGE_DIR}
  COMPONENT air-cmake-exports)
