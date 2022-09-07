# (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${AIE_INCLUDE_DIRS}/../runtime_lib
)

add_definitions(-DLIBXAIENGINEV2)

if(LIBXAIE_FOUND)
add_library(airhost STATIC
    memory.cpp
    queue.cpp
    host.cpp
)
set_property(TARGET airhost PROPERTY POSITION_INDEPENDENT_CODE ON)

add_library(airhost_shared SHARED
    memory.cpp
    queue.cpp
    host.cpp
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
set(INSTALLS memory.cpp queue.cpp host.cpp)
install(FILES ${INSTALLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/airhost)

else()
message("LIBXAIE_FOUND not set.  Skipping building airhost library")
endif()
