if(LIBXAIE_FOUND)
add_library(airhost STATIC
    memory.cpp
    queue.cpp
    host.cpp
)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(airhost
  ${AIR_LIBXAIE_LIBS}
  dl
)

set_target_properties(airhost PROPERTIES
         LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/runtime_lib)
install(TARGETS airhost DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib)

add_subdirectory(include)

# Install files
set(INSTALLS memory.cpp queue.cpp host.cpp)
install(FILES ${INSTALLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/airhost)

else()
message("LIBXAIE_FOUND not set.  Skipping building airhost library")
endif()
