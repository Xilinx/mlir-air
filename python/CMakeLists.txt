

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  ${Python3_INCLUDE_DIRS}
  ${pybind11_INCLUDE_DIR}
)
add_library(_air SHARED
  AirModule.cpp
)
set_target_properties(_air PROPERTIES PREFIX "")

target_link_libraries(_air
  ${Python3_LIBRARIES}
  ${AIR_LIBXAIE_LIBS}
)
