MACRO(SUBDIRLIST result curdir)
  file(GLOB children RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child} AND EXISTS ${curdir}/${child}/CMakeLists.txt)
      LIST(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
ENDMACRO()

SUBDIRLIST(SUBDIRS ${CMAKE_CURRENT_SOURCE_DIR})

include_directories(
  ${CMAKE_SOURCE_DIR}/runtime_lib/airhost/include
  ${CMAKE_SOURCE_DIR}/../aie/runtime_lib
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TEST_DEPENDS
  FileCheck count not
)

add_lit_testsuite(check-air "Running the air board regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${TEST_DEPENDS}
  )
set_target_properties(check-air PROPERTIES FOLDER "Tests")

add_lit_testsuites(AIR_TEST ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TEST_DEPENDS})

add_custom_target(build-check-air)
add_dependencies(check-air build-check-air)
add_dependencies(check-all check-air)

if(LIBXAIE_FOUND)

  link_libraries(
    ${AIR_LIBXAIE_LIBS}
    airhost
  )

  foreach(subdir ${SUBDIRS})
    add_subdirectory(${subdir})
    message(${subdir})
  endforeach()

endif()
