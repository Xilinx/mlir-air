set(PYTHON_INSTALL_PATH ${CMAKE_INSTALL_PREFIX}/bin)

set(AIRCC_SUBFILES
  cl_arguments.py
  __init__.py
  main.py)

set(AIRCC_FILES
  aircc.py
  aircc/cl_arguments.py
  aircc/__init__.py
  aircc/main.py)

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
foreach(file ${AIRCC_SUBFILES})
  install(PROGRAMS aircc/${file} DESTINATION ${PYTHON_INSTALL_PATH}/aircc)
endforeach()
