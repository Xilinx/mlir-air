set(INSTALLS air_tensor.h air_host.h acdc_queue.h hsa_defs.h)
# Stuff into the build area:
add_custom_target(copy-runtime-includes ALL)
foreach(file ${INSTALLS})
    add_custom_target(copy-runtime-includes-${file} ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${file})
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file}
                    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/${file}
                    ${CMAKE_CURRENT_BINARY_DIR}/${file}
                    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    add_dependencies(copy-runtime-includes copy-runtime-includes-${file} )
endforeach()

# Install too
install(FILES ${INSTALLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/airhost/include)