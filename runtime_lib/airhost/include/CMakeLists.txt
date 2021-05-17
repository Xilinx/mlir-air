set(INSTALLS air_tensor.h air_host.h acdc_queue.h hsa_defs.h)
install(FILES ${INSTALLS} DESTINATION ${CMAKE_INSTALL_PREFIX}/runtime_lib/airhost/include)