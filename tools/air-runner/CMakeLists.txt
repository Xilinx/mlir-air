# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs support core irreader)

# Now build our tools
add_llvm_tool(air-runner air-runner.cpp)
llvm_update_compile_flags(air-runner)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
AIRDialect
AIRRtDialect
AIRUtil
AIRConversionPasses
AIRTransformPasses
AIRInitAll
)

target_link_libraries(air-runner PRIVATE ${LIBS})
