# Find the libraries that correspond to the LLVM components
# that we wish to use
llvm_map_components_to_libnames(llvm_libs support core irreader)

# Now build our tools
add_llvm_tool(air-opt air-opt.cpp)
llvm_update_compile_flags(air-opt)

set(LIBS
  NPCOMPATenDialect
  NPCOMPBasicpyDialect
  NPCOMPTypingCPA
  ATen
  AIE
  MLIR
  )

target_link_libraries(air-opt PRIVATE ${LIBS})
