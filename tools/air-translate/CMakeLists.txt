# Find the libraries that correspond to the LLVM components
# that we wish to use
set(LLVM_LINK_COMPONENTS
  Support
  )


get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

add_llvm_tool(air-translate
  air-translate.cpp
  )
install(TARGETS air-translate
  EXPORT AIRTargets
  RUNTIME DESTINATION ${LLVM_TOOLS_INSTALL_DIR}
  COMPONENT air-translate)

add_dependencies(air-translate AIRTargets)

set(LIBS
  AIRDialect
  AIRRtDialect
  AIRTargets
  NPCOMPATenDialect
  NPCOMPBasicpyDialect
  NPCOMPTypingCPA
  AIE
  MLIR
)

llvm_update_compile_flags(air-translate)
target_link_libraries(air-translate PRIVATE ${LIBS})

 