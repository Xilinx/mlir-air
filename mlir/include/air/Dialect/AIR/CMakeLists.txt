set(dialect AIR)
set(dialect_namespace air)
set(LLVM_TARGET_DEFINITIONS ${dialect}.td)


mlir_tablegen(${dialect}.h.inc -gen-op-decls)
mlir_tablegen(${dialect}.cpp.inc -gen-op-defs)
mlir_tablegen(${dialect}Types.h.inc -gen-typedef-decls)
mlir_tablegen(${dialect}Types.cpp.inc -gen-typedef-defs)
mlir_tablegen(${dialect}Dialect.h.inc -gen-dialect-decls -dialect=${dialect_namespace})
mlir_tablegen(${dialect}Dialect.cpp.inc -gen-dialect-defs -dialect=${dialect_namespace})
mlir_tablegen(${dialect}Enums.h.inc -gen-enum-decls)
mlir_tablegen(${dialect}Enums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRAIRIncGen)

set(LLVM_TARGET_DEFINITIONS AIROpBase.td)
mlir_tablegen(AIROpInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(AIROpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(MLIRAIROpInterfacesIncGen)
add_dependencies(mlir-generic-headers MLIRAIROpInterfacesIncGen)

add_mlir_doc(AIR AIRDialect ./ -gen-dialect-doc)