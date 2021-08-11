set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AIRConversion)
add_public_tablegen_target(AIRConversionPassIncGen)

set(LLVM_TARGET_DEFINITIONS ATenToXTen.td)
mlir_tablegen(ATenToXTen.cpp.inc -gen-rewriters)
add_public_tablegen_target(AIRATenToXTenIncGen)

#add_mlir_doc(Passes AIRConversionPasses ./ -gen-pass-doc)