set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name AIRTransform)
add_public_tablegen_target(AIRTransformPassIncGen)

#add_mlir_doc(Passes AIRTransformPasses ./ -gen-pass-doc)