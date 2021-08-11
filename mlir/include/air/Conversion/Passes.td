#ifndef AIR_CONVERSION_PASSES
#define AIR_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def AffineToAIR : Pass<"affine-to-air", "ModuleOp"> {
  let summary = "Affine to AIR";
  let constructor = "xilinx::air::createAffineToAIRPass()";
}

def AIRLowering : Pass<"air-to-std", "ModuleOp"> {
  let summary = "AIR dialect lowering";
  let constructor = "xilinx::air::createAIRLoweringPass()";
}

def AIRToAIE : Pass<"air-to-aie", "ModuleOp"> {
  let summary = "Lower air.launch_herd to AIE dialect";
  let constructor = "xilinx::air::createAIRToAIEPass()";
}

def ATenToXTen : Pass<"aten-to-xten", "ModuleOp"> {
  let summary = "Convert ATen dialect to XTen dialect";
  let constructor = "xilinx::xten::createATenToXTenPass()";
}

def AIRRtToLLVM : Pass<"airrt-to-llvm", "ModuleOp"> {
  let summary = "Lower AIRRt dialect to LLVM dialect";
  let constructor = "xilinx::airrt::createAIRRtToLLVMPass()";
}

def XTenToAffine : Pass<"xten-to-affine", "ModuleOp"> {
  let summary = "Convert XTen dialect to Affine dialect";
  let constructor = "xilinx::xten::createXTenToAffinePass()";
}

def XTenToLinalg : Pass<"xten-to-linalg", "ModuleOp"> {
  let summary = "Convert XTen dialect to Linalg dialect";
  let constructor = "xilinx::xten::createXTenToLinalgPass()";
}

#endif
