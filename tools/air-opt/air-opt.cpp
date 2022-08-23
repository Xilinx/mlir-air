// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/InitAll.h"

#include "aie/AIEDialect.h"

using namespace llvm;
using namespace mlir;

namespace test {
void registerTestDialect(DialectRegistry &);
void registerTestTransformDialectExtension(DialectRegistry &);
} // namespace test

namespace mlir {
namespace test {
void registerTestTransformDialectInterpreterPass();
}
} // namespace mlir

int main(int argc, char **argv) {
  registerAllPasses();
  xilinx::air::registerAllPasses();
  mlir::test::registerTestTransformDialectInterpreterPass();
  DialectRegistry registry;
  registerAllDialects(registry);
  ::test::registerTestDialect(registry);
  ::test::registerTestTransformDialectExtension(registry);

  xilinx::air::registerAllDialects(registry);
  registry.insert<xilinx::AIE::AIEDialect>();

  return failed(MlirOptMain(argc, argv, "MLIR-AIR modular optimizer driver\n",
                            registry,
                            /*preloadDialectsInContext=*/false));
}
