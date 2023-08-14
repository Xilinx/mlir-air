//===- air-opt.cpp ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/InitAll.h"

#ifdef BUILD_WITH_AIE
#include <aie/Dialect/AIE/IR/AIEDialect.h>
#endif

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

  DialectRegistry registry;
  registerAllDialects(registry);
  xilinx::air::registerAllDialects(registry);
#ifdef BUILD_WITH_AIE
  registry.insert<xilinx::AIE::AIEDialect>();
#endif

  registerAllPasses();
  xilinx::air::registerAllPasses();

  mlir::test::registerTestTransformDialectInterpreterPass();
  ::test::registerTestDialect(registry);
  ::test::registerTestTransformDialectExtension(registry);

  return failed(MlirOptMain(argc, argv, "MLIR-AIR modular optimizer driver\n",
                            registry,
                            /*preloadDialectsInContext=*/true));
}
