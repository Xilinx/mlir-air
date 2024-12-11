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

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
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

int main(int argc, char **argv) {

  DialectRegistry registry;
  registerAllDialects(registry);
  xilinx::air::registerAllDialects(registry);
  registry.insert<xilinx::AIE::AIEDialect>();
  registry.insert<xilinx::AIEX::AIEXDialect>();

  registerAllExtensions(registry);

  registerAllPasses();
  xilinx::air::registerAllPasses();

  return failed(
      MlirOptMain(argc, argv, "MLIR-AIR modular optimizer driver\n", registry));
}
