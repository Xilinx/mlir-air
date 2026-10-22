//===- AIRTargets.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "AIRTargets.h"

#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Tools/mlir-translate/Translation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/IR/Module.h"
#include "llvm/Support/TargetSelect.h"

using namespace mlir;

namespace xilinx {
namespace air {

namespace {

static llvm::cl::opt<int>
    gridNumRows("num-rows",
                llvm::cl::desc("Number of rows of AIEs in the grid"),
                llvm::cl::init(0));
static llvm::cl::opt<int>
    gridNumCols("num-cols",
                llvm::cl::desc("Number of columns of AIEs in the grid"),
                llvm::cl::init(0));

} // namespace

void registerAIRRtTranslations() {

  TranslateFromMLIRRegistration registrationXJSON(
      "air-herds-to-json", "Transform herd information to JSON",
      [](ModuleOp module, raw_ostream &output) {
        // boilerplate to give dimensions to the visualizer
        output << "{\n\t\"switchbox00\": {\n\t\t\"row\": " << gridNumRows - 1
               << ", "
               << "\n\t\t\"col\": " << gridNumCols - 1 << "\n\t}, ";
        output << "\n\t\"segment\": [ ";
        return AIRHerdsToJSON(module, output);
      },
      [](DialectRegistry &registry) {
        registry.insert<air::airDialect, func::FuncDialect, arith::ArithDialect,
                        memref::MemRefDialect, scf::SCFDialect,
                        affine::AffineDialect, linalg::LinalgDialect>();
      });
}

} // namespace air
} // namespace xilinx
