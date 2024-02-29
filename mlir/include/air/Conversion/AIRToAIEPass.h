//===- AIRToAIEPass.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_TO_AIE_PASS_H
#define AIR_TO_AIE_PASS_H

#include "air/Conversion/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace mlir {
class ModuleOp;
class RewriterBase;
} // namespace mlir

namespace xilinx {
namespace air {
class SegmentOp;

mlir::FailureOr<mlir::ModuleOp> convertAIRToAIE(mlir::RewriterBase &rewriter,
                                                air::SegmentOp segment);
std::unique_ptr<mlir::Pass> createAIRToAIEPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIRToAIEPass(const AIRToAIEOptions &options);

std::unique_ptr<mlir::Pass> createAIRSplitDevicesPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAIRLinalgToFuncPass();

} // namespace air
} // namespace xilinx

#endif // AIR_TO_AIE_PASS_H
