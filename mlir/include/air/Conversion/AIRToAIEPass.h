//===- AIRToAIEPass.h -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_TO_AIE_PASS_H
#define AIR_TO_AIE_PASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace mlir {
class ModuleOp;
class RewriterBase;
} // namespace mlir

namespace xilinx {
namespace air {
class PartitionOp;

mlir::FailureOr<mlir::ModuleOp> convertAIRToAIE(mlir::RewriterBase &rewriter,
                                                air::PartitionOp partition);
std::unique_ptr<mlir::Pass> createAIRToAIEPass();

} // namespace air
} // namespace xilinx

#endif // AIR_TO_AIE_PASS_H