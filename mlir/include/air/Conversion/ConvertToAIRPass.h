//===- ConvertToAIRPass.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef CONVERT_TO_AIR_H
#define CONVERT_TO_AIR_H

#include "air/Conversion/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createParallelToHerdPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createParallelToHerdPass(const ParallelToHerdOptions &options);

std::unique_ptr<mlir::Pass> createParallelToLaunchPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createParallelToLaunchPass(const ParallelToLaunchOptions &options);

std::unique_ptr<mlir::Pass> createParallelToSegmentPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createParallelToSegmentPass(const ParallelToSegmentOptions &options);

std::unique_ptr<mlir::Pass> createCopyToDmaPass();
std::unique_ptr<mlir::Pass> createInsertEmptyLaunchOverHerdPass();

std::unique_ptr<Pass> createAIRWrapFuncWithParallelPass();
std::unique_ptr<mlir::Pass>
createAIRWrapFuncWithParallelPass(AIRWrapFuncWithParallelPassOptions options);

} // namespace air
} // namespace xilinx

#endif // CONVERT_TO_AIR_H
