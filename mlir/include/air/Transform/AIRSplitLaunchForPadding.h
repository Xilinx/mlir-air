//===- AIRSplitLaunchForPadding.h --------------------------------*- C++
//-*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_SPLIT_LAUNCH_FOR_PADDING_H
#define AIR_SPLIT_LAUNCH_FOR_PADDING_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRSplitLaunchForPadding();
std::unique_ptr<OperationPass<ModuleOp>>
createAIRSplitLaunchForPadding(const AIRSplitLaunchForPaddingOptions &);

} // namespace air
} // namespace xilinx

#endif // AIR_SPLIT_LAUNCH_FOR_PADDING_H
