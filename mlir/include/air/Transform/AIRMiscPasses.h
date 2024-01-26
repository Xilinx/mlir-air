//===- AIRMiscPasses.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_MISC_PASSES_H
#define AIR_MISC_PASSES_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRExamplePass();
std::unique_ptr<mlir::Pass> createAIRSpecializeDma();
std::unique_ptr<mlir::Pass> createAIRSpecializeDmaBroadcast();
std::unique_ptr<mlir::Pass> createAIRPromoteUniformL1Dma();
std::unique_ptr<mlir::Pass> createAIRLinalgNamePass();
std::unique_ptr<mlir::Pass> createAIRRemoveLinalgNamePass();
std::unique_ptr<mlir::Pass> createAIRPipelineReducePass();
std::unique_ptr<mlir::Pass> createAIRFuseParallelHerdPass();
std::unique_ptr<mlir::Pass> createAIRRenumberDmaIdPass();
std::unique_ptr<mlir::Pass> createAIRLowerHerdParallelPass();
std::unique_ptr<mlir::Pass> createAIRLabelBroadcastChannelWithTilePass();
std::unique_ptr<mlir::Pass> createAIRCollapseHerdPass();
std::unique_ptr<mlir::Pass> createAIRUnrollOuterPerfectlyNestedLoopsPass();

} // namespace air
} // namespace xilinx

#endif // AIR_MISC_PASSES_H