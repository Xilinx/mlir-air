//===- AIRDependencyScheduleOpt.h -------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DEPENDENCY_SCHEDULE_OPT_H
#define AIR_DEPENDENCY_SCHEDULE_OPT_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRHoistDmaInAccumPattern();

std::unique_ptr<mlir::Pass> createAIRAnnotateFrontAndBackOpsInForPattern();

std::unique_ptr<mlir::Pass> createAIRHoistMemallocInForPattern();

std::unique_ptr<mlir::Pass> createAIRUnrollLoopForPipeliningPattern();

std::unique_ptr<mlir::Pass> createAIRConstructPingPongDependencyPattern();

std::unique_ptr<mlir::Pass> createAIRHoistOpsNotUsingPingPongPattern();

std::unique_ptr<mlir::Pass> createAIRBroadcastDetection();

std::unique_ptr<mlir::Pass> createAIRPruneLinalgGenericInputDma();

std::unique_ptr<mlir::Pass> createAIRPingPongTransformationPattern();
std::unique_ptr<OperationPass<ModuleOp>> createAIRPingPongTransformationPattern(
    const AIRPingPongTransformationPatternOptions &);

std::unique_ptr<mlir::Pass> createAIRLabelScfForLoopForPingPongPattern();

std::unique_ptr<mlir::Pass> createAIRLabelScfForLoopInAIRSegmentPattern();

std::unique_ptr<mlir::Pass> createAIRSpecializeChannelWrapAndStridePattern();

std::unique_ptr<mlir::Pass> createAIRDependencyScheduleOptPass();

std::unique_ptr<mlir::Pass> createAIRUnrollChannelByFactorPattern();

std::unique_ptr<mlir::Pass> createAIREnforceLoopCarriedMemrefDeallocPattern();

std::unique_ptr<mlir::Pass> createAIRDeAliasMemref();

std::unique_ptr<mlir::Pass> createAIRFuseChannels();

std::unique_ptr<mlir::Pass> createAIRIsolateAsyncDmaLoopNests();

std::unique_ptr<mlir::Pass> createAIRSegmentLoopFusion();

// Populate patterns for canonicalizing index operations on loop index
// variables. At the moment, only affine.apply computations on induction
// variables are canonicalized
void populateAIRLoopIndexCanonicalizationPatterns(RewritePatternSet &patterns);

} // namespace air
} // namespace xilinx

#endif // AIR_DEPENDENCY_SCHEDULE_OPT_H
