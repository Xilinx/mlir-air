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

std::unique_ptr<mlir::Pass> createAIRUnrollChannelByFactorPattern();

std::unique_ptr<mlir::Pass> createAIRFuseChannels();
std::unique_ptr<OperationPass<ModuleOp>>
createAIRFuseChannels(const AIRFuseChannelsOptions &);

std::unique_ptr<mlir::Pass> createAIRIsolateAsyncDmaLoopNests();

std::unique_ptr<mlir::Pass> createAIRLoopFusion();

std::unique_ptr<mlir::Pass> createAIROptimizeShimDMABDs();
std::unique_ptr<Pass>
createAIROptimizeShimDMABDs(AIROptimizeShimDMABDsOptions options);

std::unique_ptr<mlir::Pass> createAIROptimizeMemtileDMABDs();
std::unique_ptr<Pass>
createAIROptimizeMemtileDMABDs(AIROptimizeMemtileDMABDsOptions options);

std::unique_ptr<mlir::Pass> createAIRFuseAllocDealloc();

std::unique_ptr<mlir::Pass> createAIRShrinkMemrefSizesByAccess();

// Populate patterns for canonicalizing index operations on loop index
// variables. At the moment, only affine.apply computations on induction
// variables are canonicalized
void populateAIRLoopIndexCanonicalizationPatterns(RewritePatternSet &patterns);

// Populate patterns for canonicalizing offsets, sizes and strides in air
// channel_interface operations.
void populateAIRCanonicalizeChannelWrapAndStridePatterns(
    RewritePatternSet &patterns, int &maxSize, int &maxNumDims,
    bool &enableRepeatAtHighestDim);

// Apply AIRSpecializeChannelWrapAndStridePattern on region.
void applyAIRSpecializeChannelWrapAndStridePattern(
    Region *region, int maxNumDims, int maxSize, bool enableForLoopUnrolling,
    bool enableRepeatAtHighestDim);

// Populate patterns for fusing scf.for loops within air.launch.
void populateAIRLoopFusionPattern(RewritePatternSet &patterns);

// Apply AIRIsolateAsyncDmaLoopNestsPattern on region.
void applyAIRIsolateAsyncDmaLoopNestsPattern(Region *region);

// Populate patterns for fusing memref.alloc and dealloc ops into air.herarchy
// ops.
void populateAIRFuseAllocDeallocToAIRHierPatterns(RewritePatternSet &patterns);

} // namespace air
} // namespace xilinx

#endif // AIR_DEPENDENCY_SCHEDULE_OPT_H
