//===- AIRDmaToChannel.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DMA_TO_CHANNEL_H
#define AIR_DMA_TO_CHANNEL_H

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/PassDetail.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createDmaToChannelPass();

SmallVector<Operation *> cloneOpsInBlock(Block *blk, OpBuilder &builder,
                                         IRMapping &remap);
SmallVector<Operation *> cloneAffineIfUsingRemap(OpBuilder builder,
                                                 IRMapping &remap,
                                                 affine::AffineIfOp aif_op);

template <typename T>
SmallVector<Operation *>
cloneScfLoopUsingRemap(OpBuilder builder, IRMapping &remap, T loop_op,
                       air::ChannelInterface externalGetPut = nullptr);

template <>
SmallVector<Operation *> cloneScfLoopUsingRemap<LoopLikeOpInterface>(
    OpBuilder builder, IRMapping &remap, LoopLikeOpInterface loop_op,
    air::ChannelInterface externalGetPut);

} // namespace air
} // namespace xilinx

#endif // AIR_DMA_TO_CHANNEL_H