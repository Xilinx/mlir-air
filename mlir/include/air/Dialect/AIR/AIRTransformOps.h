//===- AIRTransformOps.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_AIR_TRANSFORM_OPS_H
#define MLIR_AIR_TRANSFORM_OPS_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/PDL/IR/PDLTypes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/SmallSet.h"

#include "air/Dialect/AIR/AIRDialect.h"

using namespace mlir;

namespace mlir {
class DialectRegistry;
namespace func {
class FuncOp;
} // namespace func
} // namespace mlir

#define GET_OP_CLASSES
#include "air/Dialect/AIR/AIRTransformOps.h.inc"

namespace xilinx {
namespace air {
void registerTransformDialectExtension(mlir::DialectRegistry &registry);

void replaceAllUsesOfConstsInRegionWithNew(SmallVector<Value, 4> constants,
                                           OpBuilder builder, Region &region);

LogicalResult normalizeScfParallel(scf::ParallelOp parOp,
                                   PatternRewriter &rewriter);

void getHerdNames(ModuleOp module);

std::optional<Value> allocBufferCallBack(OpBuilder &b,
                                         memref::SubViewOp subView,
                                         ArrayRef<Value> boundingSubViewSize,
                                         DataLayout &layout);

LogicalResult deallocBufferCallBack(OpBuilder &b, Value buffer);

FailureOr<linalg::TiledLinalgOp> pipelineReduceLinalgOp(
    RewriterBase &b, linalg::LinalgOp op, ArrayRef<int64_t> static_tile_sizes,
    unsigned int pipeline_depth, std::string pipeline_direction, bool promote);

void populateScfParToHerdConversionPattern(
    RewritePatternSet &patterns, SmallPtrSet<Operation *, 8> &filteredOps,
    llvm::SmallSet<HerdOp, 2> &replacementOps);

void populateScfParToLaunchConversionPattern(
    RewritePatternSet &patterns, llvm::SmallSet<Operation *, 8> &filteredOps,
    llvm::SmallSet<LaunchOp, 2> &replacementOps);

void populateRemoveSubViewOpsPattern(RewritePatternSet &patterns,
                                     unsigned int fast_memory_space = 1);

void populateRemoveViewOpsPattern(RewritePatternSet &patterns,
                                  unsigned int fast_memory_space = 1);

void populateFoldSubViewOpsPattern(RewritePatternSet &patterns);

void populateRemoveExtraAllocPattern(RewritePatternSet &patterns);

void populateRemoveDeadCopyPattern(RewritePatternSet &patterns);

void populateRemoveDeadCopyPattern(RewritePatternSet &patterns);

void populateRemoveAllocCopyLinalgOpCopyPattern(RewritePatternSet &patterns);

} // namespace air
} // namespace xilinx

#endif // MLIR_AIR_TRANSFORM_OPS_H
