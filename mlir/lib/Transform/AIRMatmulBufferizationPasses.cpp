//===- AIRMatmulBufferizationPasses.cpp -------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Bufferization phases of the air-matmul-codegen orchestrator: bufferize-
// output-l2, bufferize-l1-inputs, bufferize-l1-output, post-bufferize
// cleanup, ping-pong sibling fusion, and bf16-output truncf fusion.
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRMatmulBufferizationPasses.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/AIRLinalgBufferize.h"
#include "air/Transform/AIRMatmulCodegenHelpers.h"
#include "air/Transform/AIRMatmulTileL3ToL2Copies.h"
#include "air/Util/Util.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"

#define DEBUG_TYPE "air-matmul-bufferization-passes"

using namespace mlir;
using namespace xilinx::air;

namespace xilinx {
namespace air {

namespace {

// `findMarkedOp` / `findMarkedForLoop` live in air/Util/Util.h as
// `xilinx::air::findOpWithAttr` and `findOpOfTypeWithAttr<scf::ForOp>`.

/// Bufferize `target` into a new allocation in `memorySpace`.
/// `bufferizeDestinationOnly=true` so the targeted op itself is not rewritten;
/// only its destination operand is materialized as a fresh memref alloc.
static LogicalResult bufferizeOpToAllocation(
    Operation *target, int64_t memorySpace,
    linalg::BufferizeToAllocationOptions ::MemcpyOp memcpyOp,
    RewriterBase &rewriter) {
  linalg::BufferizeToAllocationOptions options;
  options.bufferizeDestinationOnly = true;
  options.emitDealloc = true;
  options.memcpyOp = memcpyOp;
  Attribute memSpaceAttr =
      IntegerAttr::get(IntegerType::get(target->getContext(), 64), memorySpace);
  Value buffer =
      linalg::bufferizeToAllocation(rewriter, options, target, memSpaceAttr);
  return success(buffer != nullptr);
}

} // namespace

//===----------------------------------------------------------------------===//
// runBufferizeOutputL2Impl  (Phase 2)
//===----------------------------------------------------------------------===//

LogicalResult runBufferizeOutputL2Impl(func::FuncOp f, int64_t memorySpace,
                                       bool fuseOutputTruncfFirst,
                                       bool doTileL3ToL2Copies, int64_t kL2Tile,
                                       StringRef copyALoopMarker,
                                       StringRef copyBLoopMarker,
                                       RewriterBase &rewriter) {
  // Optional pre-step 1: convert memref.copy L3->L2 stagings to linalg.copy
  // and tile by k-l2-tile (with copy_a_loop / copy_b_loop annotations).
  if (doTileL3ToL2Copies)
    if (failed(runTileL3ToL2CopiesImpl(f, kL2Tile, copyALoopMarker,
                                       copyBLoopMarker)))
      return failure();

  // Optional pre-step 2: fuse a single-truncf linalg.generic consumer of
  // the matmul into the matmul itself before bufferizing the fill, so the
  // fill's element type matches the post-fuse matmul.
  if (fuseOutputTruncfFirst)
    runFuseOutputTruncfImpl(f, rewriter);

  SmallVector<linalg::FillOp> fills;
  f.walk([&](linalg::FillOp op) { fills.push_back(op); });
  if (fills.empty())
    return success(); // no-op if no fill.
  for (linalg::FillOp fill : fills) {
    if (!fill.getOperation()->getBlock())
      continue; // erased by a prior iteration's bufferization
    if (failed(bufferizeOpToAllocation(
            fill, memorySpace,
            linalg::BufferizeToAllocationOptions::MemcpyOp::LinalgCopy,
            rewriter)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AIRMatmulBufferizeL1Output  (Phase 3 tail)
//===----------------------------------------------------------------------===//

// Free-function body for the former `air-matmul-bufferize-l1-output` pass.
// Now invoked from `air-matmul-pack-and-transpose` when its
// `do-bufferize-l1-output` option is set.
LogicalResult runBufferizeL1OutputImpl(func::FuncOp f, int64_t memorySpace,
                                       StringRef packedMatmulMarker,
                                       RewriterBase &rewriter) {
  Operation *packedMatmul = xilinx::air::findOpWithAttr(f, packedMatmulMarker);
  if (!packedMatmul)
    return success();
  auto linalgOp = dyn_cast<linalg::LinalgOp>(packedMatmul);
  if (!linalgOp || linalgOp.getNumDpsInits() != 1)
    return packedMatmul->emitError(
        "packed_matmul op must be a LinalgOp with one DPS init");
  Operation *packC = linalgOp.getDpsInits()[0].getDefiningOp();
  if (!isa_and_nonnull<linalg::PackOp>(packC))
    return success(); // pack already bufferized or absent.
  if (failed(bufferizeOpToAllocation(
          packC, memorySpace,
          linalg::BufferizeToAllocationOptions::MemcpyOp::LinalgCopy,
          rewriter)))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// runBufferizeL1InputsImpl  (Phase 6a)
//===----------------------------------------------------------------------===//

LogicalResult runBufferizeL1InputsImpl(func::FuncOp f, int64_t memorySpace,
                                       StringRef memcpyOp, StringRef lhsMarker,
                                       StringRef rhsMarker,
                                       RewriterBase &rewriter) {
  auto memcpy =
      linalg::BufferizeToAllocationOptions::MemcpyOp::MaterializeInDestination;
  if (memcpyOp == "linalg-copy")
    memcpy = linalg::BufferizeToAllocationOptions::MemcpyOp::LinalgCopy;
  for (StringRef marker : {lhsMarker, rhsMarker}) {
    Operation *target = xilinx::air::findOpWithAttr(f, marker);
    if (!target)
      continue;
    if (failed(bufferizeOpToAllocation(target, memorySpace, memcpy, rewriter)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AIRMatmulPostBufferizeCleanup  (Phase 7+8: remove uninitialized copies,
// eliminate cascade memcpys, then sibling-fuse the K-reduction loop with the
// L3->L2 copy loops for ping-pong buffering. Combined into one pass since
// the two halves are always run back-to-back.)
//===----------------------------------------------------------------------===//

namespace {

/// Hoist any same-block ops between `target` and `source` that are used
/// inside *either* loop's body. Required because
/// `fuseIndependentSiblingForLoops` may place the merged loop at the
/// earlier of the two source positions, leaving any in-between ops
/// (including allocs/casts the merged loop depends on) below the new
/// merged-loop position.
static void hoistInterveningDeps(scf::ForOp target, scf::ForOp source) {
  Operation *first = target->isBeforeInBlock(source) ? target.getOperation()
                                                     : source.getOperation();
  Operation *second = (first == target.getOperation()) ? source.getOperation()
                                                       : target.getOperation();
  Block *block = target->getBlock();
  if (block != source->getBlock())
    return;

  llvm::SetVector<Operation *> toHoist;
  auto collect = [&](Operation *loopRoot) {
    loopRoot->walk([&](Operation *op) {
      for (Value v : op->getOperands()) {
        Operation *defOp = v.getDefiningOp();
        if (!defOp || defOp->getBlock() != block)
          continue;
        if (defOp == source.getOperation() || defOp == target.getOperation())
          continue;
        if (defOp->isBeforeInBlock(first) || defOp == first)
          continue;
        if (second->isBeforeInBlock(defOp) || defOp == second)
          continue;
        toHoist.insert(defOp);
      }
    });
  };
  collect(target.getOperation());
  collect(source.getOperation());

  // Sort the to-hoist set topologically and move each above `first` in
  // dependency order. Operands defined outside `toHoist` are treated as
  // already-ready by computeTopologicalSorting (incomplete-chain semantics).
  SmallVector<Operation *> sorted(toHoist.begin(), toHoist.end());
  (void)mlir::computeTopologicalSorting(sorted);
  for (Operation *op : sorted)
    op->moveBefore(first);
}

} // namespace

// Free-function bodies for the prior `fuse-pingpong-loops`,
// `fuse-output-truncf`, and `hoist-static-alloc` passes. Exposed via
// AIRMatmulBufferizationPasses.h so they can be called either from the
// combined post-bufferize-cleanup pass or as option-driven steps inside
// the parametric passes (pack-and-transpose, prologue-epilogue).

LogicalResult runFusePingpongLoopsImpl(func::FuncOp f, RewriterBase &rewriter) {
  scf::ForOp copyA =
      xilinx::air::findOpOfTypeWithAttr<scf::ForOp>(f, "copy_a_loop");
  scf::ForOp copyB =
      xilinx::air::findOpOfTypeWithAttr<scf::ForOp>(f, "copy_b_loop");
  scf::ForOp kRed =
      xilinx::air::findOpOfTypeWithAttr<scf::ForOp>(f, "k_reduction_loop");
  if (!copyA || !copyB || !kRed)
    return success(); // not in the right shape; no-op.

  scf::ForOp normalized = runNormalizeForBounds(kRed, rewriter);
  hoistInterveningDeps(normalized, copyB);
  if (copyB->isBeforeInBlock(normalized))
    copyB->moveBefore(normalized);
  scf::ForOp afterB =
      fuseIndependentSiblingForLoops(normalized, copyB, rewriter);
  if (!afterB)
    return failure();
  hoistInterveningDeps(afterB, copyA);
  if (copyA->isBeforeInBlock(afterB))
    copyA->moveBefore(afterB);
  scf::ForOp afterA = fuseIndependentSiblingForLoops(afterB, copyA, rewriter);
  if (!afterA)
    return failure();
  return success();
}

void runFuseOutputTruncfImpl(func::FuncOp f, RewriterBase &rewriter) {
  // Collect all (producer, truncf_only_consumer) pairs first; fusing in-
  // place mutates the IR and would invalidate a live walk.
  SmallVector<std::pair<linalg::LinalgOp, linalg::LinalgOp>> pairs;
  f.walk([&](linalg::LinalgOp op) {
    if (!containsOnlyTruncfOp(op))
      return;
    if (op.getNumDpsInputs() != 1)
      return;
    auto producerOp = op.getDpsInputs()[0].getDefiningOp<linalg::LinalgOp>();
    if (!producerOp)
      return;
    if (!producesResultForOp(producerOp, op))
      return;
    pairs.emplace_back(producerOp, op);
  });
  for (auto &p : pairs) {
    if (!p.first->getBlock() || !p.second->getBlock())
      continue;
    (void)runFuseTruncfLinalg(p.first, p.second, rewriter);
  }
}

void runHoistStaticAllocImpl(func::FuncOp f, RewriterBase &rewriter) {
  hoistStaticAllocsInFunc(rewriter,
                          cast<mlir::FunctionOpInterface>(f.getOperation()));
}

// Composite of post-bufferize-cleanup: remove uninitialized copies +
// eliminate cascade memcpys + sibling-fuse pingpong loops. Now invoked
// from `air-matmul-tile-for-vectorize` when its
// `do-post-bufferize-cleanup-first` option is set.
LogicalResult runPostBufferizeCleanupImpl(func::FuncOp f,
                                          RewriterBase &rewriter) {
  if (failed(runRemoveUninitializedCopy(f)))
    return failure();
  if (failed(runEliminateCascadeMemcpy(f)))
    return failure();
  return runFusePingpongLoopsImpl(f, rewriter);
}

} // namespace air
} // namespace xilinx
