//===- AIRMatmulBufferizationPasses.cpp -------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// M2 (Group A tail) passes. Each pass wraps a small subset of the legacy
// transform-script Phases 2/7/8: post-bufferize cleanup, ping-pong sibling
// fusion, and bf16-output truncf fusion.
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRMatmulBufferizationPasses.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/AIRLinalgBufferize.h"
#include "air/Transform/AIRMatmulCodegenHelpers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "air-matmul-bufferization-passes"

using namespace mlir;
using namespace xilinx::air;

namespace xilinx {
namespace air {

namespace {

/// Find the first op in `f` carrying `marker` as a discardable attribute.
static Operation *findMarkedOp(func::FuncOp f, StringRef marker) {
  Operation *found = nullptr;
  f.walk([&](Operation *op) {
    if (op->hasAttr(marker)) {
      found = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// Bufferize `target` into a new allocation in `memorySpace`.
/// `bufferizeDestinationOnly=true` so the targeted op itself is not rewritten;
/// only its destination operand is materialized as a fresh memref alloc.
static LogicalResult bufferizeOpToAllocation(Operation *target,
                                             int64_t memorySpace,
                                             linalg::BufferizeToAllocationOptions
                                                 ::MemcpyOp memcpyOp,
                                             RewriterBase &rewriter) {
  linalg::BufferizeToAllocationOptions options;
  options.bufferizeDestinationOnly = true;
  options.emitDealloc = true;
  options.memcpyOp = memcpyOp;
  Attribute memSpaceAttr =
      IntegerAttr::get(IntegerType::get(target->getContext(), 64), memorySpace);
  Value buffer = linalg::bufferizeToAllocation(rewriter, options, target,
                                               memSpaceAttr);
  return success(buffer != nullptr);
}

} // namespace

//===----------------------------------------------------------------------===//
// AIRMatmulBufferizeOutputL2  (Phase 2)
//===----------------------------------------------------------------------===//

namespace {
class AIRMatmulBufferizeOutputL2
    : public impl::AIRMatmulBufferizeOutputL2Base<AIRMatmulBufferizeOutputL2> {
public:
  AIRMatmulBufferizeOutputL2() = default;
  AIRMatmulBufferizeOutputL2(const AIRMatmulBufferizeOutputL2Options &opts)
      : AIRMatmulBufferizeOutputL2Base(opts) {}

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    SmallVector<linalg::FillOp> fills;
    f.walk([&](linalg::FillOp op) { fills.push_back(op); });
    if (fills.empty())
      return; // no-op if no fill.
    IRRewriter rewriter(&getContext());
    for (linalg::FillOp fill : fills) {
      if (!fill.getOperation()->getBlock())
        continue; // erased by a prior iteration's bufferization
      if (failed(bufferizeOpToAllocation(
              fill, clMemorySpace,
              linalg::BufferizeToAllocationOptions::MemcpyOp::LinalgCopy,
              rewriter)))
        return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeOutputL2Pass() {
  return std::make_unique<AIRMatmulBufferizeOutputL2>();
}
std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeOutputL2Pass(
    const AIRMatmulBufferizeOutputL2Options &opts) {
  return std::make_unique<AIRMatmulBufferizeOutputL2>(opts);
}

//===----------------------------------------------------------------------===//
// AIRMatmulBufferizeL1Output  (Phase 3 tail)
//===----------------------------------------------------------------------===//

namespace {
class AIRMatmulBufferizeL1Output
    : public impl::AIRMatmulBufferizeL1OutputBase<AIRMatmulBufferizeL1Output> {
public:
  AIRMatmulBufferizeL1Output() = default;
  AIRMatmulBufferizeL1Output(const AIRMatmulBufferizeL1OutputOptions &opts)
      : AIRMatmulBufferizeL1OutputBase(opts) {}

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    Operation *packedMatmul = findMarkedOp(f, clPackedMatmulMarker);
    if (!packedMatmul)
      return;
    auto linalgOp = dyn_cast<linalg::LinalgOp>(packedMatmul);
    if (!linalgOp || linalgOp.getNumDpsInits() != 1) {
      packedMatmul->emitError("packed_matmul op must be a LinalgOp with one "
                              "DPS init");
      return signalPassFailure();
    }
    Operation *packC = linalgOp.getDpsInits()[0].getDefiningOp();
    if (!isa_and_nonnull<linalg::PackOp>(packC))
      return; // pack already bufferized or absent.
    IRRewriter rewriter(&getContext());
    if (failed(bufferizeOpToAllocation(
            packC, clMemorySpace,
            linalg::BufferizeToAllocationOptions::MemcpyOp::LinalgCopy,
            rewriter)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeL1OutputPass() {
  return std::make_unique<AIRMatmulBufferizeL1Output>();
}
std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeL1OutputPass(
    const AIRMatmulBufferizeL1OutputOptions &opts) {
  return std::make_unique<AIRMatmulBufferizeL1Output>(opts);
}

//===----------------------------------------------------------------------===//
// AIRMatmulBufferizeL1Inputs  (Phase 6a)
//===----------------------------------------------------------------------===//

namespace {
class AIRMatmulBufferizeL1Inputs
    : public impl::AIRMatmulBufferizeL1InputsBase<AIRMatmulBufferizeL1Inputs> {
public:
  AIRMatmulBufferizeL1Inputs() = default;
  AIRMatmulBufferizeL1Inputs(const AIRMatmulBufferizeL1InputsOptions &opts)
      : AIRMatmulBufferizeL1InputsBase(opts) {}

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    IRRewriter rewriter(&getContext());
    auto memcpy = linalg::BufferizeToAllocationOptions::MemcpyOp::
        MaterializeInDestination;
    if (StringRef(clMemcpyOp) == "linalg-copy")
      memcpy = linalg::BufferizeToAllocationOptions::MemcpyOp::LinalgCopy;
    for (StringRef marker : {StringRef(clLhsMarker), StringRef(clRhsMarker)}) {
      Operation *target = findMarkedOp(f, marker);
      if (!target)
        continue;
      if (failed(bufferizeOpToAllocation(target, clMemorySpace, memcpy,
                                         rewriter)))
        return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeL1InputsPass() {
  return std::make_unique<AIRMatmulBufferizeL1Inputs>();
}
std::unique_ptr<mlir::Pass> createAIRMatmulBufferizeL1InputsPass(
    const AIRMatmulBufferizeL1InputsOptions &opts) {
  return std::make_unique<AIRMatmulBufferizeL1Inputs>(opts);
}

//===----------------------------------------------------------------------===//
// AIRMatmulCleanupBufferize  (Phase 7 tail)
//===----------------------------------------------------------------------===//

namespace {
class AIRMatmulCleanupBufferize
    : public impl::AIRMatmulCleanupBufferizeBase<AIRMatmulCleanupBufferize> {
public:
  AIRMatmulCleanupBufferize() = default;

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    if (failed(runRemoveUninitializedCopy(f)))
      return signalPassFailure();
    if (failed(runEliminateCascadeMemcpy(f)))
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulCleanupBufferizePass() {
  return std::make_unique<AIRMatmulCleanupBufferize>();
}

//===----------------------------------------------------------------------===//
// AIRMatmulFusePingpongLoops  (Phase 8)
//===----------------------------------------------------------------------===//

namespace {

/// Find the first scf.for in `f` whose `marker` discardable attribute is set.
static scf::ForOp findMarkedForLoop(func::FuncOp f, StringRef marker) {
  scf::ForOp found;
  f.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr(marker)) {
      found = forOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

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

  // Iteratively move ops with all-resolved operands above `first`.
  bool progress = true;
  while (progress && !toHoist.empty()) {
    progress = false;
    for (Operation *op : llvm::to_vector(toHoist)) {
      bool ready = true;
      for (Value v : op->getOperands()) {
        Operation *defOp = v.getDefiningOp();
        if (defOp && defOp->getBlock() == block &&
            !defOp->isBeforeInBlock(first) && defOp != first) {
          ready = false;
          break;
        }
      }
      if (ready) {
        op->moveBefore(first);
        toHoist.remove(op);
        progress = true;
      }
    }
  }
}

class AIRMatmulFusePingpongLoops
    : public impl::AIRMatmulFusePingpongLoopsBase<AIRMatmulFusePingpongLoops> {
public:
  AIRMatmulFusePingpongLoops() = default;

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    IRRewriter rewriter(&getContext());

    scf::ForOp copyA = findMarkedForLoop(f, "copy_a_loop");
    scf::ForOp copyB = findMarkedForLoop(f, "copy_b_loop");
    scf::ForOp kRed = findMarkedForLoop(f, "k_reduction_loop");

    // No-op if the IR is not in the post-Phase-4 shape (e.g. running on a
    // function that didn't go through tile-l3-to-l2 + tile-k-and-fuse).
    if (!copyA || !copyB || !kRed)
      return;

    scf::ForOp normalized = runNormalizeForBounds(kRed, rewriter);

    // Fuse copy_b first, then copy_a, matching the legacy transform script.
    // `fuseIndependentSiblingForLoops` may place the merged loop at the
    // earlier of the two source positions; if the source is earlier than the
    // target, that drags the merged loop above any intervening prologue/
    // epilogue scf.forall ops. To avoid that, MOVE the source loop to
    // immediately before the target first, so the merged loop stays at the
    // target's position. (`hoistInterveningDeps` is still called for any
    // allocs/casts the source loop body uses.)
    hoistInterveningDeps(normalized, copyB);
    if (copyB->isBeforeInBlock(normalized))
      copyB->moveBefore(normalized);
    scf::ForOp afterB =
        fuseIndependentSiblingForLoops(normalized, copyB, rewriter);
    if (!afterB)
      return signalPassFailure();
    hoistInterveningDeps(afterB, copyA);
    if (copyA->isBeforeInBlock(afterB))
      copyA->moveBefore(afterB);
    scf::ForOp afterA =
        fuseIndependentSiblingForLoops(afterB, copyA, rewriter);
    if (!afterA)
      return signalPassFailure();
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulFusePingpongLoopsPass() {
  return std::make_unique<AIRMatmulFusePingpongLoops>();
}

//===----------------------------------------------------------------------===//
// AIRMatmulFuseOutputTruncf  (Phase 2, test 53 / bf16-out flow)
//===----------------------------------------------------------------------===//

namespace {
class AIRMatmulFuseOutputTruncf
    : public impl::AIRMatmulFuseOutputTruncfBase<AIRMatmulFuseOutputTruncf> {
public:
  AIRMatmulFuseOutputTruncf() = default;

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    IRRewriter rewriter(&getContext());

    // Collect all (producer, truncf_only_consumer) pairs first; fusing in-
    // place mutates the IR and would invalidate a live walk.
    SmallVector<std::pair<linalg::LinalgOp, linalg::LinalgOp>> pairs;
    f.walk([&](linalg::LinalgOp op) {
      if (!containsOnlyTruncfOp(op))
        return;
      if (op.getNumDpsInputs() != 1)
        return;
      auto producerOp =
          op.getDpsInputs()[0].getDefiningOp<linalg::LinalgOp>();
      if (!producerOp)
        return;
      if (!producesResultForOp(producerOp, op))
        return;
      pairs.emplace_back(producerOp, op);
    });

    for (auto &p : pairs) {
      // Skip if either op was erased by a prior fusion in this loop.
      if (!p.first->getBlock() || !p.second->getBlock())
        continue;
      (void)runFuseTruncfLinalg(p.first, p.second, rewriter);
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulFuseOutputTruncfPass() {
  return std::make_unique<AIRMatmulFuseOutputTruncf>();
}

//===----------------------------------------------------------------------===//
// AIRHoistStaticAlloc (M4 helper for the K-peel flow)
//===----------------------------------------------------------------------===//

namespace {
class AIRHoistStaticAlloc
    : public impl::AIRHoistStaticAllocBase<AIRHoistStaticAlloc> {
public:
  AIRHoistStaticAlloc() = default;

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    IRRewriter rewriter(&getContext());
    hoistStaticAllocsInFunc(rewriter,
                            cast<mlir::FunctionOpInterface>(f.getOperation()));
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createAIRHoistStaticAllocPass() {
  return std::make_unique<AIRHoistStaticAlloc>();
}

} // namespace air
} // namespace xilinx
