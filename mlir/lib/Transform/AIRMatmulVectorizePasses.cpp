//===- AIRMatmulVectorizePasses.cpp ----------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// M1a passes of the matmul codegen pipeline. Each pass is a thin wrapper that
// walks a func::FuncOp and dispatches to a runFoo helper in
// AIRMatmulCodegenHelpers; the same helper is shared with the corresponding
// transform.air.* op apply() in AIRLinalgCodegen.cpp.
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRMatmulVectorizePasses.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/AIRMatmulCodegenHelpers.h"
#include "air/Util/MatmulCodegenConfig.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "air-matmul-vectorize-passes"

using namespace mlir;
using namespace xilinx::air;

namespace xilinx {
namespace air {

namespace {

//===----------------------------------------------------------------------===//
// AIRFoldUnitExtentDims
//===----------------------------------------------------------------------===//

class AIRFoldUnitExtentDims
    : public impl::AIRFoldUnitExtentDimsBase<AIRFoldUnitExtentDims> {
public:
  AIRFoldUnitExtentDims() = default;

  void runOnOperation() override {
    if (failed(runFoldUnitExtentDimsOnFunc(getOperation())))
      return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAIRFoldUnitExtentDimsPass() {
  return std::make_unique<AIRFoldUnitExtentDims>();
}

//===----------------------------------------------------------------------===//
// AIREliminateRedundantVectorTransfers
//===----------------------------------------------------------------------===//

namespace {

class AIREliminateRedundantVectorTransfers
    : public impl::AIREliminateRedundantVectorTransfersBase<
          AIREliminateRedundantVectorTransfers> {
public:
  AIREliminateRedundantVectorTransfers() = default;

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    (void)runEliminateRedundantVectorTransfers(getOperation(), rewriter);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAIREliminateRedundantVectorTransfersPass() {
  return std::make_unique<AIREliminateRedundantVectorTransfers>();
}

//===----------------------------------------------------------------------===//
// AIRFlattenForIterArgs
//===----------------------------------------------------------------------===//

namespace {

class AIRFlattenForIterArgs
    : public impl::AIRFlattenForIterArgsBase<AIRFlattenForIterArgs> {
public:
  AIRFlattenForIterArgs() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    // Collect first to avoid invalidation when scf.for is replaced.
    SmallVector<mlir::scf::ForOp> targets;
    getOperation().walk([&](mlir::scf::ForOp forOp) {
      // Only target loops with at least one vector-typed iter_arg; runFlatten
      // is a no-op otherwise but we skip them to keep IR diff minimal.
      for (Value v : forOp.getInitArgs())
        if (isa<VectorType>(v.getType())) {
          targets.push_back(forOp);
          break;
        }
    });
    for (mlir::scf::ForOp forOp : targets) {
      auto res = runFlattenForIterArgs(forOp, rewriter);
      if (failed(res)) {
        forOp->emitError("flatten-for-iter-args failed");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAIRFlattenForIterArgsPass() {
  return std::make_unique<AIRFlattenForIterArgs>();
}

//===----------------------------------------------------------------------===//
// AIRHoistLoopInvariantTransfers
//===----------------------------------------------------------------------===//

namespace {

// Find the outermost scf.for that lives directly inside `scope`'s region
// (i.e., not nested within another scf.for). Returns nullptr if none.
// True if the herd contains at least one vector.contract — i.e., it's a
// compute herd, not a fill/epilogue herd. Mirrors the script's targeting of
// `herd2_1` specifically (the compute herd).
static bool herdHasVectorContract(xilinx::air::HerdOp herd) {
  bool found = false;
  herd->walk([&](mlir::vector::ContractionOp) {
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

[[maybe_unused]] static mlir::scf::ForOp findOutermostForIn(Operation *scope) {
  mlir::scf::ForOp result;
  scope->walk([&](mlir::scf::ForOp forOp) {
    if (result)
      return WalkResult::skip();
    // Skip nested-within-other-for cases — the outermost-in-scope is the
    // first one whose nearest enclosing scf.for is outside `scope`.
    auto parentFor = forOp->getParentOfType<mlir::scf::ForOp>();
    if (!parentFor || !scope->isProperAncestor(parentFor)) {
      result = forOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;
}

class AIRHoistLoopInvariantTransfers
    : public impl::AIRHoistLoopInvariantTransfersBase<
          AIRHoistLoopInvariantTransfers> {
public:
  AIRHoistLoopInvariantTransfers() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    // Target every innermost scf.for inside each herd: an scf.for is
    // "innermost" if its body contains no nested scf.for. The helper checks
    // that vector.transfer_read/write pairs live in the loop's immediate
    // body, so we must call it on the loop where the transfers actually are.
    SmallVector<mlir::scf::ForOp> innermost;
    getOperation().walk([&](xilinx::air::HerdOp herd) {
      herd->walk([&](mlir::scf::ForOp forOp) {
        bool hasInnerFor = false;
        for (Operation &nested : forOp.getBody()->without_terminator()) {
          if (isa<mlir::scf::ForOp>(nested)) {
            hasInnerFor = true;
            break;
          }
          // Check one level deeper too (scf.for nested in another scf op
          // counts as inner).
          nested.walk([&](mlir::scf::ForOp) { hasInnerFor = true; });
          if (hasInnerFor)
            break;
        }
        if (!hasInnerFor)
          innermost.push_back(forOp);
      });
    });
    for (mlir::scf::ForOp loopOp : innermost) {
      auto scopeOp = loopOp->getParentOfType<xilinx::air::HerdOp>();
      auto res =
          runHoistLoopInvariantTransfers(scopeOp, loopOp, rewriter);
      if (failed(res)) {
        loopOp->emitError("hoist-loop-invariant-transfers failed");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAIRHoistLoopInvariantTransfersPass() {
  return std::make_unique<AIRHoistLoopInvariantTransfers>();
}

//===----------------------------------------------------------------------===//
// AIRHoistVectorTransferPointers
//===----------------------------------------------------------------------===//

namespace {

class AIRHoistVectorTransferPointers
    : public impl::AIRHoistVectorTransferPointersBase<
          AIRHoistVectorTransferPointers> {
public:
  AIRHoistVectorTransferPointers() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::scf::SCFDialect, mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    // Target every innermost scf.for inside each herd. The helper iterates
    // forOp.getBody()->without_terminator() looking for vector.transfer ops
    // — only effective when called on the loop where the transfers live.
    SmallVector<mlir::scf::ForOp> innermost;
    getOperation().walk([&](xilinx::air::HerdOp herd) {
      // Only target compute herds (containing vector.contract). Skipping
      // fill/epilogue herds preserves their 6D memref access patterns so
      // downstream `air-shrink-memref-sizes-by-access` can split L1 buffers
      // across cores; flattening the fill herd's access via this pass would
      // produce a 1D access pattern shrink can't analyze.
      if (!herdHasVectorContract(herd))
        return;
      herd->walk([&](mlir::scf::ForOp forOp) {
        bool hasInnerFor = false;
        for (Operation &nested : forOp.getBody()->without_terminator()) {
          if (isa<mlir::scf::ForOp>(nested)) {
            hasInnerFor = true;
            break;
          }
          nested.walk([&](mlir::scf::ForOp) { hasInnerFor = true; });
          if (hasInnerFor)
            break;
        }
        if (!hasInnerFor)
          innermost.push_back(forOp);
      });
    });
    for (mlir::scf::ForOp forOp : innermost) {
      if (failed(runHoistVectorTransferPointers(forOp, rewriter))) {
        forOp->emitError("hoist-vector-transfer-pointers failed");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAIRHoistVectorTransferPointersPass() {
  return std::make_unique<AIRHoistVectorTransferPointers>();
}

//===----------------------------------------------------------------------===//
// AIRVectorCastForEmulation
//===----------------------------------------------------------------------===//

namespace {

class AIRVectorCastForEmulation
    : public impl::AIRVectorCastForEmulationBase<AIRVectorCastForEmulation> {
public:
  AIRVectorCastForEmulation() = default;
  AIRVectorCastForEmulation(const AIRVectorCastForEmulationOptions &opts)
      : AIRVectorCastForEmulationBase(opts) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    Type targetTy =
        llvm::StringSwitch<Type>(clTargetElementType)
            .Case("f32", Float32Type::get(ctx))
            .Case("bf16", BFloat16Type::get(ctx))
            .Case("f16", Float16Type::get(ctx))
            .Case("i32", IntegerType::get(ctx, 32))
            .Case("i16", IntegerType::get(ctx, 16))
            .Case("i8", IntegerType::get(ctx, 8))
            .Default(Type());
    if (!targetTy) {
      getOperation()->emitError("unknown target-element-type '")
          << clTargetElementType << "'";
      return signalPassFailure();
    }

    SmallVector<int64_t> inIdx(clInputIndices.begin(), clInputIndices.end());
    SmallVector<int64_t> outIdx(clOutputIndices.begin(), clOutputIndices.end());

    IRRewriter rewriter(ctx);
    SmallVector<mlir::vector::ContractionOp> targets;
    getOperation().walk(
        [&](mlir::vector::ContractionOp c) { targets.push_back(c); });
    for (mlir::vector::ContractionOp c : targets) {
      if (failed(runVectorTypeCastOnTarget(c.getOperation(), targetTy, inIdx,
                                            outIdx, rewriter))) {
        c->emitError("vector_type_cast failed");
        return signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAIRVectorCastForEmulationPass() {
  return std::make_unique<AIRVectorCastForEmulation>();
}

std::unique_ptr<mlir::Pass> createAIRVectorCastForEmulationPass(
    const AIRVectorCastForEmulationOptions &opts) {
  return std::make_unique<AIRVectorCastForEmulation>(opts);
}

//===----------------------------------------------------------------------===//
// AIRHoistCastPairs (fixed-point wrapper around runHoistCastPair)
//===----------------------------------------------------------------------===//

namespace {

// For each vector iter_arg of `forOp`, look for an extension that operates
// on it (directly or through a single shape_cast) and a truncation whose
// result is yielded back at the same iter_arg position. Returns the first
// such pair.
static bool findNextPair(mlir::Operation *funcOp, mlir::Operation *&extOp,
                          mlir::Operation *&truncOp,
                          mlir::scf::ForOp &loopOp) {
  bool found = false;
  funcOp->walk([&](xilinx::air::HerdOp herd) {
    if (found)
      return WalkResult::interrupt();
    herd->walk([&](mlir::scf::ForOp forOp) {
      if (found)
        return WalkResult::interrupt();
      auto yieldOp =
          dyn_cast<mlir::scf::YieldOp>(forOp.getBody()->getTerminator());
      if (!yieldOp)
        return WalkResult::advance();
      // For each vector-typed iter_arg, search for a matching ext/trunc pair.
      mlir::Block *body = forOp.getBody();
      for (auto [argIdx, blockArg] :
           llvm::enumerate(body->getArguments().drop_front(1))) {
        if (!isa<mlir::VectorType>(blockArg.getType()))
          continue;
        // Find an extension whose input is `blockArg` (directly or via a
        // single shape_cast).
        mlir::Operation *foundExt = nullptr;
        for (mlir::Operation *user : blockArg.getUsers()) {
          if (isa<mlir::arith::ExtFOp, mlir::arith::ExtSIOp,
                  mlir::arith::ExtUIOp>(user)) {
            foundExt = user;
            break;
          }
          if (auto sc = dyn_cast<mlir::vector::ShapeCastOp>(user)) {
            for (mlir::Operation *u2 : sc.getResult().getUsers()) {
              if (isa<mlir::arith::ExtFOp, mlir::arith::ExtSIOp,
                      mlir::arith::ExtUIOp>(u2)) {
                foundExt = u2;
                break;
              }
            }
            if (foundExt)
              break;
          }
        }
        if (!foundExt)
          continue;
        // Find the truncation whose output is yielded at the same iter_arg
        // position (directly or via a single shape_cast).
        mlir::Value yieldedVal = yieldOp.getOperand((unsigned)argIdx);
        mlir::Operation *foundTrunc = yieldedVal.getDefiningOp();
        if (auto sc = dyn_cast_if_present<mlir::vector::ShapeCastOp>(foundTrunc))
          foundTrunc = sc.getSource().getDefiningOp();
        if (!foundTrunc ||
            !isa<mlir::arith::TruncFOp, mlir::arith::TruncIOp>(foundTrunc))
          continue;
        extOp = foundExt;
        truncOp = foundTrunc;
        loopOp = forOp;
        found = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return WalkResult::advance();
  });
  return found;
}

class AIRHoistCastPairs
    : public impl::AIRHoistCastPairsBase<AIRHoistCastPairs> {
public:
  AIRHoistCastPairs() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                    mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    int64_t budget = clMaxIterations;
    while (budget-- > 0) {
      mlir::Operation *extOp = nullptr;
      mlir::Operation *truncOp = nullptr;
      mlir::scf::ForOp loopOp;
      if (!findNextPair(getOperation(), extOp, truncOp, loopOp))
        return;
      auto res = runHoistCastPair(extOp, truncOp, loopOp, rewriter);
      if (failed(res)) {
        getOperation()->emitError("hoist-cast-pair failed");
        return signalPassFailure();
      }
    }
    getOperation()->emitWarning(
        "air-hoist-cast-pairs hit max-iterations cap; remaining pairs not "
        "hoisted");
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAIRHoistCastPairsPass() {
  return std::make_unique<AIRHoistCastPairs>();
}

// Stubs for the remaining 5 passes (M1a-2..6) — implemented in a follow-up.
// Defined here so the pass registration in Passes.td/.cpp links.

#define UNIMPL_PASS(ClassName, CreateName)                                     \
  namespace {                                                                  \
  class ClassName : public impl::ClassName##Base<ClassName> {                  \
  public:                                                                      \
    ClassName() = default;                                                     \
    void runOnOperation() override {                                           \
      getOperation()->emitError(#CreateName " is not yet implemented");        \
      signalPassFailure();                                                     \
    }                                                                          \
  };                                                                           \
  }                                                                            \
  std::unique_ptr<mlir::Pass> create##ClassName##Pass() {                      \
    return std::make_unique<ClassName>();                                      \
  }


#undef UNIMPL_PASS

namespace {

// Tile a TilingInterface op by the given sizes, using scf.for. If `sizes`
// is shorter than the op's iteration domain rank, pads with zeros (matching
// `transform.structured.tile_using_for` semantics). Returns the produced
// loops on success.
static FailureOr<SmallVector<mlir::LoopLikeOpInterface>>
tileWithScfFor(mlir::Operation *op, ArrayRef<int64_t> sizes,
               IRRewriter &rewriter) {
  auto iface = dyn_cast<mlir::TilingInterface>(op);
  if (!iface)
    return op->emitError("op does not implement TilingInterface");
  rewriter.setInsertionPoint(op);
  mlir::scf::SCFTilingOptions opts;
  SmallVector<OpFoldResult> sizeFolds;
  for (int64_t s : sizes)
    sizeFolds.push_back(rewriter.getIndexAttr(s));
  // Pad with zeros to match iteration domain rank.
  unsigned numLoops = iface.getLoopIteratorTypes().size();
  while (sizeFolds.size() < numLoops)
    sizeFolds.push_back(rewriter.getIndexAttr(0));
  opts.setTileSizes(sizeFolds);
  auto res = mlir::scf::tileUsingSCF(rewriter, iface, opts);
  if (failed(res))
    return op->emitError("tileUsingSCF failed");
  rewriter.replaceOp(op, res->replacements);
  return res->loops;
}

class AIRMatmulTileForVectorize
    : public impl::AIRMatmulTileForVectorizeBase<AIRMatmulTileForVectorize> {
public:
  AIRMatmulTileForVectorize() = default;
  AIRMatmulTileForVectorize(const AIRMatmulTileForVectorizeOptions &opts)
      : AIRMatmulTileForVectorizeBase(opts) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::linalg::LinalgDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    IRRewriter rewriter(&getContext());

    SmallVector<int64_t> matmulTile = clMatmulTileSizes.empty()
                                          ? SmallVector<int64_t>{2, 2, 1, 0, 0, 0}
                                          : llvm::to_vector(clMatmulTileSizes);
    SmallVector<int64_t> matmulUnroll =
        clMatmulUnrollTileSizes.empty()
            ? SmallVector<int64_t>{1, 1, 0, 0, 0, 0}
            : llvm::to_vector(clMatmulUnrollTileSizes);
    SmallVector<int64_t> fillTile = clFillTileSizes.empty()
                                        ? SmallVector<int64_t>{1, 1, 0, 0}
                                        : llvm::to_vector(clFillTileSizes);
    int64_t unrollFactor = clMatmulUnrollFactor;
    if (auto cfg = xilinx::air::findMatmulCodegenConfig(getOperation())) {
      auto take = [&](StringRef key, SmallVector<int64_t> &dst) {
        auto v = xilinx::air::getI64Array(*cfg, key);
        if (!v.empty())
          dst = std::move(v);
      };
      take("vector_tile", matmulTile);
      take("vector_unroll_tile", matmulUnroll);
      take("fill_vector_tile", fillTile);
      unrollFactor = xilinx::air::getI64(*cfg, "vector_unroll_factor",
                                         unrollFactor);
    }

    // Phase 1: tile each linalg.generic packed-matmul body by matmulTile.
    // Accept ops that either (a) live inside an air.herd (M1 iron-built flow)
    // or (b) carry the `matmul_compute` marker (M2 linalg-input flow runs
    // this pass BEFORE the forall->herd materialization).
    SmallVector<mlir::linalg::GenericOp> matmulGenerics;
    getOperation().walk([&](mlir::linalg::GenericOp op) {
      bool inHerd = op->getParentOfType<xilinx::air::HerdOp>() != nullptr;
      bool isMatmulCompute = op->hasAttr("matmul_compute");
      if (!inHerd && !isMatmulCompute)
        return;
      if (op.getNumLoops() < (int64_t)matmulTile.size())
        return;
      matmulGenerics.push_back(op);
    });
    for (mlir::linalg::GenericOp gen : matmulGenerics) {
      auto loops1 = tileWithScfFor(gen.getOperation(), matmulTile, rewriter);
      if (failed(loops1))
        return signalPassFailure();
      // After first tile, find the new inner linalg.generic (the only
      // descendant of the produced loops).
      mlir::linalg::GenericOp inner;
      if (!loops1->empty()) {
        loops1->back()->walk([&](mlir::linalg::GenericOp g) {
          inner = g;
          return WalkResult::interrupt();
        });
      } else {
        inner = gen; // No tiling happened (zero sizes). Skip second tile.
      }
      if (!inner)
        continue;
      auto loops2 =
          tileWithScfFor(inner.getOperation(), matmulUnroll, rewriter);
      if (failed(loops2))
        return signalPassFailure();
      // Unroll the two innermost produced loops.
      // loops2->back() is the innermost; loops2 is in outer→inner order.
      uint64_t factor = unrollFactor;
      if (factor > 1) {
        SmallVector<mlir::scf::ForOp> toUnroll;
        for (auto loop : *loops2)
          if (auto sf = dyn_cast<mlir::scf::ForOp>(loop.getOperation()))
            toUnroll.push_back(sf);
        // Unroll from innermost outward (last two).
        for (auto it = toUnroll.rbegin();
             it != toUnroll.rend() && std::distance(toUnroll.rbegin(), it) < 2;
             ++it) {
          if (failed(mlir::loopUnrollByFactor(*it, factor))) {
            it->emitError("loopUnrollByFactor failed");
            return signalPassFailure();
          }
        }
      }
    }

    // Phase 2: tile each linalg.fill (or linalg.generic carrying the
    // `init_fill` marker, set by the M2 prologue-epilogue pass after
    // generalize+interchange) by fillTile.
    SmallVector<mlir::Operation *> fills;
    getOperation().walk([&](mlir::linalg::FillOp f) {
      if (f->getParentOfType<xilinx::air::HerdOp>())
        fills.push_back(f.getOperation());
    });
    getOperation().walk([&](mlir::linalg::GenericOp g) {
      if (g->hasAttr("init_fill"))
        fills.push_back(g.getOperation());
    });
    for (mlir::Operation *f : fills) {
      auto loops = tileWithScfFor(f, fillTile, rewriter);
      if (failed(loops))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulTileForVectorizePass() {
  return std::make_unique<AIRMatmulTileForVectorize>();
}

std::unique_ptr<mlir::Pass> createAIRMatmulTileForVectorizePass(
    const AIRMatmulTileForVectorizeOptions &opts) {
  return std::make_unique<AIRMatmulTileForVectorize>(opts);
}

} // namespace air
} // namespace xilinx
