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

namespace {

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

// Per-step bodies. Extracted from the previously-individual AIR passes; now
// invoked in fixed order from the AIRMatmulCodegenVecPrep composite below.

static LogicalResult runFlattenForIterArgsStep(func::FuncOp func,
                                                IRRewriter &rewriter) {
  SmallVector<mlir::scf::ForOp> targets;
  func.walk([&](mlir::scf::ForOp forOp) {
    for (Value v : forOp.getInitArgs())
      if (isa<VectorType>(v.getType())) {
        targets.push_back(forOp);
        break;
      }
  });
  for (mlir::scf::ForOp forOp : targets) {
    auto res = runFlattenForIterArgs(forOp, rewriter);
    if (failed(res))
      return forOp->emitError("flatten-for-iter-args failed");
  }
  return success();
}

static LogicalResult runHoistLoopInvariantTransfersStep(func::FuncOp func,
                                                        IRRewriter &rewriter) {
  // Innermost scf.for inside each herd; the helper requires vector.transfer
  // pairs in the loop's immediate body.
  SmallVector<mlir::scf::ForOp> innermost;
  func.walk([&](xilinx::air::HerdOp herd) {
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
  for (mlir::scf::ForOp loopOp : innermost) {
    auto scopeOp = loopOp->getParentOfType<xilinx::air::HerdOp>();
    auto res = runHoistLoopInvariantTransfers(scopeOp, loopOp, rewriter);
    if (failed(res))
      return loopOp->emitError("hoist-loop-invariant-transfers failed");
  }
  return success();
}

static LogicalResult runHoistVectorTransferPointersStep(func::FuncOp func,
                                                        IRRewriter &rewriter) {
  // Compute-herd-only filter: skip fill/epilogue herds so downstream
  // air-shrink-memref-sizes-by-access can still split L1 buffers per-core.
  SmallVector<mlir::scf::ForOp> innermost;
  func.walk([&](xilinx::air::HerdOp herd) {
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
    if (failed(runHoistVectorTransferPointers(forOp, rewriter)))
      return forOp->emitError("hoist-vector-transfer-pointers failed");
  }
  return success();
}

static LogicalResult runVectorCastForEmulationStep(func::FuncOp func,
                                                   StringRef targetElementType,
                                                   ArrayRef<int64_t> inIdx,
                                                   ArrayRef<int64_t> outIdx,
                                                   IRRewriter &rewriter) {
  if (targetElementType.empty())
    return success(); // skip
  MLIRContext *ctx = func.getContext();
  Type targetTy = llvm::StringSwitch<Type>(targetElementType)
                      .Case("f32", Float32Type::get(ctx))
                      .Case("bf16", BFloat16Type::get(ctx))
                      .Case("f16", Float16Type::get(ctx))
                      .Case("i32", IntegerType::get(ctx, 32))
                      .Case("i16", IntegerType::get(ctx, 16))
                      .Case("i8", IntegerType::get(ctx, 8))
                      .Default(Type());
  if (!targetTy)
    return func->emitError("unknown target-element-type '")
           << targetElementType << "'";
  SmallVector<mlir::vector::ContractionOp> targets;
  func.walk([&](mlir::vector::ContractionOp c) { targets.push_back(c); });
  for (mlir::vector::ContractionOp c : targets) {
    if (failed(runVectorTypeCastOnTarget(c.getOperation(), targetTy, inIdx,
                                          outIdx, rewriter)))
      return c->emitError("vector_type_cast failed");
  }
  return success();
}

// For each vector iter_arg of `forOp`, look for an extension that operates
// on it (directly or through a single shape_cast) and a truncation whose
// result is yielded back at the same iter_arg position.
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
      mlir::Block *body = forOp.getBody();
      for (auto [argIdx, blockArg] :
           llvm::enumerate(body->getArguments().drop_front(1))) {
        if (!isa<mlir::VectorType>(blockArg.getType()))
          continue;
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

static LogicalResult runHoistCastPairsStep(func::FuncOp func,
                                           int64_t maxIterations,
                                           IRRewriter &rewriter) {
  int64_t budget = maxIterations;
  while (budget-- > 0) {
    mlir::Operation *extOp = nullptr;
    mlir::Operation *truncOp = nullptr;
    mlir::scf::ForOp loopOp;
    if (!findNextPair(func.getOperation(), extOp, truncOp, loopOp))
      return success();
    auto res = runHoistCastPair(extOp, truncOp, loopOp, rewriter);
    if (failed(res))
      return func->emitError("hoist-cast-pair failed");
  }
  func->emitWarning(
      "air-matmul-codegen-vec-prep hit hoist-cast-pairs-max-iterations cap; "
      "remaining pairs not hoisted");
  return success();
}

class AIRMatmulCodegenVecPrep
    : public impl::AIRMatmulCodegenVecPrepBase<AIRMatmulCodegenVecPrep> {
public:
  AIRMatmulCodegenVecPrep() = default;
  AIRMatmulCodegenVecPrep(const AIRMatmulCodegenVecPrepOptions &opts)
      : AIRMatmulCodegenVecPrepBase(opts) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                    mlir::vector::VectorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());

    if (clDoFoldUnitExtentDims)
      if (failed(runFoldUnitExtentDimsOnFunc(func)))
        return signalPassFailure();
    if (clDoEliminateRedundantVectorTransfers)
      (void)runEliminateRedundantVectorTransfers(func, rewriter);
    SmallVector<int64_t> cast1In(clCast1InputIndices.begin(),
                                  clCast1InputIndices.end());
    SmallVector<int64_t> cast1Out(clCast1OutputIndices.begin(),
                                   clCast1OutputIndices.end());
    if (failed(runVectorCastForEmulationStep(func, clCast1TargetElementType,
                                             cast1In, cast1Out, rewriter)))
      return signalPassFailure();
    SmallVector<int64_t> cast2In(clCast2InputIndices.begin(),
                                  clCast2InputIndices.end());
    SmallVector<int64_t> cast2Out(clCast2OutputIndices.begin(),
                                   clCast2OutputIndices.end());
    if (failed(runVectorCastForEmulationStep(func, clCast2TargetElementType,
                                             cast2In, cast2Out, rewriter)))
      return signalPassFailure();
    if (clDoHoistLoopInvariantTransfers)
      if (failed(runHoistLoopInvariantTransfersStep(func, rewriter)))
        return signalPassFailure();
    if (clDoFlattenForIterArgs)
      if (failed(runFlattenForIterArgsStep(func, rewriter)))
        return signalPassFailure();
    if (clDoHoistVectorTransferPointers)
      if (failed(runHoistVectorTransferPointersStep(func, rewriter)))
        return signalPassFailure();
    if (clDoHoistCastPairs)
      if (failed(runHoistCastPairsStep(func, clHoistCastPairsMaxIterations,
                                       rewriter)))
        return signalPassFailure();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulCodegenVecPrepPass() {
  return std::make_unique<AIRMatmulCodegenVecPrep>();
}

std::unique_ptr<mlir::Pass> createAIRMatmulCodegenVecPrepPass(
    const AIRMatmulCodegenVecPrepOptions &opts) {
  return std::make_unique<AIRMatmulCodegenVecPrep>(opts);
}

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
