//===- AIRMatmulVectorizePasses.cpp ----------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Vectorization-prep phases of the air-matmul-codegen orchestrator:
// tile-for-vectorize and the vec-prep composite. Each free function walks
// a func::FuncOp and dispatches to a runFoo helper in
// AIRMatmulCodegenHelpers; the helpers are shared with the corresponding
// transform.air.* op apply() in AIRLinalgCodegen.cpp.
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRMatmulVectorizePasses.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/AIRMatmulBufferizationPasses.h"
#include "air/Transform/AIRMatmulCodegenHelpers.h"
#include "air/Transform/PassDetail.h"

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

// Collect every scf.for that lives inside an air.herd in `func` and has no
// further scf.for in its subtree. Optional `herdFilter` skips entire herds.
static SmallVector<mlir::scf::ForOp>
findInnermostForsInHerds(func::FuncOp func,
                         function_ref<bool(HerdOp)> herdFilter = nullptr) {
  SmallVector<mlir::scf::ForOp> innermost;
  func.walk([&](HerdOp herd) {
    if (herdFilter && !herdFilter(herd))
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
  return innermost;
}

// Per-step bodies. Extracted from the previously-individual AIR passes; now
// invoked in fixed order from runCodegenVecPrepImpl below.

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
  for (mlir::scf::ForOp loopOp : findInnermostForsInHerds(func)) {
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
  for (mlir::scf::ForOp forOp :
       findInnermostForsInHerds(func, herdHasVectorContract)) {
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
                         mlir::Operation *&truncOp, mlir::scf::ForOp &loopOp) {
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
        if (auto sc =
                dyn_cast_if_present<mlir::vector::ShapeCastOp>(foundTrunc))
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

} // namespace

LogicalResult runCodegenVecPrepImpl(
    func::FuncOp func, StringRef cast1TargetElementType,
    ArrayRef<int64_t> cast1InputIndices, ArrayRef<int64_t> cast1OutputIndices,
    StringRef cast2TargetElementType, ArrayRef<int64_t> cast2InputIndices,
    ArrayRef<int64_t> cast2OutputIndices, bool doHoistCastPairs,
    int64_t hoistCastPairsMaxIterations, RewriterBase &rewriter) {
  IRRewriter &irRewriter = static_cast<IRRewriter &>(rewriter);

  if (failed(runFoldUnitExtentDimsOnFunc(func)))
    return failure();
  (void)runEliminateRedundantVectorTransfers(func, irRewriter);
  if (failed(runVectorCastForEmulationStep(func, cast1TargetElementType,
                                           cast1InputIndices,
                                           cast1OutputIndices, irRewriter)))
    return failure();
  if (failed(runVectorCastForEmulationStep(func, cast2TargetElementType,
                                           cast2InputIndices,
                                           cast2OutputIndices, irRewriter)))
    return failure();
  if (failed(runHoistLoopInvariantTransfersStep(func, irRewriter)))
    return failure();
  if (failed(runFlattenForIterArgsStep(func, irRewriter)))
    return failure();
  if (failed(runHoistVectorTransferPointersStep(func, irRewriter)))
    return failure();
  if (doHoistCastPairs)
    if (failed(runHoistCastPairsStep(func, hoistCastPairsMaxIterations,
                                     irRewriter)))
      return failure();
  return success();
}

LogicalResult runTileForVectorizeImpl(func::FuncOp func,
                                      ArrayRef<int64_t> matmulTileSizes,
                                      ArrayRef<int64_t> matmulUnrollTileSizes,
                                      int64_t matmulUnrollFactor,
                                      ArrayRef<int64_t> fillTileSizes,
                                      bool doPostBufferizeCleanupFirst,
                                      RewriterBase &rewriter) {
  IRRewriter &irRewriter = static_cast<IRRewriter &>(rewriter);

  // Optional pre-step: post-bufferize cleanup (remove uninitialized
  // copies + eliminate cascade memcpys + sibling-fuse pingpong loops).
  // Replaces the former standalone `air-matmul-post-bufferize-cleanup`
  // pass.
  if (doPostBufferizeCleanupFirst)
    if (failed(runPostBufferizeCleanupImpl(func, rewriter)))
      return failure();

  // Phase 1: tile each linalg.generic packed-matmul body by matmulTileSizes.
  // Accept ops that either (a) live inside an air.herd (iron-built flow)
  // or (b) carry the `matmul_compute` marker (linalg-input flow runs this
  // pass BEFORE the forall->herd materialization).
  SmallVector<mlir::linalg::GenericOp> matmulGenerics;
  func.walk([&](mlir::linalg::GenericOp op) {
    bool inHerd = op->getParentOfType<xilinx::air::HerdOp>() != nullptr;
    bool isMatmulCompute = op->hasAttr("matmul_compute");
    if (!inHerd && !isMatmulCompute)
      return;
    if (op.getNumLoops() < (int64_t)matmulTileSizes.size())
      return;
    matmulGenerics.push_back(op);
  });
  for (mlir::linalg::GenericOp gen : matmulGenerics) {
    auto loops1 =
        tileWithScfFor(gen.getOperation(), matmulTileSizes, irRewriter);
    if (failed(loops1))
      return failure();
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
        tileWithScfFor(inner.getOperation(), matmulUnrollTileSizes, irRewriter);
    if (failed(loops2))
      return failure();
    // Unroll the two innermost produced loops.
    // loops2->back() is the innermost; loops2 is in outer→inner order.
    uint64_t factor = matmulUnrollFactor;
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
          return failure();
        }
      }
    }
  }

  // Phase 2: tile each linalg.fill (or linalg.generic carrying the
  // `init_fill` marker, set by the prologue-epilogue phase after
  // generalize+interchange) by fillTileSizes.
  SmallVector<mlir::Operation *> fills;
  func.walk([&](mlir::linalg::FillOp f) {
    if (f->getParentOfType<xilinx::air::HerdOp>())
      fills.push_back(f.getOperation());
  });
  func.walk([&](mlir::linalg::GenericOp g) {
    if (g->hasAttr("init_fill"))
      fills.push_back(g.getOperation());
  });
  for (mlir::Operation *f : fills) {
    auto loops = tileWithScfFor(f, fillTileSizes, irRewriter);
    if (failed(loops))
      return failure();
  }
  return success();
}

} // namespace air
} // namespace xilinx
