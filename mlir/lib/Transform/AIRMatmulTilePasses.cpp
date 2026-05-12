//===- AIRMatmulTilePasses.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Tiling phases of the air-matmul-codegen orchestrator: launch-tile,
// tile-k-and-fuse-packs, tile-cores, prologue-epilogue. Each tiles the
// (packed) matmul on a different axis and fuses its operand-producing
// pack ops into the new loop. Markers wired so downstream phases
// (bufferize-l1-inputs, fuse-pingpong-loops) can find their targets.
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRMatmulTilePasses.h"
#include "air/Transform/AIRMatmulBufferizationPasses.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/TilingInterface.h"

#include "llvm/ADT/StringRef.h"

#define DEBUG_TYPE "air-matmul-tile-passes"

using namespace mlir;

namespace xilinx {
namespace air {

namespace {

// `findMarkedOp` lives in air/Util/Util.h as `xilinx::air::findOpWithAttr`.

/// Build OpFoldResult-typed tile sizes (one per iterator dim) from int64s.
/// Pads with 0 if shorter than `numIters`; truncates if longer.
static SmallVector<OpFoldResult>
buildTileSizes(ArrayRef<int64_t> sizes, int64_t numIters, MLIRContext *ctx) {
  SmallVector<OpFoldResult> out;
  out.reserve(numIters);
  OpBuilder b(ctx);
  for (int64_t i = 0; i < numIters; ++i) {
    int64_t v = (i < (int64_t)sizes.size()) ? sizes[i] : 0;
    out.push_back(b.getIndexAttr(v));
  }
  return out;
}

/// Fuse a linalg.fill that lives just outside `forall` into the forall body
/// when its result feeds a `shared_outs` operand. After fusion the shared_outs
/// operand becomes the original fill destination (e.g. tensor.empty) and a
/// per-iter linalg.fill is cloned inside the body, before the consuming
/// linalg op, filling the corresponding extract_slice. Returns success when
/// the pattern matched and was fused.
static LogicalResult fuseFillIntoForallSharedOuts(linalg::FillOp fillOp,
                                                  scf::ForallOp forall,
                                                  RewriterBase &rewriter) {
  Value fillResult = fillOp.getResult(0);
  int64_t sharedOutIdx = -1;
  for (auto [idx, val] : llvm::enumerate(forall.getOutputs())) {
    if (val == fillResult) {
      sharedOutIdx = idx;
      break;
    }
  }
  if (sharedOutIdx < 0)
    return failure();

  BlockArgument blockArg = forall.getRegionIterArgs()[sharedOutIdx];
  Value fillDest = fillOp.getOutputs()[0]; // typically tensor.empty
  Value fillValue = fillOp.getInputs()[0];

  // Find consumer of the block arg (or extract_slice on it) inside the body
  // that should be re-initialized per-iter. Match a linalg op whose init
  // operand is an extract_slice on blockArg.
  linalg::LinalgOp consumer;
  tensor::ExtractSliceOp consumerSlice;
  forall.getBody()->walk([&](linalg::LinalgOp op) {
    if (op.getNumDpsInits() != 1)
      return WalkResult::advance();
    auto es = op.getDpsInits()[0].getDefiningOp<tensor::ExtractSliceOp>();
    if (!es || es.getSource() != blockArg)
      return WalkResult::advance();
    consumer = op;
    consumerSlice = es;
    return WalkResult::interrupt();
  });
  if (!consumer)
    return failure();

  // Re-source the shared_outs from the original empty (the fill destination).
  forall.getOutputsMutable()[sharedOutIdx].set(fillDest);

  // Clone a per-iter fill into the body, filling the extract_slice.
  rewriter.setInsertionPoint(consumer);
  auto newFill =
      linalg::FillOp::create(rewriter, fillOp.getLoc(), ValueRange{fillValue},
                             ValueRange{consumerSlice.getResult()});
  rewriter.modifyOpInPlace(consumer, [&]() {
    consumer.getDpsInitsMutable()[0].set(newFill.getResult(0));
  });

  // Erase the outside fill (its only use is the shared_outs slot we just
  // re-sourced, plus any tensor.empty chain — leave the empty for DCE).
  if (fillOp.getResult(0).use_empty())
    rewriter.eraseOp(fillOp);
  return success();
}

/// Fuse a producer LinalgOp's first tensor.extract_slice user inside `loop`
/// into the loop, returning the fused (tiled) op. This mirrors what
/// `transform.structured.fuse_into_containing_op` does for tensor producers.
static Operation *fuseProducerIntoLoop(Operation *producerOp,
                                       LoopLikeOpInterface loop,
                                       RewriterBase &rewriter) {
  if (!producerOp || !loop)
    return nullptr;
  ResultRange producerResults = producerOp->getResults();
  tensor::ExtractSliceOp slice;
  loop->walk([&](tensor::ExtractSliceOp s) {
    if (llvm::is_contained(producerResults, s.getSource())) {
      slice = s;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (!slice)
    return nullptr;
  SmallVector<LoopLikeOpInterface> loops{loop};
  auto res = scf::tileAndFuseProducerOfSlice(rewriter, slice, loops);
  if (!res || res->tiledOps.empty())
    return nullptr;
  return res->tiledOps.front();
}

/// Tile `target` with `LoopType::ForallOp` and pre-built `tileSizes`. Returns
/// the full `SCFTilingResult` on success; the original op is `replaceOp`d.
static FailureOr<scf::SCFTilingResult>
tileAsForallResult(Operation *target, ArrayRef<OpFoldResult> tileSizes,
                   RewriterBase &rewriter) {
  auto tileable = dyn_cast_if_present<TilingInterface>(target);
  if (!tileable)
    return failure();
  rewriter.setInsertionPoint(target);
  scf::SCFTilingOptions opts;
  opts.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  opts.setTileSizes(tileSizes);
  auto res = scf::tileUsingSCF(rewriter, tileable, opts);
  if (failed(res))
    return failure();
  rewriter.replaceOp(target, res->replacements);
  return res;
}

/// Convenience wrapper around `tileAsForallResult` for callers that only need
/// the new forall loop and accept padded raw int64_t tile sizes.
static LoopLikeOpInterface tileAsForall(Operation *target,
                                        ArrayRef<int64_t> tileSizes,
                                        RewriterBase &rewriter) {
  if (!target)
    return {};
  auto tileable = dyn_cast<TilingInterface>(target);
  if (!tileable)
    return {};
  auto folded = buildTileSizes(
      tileSizes, tileable.getLoopIteratorTypes().size(), target->getContext());
  auto res = tileAsForallResult(target, folded, rewriter);
  if (failed(res))
    return {};
  return res->loops.empty() ? LoopLikeOpInterface() : res->loops.front();
}

} // namespace

//===----------------------------------------------------------------------===//
// runTileKAndFusePacksImpl (Phase 4)
//===----------------------------------------------------------------------===//

LogicalResult runTileKAndFusePacksImpl(
    func::FuncOp f, int64_t kTileFactor, int64_t kIterIndex,
    StringRef packedMatmulMarker, StringRef kReductionLoopMarker,
    StringRef lhsPackMarker, StringRef rhsPackMarker, StringRef lhsL2PackMarker,
    StringRef rhsL2PackMarker, RewriterBase &rewriter) {
  Operation *packedMatmulOp =
      xilinx::air::findOpWithAttr(f, packedMatmulMarker);
  if (!packedMatmulOp)
    return success();
  auto matmul = dyn_cast<linalg::LinalgOp>(packedMatmulOp);
  if (!matmul) {
    packedMatmulOp->emitError("packed_matmul op must be a LinalgOp");
    return failure();
  }

  // Identify pack producers of operand 0 (LHS) and operand 1 (RHS) BEFORE
  // tiling — tiling rewrites the operands and would invalidate these.
  Operation *packA = matmul.getDpsInputs()[0].getDefiningOp();
  Operation *packB = matmul.getDpsInputs()[1].getDefiningOp();

  // Tile on the K iterator. Matmul iterators after pack: m0,n0,k0,m1,n1,k1
  // (3 outer + 3 inner) for standard pack [m,n,k]. K iterator index = 2.
  int64_t numIters = matmul.getNumLoops();
  SmallVector<int64_t> raw(numIters, 0);
  if (numIters < 3) {
    packedMatmulOp->emitError(
        "packed_matmul has fewer than 3 iterators; expected M, N, K");
    return failure();
  }
  if (kIterIndex < 0 || kIterIndex >= numIters) {
    packedMatmulOp->emitError("k-iter-index ")
        << kIterIndex << " out of range [0, " << numIters << ")";
    return failure();
  }
  raw[kIterIndex] = kTileFactor;
  auto tileSizes = buildTileSizes(raw, numIters, f.getContext());

  auto tileable = cast<TilingInterface>(packedMatmulOp);
  rewriter.setInsertionPoint(packedMatmulOp);
  scf::SCFTilingOptions opts;
  opts.setTileSizes(tileSizes);
  auto tilingResult = scf::tileUsingSCF(rewriter, tileable, opts);
  if (failed(tilingResult)) {
    packedMatmulOp->emitError("scf::tileUsingSCF on K failed");
    return failure();
  }
  rewriter.replaceOp(packedMatmulOp, tilingResult->replacements);

  if (tilingResult->loops.empty())
    return success(); // K tile of 0; nothing more to do.
  LoopLikeOpInterface kLoop = tilingResult->loops.front();
  kLoop->setAttr(kReductionLoopMarker, rewriter.getUnitAttr());

  // Fuse pack_a and pack_b into the K loop. Annotate. For two-pack-level
  // flows where the matmul's immediate operand pack (L1) has a grandparent
  // pack (L2) feeding it, recursively fuse the producer chain so the L2
  // pack ends up at K-loop scope too.
  auto fuseChain = [&](Operation *pack, StringRef l1Marker,
                       StringRef l2Marker) {
    bool producerHadL1Marker = pack && pack->hasAttr(l1Marker);
    Operation *fused = fuseProducerIntoLoop(pack, kLoop, rewriter);
    if (!fused)
      return;
    if (producerHadL1Marker && pack->getBlock())
      pack->removeAttr(l1Marker);
    fused->setAttr(l1Marker, rewriter.getUnitAttr());
    if (auto innerPack = dyn_cast<linalg::PackOp>(fused)) {
      Value src = innerPack.getSource();
      while (auto es = src.getDefiningOp<tensor::ExtractSliceOp>())
        src = es.getSource();
      if (auto gp = src.getDefiningOp<linalg::PackOp>()) {
        if (!kLoop->isProperAncestor(gp)) {
          if (Operation *l2Fused = fuseProducerIntoLoop(gp, kLoop, rewriter))
            l2Fused->setAttr(l2Marker, rewriter.getUnitAttr());
        }
      }
    }
  };
  fuseChain(packA, lhsPackMarker, lhsL2PackMarker);
  fuseChain(packB, rhsPackMarker, rhsL2PackMarker);
  return success();
}

//===----------------------------------------------------------------------===//
// runTileCoresImpl (Phase 5)
//===----------------------------------------------------------------------===//

LogicalResult
runTileCoresImpl(func::FuncOp f, ArrayRef<int64_t> tileSizes,
                 StringRef packedMatmulMarker, StringRef lhsPackInKMarker,
                 StringRef rhsPackInKMarker, StringRef computeForallMarker,
                 StringRef matmulComputeMarker, StringRef lhsL1PackMarker,
                 StringRef rhsL1PackMarker, RewriterBase &rewriter) {
  Operation *packedMatmulOp =
      xilinx::air::findOpWithAttr(f, packedMatmulMarker);
  if (!packedMatmulOp)
    return success();
  auto matmul = dyn_cast<linalg::LinalgOp>(packedMatmulOp);
  if (!matmul) {
    packedMatmulOp->emitError("packed_matmul op must be a LinalgOp");
    return failure();
  }

  auto folded = buildTileSizes(tileSizes, matmul.getNumLoops(), f.getContext());

  auto tilingResult = tileAsForallResult(packedMatmulOp, folded, rewriter);
  if (failed(tilingResult)) {
    packedMatmulOp->emitError("scf::tileUsingSCF (forall) failed");
    return failure();
  }

  if (tilingResult->loops.empty())
    return success();
  LoopLikeOpInterface forall = tilingResult->loops.front();
  forall->setAttr(computeForallMarker, rewriter.getUnitAttr());

  // Per-core matmul body: only one tiledOp expected.
  if (!tilingResult->tiledOps.empty())
    tilingResult->tiledOps.front()->setAttr(matmulComputeMarker,
                                            rewriter.getUnitAttr());

  // Fuse the K-loop-fused packs into the forall.
  Operation *lhsPack = xilinx::air::findOpWithAttr(f, lhsPackInKMarker);
  Operation *rhsPack = xilinx::air::findOpWithAttr(f, rhsPackInKMarker);
  if (Operation *fusedA = fuseProducerIntoLoop(lhsPack, forall, rewriter))
    fusedA->setAttr(lhsL1PackMarker, rewriter.getUnitAttr());
  if (Operation *fusedB = fuseProducerIntoLoop(rhsPack, forall, rewriter))
    fusedB->setAttr(rhsL1PackMarker, rewriter.getUnitAttr());
  return success();
}

//===----------------------------------------------------------------------===//
// runPrologueEpilogueImpl (Phase 6 prologue/epilogue)
//===----------------------------------------------------------------------===//

LogicalResult runPrologueEpilogueImpl(
    func::FuncOp f, ArrayRef<int64_t> prologueTileSizes,
    ArrayRef<int64_t> epilogueTileSizes,
    ArrayRef<int64_t> fillIteratorInterchange, StringRef initFillMarker,
    StringRef prologueForallMarker, StringRef epilogueForallMarker,
    bool hoistStaticAllocFirst, RewriterBase &rewriter) {
  // Optional pre-step: hoist statically-bound memref.alloc ops out of
  // nested loops to the function entry block. Used by two-pack-level flows
  // so the L1 acc alloc lives outside the K-reduction loop (K-peel flow).
  if (hoistStaticAllocFirst)
    runHoistStaticAllocImpl(f, rewriter);

  // ---- Prologue: generalize+interchange+tile the linalg.fill ----
  // The prologue must execute BEFORE the compute work. Find the compute
  // forall (or its ancestor scf.for) and move the fill in front of it
  // before generalizing/tiling so the resulting prologue forall lands at
  // the correct position.
  linalg::FillOp fill;
  f.walk([&](linalg::FillOp op) {
    fill = op;
    return WalkResult::interrupt();
  });
  if (fill) {
    Operation *anchor = nullptr;
    f.walk([&](scf::ForOp forOp) {
      if (forOp->hasAttr("k_reduction_loop")) {
        anchor = forOp.getOperation();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!anchor) {
      f.walk([&](scf::ForallOp forallOp) {
        if (forallOp->hasAttr("compute_forall")) {
          anchor = forallOp.getOperation();
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    }
    if (anchor) {
      Block *fillBlock = fill->getBlock();
      while (anchor && anchor->getBlock() != fillBlock)
        anchor = anchor->getParentOp();
      if (anchor && !fill->isBeforeInBlock(anchor))
        fill->moveBefore(anchor);
    }
    rewriter.setInsertionPoint(fill);
    FailureOr<linalg::GenericOp> generic =
        linalg::generalizeNamedOp(rewriter, fill);
    if (failed(generic)) {
      fill->emitError("generalizeNamedOp failed");
      return failure();
    }
    generic->getOperation()->setAttr(initFillMarker, rewriter.getUnitAttr());

    Operation *fillTileTarget = generic->getOperation();
    // Interchange iterators if a non-empty perm was provided.
    if (!fillIteratorInterchange.empty()) {
      SmallVector<unsigned> permUnsigned(fillIteratorInterchange.begin(),
                                         fillIteratorInterchange.end());
      FailureOr<linalg::GenericOp> interchanged =
          linalg::interchangeGenericOp(rewriter, *generic, permUnsigned);
      if (failed(interchanged)) {
        generic->getOperation()->emitError("interchangeGenericOp failed");
        return failure();
      }
      // Re-stamp the marker on the new op.
      interchanged->getOperation()->setAttr(initFillMarker,
                                            rewriter.getUnitAttr());
      fillTileTarget = interchanged->getOperation();
    }

    LoopLikeOpInterface prologueForall =
        tileAsForall(fillTileTarget, prologueTileSizes, rewriter);
    if (prologueForall)
      prologueForall->setAttr(prologueForallMarker, rewriter.getUnitAttr());
  }

  // ---- Epilogue: tile the linalg.unpack ----
  linalg::UnPackOp unpack;
  f.walk([&](linalg::UnPackOp op) {
    unpack = op;
    return WalkResult::interrupt();
  });
  if (unpack) {
    LoopLikeOpInterface epilogueForall =
        tileAsForall(unpack, epilogueTileSizes, rewriter);
    if (epilogueForall)
      epilogueForall->setAttr(epilogueForallMarker, rewriter.getUnitAttr());
  }
  return success();
}

//===----------------------------------------------------------------------===//
// runTileLaunchTileImpl
//===----------------------------------------------------------------------===//

LogicalResult runTileLaunchTileImpl(func::FuncOp f, ArrayRef<int64_t> tileSizes,
                                    StringRef launchTileForallMarker,
                                    RewriterBase &rewriter) {
  linalg::MatmulOp matmul;
  f.walk([&](linalg::MatmulOp op) {
    matmul = op;
    return WalkResult::interrupt();
  });
  if (!matmul)
    return success();

  auto folded = buildTileSizes(tileSizes,
                               cast<TilingInterface>(matmul.getOperation())
                                   .getLoopIteratorTypes()
                                   .size(),
                               f.getContext());

  // Capture the linalg.fill producer of the matmul's accumulator BEFORE
  // tiling (after which the matmul is rewritten and producer linkage may
  // shift through extract_slice).
  Operation *fillProducer =
      matmul.getOutputs()[0].getDefiningOp<linalg::FillOp>();

  auto tilingResult =
      tileAsForallResult(matmul.getOperation(), folded, rewriter);
  if (failed(tilingResult)) {
    matmul->emitError("scf::tileUsingSCF (forall) on launch-tile failed");
    return failure();
  }

  if (tilingResult->loops.empty())
    return success();
  LoopLikeOpInterface forall = tilingResult->loops.front();
  forall->setAttr(launchTileForallMarker, rewriter.getUnitAttr());

  // Tag the inner (per-launch-tile) matmul with `matmul_compute` so that
  // downstream tile-for-vectorize (which only matches inHerd ops or
  // `matmul_compute`-tagged ops) can find it in launch-tile-only flows
  // where there is no separate tile-cores step. The marker is preserved
  // by linalg::pack (which copies discardable attrs).
  if (!tilingResult->tiledOps.empty())
    tilingResult->tiledOps.front()->setAttr("matmul_compute",
                                            rewriter.getUnitAttr());

  if (fillProducer) {
    auto fillOp = dyn_cast<linalg::FillOp>(fillProducer);
    auto forallOp = dyn_cast<scf::ForallOp>(forall.getOperation());
    if (fillOp && forallOp)
      (void)fuseFillIntoForallSharedOuts(fillOp, forallOp, rewriter);
  }
  return success();
}

} // namespace air
} // namespace xilinx
