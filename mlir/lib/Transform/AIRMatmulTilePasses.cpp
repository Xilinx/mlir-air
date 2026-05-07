//===- AIRMatmulTilePasses.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// M2 Phase 4 / Phase 5 passes. Each tiles the packed matmul (on K, then on
// the per-core forall) and fuses the LHS/RHS L1 pack producers into the new
// loop. Markers wired so downstream passes (bufferize-l1-inputs,
// fuse-pingpong-loops) can find their targets.
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRMatmulTilePasses.h"
#include "air/Util/MatmulCodegenConfig.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/StringRef.h"

#define DEBUG_TYPE "air-matmul-tile-passes"

using namespace mlir;

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

/// Parse a comma-separated list of integers (e.g. "8,4,0") into a vector.
static SmallVector<int64_t> parseIntList(StringRef s) {
  SmallVector<int64_t> out;
  SmallVector<StringRef> tokens;
  s.split(tokens, ',');
  for (StringRef t : tokens) {
    t = t.trim();
    if (t.empty())
      continue;
    int64_t v = 0;
    if (!t.getAsInteger(10, v))
      out.push_back(v);
  }
  return out;
}

/// Build OpFoldResult-typed tile sizes (one per iterator dim) from int64s.
/// Pads with 0 if shorter than `numIters`; truncates if longer.
static SmallVector<OpFoldResult> buildTileSizes(ArrayRef<int64_t> sizes,
                                                int64_t numIters,
                                                MLIRContext *ctx) {
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
    auto es =
        op.getDpsInits()[0].getDefiningOp<tensor::ExtractSliceOp>();
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
  auto newFill = linalg::FillOp::create(rewriter, fillOp.getLoc(),
                                        ValueRange{fillValue},
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

} // namespace

//===----------------------------------------------------------------------===//
// AIRMatmulTileKAndFusePacks (Phase 4)
//===----------------------------------------------------------------------===//

namespace {
class AIRMatmulTileKAndFusePacks
    : public impl::AIRMatmulTileKAndFusePacksBase<AIRMatmulTileKAndFusePacks> {
public:
  AIRMatmulTileKAndFusePacks() = default;
  AIRMatmulTileKAndFusePacks(const AIRMatmulTileKAndFusePacksOptions &opts)
      : AIRMatmulTileKAndFusePacksBase(opts) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    Operation *packedMatmulOp = findMarkedOp(f, clPackedMatmulMarker);
    if (!packedMatmulOp)
      return;
    auto matmul = dyn_cast<linalg::LinalgOp>(packedMatmulOp);
    if (!matmul) {
      packedMatmulOp->emitError("packed_matmul op must be a LinalgOp");
      return signalPassFailure();
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
      return signalPassFailure();
    }
    int64_t kTileFactor = clKTileFactor;
    if (auto cfg = xilinx::air::findMatmulCodegenConfig(f))
      kTileFactor = xilinx::air::getI64(*cfg, "tile_k_factor", kTileFactor);
    int64_t kIdx = std::min<int64_t>(clKIterIndex, numIters - 1);
    raw[kIdx] = kTileFactor;
    auto tileSizes = buildTileSizes(raw, numIters, &getContext());

    auto tileable = cast<TilingInterface>(packedMatmulOp);
    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPoint(packedMatmulOp);
    scf::SCFTilingOptions opts;
    opts.setTileSizes(tileSizes);
    auto tilingResult = scf::tileUsingSCF(rewriter, tileable, opts);
    if (failed(tilingResult)) {
      packedMatmulOp->emitError("scf::tileUsingSCF on K failed");
      return signalPassFailure();
    }
    rewriter.replaceOp(packedMatmulOp, tilingResult->replacements);

    if (tilingResult->loops.empty())
      return; // K tile of 0; nothing more to do.
    LoopLikeOpInterface kLoop = tilingResult->loops.front();
    kLoop->setAttr(clKReductionLoopMarker, rewriter.getUnitAttr());

    // The marker on the matmul body is preserved by tileUsingSCF (it clones
    // ops and their attributes). Re-find the new packed matmul as a sanity
    // check; if missing, downstream passes will no-op correctly.

    // Fuse pack_a and pack_b into the K loop. Annotate. For M4 two-pack-
    // level flows where the matmul's immediate operand pack (L1) has a
    // grandparent pack (L2) feeding it, recursively fuse the producer
    // chain so the L2 pack ends up at K-loop scope too (matching the
    // legacy script's "fuse 4 packs into K-loop" pattern).
    auto fuseChain = [&](Operation *pack, StringRef l1Marker,
                         StringRef l2Marker) {
      // If the producer already carries `l1Marker` from a previous phase
      // (e.g. tile-cores set `fused_lhs_l1_pack` on the cores-scope pack
      // before this inner tile-k fuses it again), strip that marker first
      // so the post-fusion `setAttr` doesn't leave both producer and fused
      // copy claiming to be the live one — bufferize-l1-inputs would then
      // pick the orphan and canonicalize would DCE its L1 alloc.
      bool producerHadL1Marker = pack && pack->hasAttr(l1Marker);
      Operation *fused = fuseProducerIntoLoop(pack, kLoop, rewriter);
      if (!fused)
        return;
      if (producerHadL1Marker && pack->getBlock())
        pack->removeAttr(l1Marker);
      fused->setAttr(l1Marker, rewriter.getUnitAttr());
      // If the inner (just-fused) pack's source is another linalg.pack
      // outside the loop, fuse THAT too and mark it with l2Marker. After
      // fusion the source is typically `tensor.extract_slice(L2 pack)`,
      // so walk through extract_slice ops to reach the grandparent.
      if (auto innerPack = dyn_cast<linalg::PackOp>(fused)) {
        Value src = innerPack.getSource();
        while (auto es = src.getDefiningOp<tensor::ExtractSliceOp>())
          src = es.getSource();
        if (auto gp = src.getDefiningOp<linalg::PackOp>()) {
          if (!kLoop->isProperAncestor(gp)) {
            if (Operation *l2Fused =
                    fuseProducerIntoLoop(gp, kLoop, rewriter))
              l2Fused->setAttr(l2Marker, rewriter.getUnitAttr());
          }
        }
      }
    };
    fuseChain(packA, clLhsPackMarker, clLhsL2PackMarker);
    fuseChain(packB, clRhsPackMarker, clRhsL2PackMarker);
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulTileKAndFusePacksPass() {
  return std::make_unique<AIRMatmulTileKAndFusePacks>();
}
std::unique_ptr<mlir::Pass> createAIRMatmulTileKAndFusePacksPass(
    const AIRMatmulTileKAndFusePacksOptions &opts) {
  return std::make_unique<AIRMatmulTileKAndFusePacks>(opts);
}

//===----------------------------------------------------------------------===//
// AIRMatmulTileCores (Phase 5)
//===----------------------------------------------------------------------===//

namespace {
class AIRMatmulTileCores
    : public impl::AIRMatmulTileCoresBase<AIRMatmulTileCores> {
public:
  AIRMatmulTileCores() = default;
  AIRMatmulTileCores(const AIRMatmulTileCoresOptions &opts)
      : AIRMatmulTileCoresBase(opts) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    Operation *packedMatmulOp = findMarkedOp(f, clPackedMatmulMarker);
    if (!packedMatmulOp)
      return;
    auto matmul = dyn_cast<linalg::LinalgOp>(packedMatmulOp);
    if (!matmul) {
      packedMatmulOp->emitError("packed_matmul op must be a LinalgOp");
      return signalPassFailure();
    }

    SmallVector<int64_t> rawSizes = parseIntList(clTileSizes);
    if (auto cfg = xilinx::air::findMatmulCodegenConfig(f)) {
      auto v = xilinx::air::getI64Array(*cfg, "tile_cores");
      if (!v.empty())
        rawSizes = std::move(v);
    }
    auto tileSizes =
        buildTileSizes(rawSizes, matmul.getNumLoops(), &getContext());

    auto tileable = cast<TilingInterface>(packedMatmulOp);
    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPoint(packedMatmulOp);
    scf::SCFTilingOptions opts;
    opts.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
    opts.setTileSizes(tileSizes);
    auto tilingResult = scf::tileUsingSCF(rewriter, tileable, opts);
    if (failed(tilingResult)) {
      packedMatmulOp->emitError("scf::tileUsingSCF (forall) failed");
      return signalPassFailure();
    }
    rewriter.replaceOp(packedMatmulOp, tilingResult->replacements);

    if (tilingResult->loops.empty())
      return;
    LoopLikeOpInterface forall = tilingResult->loops.front();
    forall->setAttr(clComputeForallMarker, rewriter.getUnitAttr());

    // Per-core matmul body: only one tiledOp expected.
    if (!tilingResult->tiledOps.empty())
      tilingResult->tiledOps.front()->setAttr(clMatmulComputeMarker,
                                              rewriter.getUnitAttr());

    // Fuse the K-loop-fused packs into the forall.
    Operation *lhsPack = findMarkedOp(f, clLhsPackInKMarker);
    Operation *rhsPack = findMarkedOp(f, clRhsPackInKMarker);
    if (Operation *fusedA = fuseProducerIntoLoop(lhsPack, forall, rewriter))
      fusedA->setAttr(clLhsL1PackMarker, rewriter.getUnitAttr());
    if (Operation *fusedB = fuseProducerIntoLoop(rhsPack, forall, rewriter))
      fusedB->setAttr(clRhsL1PackMarker, rewriter.getUnitAttr());
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulTileCoresPass() {
  return std::make_unique<AIRMatmulTileCores>();
}
std::unique_ptr<mlir::Pass>
createAIRMatmulTileCoresPass(const AIRMatmulTileCoresOptions &opts) {
  return std::make_unique<AIRMatmulTileCores>(opts);
}

//===----------------------------------------------------------------------===//
// AIRMatmulPrologueEpilogue (Phase 6 prologue/epilogue)
//===----------------------------------------------------------------------===//

namespace {
/// Tile `target` (which must implement TilingInterface) with `LoopType::ForallOp`
/// and `tileSizes`. Returns the new forall loop on success.
static LoopLikeOpInterface tileAsForall(Operation *target,
                                        ArrayRef<int64_t> tileSizes,
                                        RewriterBase &rewriter) {
  if (!target)
    return {};
  auto tileable = dyn_cast<TilingInterface>(target);
  if (!tileable)
    return {};
  auto numIters = tileable.getLoopIteratorTypes().size();
  auto folded = buildTileSizes(tileSizes, numIters, target->getContext());
  rewriter.setInsertionPoint(target);
  scf::SCFTilingOptions opts;
  opts.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
  opts.setTileSizes(folded);
  auto res = scf::tileUsingSCF(rewriter, tileable, opts);
  if (failed(res))
    return {};
  rewriter.replaceOp(target, res->replacements);
  return res->loops.empty() ? LoopLikeOpInterface() : res->loops.front();
}

class AIRMatmulPrologueEpilogue
    : public impl::AIRMatmulPrologueEpilogueBase<AIRMatmulPrologueEpilogue> {
public:
  AIRMatmulPrologueEpilogue() = default;
  AIRMatmulPrologueEpilogue(const AIRMatmulPrologueEpilogueOptions &opts)
      : AIRMatmulPrologueEpilogueBase(opts) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<int64_t> prologueTile = parseIntList(clPrologueTileSizes);
    SmallVector<int64_t> epilogueTile = parseIntList(clEpilogueTileSizes);
    SmallVector<int64_t> fillIterPerm = parseIntList(clFillIteratorInterchange);
    if (auto cfg = xilinx::air::findMatmulCodegenConfig(f)) {
      auto take = [&](StringRef key, SmallVector<int64_t> &dst) {
        auto v = xilinx::air::getI64Array(*cfg, key);
        if (!v.empty())
          dst = std::move(v);
      };
      take("prologue_tile", prologueTile);
      take("epilogue_tile", epilogueTile);
      take("fill_iter_perm", fillIterPerm);
    }

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
      // Find the K-reduction scf.for (set by Phase 4 tile-k-and-fuse-packs)
      // or, failing that, the compute_forall scf.forall (set by Phase 5).
      // Walk up to the same block as the fill and move the fill in front
      // of that ancestor so the resulting prologue lands BEFORE compute.
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
        return signalPassFailure();
      }
      generic->getOperation()->setAttr(clInitFillMarker,
                                       rewriter.getUnitAttr());

      Operation *fillTileTarget = generic->getOperation();
      // Interchange iterators if a non-empty perm was provided.
      if (!fillIterPerm.empty()) {
        SmallVector<unsigned> permUnsigned(fillIterPerm.begin(),
                                           fillIterPerm.end());
        FailureOr<linalg::GenericOp> interchanged =
            linalg::interchangeGenericOp(rewriter, *generic, permUnsigned);
        if (failed(interchanged)) {
          generic->getOperation()->emitError("interchangeGenericOp failed");
          return signalPassFailure();
        }
        // Re-stamp the marker on the new op.
        interchanged->getOperation()->setAttr(clInitFillMarker,
                                              rewriter.getUnitAttr());
        fillTileTarget = interchanged->getOperation();
      }

      LoopLikeOpInterface prologueForall =
          tileAsForall(fillTileTarget, prologueTile, rewriter);
      if (prologueForall)
        prologueForall->setAttr(clPrologueForallMarker, rewriter.getUnitAttr());
    }

    // ---- Epilogue: tile the linalg.unpack ----
    linalg::UnPackOp unpack;
    f.walk([&](linalg::UnPackOp op) {
      unpack = op;
      return WalkResult::interrupt();
    });
    if (unpack) {
      LoopLikeOpInterface epilogueForall =
          tileAsForall(unpack, epilogueTile, rewriter);
      if (epilogueForall)
        epilogueForall->setAttr(clEpilogueForallMarker, rewriter.getUnitAttr());
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulPrologueEpiloguePass() {
  return std::make_unique<AIRMatmulPrologueEpilogue>();
}
std::unique_ptr<mlir::Pass> createAIRMatmulPrologueEpiloguePass(
    const AIRMatmulPrologueEpilogueOptions &opts) {
  return std::make_unique<AIRMatmulPrologueEpilogue>(opts);
}

//===----------------------------------------------------------------------===//
// AIRMatmulSetCodegenConfig (M3a heuristic)
//===----------------------------------------------------------------------===//

namespace {

/// Element-type category. Used by the heuristic lookup table.
enum class ElemKind { Bf16, F32, I8, I16, I32, Other };

static ElemKind classify(Type t) {
  if (t.isBF16())
    return ElemKind::Bf16;
  if (t.isF32())
    return ElemKind::F32;
  if (auto i = dyn_cast<IntegerType>(t)) {
    switch (i.getWidth()) {
    case 8:
      return ElemKind::I8;
    case 16:
      return ElemKind::I16;
    case 32:
      return ElemKind::I32;
    default:
      return ElemKind::Other;
    }
  }
  return ElemKind::Other;
}

class AIRMatmulSetCodegenConfig
    : public impl::AIRMatmulSetCodegenConfigBase<AIRMatmulSetCodegenConfig> {
public:
  AIRMatmulSetCodegenConfig() = default;
  AIRMatmulSetCodegenConfig(const AIRMatmulSetCodegenConfigOptions &opts)
      : AIRMatmulSetCodegenConfigBase(opts) {}

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    MLIRContext *ctx = &getContext();
    Builder b(ctx);

    // Locate the first linalg.matmul.
    linalg::MatmulOp matmul;
    f.walk([&](linalg::MatmulOp op) {
      matmul = op;
      return WalkResult::interrupt();
    });
    if (!matmul)
      return;

    auto lhsTy = cast<RankedTensorType>(matmul.getInputs()[0].getType());
    auto rhsTy = cast<RankedTensorType>(matmul.getInputs()[1].getType());
    auto outTy = cast<RankedTensorType>(matmul.getOutputs()[0].getType());
    ElemKind inK = classify(lhsTy.getElementType());
    ElemKind accK = classify(outTy.getElementType());
    // The "effective" output type after any downstream truncf-only consumer:
    // bf16-out is detected by walking the matmul's consumers for a
    // linalg.generic whose body contains only arith.truncf (the test-53
    // pattern). If found and its output is bf16, the codegen flow follows
    // the bf16-output path even though the matmul itself accumulates in f32.
    Type effOutEltTy = outTy.getElementType();
    for (Operation *user : matmul->getUsers()) {
      auto g = dyn_cast<linalg::GenericOp>(user);
      if (!g)
        continue;
      bool onlyTruncf = false;
      Block *body = g.getBody();
      if (body && std::distance(body->begin(), body->end()) == 2) {
        Operation &op0 = body->front();
        if (isa<arith::TruncFOp>(op0))
          onlyTruncf = true;
      }
      if (!onlyTruncf)
        continue;
      auto outT = dyn_cast<RankedTensorType>(g.getDpsInits()[0].getType());
      if (!outT || !outT.getElementType().isBF16())
        continue;
      effOutEltTy = outT.getElementType();
      break;
    }
    bool bf16Out = effOutEltTy.isBF16();

    StringRef target(clTargetDevice);
    bool isAie2p = target.equals_insensitive("aie2p");

    // --- Pack sizes from device + element types -----------------------
    // AIE2 bf16/f32 -> [4,8,4]; AIE2P -> [8,8,8] for all dtypes we cover.
    SmallVector<int64_t, 3> packSizes = {8, 8, 8};
    if (!isAie2p && (inK == ElemKind::Bf16 || inK == ElemKind::F32))
      packSizes = {4, 8, 4};

    // --- Per-operand pack transpose perms (constant across modes) -----
    SmallVector<int64_t, 2> p10 = {1, 0};
    SmallVector<int64_t, 2> p01 = {0, 1};

    // --- L2 K tile + K-loop tile factor ------------------------------
    // Preferred: 64 for narrow types (bf16/i8), 16 for f32. Halve until it
    // both divides K and is a multiple of packK (= 8). Floor at packK.
    int64_t shapeK = lhsTy.getShape()[1];
    int64_t packK = packSizes[2];
    int64_t tileL3L2K = clTileL3L2K;
    if (tileL3L2K == 0) {
      int64_t preferred = (inK == ElemKind::F32) ? 16 : 64;
      tileL3L2K = preferred;
      while (tileL3L2K > packK &&
             (shapeK % tileL3L2K != 0 || tileL3L2K % packK != 0))
        tileL3L2K /= 2;
      if (tileL3L2K < packK)
        tileL3L2K = packK;
    }
    int64_t tileKFactor = std::max<int64_t>(1, tileL3L2K / packK);

    // --- Per-core (compute forall) tile sizes ------------------------
    // After pack with outer_perm=[1,0], packed iter space is
    // [N/packN, M/packM, K/packK, packM, packN, packK]. tile_using_forall
    // with [t0, t1, 0] produces forall(packedN/t0, packedM/t1) outer
    // iterations, which become air.herd cores.
    //
    // M3a/M3b: empirical lookup based on (target, in/out elt-type) plus an
    // L1-fit guardrail. The lookup matches the hand-tuned tests 53/54
    // values; the guardrail halves coreTile1 (then coreTile0) when the
    // chosen tile would overflow per-tile L1. A fully derivation-driven
    // heuristic would require modelling the downstream `air-collapse-herd`
    // remap; left for a future M3c.
    int64_t shapeM = lhsTy.getShape()[0];
    int64_t shapeN = rhsTy.getShape()[1];
    int64_t packedM = shapeM / packSizes[0];
    int64_t packedN = shapeN / packSizes[1];
    int64_t coreTile0, coreTile1; // tile sizes for the outer two dims.
    if (isAie2p && bf16Out) {
      // Test 53 profile: bf16-in/bf16-out, 4×2 herd, square per-core mmul.
      coreTile0 = 8;
      coreTile1 = 8;
    } else if (isAie2p && inK == ElemKind::F32) {
      // Test 54 profile: f32-in/out + BFP16 emul, 4×4 herd via collapse.
      coreTile0 = 8;
      coreTile1 = 4;
    } else {
      // Generic fallback: map matmul tile to ~16 forall cores total.
      int64_t targetCores = std::max<int64_t>(1, clHerdM * clHerdN);
      coreTile0 = std::max<int64_t>(1, packedN * packedM / targetCores / 4);
      coreTile1 = 4;
    }
    coreTile0 = std::min(coreTile0, packedN);
    coreTile1 = std::min(coreTile1, packedM);

    // L1-fit guardrail: halve coreTile1 (M dim) then coreTile0 (N dim)
    // until per-core L1 footprint is below the AIE tile budget.
    auto bytesOf = [](Type t) -> int64_t {
      return std::max<int64_t>(1, t.getIntOrFloatBitWidth() / 8);
    };
    int64_t bytesIn = bytesOf(lhsTy.getElementType());
    int64_t bytesAcc = bytesOf(effOutEltTy);
    auto l1FitBytes = [&](int64_t t0, int64_t t1) -> int64_t {
      int64_t lhs = t1 * packSizes[0] * tileKFactor * packK * bytesIn;
      int64_t rhs = t0 * packSizes[1] * tileKFactor * packK * bytesIn;
      int64_t acc = t0 * t1 * packSizes[0] * packSizes[1] * bytesAcc;
      return lhs + rhs + acc;
    };
    constexpr int64_t kL1BudgetBytes = 64 * 1024; // 64KB AIE tile L1.
    while (l1FitBytes(coreTile0, coreTile1) > kL1BudgetBytes &&
           coreTile1 > 1)
      coreTile1 /= 2;
    while (l1FitBytes(coreTile0, coreTile1) > kL1BudgetBytes &&
           coreTile0 > 1)
      coreTile0 /= 2;

    SmallVector<int64_t, 3> tileCores = {coreTile0, coreTile1, 0};

    // --- Prologue (fill) tile (matches tile_cores per dim) -----------
    SmallVector<int64_t, 2> prologueTile = {coreTile0, coreTile1};
    SmallVector<int64_t, 4> fillIterPerm = {1, 0, 2, 3};

    // --- Epilogue (unpack) tile --------------------------------------
    // Unpack iter is (M, N). Empirically matches both tests' hand-tuned
    // values:
    //   epM = max(coreTile1 × packM, M / herdM_user)
    //   epN = N / herdN_user
    // The max() handles the case where the per-core natural M-row span
    // (= coreTile1 × packM) exceeds M/herdM; this happens for tests where
    // the matmul shape forces fewer compute cores than the requested herd
    // (e.g. test 53 ends up with 8 compute cores in a 4×2 layout despite
    // herd-m=herd-n=4 being passed). For such cases the unpack still tiles
    // M by the per-core span so the resulting forall iter count matches
    // compute's actual core count.
    int64_t herdM = std::max<int64_t>(1, clHerdM);
    int64_t herdN = std::max<int64_t>(1, clHerdN);
    int64_t epM = std::max<int64_t>(coreTile1 * packSizes[0],
                                    shapeM / herdM);
    int64_t epN = std::max<int64_t>(1, shapeN / herdN);
    SmallVector<int64_t, 2> epilogueTile = {epM, epN};

    // --- Vectorize tiles (constant across tests so far) ---------------
    SmallVector<int64_t, 6> vectorTile = {2, 2, 1, 0, 0, 0};
    SmallVector<int64_t, 6> vectorUnrollTile = {1, 1, 0, 0, 0, 0};
    int64_t vectorUnrollFactor = 2;
    SmallVector<int64_t, 4> fillVectorTile = {1, 1, 0, 0};

    // --- Mode flags ---------------------------------------------------
    // f32 in + AIE2P + bfp16-emulation requested -> BFP16 mmul emulation
    // (test 54).
    bool bfp16Emul =
        clBfp16Emulation && isAie2p && (inK == ElemKind::F32);
    // bf16 out + f32 acc -> truncf-fuse + hoist-cast-pairs (test 53).
    bool fuseTruncf = bf16Out && (accK == ElemKind::F32);
    // For test 53, the output op is bf16 but the inner matmul accumulates
    // in f32 via the truncf-fused matmul body — same flag covers both.
    bool hoistCastPairs = bf16Out;
    bool threeHerd = clThreeHerd;

    // --- Build dictionary --------------------------------------------
    auto i64Attr = [&](int64_t v) { return b.getI64IntegerAttr(v); };
    auto i64Arr = [&](ArrayRef<int64_t> a) {
      SmallVector<int64_t> v(a);
      return b.getI64ArrayAttr(v);
    };
    auto boolAttr = [&](bool v) { return b.getBoolAttr(v); };

    SmallVector<NamedAttribute> entries = {
        b.getNamedAttr("pack_sizes", i64Arr(packSizes)),
        b.getNamedAttr("lhs_outer_perm", i64Arr(p10)),
        b.getNamedAttr("lhs_inner_perm", i64Arr(p01)),
        b.getNamedAttr("rhs_outer_perm", i64Arr(p10)),
        b.getNamedAttr("rhs_inner_perm", i64Arr(p10)),
        b.getNamedAttr("acc_outer_perm", i64Arr(p10)),
        b.getNamedAttr("acc_inner_perm", i64Arr(p01)),
        b.getNamedAttr("tile_l3_l2_k", i64Attr(tileL3L2K)),
        b.getNamedAttr("tile_k_factor", i64Attr(tileKFactor)),
        b.getNamedAttr("tile_cores", i64Arr(tileCores)),
        b.getNamedAttr("prologue_tile", i64Arr(prologueTile)),
        b.getNamedAttr("epilogue_tile", i64Arr(epilogueTile)),
        b.getNamedAttr("fill_iter_perm", i64Arr(fillIterPerm)),
        b.getNamedAttr("vector_tile", i64Arr(vectorTile)),
        b.getNamedAttr("vector_unroll_tile", i64Arr(vectorUnrollTile)),
        b.getNamedAttr("vector_unroll_factor", i64Attr(vectorUnrollFactor)),
        b.getNamedAttr("fill_vector_tile", i64Arr(fillVectorTile)),
        b.getNamedAttr("bfp16_emulation", boolAttr(bfp16Emul)),
        b.getNamedAttr("fuse_output_truncf", boolAttr(fuseTruncf)),
        b.getNamedAttr("bf16_output_hoist_pairs", boolAttr(hoistCastPairs)),
        b.getNamedAttr("three_herd_prologue_epilogue", boolAttr(threeHerd)),
    };
    auto dict = buildMatmulCodegenConfig(ctx, entries);
    matmul->setAttr(getMatmulCodegenConfigAttrName(), dict);
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulSetCodegenConfigPass() {
  return std::make_unique<AIRMatmulSetCodegenConfig>();
}
std::unique_ptr<mlir::Pass> createAIRMatmulSetCodegenConfigPass(
    const AIRMatmulSetCodegenConfigOptions &opts) {
  return std::make_unique<AIRMatmulSetCodegenConfig>(opts);
}

//===----------------------------------------------------------------------===//
// AIRMatmulTileLaunchTile (M4 Phase 0)
//===----------------------------------------------------------------------===//

namespace {
class AIRMatmulTileLaunchTile
    : public impl::AIRMatmulTileLaunchTileBase<AIRMatmulTileLaunchTile> {
public:
  AIRMatmulTileLaunchTile() = default;
  AIRMatmulTileLaunchTile(const AIRMatmulTileLaunchTileOptions &opts)
      : AIRMatmulTileLaunchTileBase(opts) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    tensor::TensorDialect>();
  }

  void runOnOperation() override {
    func::FuncOp f = getOperation();
    linalg::MatmulOp matmul;
    f.walk([&](linalg::MatmulOp op) {
      matmul = op;
      return WalkResult::interrupt();
    });
    if (!matmul)
      return;

    SmallVector<int64_t> rawSizes = parseIntList(clTileSizes);
    auto tileSizes = buildTileSizes(rawSizes,
                                    cast<TilingInterface>(matmul.getOperation())
                                        .getLoopIteratorTypes()
                                        .size(),
                                    &getContext());

    // Capture the linalg.fill producer of the matmul's accumulator BEFORE
    // tiling (after which the matmul is rewritten and producer linkage may
    // shift through extract_slice).
    Operation *fillProducer =
        matmul.getOutputs()[0].getDefiningOp<linalg::FillOp>();

    auto tileable = cast<TilingInterface>(matmul.getOperation());
    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPoint(matmul);
    scf::SCFTilingOptions opts;
    opts.setLoopType(scf::SCFTilingOptions::LoopType::ForallOp);
    opts.setTileSizes(tileSizes);
    auto tilingResult = scf::tileUsingSCF(rewriter, tileable, opts);
    if (failed(tilingResult)) {
      matmul->emitError("scf::tileUsingSCF (forall) on launch-tile failed");
      return signalPassFailure();
    }
    rewriter.replaceOp(matmul, tilingResult->replacements);

    if (tilingResult->loops.empty())
      return;
    LoopLikeOpInterface forall = tilingResult->loops.front();
    forall->setAttr(clLaunchTileForallMarker, rewriter.getUnitAttr());

    if (fillProducer) {
      auto fillOp = dyn_cast<linalg::FillOp>(fillProducer);
      auto forallOp = dyn_cast<scf::ForallOp>(forall.getOperation());
      if (fillOp && forallOp)
        (void)fuseFillIntoForallSharedOuts(fillOp, forallOp, rewriter);
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulTileLaunchTilePass() {
  return std::make_unique<AIRMatmulTileLaunchTile>();
}
std::unique_ptr<mlir::Pass> createAIRMatmulTileLaunchTilePass(
    const AIRMatmulTileLaunchTileOptions &opts) {
  return std::make_unique<AIRMatmulTileLaunchTile>(opts);
}

} // namespace air
} // namespace xilinx
