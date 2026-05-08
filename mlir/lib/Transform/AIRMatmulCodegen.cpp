//===- AIRMatmulCodegen.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// AIRMatmulCodegen: single public matmul codegen pass. Internal phases are
// gated by their config (skip-if-empty) and chained with canonicalize/cse +
// upstream one-shot-bufferize.
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRMatmulCodegen.h"
#include "air/Transform/AIRMatmulBufferizationPasses.h"
#include "air/Transform/AIRMatmulCodegenHelpers.h"
#include "air/Transform/AIRMatmulPackAndTranspose.h"
#include "air/Transform/AIRMatmulTilePasses.h"
#include "air/Transform/AIRMatmulVectorizePasses.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#define DEBUG_TYPE "air-matmul-codegen"

using namespace mlir;
using namespace xilinx::air;

namespace xilinx {
namespace air {

namespace {

// Internal marker constants. The orchestrator owns the marker namespace —
// each phase tags ops with names known to the next consumer phase. Not
// configurable: callers don't need to compose phases out-of-order.
static constexpr llvm::StringLiteral kPackedMatmul = "packed_matmul";
static constexpr llvm::StringLiteral kLaunchTileForall = "launch_tile_forall";
static constexpr llvm::StringLiteral kCopyALoop = "copy_a_loop";
static constexpr llvm::StringLiteral kCopyBLoop = "copy_b_loop";
static constexpr llvm::StringLiteral kKReductionLoop = "k_reduction_loop";
static constexpr llvm::StringLiteral kKReductionLoopInner =
    "k_reduction_loop_inner";
static constexpr llvm::StringLiteral kLhsPackInK = "lhs_pack_in_k";
static constexpr llvm::StringLiteral kRhsPackInK = "rhs_pack_in_k";
static constexpr llvm::StringLiteral kLhsL2PackInK = "lhs_l2_pack_in_k";
static constexpr llvm::StringLiteral kRhsL2PackInK = "rhs_l2_pack_in_k";
static constexpr llvm::StringLiteral kComputeForall = "compute_forall";
static constexpr llvm::StringLiteral kMatmulCompute = "matmul_compute";
static constexpr llvm::StringLiteral kFusedLhsL1Pack = "fused_lhs_l1_pack";
static constexpr llvm::StringLiteral kFusedRhsL1Pack = "fused_rhs_l1_pack";
static constexpr llvm::StringLiteral kInitFill = "init_fill";
static constexpr llvm::StringLiteral kPrologueForall = "prologue_forall";
static constexpr llvm::StringLiteral kEpilogueForall = "epilogue_forall";

class AIRMatmulCodegen : public impl::AIRMatmulCodegenBase<AIRMatmulCodegen> {
public:
  AIRMatmulCodegen() = default;
  AIRMatmulCodegen(const AIRMatmulCodegenOptions &opts)
      : AIRMatmulCodegenBase(opts) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect>();
  }

  // Run a small pipeline at func or module scope. AIRMatmulCodegen runs at
  // ModuleOp so dynamic scheduling at either scope is permitted.
  bool runFuncScoped(func::FuncOp f,
                     llvm::function_ref<void(OpPassManager &)> populate) {
    OpPassManager pm(func::FuncOp::getOperationName());
    populate(pm);
    return succeeded(runPipeline(pm, f));
  }

  bool runModuleScoped(ModuleOp m,
                       llvm::function_ref<void(OpPassManager &)> populate) {
    OpPassManager pm(ModuleOp::getOperationName());
    populate(pm);
    return succeeded(runPipeline(pm, m));
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> funcs(module.getOps<func::FuncOp>());
    for (func::FuncOp f : funcs)
      if (failed(runOnFunc(f)))
        return;
  }

  LogicalResult runOnFunc(func::FuncOp f) {
    IRRewriter rewriter(&getContext());
    ModuleOp module = f->getParentOfType<ModuleOp>();
    auto fail = [&]() {
      signalPassFailure();
      return failure();
    };

    auto canonicalizeCse = [&]() {
      return runFuncScoped(f, [](OpPassManager &pm) {
        pm.addPass(createCanonicalizerPass());
        pm.addPass(createCSEPass());
      });
    };

    // ---------- Phase 0: pre-fold unit-extent dims (opt-in) ----------
    if (clDoPreFoldUnitExtentDims)
      if (failed(runFoldUnitExtentDimsOnFunc(f)))
        return fail();

    // Phase C placement: single-pack flows (no L1 pack) run bufferize-output-l2
    // BEFORE Phase A and Phase B — required by the tile-l3-to-l2-copies and
    // fuse-output-truncf-first pre-steps (which must operate on un-packed IR)
    // and so that the L2 alloc lands at LAUNCH scope, outside any per-core
    // forall created by Phase A.
    // Two-pack flows run Phase C AFTER Phase B (L2 pack) so the L2 alloc
    // takes the packed shape matching the L1 pack's expected operand layout.
    bool singlePackLevel = clL1PackSizes.empty();
    auto runPhaseC = [&]() -> LogicalResult {
      if (!clBufferizeOutputL2)
        return success();
      return runBufferizeOutputL2Impl(
          f, clBufferizeOutputL2MemorySpace, clFuseOutputTruncfFirst,
          clTileL3ToL2Copies, clKL2Tile, kCopyALoop, kCopyBLoop, rewriter);
    };

    if (singlePackLevel)
      if (failed(runPhaseC()))
        return fail();

    // ---------- Phase A: launch tile (skip if empty) ----------
    if (!clLaunchTile.empty()) {
      if (failed(runTileLaunchTileImpl(f, clLaunchTile, kLaunchTileForall,
                                       rewriter)))
        return fail();
    }

    // ---------- Phase B: L2 pack (skip if empty) ----------
    // The L2 pack bufferizes its output to L1 only in single-pack-level flows
    // (l1-pack-sizes empty) AND when bufferize-last-pack-output is true.
    // Two-pack-level flows defer L1 output bufferization to Phase D (L1 pack).
    if (!clL2PackSizes.empty()) {
      bool bufferizeL2OutputToL1 = singlePackLevel && clBufferizeLastPackOutput;
      if (failed(runPackAndTransposeImpl(
              f, clL2PackSizes, clL2LhsOuterPerm, clL2LhsInnerPerm,
              clL2RhsOuterPerm, clL2RhsInnerPerm, clL2AccOuterPerm,
              clL2AccInnerPerm, kPackedMatmul,
              /*doBufferizeL1Output=*/bufferizeL2OutputToL1,
              /*memSpace=*/clL1OutputMemorySpace, rewriter)))
        return fail();
      if (!canonicalizeCse())
        return fail();
    }

    if (!singlePackLevel)
      if (failed(runPhaseC()))
        return fail();

    // ---------- Phase D: L1 pack (skip if empty) ----------
    // The L1 pack is the LAST pack in two-pack flows, so its output is
    // bufferized to L1 when bufferize-last-pack-output is true.
    if (!clL1PackSizes.empty()) {
      if (failed(runPackAndTransposeImpl(
              f, clL1PackSizes, clL1LhsOuterPerm, clL1LhsInnerPerm,
              clL1RhsOuterPerm, clL1RhsInnerPerm, clL1AccOuterPerm,
              clL1AccInnerPerm, kPackedMatmul,
              /*doBufferizeL1Output=*/clBufferizeLastPackOutput,
              /*memSpace=*/clL1OutputMemorySpace, rewriter)))
        return fail();
    }

    // ---------- Phase E: outer K-tile + fuse packs (skip if 0) ----------
    if (clOuterKTileFactor > 0) {
      if (failed(runTileKAndFusePacksImpl(
              f, clOuterKTileFactor, clOuterKIterIndex, kPackedMatmul,
              kKReductionLoop, kLhsPackInK, kRhsPackInK, kLhsL2PackInK,
              kRhsL2PackInK, rewriter)))
        return fail();
      // Phase F: bufferize L2 inputs (always paired with two-pack outer-K-tile
      // since the L2 packs were chain-fused). Skip if no L1 pack was done
      // (single-pack-level flow doesn't have L2 packs to bufferize here).
      if (!clL1PackSizes.empty()) {
        if (failed(runBufferizeL1InputsImpl(f, /*memSpace=*/1,
                                            /*memcpyOp=*/"linalg-copy",
                                            kLhsL2PackInK, kRhsL2PackInK,
                                            rewriter)))
          return fail();
      } else if (clCoreTile.empty()) {
        // Phase F': single-pack flow with NO tile-cores (e.g. a launch-tile-
        // only flow). The L1 packs from Phase E are tagged lhs_pack_in_k /
        // rhs_pack_in_k and need bufferization to L1 here, since Phase J
        // (which uses fused_*_l1_pack markers) won't fire.
        if (failed(runBufferizeL1InputsImpl(f, /*memSpace=*/2,
                                            /*memcpyOp=*/"materialize",
                                            kLhsPackInK, kRhsPackInK,
                                            rewriter)))
          return fail();
      }
      if (!canonicalizeCse())
        return fail();
    }

    // ---------- Phase H: tile cores (skip if empty) ----------
    if (!clCoreTile.empty()) {
      if (failed(runTileCoresImpl(f, clCoreTile, kPackedMatmul, kLhsPackInK,
                                  kRhsPackInK, kComputeForall, kMatmulCompute,
                                  kFusedLhsL1Pack, kFusedRhsL1Pack, rewriter)))
        return fail();
      if (!canonicalizeCse())
        return fail();
    }

    // ---------- Phase I: inner K-tile (skip if 0) ----------
    if (clInnerKTileFactor > 0) {
      if (failed(runTileKAndFusePacksImpl(
              f, clInnerKTileFactor, clInnerKIterIndex, kPackedMatmul,
              kKReductionLoopInner, kFusedLhsL1Pack, kFusedRhsL1Pack,
              kLhsL2PackInK, kRhsL2PackInK, rewriter)))
        return fail();
    }

    // ---------- Phase J: bufferize L1 inputs (skip if no tile-cores)
    // ----------
    if (!clCoreTile.empty()) {
      if (failed(runBufferizeL1InputsImpl(f, /*memSpace=*/2,
                                          /*memcpyOp=*/"materialize",
                                          kFusedLhsL1Pack, kFusedRhsL1Pack,
                                          rewriter)))
        return fail();
      if (!canonicalizeCse())
        return fail();
    }

    // ---------- Phase K: prologue/epilogue (skip if both tiles empty)
    // ----------
    if (!clPrologueTile.empty() || !clEpilogueTile.empty()) {
      if (failed(runPrologueEpilogueImpl(f, clPrologueTile, clEpilogueTile,
                                         clFillIterPerm, kInitFill,
                                         kPrologueForall, kEpilogueForall,
                                         clHoistStaticAllocFirst, rewriter)))
        return fail();
      if (!canonicalizeCse())
        return fail();
    }

    // ---------- Phase L: one-shot bufferize (gated; default true) ----------
    if (clOneShotBufferize) {
      if (!runModuleScoped(module, [](OpPassManager &pm) {
            bufferization::OneShotBufferizePassOptions opts;
            opts.bufferizeFunctionBoundaries = true;
            opts.functionBoundaryTypeConversion =
                bufferization::LayoutMapOption::IdentityLayoutMap;
            opts.unknownTypeConversion =
                bufferization::LayoutMapOption::IdentityLayoutMap;
            pm.addPass(bufferization::createOneShotBufferizePass(opts));
          }))
        return fail();
      // canonicalize, cse, canonicalize (mirrors the legacy pipeline).
      if (!runFuncScoped(f, [](OpPassManager &pm) {
            pm.addPass(createCanonicalizerPass());
            pm.addPass(createCSEPass());
            pm.addPass(createCanonicalizerPass());
          }))
        return fail();
    }

    // ---------- Phase M: tile for vectorize (skip if empty) ----------
    if (!clMatmulVecTile.empty()) {
      if (failed(runTileForVectorizeImpl(
              f, clMatmulVecTile, clMatmulUnrollVecTile, clMatmulUnrollFactor,
              clFillVecTile, clPostBufferizeCleanupFirst, rewriter)))
        return fail();
    }

    // ---------- Phase N: vec prep composite (gated; default true) ----------
    if (clDoVecPrep) {
      if (failed(runCodegenVecPrepImpl(
              f, clVecPrepFoldUnitExtentDims,
              clVecPrepEliminateRedundantVectorTransfers,
              clVecPrepCast1TargetElementType, clVecPrepCast1InputIndices,
              clVecPrepCast1OutputIndices, clVecPrepCast2TargetElementType,
              clVecPrepCast2InputIndices, clVecPrepCast2OutputIndices,
              clVecPrepHoistLoopInvariantTransfers, clVecPrepFlattenForIterArgs,
              clVecPrepHoistVectorTransferPointers, clVecPrepHoistCastPairs,
              clVecPrepHoistCastPairsMaxIterations, rewriter)))
        return fail();
    }

    return success();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulCodegenPass() {
  return std::make_unique<AIRMatmulCodegen>();
}

std::unique_ptr<mlir::Pass>
createAIRMatmulCodegenPass(const AIRMatmulCodegenOptions &opts) {
  return std::make_unique<AIRMatmulCodegen>(opts);
}

} // namespace air
} // namespace xilinx
