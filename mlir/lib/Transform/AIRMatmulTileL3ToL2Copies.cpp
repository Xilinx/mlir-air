//===- AIRMatmulTileL3ToL2Copies.cpp ---------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRMatmulTileL3ToL2Copies.h"
#include "air/Util/MatmulCodegenConfig.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "air-matmul-tile-l3-to-l2-copies"

using namespace mlir;
using namespace xilinx::air;

namespace xilinx {
namespace air {

namespace {

// Convert memref.copy → linalg.copy. Local copy of the pattern in
// AIRLinalgCodegen.cpp's anonymous namespace; reproduced here to avoid
// exposing it as public API just for one user.
struct ConvertMemrefCopyToLinalgCopyPattern
    : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<linalg::CopyOp>(copyOp, copyOp.getSource(),
                                                copyOp.getTarget());
    return success();
  }
};

// Walk back from a matmul tensor operand to the linalg.copy that fills the
// memref later read by `bufferization.to_tensor`. Returns nullptr if the
// chain doesn't match the expected shape (pre-bufferization Triton-XDNA-style
// IR).
static linalg::CopyOp findCopyForOperand(Value matmulOperand) {
  auto toTensor = matmulOperand.getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensor)
    return nullptr;
  Value memref = toTensor.getBuffer();
  // The linalg.copy targets `memref` as its DPS output.
  for (Operation *user : memref.getUsers()) {
    auto copyOp = dyn_cast<linalg::CopyOp>(user);
    if (!copyOp)
      continue;
    if (copyOp.getDpsInits().size() != 1)
      continue;
    if (copyOp.getDpsInits()[0] == memref)
      return copyOp;
  }
  return nullptr;
}

// Tile a 2D linalg.copy by `tileSizes` (one OpFoldResult per dim; zero means
// not tiled). Annotates the produced scf.for with `marker` (unit attr).
static LogicalResult tileCopyAndAnnotate(linalg::CopyOp copyOp,
                                         ArrayRef<OpFoldResult> tileSizes,
                                         StringRef marker) {
  IRRewriter rewriter(copyOp.getContext());
  rewriter.setInsertionPoint(copyOp);
  auto tilingIface = cast<TilingInterface>(copyOp.getOperation());
  scf::SCFTilingOptions tilingOpts;
  tilingOpts.setTileSizes(tileSizes);
  auto result = scf::tileUsingSCF(rewriter, tilingIface, tilingOpts);
  if (failed(result))
    return copyOp->emitError() << "scf::tileUsingSCF failed";
  rewriter.replaceOp(copyOp, result->replacements);

  if (marker.empty() || result->loops.empty())
    return success();
  // Annotate the outermost generated loop with the marker.
  Operation *outerLoop = result->loops.front().getOperation();
  outerLoop->setAttr(marker, rewriter.getUnitAttr());
  return success();
}

class AIRMatmulTileL3ToL2Copies
    : public impl::AIRMatmulTileL3ToL2CopiesBase<AIRMatmulTileL3ToL2Copies> {

public:
  AIRMatmulTileL3ToL2Copies() = default;
  AIRMatmulTileL3ToL2Copies(const AIRMatmulTileL3ToL2CopiesOptions &opts)
      : AIRMatmulTileL3ToL2CopiesBase(opts) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, scf::SCFDialect,
                    bufferization::BufferizationDialect>();
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    // Step 1: convert any memref.copy to linalg.copy. Greedy walk over the
    // function. Idempotent — passes that have already converted upstream
    // contribute no work.
    {
      RewritePatternSet patterns(&getContext());
      patterns.insert<ConvertMemrefCopyToLinalgCopyPattern>(&getContext());
      (void)applyPatternsGreedily(func, std::move(patterns));
    }

    // Step 2: locate the first linalg.matmul.
    linalg::MatmulOp matmul;
    func.walk([&](linalg::MatmulOp op) {
      matmul = op;
      return WalkResult::interrupt();
    });
    if (!matmul) {
      // No matmul; nothing more to do.
      return;
    }

    // Step 3: find the LHS and RHS L3-staging copies.
    linalg::CopyOp copyA = findCopyForOperand(matmul->getOperand(0));
    linalg::CopyOp copyB = findCopyForOperand(matmul->getOperand(1));

    int64_t kL2Tile = clKL2Tile;
    if (auto cfg = xilinx::air::findMatmulCodegenConfig(func))
      kL2Tile = xilinx::air::getI64(*cfg, "tile_l3_l2_k", kL2Tile);

    OpFoldResult zero = OpBuilder(&getContext()).getIndexAttr(0);
    OpFoldResult kTile = OpBuilder(&getContext()).getIndexAttr(kL2Tile);

    // LHS layout is (M, K): tile dim 1 (= K). RHS layout is (K, N): tile
    // dim 0 (= K). If a copy isn't found (e.g., upstream already tiled it),
    // skip silently — re-running the pass should be a no-op.
    if (copyA) {
      if (failed(tileCopyAndAnnotate(copyA, {zero, kTile}, clCopyALoopMarker)))
        return signalPassFailure();
    }
    if (copyB) {
      if (failed(tileCopyAndAnnotate(copyB, {kTile, zero}, clCopyBLoopMarker)))
        return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createAIRMatmulTileL3ToL2CopiesPass() {
  return std::make_unique<AIRMatmulTileL3ToL2Copies>();
}

std::unique_ptr<mlir::Pass> createAIRMatmulTileL3ToL2CopiesPass(
    const AIRMatmulTileL3ToL2CopiesOptions &opts) {
  return std::make_unique<AIRMatmulTileL3ToL2Copies>(opts);
}

} // namespace air
} // namespace xilinx
