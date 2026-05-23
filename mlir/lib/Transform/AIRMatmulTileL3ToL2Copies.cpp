//===- AIRMatmulTileL3ToL2Copies.cpp ---------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Free-function body for the former `air-matmul-tile-l3-to-l2-copies` pass.
// Now invoked from `air-matmul-bufferize-output-l2` when its
// `do-tile-l3-to-l2-copies` option is set.
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRMatmulTileL3ToL2Copies.h"
#include "air/Transform/AIRMatmulCodegenHelpers.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/TilingInterface.h"

#define DEBUG_TYPE "air-matmul-tile-l3-to-l2-copies"

using namespace mlir;
using namespace xilinx::air;

namespace xilinx {
namespace air {

namespace {

// Walk back from a matmul tensor operand to the linalg.copy that fills the
// memref later read by `bufferization.to_tensor`. Returns nullptr if the
// chain doesn't match the expected shape (pre-bufferization Triton-XDNA-style
// IR).
static linalg::CopyOp findCopyForOperand(Value matmulOperand) {
  auto toTensor = matmulOperand.getDefiningOp<bufferization::ToTensorOp>();
  if (!toTensor)
    return nullptr;
  Value memref = toTensor.getBuffer();
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
  Operation *outerLoop = result->loops.front().getOperation();
  outerLoop->setAttr(marker, rewriter.getUnitAttr());
  return success();
}

} // namespace

LogicalResult runTileL3ToL2CopiesImpl(func::FuncOp func, int64_t kL2Tile,
                                      StringRef copyAMarker,
                                      StringRef copyBMarker) {
  if (failed(runConvertMemrefCopyToLinalgCopy(func)))
    return failure();

  linalg::MatmulOp matmul;
  func.walk([&](linalg::MatmulOp op) {
    matmul = op;
    return WalkResult::interrupt();
  });
  if (!matmul)
    return success(); // no matmul; nothing more to do.

  linalg::CopyOp copyA = findCopyForOperand(matmul->getOperand(0));
  linalg::CopyOp copyB = findCopyForOperand(matmul->getOperand(1));

  OpBuilder b(func.getContext());
  OpFoldResult zero = b.getIndexAttr(0);
  OpFoldResult kTile = b.getIndexAttr(kL2Tile);

  // LHS layout is (M, K): tile dim 1 (= K). RHS layout is (K, N): tile dim
  // 0 (= K). If a copy isn't found, skip silently — re-running is a no-op.
  if (copyA && failed(tileCopyAndAnnotate(copyA, {zero, kTile}, copyAMarker)))
    return failure();
  if (copyB && failed(tileCopyAndAnnotate(copyB, {kTile, zero}, copyBMarker)))
    return failure();
  return success();
}

} // namespace air
} // namespace xilinx
