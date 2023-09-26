//===- AIRAutomaticTilingPass.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// ===- AIRAutomaticTilingPass.cpp - Multi-dimensional loop tiling pass ---===//
// Refers to mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp and
// mlir/lib/Dialect/Affine/Transforms/AffineLoopNormalize.cpp
// ===---------------------------------------------------------------------===//
//
// This file implements a multi-dimensional loop tiling pass that tiles all
// valid bands of loops with the same set of tiling sizes. It can also
// automatically tile a loop band, with the prime factors of the original loop
// bounds as the new loop bounds.
//
// ===---------------------------------------------------------------------===//

#include "air/Transform/AIRAutomaticTilingPass.h"
#include "air/Transform/AIRTilingUtils.h"

#include "PassDetail.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#include <set>

#define DEBUG_TYPE "air-automatic-tiling"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

namespace {

class AIRAutomaticTilingPass
    : public AIRAutomaticTilingBase<AIRAutomaticTilingPass> {

public:
  AIRAutomaticTilingPass() = default;
  AIRAutomaticTilingPass(const AIRAutomaticTilingPass &pass){};

  void runOnOperation() override;

  // Tile all bands of loops with the same set of tiling sizes.
  void tileLoopsManually(std::vector<SmallVector<AffineForOp, 6>> &bands,
                         unsigned tileSize);

  // Tile each band of loops with prime factors of the loop tripcounts.
  void tileLoopsAutomatically(std::vector<SmallVector<AffineForOp, 6>> &bands);

  SmallVector<unsigned, 6> optTileSizes;

  static const char *affineOptAttrName;

private:
};

const char *AIRAutomaticTilingPass::affineOptAttrName = "affine_opt_label";

void AIRAutomaticTilingPass::runOnOperation() {
  auto func = getOperation();

  optTileSizes.clear();
  if (loopTileSizes.size() > 0) {
    // Initialize tile sizes from the command line.
    for (unsigned i = 0; i < loopTileSizes.size(); ++i) {
      optTileSizes.push_back(loopTileSizes[i]);
    }

    for (auto tileSize : optTileSizes) {
      // Bands of loops to tile
      std::vector<SmallVector<AffineForOp, 6>> bands;
      xilinx::air::getTileableBands(
          func, bands, AIRAutomaticTilingPass::affineOptAttrName, clLabel);

      tileLoopsManually(bands, tileSize);

      // Normalize the loop space after tiling each dimension.
      func.walk([](Operation *op) {
        if (auto affineFor = dyn_cast<AffineForOp>(op))
          if (failed(normalizeAffineFor(affineFor)))
            return;
      });
    }
  } else {
    // Find the optimal tile sizes automatically.
    std::vector<SmallVector<AffineForOp, 6>> bands;
    xilinx::air::getTileableBands(
        func, bands, AIRAutomaticTilingPass::affineOptAttrName, clLabel);

    // Normalize every loop before tiling.
    for (auto band : bands)
      for (AffineForOp affineFor : band)
        if (failed(normalizeAffineFor(affineFor)))
          continue;

    tileLoopsAutomatically(bands);

    // Normalize every loop after tiling.
    bands.clear();
    xilinx::air::getTileableBands(
        func, bands, AIRAutomaticTilingPass::affineOptAttrName, clLabel);
    for (auto band : bands)
      for (AffineForOp affineFor : band)
        if (failed(normalizeAffineFor(affineFor)))
          continue;
  }
}

/// Factorizes a long number into its prime factors.
static void factorConstant(int64_t longNum,
                           SmallVectorImpl<int64_t> &primeFactors) {
  while (longNum % 2 == 0) {
    primeFactors.push_back(2);
    longNum = longNum / 2;
  }
  for (unsigned i = 3; i < longNum; i += 2) {
    while (longNum % i == 0) {
      primeFactors.push_back(i);
      longNum = longNum / i;
    }
  }
  if (longNum > 2)
    primeFactors.push_back(longNum);
}

/// Construct a tiled loop nest and set their loop range.
static void
constructTiledLoopNest(MutableArrayRef<AffineForOp> origLoops,
                       unsigned total_width,
                       MutableArrayRef<AffineForOp> tiledLoops,
                       ArrayRef<SmallVector<int64_t, 6>> setOfPrimeFactors,
                       SmallVectorImpl<unsigned> &loopLevels) {
  AffineForOp rootAffineForOp = origLoops[0];
  Location rootForLoc = rootAffineForOp.getLoc();
  Operation *topLoop = rootAffineForOp.getOperation();
  AffineForOp innermostLoop;

  // Create an Affine for loop band.
  for (unsigned i = 0; i < total_width; i++) {
    OpBuilder b(topLoop);
    AffineForOp intraLoop = b.create<AffineForOp>(rootForLoc, 0, 0);
    intraLoop.getBody()->getOperations().splice(
        intraLoop.getBody()->begin(), topLoop->getBlock()->getOperations(),
        topLoop);
    tiledLoops[total_width - 1 - i] = intraLoop;
    topLoop = intraLoop.getOperation();
    if (i == 0)
      innermostLoop = intraLoop;
  }

  // Move the innermost loop body into the specified location....
  AffineForOp src = origLoops.back();
  auto &ops = src.getBody()->getOperations();
  Block::iterator innerForLoc = innermostLoop.getBody()->begin();
  innermostLoop.getBody()->getOperations().splice(innerForLoc, ops, ops.begin(),
                                                  std::prev(ops.end()));

  // Manage the tiled loop bounds and step sizes.
  assert(!origLoops.empty());
  OpBuilder b(origLoops[0].getOperation());
  // unsigned width = origLoops.size();
  unsigned width = setOfPrimeFactors.size();
  assert(origLoops.size() == width);

  static unsigned forOpLevel = 0;
  for (unsigned i = 0; i < width; i++) {
    auto primeFactors = setOfPrimeFactors[i];
    unsigned single_width = primeFactors.size();

    // For each tiling, the outermost loop has:
    // a) upper bound = original loop upper bound
    // b) lower bound = original loop lower bound
    // c) step size = product of all successive primefactors
    OperandRange newLbOperands = origLoops[i].getLowerBoundOperands();
    OperandRange newUbOperands = origLoops[i].getUpperBoundOperands();
    tiledLoops[forOpLevel].setLowerBound(newLbOperands,
                                         origLoops[i].getLowerBoundMap());
    tiledLoops[forOpLevel].setUpperBound(newUbOperands,
                                         origLoops[i].getUpperBoundMap());
    int64_t stepSize = 1;
    for (unsigned j = 1; j < single_width; j++) {
      stepSize = stepSize * primeFactors[j];
    }
    tiledLoops[forOpLevel].setStep(stepSize);

    // For the intra tiling loops:
    for (unsigned j = 1; j < single_width; j++) {
      // a) lower bound = the outer loop IV
      AffineMap lbMap = b.getDimIdentityMap();
      tiledLoops[forOpLevel + j].setLowerBound(
          tiledLoops[forOpLevel + j - 1].getInductionVar(), lbMap);

      // b) upper bound = the outer loop IV + step size of outer loop
      int64_t shiftAmount = 1;
      for (unsigned k = j; k < single_width; k++) {
        shiftAmount = shiftAmount * primeFactors[k];
      }
      AffineMap ubMap = b.getSingleDimShiftAffineMap(shiftAmount);
      tiledLoops[forOpLevel + j].setUpperBound(
          tiledLoops[forOpLevel + j - 1].getInductionVar(), ubMap);

      // c) step size = product of *exclusively* successive prime factors
      int64_t newStepSize = 1;
      for (unsigned k = j + 1; k < single_width; k++) {
        newStepSize = newStepSize * primeFactors[k];
      }
      tiledLoops[forOpLevel + j].setStep(newStepSize);
    }

    // Note down which level to replace the loop IVs in the new loop body.
    loopLevels.push_back(forOpLevel + single_width - 1);

    forOpLevel += single_width;
  }
}

/// Tile a loop nest into multiple subloops where the new loop bounds are prime
/// factors of the original loop bounds.
/// Assume that 1) the loop is in the normalized form. The lower bound is always
/// 0, and the upper bound is the loop tripcount. The step is alwasy 1. 2) the
/// loop bounds are all constants.
/// Assume hyper-rectangular loop space. No cross-axis dependency is considered.
void AIRAutomaticTilingPass::tileLoopsAutomatically(
    std::vector<SmallVector<AffineForOp, 6>> &bands) {
  for (auto &band : bands) {
    // For each band of loops, get the array of loops and the loop bound.
    MutableArrayRef<AffineForOp> origLoops = band;
    AffineForOp outerAffineForOp = origLoops[0];

    // Factor the loop bound into prime numbers.
    unsigned total_width = 0;
    SmallVector<SmallVector<int64_t, 6>, 3> setOfPrimeFactors;
    for (auto forOp : origLoops) {
      int64_t upperLoopBound = forOp.getConstantUpperBound();
      assert(upperLoopBound > 1);

      SmallVector<int64_t, 6> primeFactors;
      factorConstant(upperLoopBound, primeFactors);
      total_width = total_width + primeFactors.size();
      setOfPrimeFactors.push_back(primeFactors);
    }

    // Construct a tiled loop nest and set the loop bounds.
    SmallVector<AffineForOp, 6> tiledLoops(total_width);
    SmallVector<unsigned, 6> loopLevels;
    constructTiledLoopNest(origLoops, total_width, tiledLoops,
                           setOfPrimeFactors, loopLevels);

    // Replace original IVs with intra-tile IVs.
    SmallVector<Value, 3> origLoopIVs;
    extractForInductionVars(band, &origLoopIVs);
    for (unsigned i = 0; i < origLoopIVs.size(); i++) {
      unsigned singleLoopLevel = loopLevels[i];
      origLoopIVs[i].replaceAllUsesWith(
          tiledLoops[singleLoopLevel].getInductionVar());
    }

    // Erase the old loop nest.
    outerAffineForOp.erase();

    auto stringAttr = band[0]->getAttrOfType<StringAttr>(
        AIRAutomaticTilingPass::affineOptAttrName);
    if (stringAttr) {
      StringAttr postLabel =
          clPostLabel.empty()
              ? stringAttr
              : StringAttr::get(clPostLabel, stringAttr.getType());
      tiledLoops[0]->setAttr(AIRAutomaticTilingPass::affineOptAttrName,
                             postLabel);
    }
  }
}

void AIRAutomaticTilingPass::tileLoopsManually(
    std::vector<SmallVector<AffineForOp, 6>> &bands, unsigned tileSize) {
  // Tile each band.
  for (auto &band : bands) {
    // Set up tile sizes; fill missing tile sizes at the end with default tile
    SmallVector<unsigned, 6> actualTileSizes;
    for (unsigned i = 0; i < band.size(); i++)
      actualTileSizes.push_back(tileSize);

    SmallVector<AffineForOp, 6> tiledNest;
    if (failed(tilePerfectlyNested(band, actualTileSizes, &tiledNest)))
      return signalPassFailure();

    // Separate full and partial tiles.
    if (clTileSeparate) {
      auto intraTileLoops =
          MutableArrayRef<AffineForOp>(tiledNest).drop_front(band.size());
      (void)separateFullTiles(intraTileLoops);
    }

    auto stringAttr = band[0]->getAttrOfType<StringAttr>(
        AIRAutomaticTilingPass::affineOptAttrName);
    // StringRef originalLabel = band[0]->getAttrOfType<StringRef>(
    //   AIRAutomaticTilingPass::affineOptAttrName);
    if (stringAttr) {
      StringAttr postLabel =
          clPostLabel.empty()
              ? stringAttr
              : StringAttr::get(clPostLabel, stringAttr.getType());
      tiledNest[0]->setAttr(AIRAutomaticTilingPass::affineOptAttrName,
                            postLabel);
    }
  }
}

} // anonymous namespace

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRAutomaticTilingPass() {
  return std::make_unique<AIRAutomaticTilingPass>();
}

} // namespace air
} // namespace xilinx
