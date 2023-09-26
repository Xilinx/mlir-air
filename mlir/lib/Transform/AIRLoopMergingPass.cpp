//===- AIRLoopMergingPass.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// ===- AIRLoopMergingPass.cpp - Merge nested loops into a single loop ----===//
//
// This pass transforms several perfectly nested subloops into a single
// loop. The trip count of the new single loop is the product of all
// trip counts in subloops. The original loop induction variables are
// restored using floordiv and modulo operations. Users can specify which
// loop levels they want to merge together.
//
// ===---------------------------------------------------------------------===//

#include "air/Transform/AIRLoopMergingPass.h"
#include "PassDetail.h"
#include "air/Transform/AIRTilingUtils.h"

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

#include <list>

#define DEBUG_TYPE "air-loop-merging"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

namespace {

class AIRLoopMergingPass : public AIRLoopMergingPassBase<AIRLoopMergingPass> {

public:
  AIRLoopMergingPass() = default;
  AIRLoopMergingPass(const AIRLoopMergingPass &pass){};

  ListOption<unsigned> clLoopMergeLevels{
      *this, "loop-merge-levels",
      llvm::cl::desc("which loop levels to merge together"), llvm::cl::Required,
      llvm::cl::OneOrMore};

  Option<std::string> clAIROptLabel{
      *this, "air-label",
      llvm::cl::desc("Transform loops with the given label"),
      llvm::cl::init("")};

  Option<std::string> clAIROptPostLabel{
      *this, "air-post-label",
      llvm::cl::desc("Label to apply to transformed loop \
                              nest"),
      llvm::cl::init("")};

  void runOnOperation() override;

  static const char *affineOptAttrName;

private:
};

const char *AIRLoopMergingPass::affineOptAttrName = "affine_opt_label";

static void
constructReducedLoopNest(MutableArrayRef<AffineForOp> origLoops,
                         unsigned total_width,
                         MutableArrayRef<AffineForOp> reducedLoops,
                         SmallVectorImpl<unsigned> &loopMergeLevels) {

  AffineForOp rootLoop = origLoops[0];
  Location rootLoopLoc = rootLoop.getLoc();
  Operation *rootLoopOp = rootLoop.getOperation();
  AffineForOp innermostLoop;

  // Create an affine loop band
  for (unsigned i = 0; i < total_width; i++) {
    OpBuilder builder(rootLoopOp);
    AffineForOp intraLoop = builder.create<AffineForOp>(rootLoopLoc, 0, 0);
    intraLoop.getBody()->getOperations().splice(
        intraLoop.getBody()->begin(), rootLoopOp->getBlock()->getOperations(),
        rootLoopOp);
    reducedLoops[total_width - 1 - i] = intraLoop;
    rootLoopOp = intraLoop.getOperation();
    if (i == 0)
      innermostLoop = intraLoop;
  }

  // Move the innermost Loop body into the specified location
  AffineForOp src = origLoops.back();
  auto &ops = src.getBody()->getOperations();
  Block::iterator innerForLoc = innermostLoop.getBody()->begin();
  innermostLoop.getBody()->getOperations().splice(innerForLoc, ops, ops.begin(),
                                                  std::prev(ops.end()));

  // Manage the reduced loop bounds
  unsigned reduceAtLevel = loopMergeLevels[0];
  unsigned reduceExitLevel = loopMergeLevels[loopMergeLevels.size() - 1] + 1;
  // Preserve the lower bounds, upper bounds and step sizes for loops prior to
  // the reduced ones.
  for (unsigned i = 0; i < reduceAtLevel; i++) {
    OperandRange newLbOperands = origLoops[i].getLowerBoundOperands();
    OperandRange newUbOperands = origLoops[i].getUpperBoundOperands();
    reducedLoops[i].setLowerBound(newLbOperands,
                                  origLoops[i].getLowerBoundMap());
    reducedLoops[i].setUpperBound(newUbOperands,
                                  origLoops[i].getUpperBoundMap());
    reducedLoops[i].setStep(origLoops[i].getStep());
  }

  // Compute the loop tripcount = product of subloop tripcounts
  unsigned reducedSize = loopMergeLevels.size();
  reducedLoops[reduceAtLevel].setStep(1);
  int64_t newUpperBound = 1;
  int64_t newLowerBound = 0;
  for (unsigned i = 0; i < reducedSize; i++) {
    unsigned loopLevel = loopMergeLevels[i];
    assert(origLoops[loopLevel].hasConstantBounds());
    auto lb_map = origLoops[loopLevel].getLowerBoundMap();
    auto ub_map = origLoops[loopLevel].getUpperBoundMap();
    int64_t lb_const = lb_map.getSingleConstantResult();
    int64_t ub_const = ub_map.getSingleConstantResult();
    assert(lb_const == newLowerBound);
    newUpperBound = newUpperBound * ub_const;
  }
  reducedLoops[reduceAtLevel].setConstantLowerBound(newLowerBound);
  reducedLoops[reduceAtLevel].setConstantUpperBound(newUpperBound);

  // Preserve the lower bounds, upper bounds and step sizes for loops following
  // the reduced ones.
  for (unsigned i = reduceAtLevel + 1; i < total_width; i++) {
    OperandRange newLbOperands =
        origLoops[i - 1 + reducedSize].getLowerBoundOperands();
    OperandRange newUbOperands =
        origLoops[i - 1 + reducedSize].getUpperBoundOperands();
    reducedLoops[i].setLowerBound(
        newLbOperands, origLoops[i - 1 + reducedSize].getLowerBoundMap());
    reducedLoops[i].setUpperBound(
        newUbOperands, origLoops[i - 1 + reducedSize].getUpperBoundMap());
    reducedLoops[i].setStep(origLoops[i - 1 + reducedSize].getStep());
  }

  // Restore the original induction variables
  AffineForOp innerFor = reducedLoops[total_width - 1];
  AffineForOp singleFor = reducedLoops[reduceAtLevel];
  SmallVector<AffineApplyOp, 3> restoredIVs;
  OpBuilder applyBuilder = OpBuilder::atBlockBegin(innerFor.getBody());
  // The IV in the outermost original reduced loop nest can be calculated as
  // i0 = i / a1*a2*...*an
  AffineExpr dim0 = applyBuilder.getAffineDimExpr(0);
  int64_t divConst = 1;
  for (unsigned i = 1; i < reducedSize; i++) {
    unsigned loopLevel = loopMergeLevels[i];
    divConst *= origLoops[loopLevel].getConstantUpperBound();
  }
  auto map_0 = AffineMap::get(1, 0, dim0.floorDiv(divConst));
  AffineApplyOp apply_0 = applyBuilder.create<AffineApplyOp>(
      innerFor.getLoc(), map_0, singleFor.getInductionVar());
  restoredIVs.push_back(apply_0);

  // The IV in the middle loop nests can be calculated as
  // ik = (i / divConst) % ak
  for (unsigned i = 1; i < reducedSize - 1; i++) {
    unsigned loopLevel = loopMergeLevels[i];
    int64_t modConst = origLoops[loopLevel].getConstantUpperBound();
    int64_t divConst = 1;
    for (unsigned j = i + 1; j < reducedSize; j++) {
      unsigned loopLevel = loopMergeLevels[j];
      divConst *= origLoops[loopLevel].getConstantUpperBound();
    }
    AffineExpr dim0 = applyBuilder.getAffineDimExpr(0);
    auto map = AffineMap::get(1, 0, dim0.floorDiv(divConst) % modConst);
    AffineApplyOp apply = applyBuilder.create<AffineApplyOp>(
        innerFor.getLoc(), map, singleFor.getInductionVar());
    restoredIVs.push_back(apply);
  }

  // The IV in the innermost original reduced loop nest can be restored as
  // in = i % an
  unsigned loopLevel = loopMergeLevels[reducedSize - 1];
  int64_t modConst = origLoops[loopLevel].getConstantUpperBound();
  auto map_1 = AffineMap::get(1, 0, dim0 % modConst);
  AffineApplyOp apply_1 = applyBuilder.create<AffineApplyOp>(
      innerFor.getLoc(), map_1, singleFor.getInductionVar());
  restoredIVs.push_back(apply_1);
  assert(restoredIVs.size() == loopMergeLevels.size());

  // Replace all original reduced subloop IVs with the new single loop IV
  SmallVector<Value, 6> origLoopIVs;
  extractForInductionVars(origLoops, &origLoopIVs);
  for (unsigned i = 0; i < reduceAtLevel; i++) {
    origLoopIVs[i].replaceAllUsesWith(reducedLoops[i].getInductionVar());
  }
  for (unsigned i = 0; i < loopMergeLevels.size(); i++) {
    unsigned loopLevel = loopMergeLevels[i];
    origLoopIVs[loopLevel].replaceAllUsesWith(restoredIVs[i]);
  }
  for (unsigned i = reduceExitLevel; i < origLoopIVs.size(); i++) {
    origLoopIVs[i].replaceAllUsesWith(
        reducedLoops[i - loopMergeLevels.size() + 1].getInductionVar());
  }

  // Erase the old loop nest.
  origLoops[0].erase();
}

void AIRLoopMergingPass::runOnOperation() {

  auto func = getOperation();
  SmallVector<unsigned, 3> reduceLoopLevels;
  for (unsigned i = 0; i < clLoopMergeLevels.size(); i++) {
    reduceLoopLevels.push_back(clLoopMergeLevels[i]);
  }

  // Assume that the pass takes in loops that are in normalized form.
  std::vector<SmallVector<AffineForOp, 6>> bands;
  xilinx::air::getTileableBands(
      func, bands, AIRLoopMergingPass::affineOptAttrName, clAIROptLabel);

  for (auto &band : bands) {
    MutableArrayRef<AffineForOp> origLoops = band;
    unsigned total_width = origLoops.size() - reduceLoopLevels.size() + 1;
    SmallVector<AffineForOp, 6> reducedLoops(total_width);

    constructReducedLoopNest(origLoops, total_width, reducedLoops,
                             reduceLoopLevels);

    // Preserve the loop band label
    auto stringAttr = band[0]->getAttrOfType<StringAttr>(
        AIRLoopMergingPass::affineOptAttrName);
    if (stringAttr) {
      StringAttr postLabel =
          clAIROptPostLabel.empty()
              ? stringAttr
              : StringAttr::get(clAIROptPostLabel, stringAttr.getType());
      reducedLoops[0]->setAttr(AIRLoopMergingPass::affineOptAttrName,
                               postLabel);
    }
  }
}

} // anonymous namespace

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRLoopMergingPass() {
  return std::make_unique<AIRLoopMergingPass>();
}

} // namespace air
} // namespace xilinx