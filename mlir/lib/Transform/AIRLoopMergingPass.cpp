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
#include "air/Transform/AIRTilingUtils.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Transforms/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#include <list>

#define DEBUG_TYPE "air-loop-merging"

using namespace mlir;

namespace {

class AIRLoopMergingPass
    : public xilinx::air::impl::AIRLoopMergingPassBase<AIRLoopMergingPass> {

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

void AIRLoopMergingPass::runOnOperation() {

  auto func = getOperation();
  SmallVector<unsigned, 3> reduceLoopLevels;
  for (unsigned i = 0; i < clLoopMergeLevels.size(); i++) {
    reduceLoopLevels.push_back(clLoopMergeLevels[i]);
  }

  // Assume that the pass takes in loops that are in normalized form.
  std::vector<SmallVector<affine::AffineForOp, 6>> bands;
  xilinx::air::getTileableBands(
      func, bands, AIRLoopMergingPass::affineOptAttrName, clAIROptLabel);

  for (auto &band : bands) {
    unsigned at = reduceLoopLevels.front();
    unsigned count = reduceLoopLevels.size();
    for (unsigned i = 1; i < count; i++) {
      if (reduceLoopLevels[i] != at + i) {
        band[0]->emitOpError("loop-merge-levels must be consecutive");
        return signalPassFailure();
      }
    }
    if (at + count > band.size()) {
      band[0]->emitOpError("loop-merge-levels exceed the loop band depth");
      return signalPassFailure();
    }
    auto stringAttr = band[0]->getAttrOfType<StringAttr>(
        AIRLoopMergingPass::affineOptAttrName);
    if (failed(affine::coalesceLoops(
            MutableArrayRef<affine::AffineForOp>(band).slice(at, count)))) {
      band[at]->emitOpError("cannot merge loops that are not normalized");
      return signalPassFailure();
    }

    // Preserve the loop band label
    if (stringAttr) {
      StringAttr postLabel =
          clAIROptPostLabel.empty()
              ? stringAttr
              : StringAttr::get(clAIROptPostLabel, stringAttr.getType());
      band[0]->setAttr(AIRLoopMergingPass::affineOptAttrName, postLabel);
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
