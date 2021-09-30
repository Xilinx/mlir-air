// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
//
// ===- AIRLoopPermutationPass.cpp - Permutate loop order in a loop nest --===//
// Refers to include/mlir/Transform/LoopUtils.h
// ===---------------------------------------------------------------------===//
//
// This pass performs a loop nest reordering according to the input mapping. 
// The i-th loop will be moved from position i -> permMap[i] where the counting 
// of i starts at the outermost loop. The pass transforms only perfect loop 
// nests. The specified ordering starts from 0, and should be of the same 
// length as the loop nest size. Each number is required to appear once in 
// the input mapping.
//
// ===---------------------------------------------------------------------===//

#include "air/Transform/AIRLoopPermutationPass.h"
#include "air/Transform/AIRTilingUtils.h"

#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/Support/Debug.h"

#include <set>

#define DEBUG_TYPE "air-loop-permutation"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

namespace {
  
class AIRLoopPermutationPass : public AIRLoopPermutationBase<AIRLoopPermutationPass> {

public:
  AIRLoopPermutationPass() = default;
  AIRLoopPermutationPass(const AIRLoopPermutationPass &pass){};


  void runOnOperation() override;

  static const char *affineOptAttrName;
private:

};

const char *AIRLoopPermutationPass::affineOptAttrName = "affine_opt_label";

void AIRLoopPermutationPass::runOnOperation() {

  auto func = getOperation();
  
  // Bands of loops to tile
  std::vector<SmallVector<AffineForOp, 6>> bands;
  xilinx::air::getTileableBands(func, bands, 
                                AIRLoopPermutationPass::affineOptAttrName,
                                AIRLabel);

  for (auto band: bands) {
    // Save and erase the previous label
    auto stringAttr = band[0]->getAttrOfType<StringAttr>(
      AIRLoopPermutationPass::affineOptAttrName);
    
    unsigned newOutermost = permuteLoops(band, loopOrder);

    if (stringAttr) { 
      StringAttr postLabel = AIRPostLabel.empty() ? 
        stringAttr:StringAttr::get(AIRPostLabel, stringAttr.getType());
      band[newOutermost]->setAttr(
          AIRLoopPermutationPass::affineOptAttrName, postLabel);
    }
    (void)band[0]->removeAttr(AIRLoopPermutationPass::affineOptAttrName);
  }
}

} // anonymous namespace

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRLoopPermutationPass() {
  return std::make_unique<AIRLoopPermutationPass>();
}

} // namespace air
} // namespace xilinx

// void xilinx::air::registerAIRLoopPermutationPass() {
//     PassRegistration<AIRLoopPermutationPass>(
//       "air-loop-permutation",
//       "Permute affine loops");
// }

// static PassRegistration<AIRLoopPermutationPass>
//     pass("air-loop-permutation", "Permute affine loops");
