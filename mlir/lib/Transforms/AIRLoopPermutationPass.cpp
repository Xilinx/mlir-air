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

#include "AIRLoopPermutationPass.h"
#include "AIRTilingUtils.h"

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

namespace {
  
class AIRLoopPermutationPass : public PassWrapper<AIRLoopPermutationPass, 
                                                  FunctionPass> {

public:
  AIRLoopPermutationPass() = default;
  AIRLoopPermutationPass(const AIRLoopPermutationPass &pass){};

  ListOption<unsigned> clLoopOrder{*this, "loop-order",
                          llvm::cl::desc("Loop permutation order"),
                          llvm::cl::OneOrMore,
                          llvm::cl::CommaSeparated};
  
  Option<std::string> clAIROptLabel{*this, "air-label",
                          llvm::cl::desc("Transform loops with the given \
                              label"),
                          llvm::cl::init("")};

  Option<std::string> clAIROptPostLabel{*this, "air-post-label",
                          llvm::cl::desc("Label to apply to transformed loop \
                              nest"),
                          llvm::cl::init("")};

  void runOnFunction() override;

  static const char *affineOptAttrName;
private:

};

const char *AIRLoopPermutationPass::affineOptAttrName = "affine_opt_label";

void AIRLoopPermutationPass::runOnFunction() {

  auto func = getFunction();
  
  // Bands of loops to tile
  std::vector<SmallVector<AffineForOp, 6>> bands;
  xilinx::air::getTileableBands(func, bands, 
                                AIRLoopPermutationPass::affineOptAttrName,
                                clAIROptLabel);

  for (auto band: bands) {
    // Save and erase the previous label
    auto stringAttr = band[0]->getAttrOfType<StringAttr>(
      AIRLoopPermutationPass::affineOptAttrName);
    
    unsigned newOutermost = permuteLoops(band, clLoopOrder);

    if (stringAttr) { 
      StringAttr postLabel = clAIROptPostLabel.empty() ? 
        stringAttr:StringAttr::get(clAIROptPostLabel, stringAttr.getType());
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

void xilinx::air::registerAIRLoopPermutationPass() {
    PassRegistration<AIRLoopPermutationPass>(
      "air-loop-permutation",
      "Permute affine loops");
}

static PassRegistration<AIRLoopPermutationPass>
    pass("air-loop-permutation", "Permute affine loops");
