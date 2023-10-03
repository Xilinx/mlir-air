//===- AIRRegularizeLoopPass.cpp --------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// ===---     AIRRegularizeLoopPass.cpp - Loop Regularization Pass      ---===//
//
// This pass regularizes loop nests by moving intermediate operations between
// subloops in a loop nest inside the innermost loop body. The pass is
// essentially the inverse of the affine loop invariant code motion pass. For
// each operation that makes the loop nest non-perfect, the pass will check
// recursively if the content of the operation is independent of the induction
// variable of the inner loop. And if it is independent, the operation will be
// moved inside the inner loop body until the induction variable of the inner
// loop is dependent on the operation or there are no loops at the same level.
//
// FIXME: This pass is the inverse of lib/Transforms/LoopInvariantCodeMotion.
// We should generalize in terms of the LICM direction in the future.
//
// ===---------------------------------------------------------------------===//

#include "air/Transform/AIRRegularizeLoopPass.h"
#include "air/Transform/AIRTilingUtils.h"

#include "PassDetail.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"

#include "llvm/Support/Debug.h"

#include <set>

#define DEBUG_TYPE "air-regularize-loop"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

namespace {

class AIRRegularizeLoopPass
    : public AIRRegularizeLoopBase<AIRRegularizeLoopPass> {

public:
  AIRRegularizeLoopPass() = default;
  AIRRegularizeLoopPass(const AIRRegularizeLoopPass &pass){};

  Option<std::string> clAIROptLabel{
      *this, "air-label", llvm::cl::desc("Transform loops with the given \
                              label"),
      llvm::cl::init("")};

  void runOnOperation() override;
  void runOnAffineForNest(SmallVector<AffineForOp, 6> &band);

  static const char *affineOptAttrName;

private:
};

const char *AIRRegularizeLoopPass::affineOptAttrName = "affine_opt_label";

static bool isIndependent(Operation *op, AffineForOp forOp,
                          SmallPtrSetImpl<Operation *> &opsWithUsers,
                          SmallPtrSetImpl<Operation *> &opsToHoist);
static bool
checkInvarianceOfNestedIfOps(Operation *op, AffineForOp forOp,
                             SmallPtrSetImpl<Operation *> &opsWithUsers,
                             SmallPtrSetImpl<Operation *> &opsToHoist);
static bool
areAllOpsInTheBlockListInvariant(Region &blockList, AffineForOp forOp,
                                 SmallPtrSetImpl<Operation *> &opsWithUsers,
                                 SmallPtrSetImpl<Operation *> &opsToHoist);

// Refer to AffineLoopInvariantCodeMotion.cpp
// Recursively check if the operation is independent of the loop induction var
bool isIndependent(Operation *op, AffineForOp forOp,
                   SmallPtrSetImpl<Operation *> &opsWithUsers,
                   SmallPtrSetImpl<Operation *> &opsToHoist) {
  auto indVar = forOp.getInductionVar();
  if (isa<AffineIfOp>(op)) {
    if (!checkInvarianceOfNestedIfOps(op, forOp, opsWithUsers, opsToHoist)) {
      return false;
    }
  } else if (auto forOp = dyn_cast<AffineForOp>(op)) {
    if (!areAllOpsInTheBlockListInvariant(forOp.getLoopBody(), forOp,
                                          opsWithUsers, opsToHoist)) {
      return false;
    }
  } else if (isa<AffineDmaWaitOp, AffineDmaStartOp>(op)) {
    return false;
  } else if (!isa<arith::ConstantOp>(op)) {
    // Register op in the set of ops that have users.
    opsWithUsers.insert(op);
    if (isa<AffineMapAccessInterface>(op)) {
      Value memref = isa<AffineReadOpInterface>(op)
                         ? cast<AffineReadOpInterface>(op).getMemRef()
                         : cast<AffineWriteOpInterface>(op).getMemRef();
      for (auto *user : memref.getUsers()) {
        // If this memref has a user that is a DMA, give up because these
        // operations write to this memref.
        if (isa<AffineDmaStartOp, AffineDmaWaitOp>(op)) {
          return false;
        }
        // If the memref used by the load/store is used in a store elsewhere in
        // the loop nest, we do not hoist. Similarly, if the memref used in a
        // load is also being stored too, we do not hoist the load.
        if (isa<AffineWriteOpInterface>(user) ||
            (isa<AffineReadOpInterface>(user) &&
             isa<AffineWriteOpInterface>(op))) {
          if (op != user) {
            SmallVector<AffineForOp, 8> userIVs;
            getAffineForIVs(*user, &userIVs);
            // Check that userIVs don't contain the for loop around the op.
            if (llvm::is_contained(userIVs, getForInductionVarOwner(indVar))) {
              return false;
            }
          }
        }
      }
    }

    if (op->getNumOperands() == 0 && !isa<AffineYieldOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "\nNon-constant op with 0 operands\n");
      return false;
    }
  }

  for (unsigned i = 0; i < op->getNumOperands(); i++) {
    auto *operandSrc = op->getOperand(i).getDefiningOp();

    if (indVar == op->getOperand(i)) {
      LLVM_DEBUG(llvm::dbgs() << "\nLoop IV is the operand.\n");
      return false;
    }

    if (operandSrc != nullptr) {
      LLVM_DEBUG(llvm::dbgs() << *operandSrc << "\nIterating on operand src\n");

      // If the value was defined in the loop (outside of the
      // if/else region), and that operation itself wasn't meant to
      // be hoisted, then mark this operation loop dependent.
      if (opsWithUsers.count(operandSrc) && opsToHoist.count(operandSrc) == 0) {
        return false;
      }
    }
  }

  return true;
}

bool checkInvarianceOfNestedIfOps(Operation *op, AffineForOp forOp,
                                  SmallPtrSetImpl<Operation *> &opsWithUsers,
                                  SmallPtrSetImpl<Operation *> &opsToHoist) {
  assert(isa<AffineIfOp>(op));
  auto ifOp = cast<AffineIfOp>(op);

  if (!areAllOpsInTheBlockListInvariant(ifOp.getThenRegion(), forOp,
                                        opsWithUsers, opsToHoist)) {
    return false;
  }

  if (!areAllOpsInTheBlockListInvariant(ifOp.getElseRegion(), forOp,
                                        opsWithUsers, opsToHoist)) {
    return false;
  }

  return true;
}

// Checks if all ops in a region (i.e. list of blocks) are loop invariant.
bool areAllOpsInTheBlockListInvariant(
    Region &blockList, AffineForOp forOp,
    SmallPtrSetImpl<Operation *> &opsWithUsers,
    SmallPtrSetImpl<Operation *> &opsToHoist) {

  for (auto &b : blockList) {
    for (auto &op : b) {
      if (!isIndependent(&op, forOp, opsWithUsers, opsToHoist)) {
        return false;
      }
    }
  }

  return true;
}

void AIRRegularizeLoopPass::runOnAffineForNest(
    SmallVector<AffineForOp, 6> &band) {

  unsigned nestLevel = band.size();
  AffineForOp innerForOp = band[nestLevel - 1];

  // Recursively find all nested loops.
  bool endOfLoopNest = false;
  SmallVector<AffineForOp, 6> innerBand;
  innerBand.push_back(innerForOp);
  while (!endOfLoopNest) {
    auto *loopBody = innerForOp.getBody();
    endOfLoopNest = true;
    for (auto &opInLoop : *loopBody) {
      if (AffineForOp forOp = dyn_cast<AffineForOp>(opInLoop)) {
        innerBand.push_back(forOp);
        endOfLoopNest = false;
        innerForOp = forOp;
      }
    }
  }

  // Move all instructions that are in the opsToMove inside the loop
  // body if they are independent of the induction variable. Then update the
  // opsToMove and move on to the next loop.
  SmallVector<Operation *, 8> opsToMove;
  SmallPtrSet<Operation *, 8> opsWithUsers;
  SmallPtrSet<Operation *, 8> opsToHoist;
  for (AffineForOp forOp : innerBand) {
    for (auto *op : opsToMove) {
      if (!op->use_empty()) {
        opsWithUsers.insert(op);
      }
      if (isIndependent(op, forOp, opsWithUsers, opsToHoist)) {
        Block *loopBody = forOp.getBody();
        Operation &firstOp = loopBody->front();
        op->moveBefore(&firstOp);
      }
    }
    auto *newloopBody = forOp.getBody();
    opsToMove.clear();
    for (auto &op : *newloopBody) {
      if (!isa<AffineForOp>(op)) {
        opsToMove.push_back(&op);
      } else
        break;
    }
  }
}

void AIRRegularizeLoopPass::runOnOperation() {
  // Walk through all loops in a function in outermost-loop-first order. This
  // way, we iteratively move operations inside loop body until we hit a
  // dependency conflict or there are no loops at the same level.

  auto func = getOperation();

  std::vector<SmallVector<AffineForOp, 6>> bands;
  xilinx::air::getTileableBands(
      func, bands, AIRRegularizeLoopPass::affineOptAttrName, clAIROptLabel);
  for (auto loopBand : bands) {
    runOnAffineForNest(loopBand);
  }
}

} // anonymous namespace

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRRegularizeLoopPass() {
  return std::make_unique<AIRRegularizeLoopPass>();
}

} // namespace air
} // namespace xilinx
