//===- AIRMiscPasses.cpp ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// ===- AIRMiscPasses.cpp -------------------------------------------------===//
//
// Miscellaneous useful and/or experimental passes
//
// ===---------------------------------------------------------------------===//

#include "air/Transform/AIRMiscPasses.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Transform/AIRTilingUtils.h"
#include "air/Util/Dependency.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include <list>
#include <numeric>

#define DEBUG_TYPE "air-misc-passes"

using namespace mlir;

namespace xilinx {

class AIRExamplePass : public air::impl::AIRExamplePassBase<AIRExamplePass> {

public:
  AIRExamplePass() = default;
  AIRExamplePass(const AIRExamplePass &pass){};

  void runOnOperation() override;

private:
};

void AIRExamplePass::runOnOperation() {}

class AIRLinalgNamePass
    : public air::impl::AIRLinalgNamePassBase<AIRLinalgNamePass> {

public:
  AIRLinalgNamePass() = default;
  AIRLinalgNamePass(const AIRLinalgNamePass &pass){};

  void runOnOperation() override;

private:
};

void AIRLinalgNamePass::runOnOperation() {
  auto module = getOperation();
  auto ctx = module.getContext();

  unsigned id = 0;
  module.walk([&](linalg::LinalgOp op) {
    auto attr = op->getAttrOfType<StringAttr>(
        air::LinalgTransforms::kLinalgTransformMarker);
    if (!attr) {
      std::string name =
          op->getName().getStringRef().str() + std::to_string(id++);
      op->setAttr(air::LinalgTransforms::kLinalgTransformMarker,
                  StringAttr::get(ctx, name));
    }
  });
}

class AIRRemoveLinalgNamePass
    : public air::impl::AIRRemoveLinalgNamePassBase<AIRRemoveLinalgNamePass> {

public:
  AIRRemoveLinalgNamePass() = default;
  AIRRemoveLinalgNamePass(const AIRRemoveLinalgNamePass &pass){};

  void runOnOperation() override;

private:
};

void AIRRemoveLinalgNamePass::runOnOperation() {
  auto module = getOperation();

  module.walk([&](linalg::LinalgOp op) {
    auto attr = op->getAttrOfType<StringAttr>(
        air::LinalgTransforms::kLinalgTransformMarker);
    if (attr) {
      op->removeAttr(air::LinalgTransforms::kLinalgTransformMarker);
    }
  });
}

// AIRSpecializeDmaBroadcast
namespace {

/**
 * Pattern to specialize air.channel ops with broadcast_shape into multiple
 * specialized channels, and rewrite all channel.put/channel.get users
 * accordingly. (Stub for implementation)
 */
class SpecializeChannelBroadcastPattern
    : public OpRewritePattern<air::ChannelOp> {
public:
  using OpRewritePattern<air::ChannelOp>::OpRewritePattern;

  // Helper: Check if all elements of an ArrayAttr are IntegerAttr and fill
  // vector
  static bool getIntArrayAttr(ArrayAttr arr, SmallVectorImpl<int64_t> &out) {
    for (Attribute a : arr) {
      if (auto intAttr = dyn_cast<IntegerAttr>(a))
        out.push_back(intAttr.getInt());
      else
        return false;
    }
    return true;
  }

  // Helper: Check if all elements of an ArrayAttr are IntegerAttr and fill
  // vector<Attribute>
  static bool getAttrArrayAttr(ArrayAttr arr, SmallVectorImpl<Attribute> &out) {
    for (Attribute a : arr) {
      if (auto intAttr = dyn_cast<IntegerAttr>(a))
        out.push_back(intAttr);
      else
        return false;
    }
    return true;
  }

  // Helper: Get IntegerAttr from ArrayAttr at index, with type check
  static IntegerAttr getIntegerAttrAt(ArrayAttr arr, size_t idx) {
    if (idx >= arr.size())
      return nullptr;
    return dyn_cast<IntegerAttr>(arr[idx]);
  }

  // Helper: Find first dimension with size > 1, with type check
  static std::optional<std::pair<size_t, int64_t>>
  findSpecializeDim(ArrayAttr sizeAttr,
                    const SmallVector<int64_t, 4> &bcastShape) {
    for (size_t d = 0; d < sizeAttr.size(); ++d) {
      if (auto chanSizeAttr = dyn_cast<IntegerAttr>(sizeAttr[d])) {
        int64_t chanSize = chanSizeAttr.getInt();
        if (chanSize > 1) {
          return std::make_pair(d, bcastShape[d]);
        }
      }
    }
    return std::nullopt;
  }

  // Helper: Rewrite all channel.put users
  static void
  rewriteChannelPutUsers(air::ChannelOp channelOp, ModuleOp moduleOp,
                         int64_t specializeDim,
                         MutableArrayRef<air::ChannelOp> specializedChannels,
                         PatternRewriter &rewriter, Location loc) {
    for (auto put : air::getChannelPutOpThroughSymbol(channelOp, moduleOp)) {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(put);
      if (put.getIndices().size() <= (size_t)specializeDim)
        continue;
      auto idxVal = put.getIndices()[specializeDim];
      auto idxOpt = getConstantIntValue(idxVal);
      if (!idxOpt || *idxOpt < 0 ||
          *idxOpt >= (int64_t)specializedChannels.size())
        continue;
      int64_t idx = *idxOpt;
      put.setChanName(specializedChannels[idx].getSymName());
      if ((int64_t)put.getIndices().size() > specializeDim)
        put->setOperand(put.getAsyncDependencies().size() + specializeDim,
                        getValueOrCreateConstantIndexOp(
                            rewriter, loc, rewriter.getIndexAttr(0)));
    }
  }

  // Helper: Build broadcast affine set constraints (extracted from lambda)
  static void makeBroadcastAffineSetConstraints(
      size_t dimCount, int64_t specializeDim, int64_t specializeIdx,
      int64_t numCols, MLIRContext *ctx, SmallVectorImpl<AffineExpr> &exprs,
      SmallVectorImpl<bool> &eqFlags) {
    for (size_t d = 0; d < dimCount; ++d) {
      if ((int64_t)d == specializeDim) {
        exprs.push_back(getAffineSymbolExpr(d, ctx) -
                        getAffineConstantExpr(specializeIdx, ctx));
        eqFlags.push_back(true);
        continue;
      }
      // Add unconstrained range for other symbols (TODO: remove this
      // requirement)
      exprs.push_back(getAffineSymbolExpr(d, ctx));
      eqFlags.push_back(false);
      exprs.push_back(numCols - 1 - getAffineSymbolExpr(d, ctx));
      eqFlags.push_back(false);
    }
  }

  // Helper: Rewrite all channel.get users
  static LogicalResult rewriteChannelGetUsers(
      air::ChannelOp channelOp, ModuleOp moduleOp, int64_t specializeDim,
      int64_t numSegments, MutableArrayRef<air::ChannelOp> specializedChannels,
      PatternRewriter &rewriter, Location loc, MLIRContext *ctx) {
    for (auto get : air::getChannelGetOpThroughSymbol(channelOp, moduleOp)) {
      auto herd = get->getParentOfType<air::HerdOp>();
      if (!herd) {
        return get->emitOpError(
            "air.channel.get with broadcast_shape must be inside air.herd");
      }
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(get);

      SmallVector<Value, 4> herdIds;
      for (BlockArgument arg : herd.getIds())
        herdIds.push_back(Value(arg));
      // Helper lambda to create and yield a ChannelGetOp
      auto createAndYieldChannelGet = [&](int idx) -> Value {
        auto newGet = air::ChannelGetOp::create(
            rewriter, loc, get.getResultTypes(), get.getAsyncDependencies(),
            rewriter.getStringAttr(specializedChannels[idx].getSymName()),
            get.getIndices(), get.getMemref(), get.getOffsets(), get.getSizes(),
            get.getStrides());
        affine::AffineYieldOp::create(rewriter, loc, newGet.getAsyncToken());
        return newGet.getAsyncToken();
      };

      for (int64_t i = 0; i < numSegments; ++i) {
        SmallVector<AffineExpr, 4> exprs;
        SmallVector<bool, 4> eqFlags;
        makeBroadcastAffineSetConstraints(herdIds.size(), specializeDim, i,
                                          herd.getNumCols(), ctx, exprs,
                                          eqFlags);
        auto intSet = IntegerSet::get(0, herdIds.size(), exprs, eqFlags);
        SmallVector<Value, 4> setArgs = herdIds;
        if (i == 0) {
          auto aif = affine::AffineIfOp::create(
              rewriter, loc, get.getResultTypes(), intSet, setArgs, true);
          rewriter.setInsertionPointToStart(aif.getThenBlock());
          createAndYieldChannelGet(i);
          rewriter.replaceAllUsesWith(get.getAsyncToken(), aif.getResult(0));
          rewriter.setInsertionPointToStart(aif.getElseBlock());
        } else if (i < numSegments - 1) {
          auto aif = affine::AffineIfOp::create(
              rewriter, loc, get.getResultTypes(), intSet, setArgs, true);
          rewriter.setInsertionPointToStart(aif.getThenBlock());
          createAndYieldChannelGet(i);
          rewriter.setInsertionPointAfter(aif);
          SmallVector<Value, 1> parentBlockYieldToken{aif.getResult(0)};
          affine::AffineYieldOp::create(rewriter, loc, parentBlockYieldToken);
          rewriter.setInsertionPointToStart(aif.getElseBlock());
        } else {
          createAndYieldChannelGet(i);
        }
      }
      rewriter.eraseOp(get);
    }
    return success();
  }

  LogicalResult matchAndRewrite(air::ChannelOp channelOp,
                                PatternRewriter &rewriter) const override {
    auto loc = rewriter.getUnknownLoc();
    auto ctx = rewriter.getContext();

    // Only match channels with a nontrivial broadcast_shape
    auto bcastShapeAttr =
        channelOp->getAttrOfType<ArrayAttr>("broadcast_shape");
    if (!bcastShapeAttr)
      return failure();
    SmallVector<int64_t, 4> bcastShape;
    if (!getIntArrayAttr(bcastShapeAttr, bcastShape))
      return failure();

    // Only specialize if shape is not empty and not all ones
    if (bcastShape.empty() ||
        llvm::all_of(bcastShape, [](int64_t d) { return d == 1; }))
      return failure();

    // Require broadcast_shape rank to match channel indices rank (empty indices
    // means rank 1)
    ArrayAttr sizeAttr = channelOp.getSize();
    unsigned channelRank = sizeAttr.empty() ? 1 : sizeAttr.size();
    if (bcastShape.size() != channelRank)
      return failure();

    // Find the first dimension with size >1 (with type check)
    auto specializeDimOpt = findSpecializeDim(sizeAttr, bcastShape);
    if (!specializeDimOpt)
      return failure();
    int64_t specializeDim = specializeDimOpt->first;
    int64_t numSegments = specializeDimOpt->second;
    if (specializeDim < 0 || numSegments <= 1)
      return failure();

    // Create specialized channels
    SmallVector<air::ChannelOp, 4> specializedChannels;
    auto moduleOp = channelOp->getParentOfType<ModuleOp>();
    auto baseName = channelOp.getSymName().str();

    // Copy the original channel's size attribute and set the specialized dim to
    // 1
    SmallVector<Attribute> newSize(sizeAttr.begin(), sizeAttr.end());
    if ((int64_t)newSize.size() > specializeDim)
      newSize[specializeDim] = rewriter.getI64IntegerAttr(1);

    // Prepare new broadcast_shape: original broadcast_shape with specialized
    // dim set to 1
    SmallVector<Attribute> newBcastShapeAttrs;
    if (!getAttrArrayAttr(bcastShapeAttr, newBcastShapeAttrs))
      return failure();
    if ((int64_t)newBcastShapeAttrs.size() > specializeDim)
      newBcastShapeAttrs[specializeDim] = rewriter.getI64IntegerAttr(1);

    for (int64_t i = 0; i < numSegments; ++i) {
      std::string newName = baseName + "_" + std::to_string(i);
      auto newChan = air::ChannelOp::create(
          rewriter, loc, rewriter.getStringAttr(newName),
          rewriter.getArrayAttr(newSize), channelOp.getChannelType());
      newChan->setAttr("broadcast_shape", ArrayAttr::get(rewriter.getContext(),
                                                         newBcastShapeAttrs));
      specializedChannels.push_back(newChan);
    }

    // Rewrite all channel.put users to use the specialized channels
    rewriteChannelPutUsers(channelOp, moduleOp, specializeDim,
                           specializedChannels, rewriter, loc);

    // Rewrite all channel.get users to use the specialized channels
    if (failed(rewriteChannelGetUsers(channelOp, moduleOp, specializeDim,
                                      numSegments, specializedChannels,
                                      rewriter, loc, ctx)))
      return failure();

    // Remove the original channel op
    rewriter.eraseOp(channelOp);
    return success();
  }
};

/**
 * Pattern to simplify DMA indices for air::DmaMemcpyNdOp with a broadcast_set
 * attribute.
 * - Propagates constant affine expressions through dependency chains.
 * - Replaces operands with constants when possible.
 * - Removes async dependencies as needed.
 */
class SimplifyDmaIndicesWithAffineSetPattern
    : public OpRewritePattern<air::DmaMemcpyNdOp> {
public:
  using OpRewritePattern<air::DmaMemcpyNdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(air::DmaMemcpyNdOp memcpyOp,
                                PatternRewriter &rewriter) const override {
    auto *ctx = memcpyOp->getContext();
    if (!memcpyOp->hasAttr("broadcast_set"))
      return failure();
    auto broadcastSet =
        memcpyOp->getAttrOfType<mlir::IntegerSetAttr>("broadcast_set");
    if (!broadcastSet)
      return failure();

    // Gather dependency history and broadcast pattern constraints
    std::vector<Operation *> depOpHistory;
    auto depTuples = traceDependentHerdId(memcpyOp);
    const auto is = broadcastSet.getValue();
    const auto &constraints = is.getConstraints();
    const auto &eqFlags = is.getEqFlags();

    // Map from herd dimension to constant affine expr and corresponding DMA
    // operand
    SmallVector<AffineExpr, 2> herdDimConstExpr = {nullptr, nullptr};
    SmallVector<Value, 2> herdDimToDmaOperand = {nullptr, nullptr};

    // Analyze dependency tuples to extract constant expressions for herd dims
    for (const auto &depTuple : depTuples) {
      for (Value v : std::get<1>(depTuple)) {
        auto herdArgOwner = air::getHerdArgOwner(v);
        if (!herdArgOwner)
          continue;
        for (unsigned j = 0; j < herdDimConstExpr.size(); ++j) {
          if (v != herdArgOwner.getIds()[j])
            continue;
          for (unsigned i = 0; i < constraints.size(); ++i) {
            const auto &c = constraints[i];
            if (!c.isFunctionOfSymbol(j) || !eqFlags[i])
              continue;
            int eval = air::evaluateSymbolEqualityInSet(c, ctx);
            herdDimConstExpr[j] = getAffineConstantExpr(eval, ctx);
            herdDimToDmaOperand[j] = std::get<0>(depTuple);
            depOpHistory.insert(depOpHistory.end(),
                                std::get<2>(depTuple).begin(),
                                std::get<2>(depTuple).end());
          }
        }
      }
    }

    // Helper lambdas for arith ops
    auto propagateAdd = [&](arith::AddIOp arithOp, unsigned j) {
      arith::ConstantIndexOp addOperand = nullptr;
      if (arithOp.getLhs().getDefiningOp() &&
          dyn_cast<arith::ConstantIndexOp>(arithOp.getLhs().getDefiningOp())) {
        addOperand =
            dyn_cast<arith::ConstantIndexOp>(arithOp.getLhs().getDefiningOp());
      } else if (arithOp.getRhs().getDefiningOp() &&
                 dyn_cast<arith::ConstantIndexOp>(
                     arithOp.getRhs().getDefiningOp())) {
        addOperand =
            dyn_cast<arith::ConstantIndexOp>(arithOp.getRhs().getDefiningOp());
      } else {
        herdDimConstExpr[j] = nullptr;
        return;
      }
      int64_t acc = addOperand.value();
      if (!isa<AffineConstantExpr>(herdDimConstExpr[j])) {
        arithOp->emitOpError("non-constant affine expression.");
        herdDimConstExpr[j] = nullptr;
        return;
      }
      acc += dyn_cast<AffineConstantExpr>(herdDimConstExpr[j]).getValue();
      herdDimConstExpr[j] = getAffineConstantExpr(acc, ctx);
    };
    auto propagateMul = [&](arith::MulIOp arithOp, unsigned j) {
      arith::ConstantIndexOp mulOperand = nullptr;
      if (arithOp.getLhs().getDefiningOp() &&
          dyn_cast<arith::ConstantIndexOp>(arithOp.getLhs().getDefiningOp())) {
        mulOperand =
            dyn_cast<arith::ConstantIndexOp>(arithOp.getLhs().getDefiningOp());
      } else if (arithOp.getRhs().getDefiningOp() &&
                 dyn_cast<arith::ConstantIndexOp>(
                     arithOp.getRhs().getDefiningOp())) {
        mulOperand =
            dyn_cast<arith::ConstantIndexOp>(arithOp.getRhs().getDefiningOp());
      } else {
        herdDimConstExpr[j] = nullptr;
        return;
      }
      int64_t mul = mulOperand.value();
      if (!isa<AffineConstantExpr>(herdDimConstExpr[j])) {
        arithOp->emitOpError("non-constant affine expression.");
        herdDimConstExpr[j] = nullptr;
        return;
      }
      mul *= dyn_cast<AffineConstantExpr>(herdDimConstExpr[j]).getValue();
      herdDimConstExpr[j] = getAffineConstantExpr(mul, ctx);
    };

    // Propagate constants through dependency op history (reverse order)
    for (auto it = depOpHistory.rbegin(); it != depOpHistory.rend(); ++it) {
      if (auto execOp = dyn_cast<air::ExecuteOp>(*it)) {
        Operation *op = &execOp.getChildOps().front();
        if (auto applyOp = dyn_cast<affine::AffineApplyOp>(op)) {
          if (applyOp.getNumOperands() != 1)
            return failure();
          auto map = applyOp.getAffineMap();
          for (unsigned j = 0; j < herdDimConstExpr.size(); ++j) {
            if (herdDimConstExpr[j]) {
              auto newMap = map.replace(getAffineSymbolExpr(0, ctx),
                                        herdDimConstExpr[j], 0, 1);
              int constInt =
                  simplifyAffineMap(newMap).getSingleConstantResult();
              herdDimConstExpr[j] = getAffineConstantExpr(constInt, ctx);
              auto asyncMemcpyOp =
                  dyn_cast<air::AsyncOpInterface>(memcpyOp.getOperation());
              eraseAsyncDependencyFromAsyncOp(asyncMemcpyOp,
                                              execOp.getAsyncToken());
            }
          }
        } else if (auto addOp = dyn_cast<arith::AddIOp>(op)) {
          for (unsigned j = 0; j < herdDimConstExpr.size(); ++j) {
            if (herdDimConstExpr[j]) {
              propagateAdd(addOp, j);
              auto asyncMemcpyOp =
                  dyn_cast<air::AsyncOpInterface>(memcpyOp.getOperation());
              eraseAsyncDependencyFromAsyncOp(asyncMemcpyOp,
                                              execOp.getAsyncToken());
            }
          }
        } else if (auto mulOp = dyn_cast<arith::MulIOp>(op)) {
          for (unsigned j = 0; j < herdDimConstExpr.size(); ++j) {
            if (herdDimConstExpr[j]) {
              propagateMul(mulOp, j);
              auto asyncMemcpyOp =
                  dyn_cast<air::AsyncOpInterface>(memcpyOp.getOperation());
              eraseAsyncDependencyFromAsyncOp(asyncMemcpyOp,
                                              execOp.getAsyncToken());
            }
          }
        }
      }
    }

    // Mutate memcpy op: replace operands with constants if possible
    OpBuilder builder(memcpyOp);
    const Location loc = memcpyOp->getLoc();
    bool opIsUpdated = false;
    for (unsigned i = 0; i < herdDimConstExpr.size(); ++i) {
      if (!herdDimConstExpr[i] || !herdDimToDmaOperand[i])
        continue;
      int opOperandId = -1;
      for (unsigned j = 0; j < memcpyOp->getNumOperands(); ++j)
        if (memcpyOp->getOperand(j) == herdDimToDmaOperand[i])
          opOperandId = j;
      if (opOperandId < 0)
        continue;
      auto val = dyn_cast<AffineConstantExpr>(herdDimConstExpr[i]).getValue();
      auto cop = arith::ConstantIndexOp::create(builder, loc, val);
      memcpyOp->getOpOperand(opOperandId).assign(cop);
      opIsUpdated = true;
    }
    // If any update was made, signal success so the pattern infra will re-run
    return opIsUpdated ? success() : failure();
  }
};

class SpecializeDmaBroadcastPattern
    : public OpRewritePattern<air::DmaMemcpyNdOp> {
public:
  using OpRewritePattern<air::DmaMemcpyNdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(air::DmaMemcpyNdOp memcpyOp,
                                PatternRewriter &rewriter) const override {
    // Check for required context: inside a herd, has broadcast_pattern
    auto *ctx = memcpyOp->getContext();
    auto herdOp = memcpyOp->getParentOfType<air::HerdOp>();
    if (!herdOp)
      return rewriter.notifyMatchFailure(memcpyOp, "not inside air.herd");
    auto broadcastPatternAttr =
        memcpyOp->getAttrOfType<mlir::IntegerSetAttr>("broadcast_pattern");
    if (!broadcastPatternAttr)
      return rewriter.notifyMatchFailure(memcpyOp, "missing broadcast_pattern");
    const auto herdIds = herdOp.getIds();
    const Location loc = memcpyOp->getLoc();
    const IntegerSet is = broadcastPatternAttr.getValue();
    const auto &constraints = is.getConstraints();
    const auto &eqFlags = is.getEqFlags();

    // Compute number of segments
    unsigned numSegments = 1;
    const SmallVector<AffineExpr, 1> zeroSyms{getAffineConstantExpr(0, ctx)};
    for (const AffineExpr &c : constraints) {
      if (c.isSymbolicOrConstant()) {
        auto newC = c.replaceSymbols(zeroSyms);
        if (auto expr =
                dyn_cast<AffineConstantExpr>(simplifyAffineExpr(newC, 0, 1))) {
          if (expr.getValue() != 0)
            numSegments = expr.getValue() + 1;
        }
      }
    }

    // Helper to clone and yield a DMA op with updated attributes
    auto cloneAndYield = [&](PatternRewriter &rewriter, Operation *origOp,
                             IntegerSet intSet) -> Value {
      auto cloned = rewriter.clone(*origOp);
      cloned->removeAttr("broadcast_pattern");
      cloned->setAttr("broadcast_set", mlir::IntegerSetAttr::get(intSet));
      auto asyncIface = dyn_cast<air::AsyncOpInterface>(cloned);
      SmallVector<Value, 1> yieldToken{asyncIface.getAsyncToken()};
      affine::AffineYieldOp::create(rewriter, cloned->getLoc(), yieldToken);
      return asyncIface.getAsyncToken();
    };

    // If only one segment, avoid unnecessary loop
    if (numSegments == 1) {
      // Build constraints for the single segment
      SmallVector<AffineExpr, 2> newConstraints;
      SmallVector<bool, 2> newEqFlags;
      const SmallVector<AffineExpr, 1> iSyms{getAffineConstantExpr(0, ctx)};
      const SmallVector<AffineExpr, 2> syms{getAffineSymbolExpr(0, ctx),
                                            getAffineSymbolExpr(1, ctx)};
      int cIter = 0;
      for (const AffineExpr &c : constraints) {
        if (!c.isSymbolicOrConstant()) {
          auto newC = c.replaceSymbols(iSyms).replaceDims(syms);
          newConstraints.push_back(newC);
          newEqFlags.push_back(eqFlags[cIter]);
        }
        ++cIter;
      }
      auto intSet = IntegerSet::get(0, 2, newConstraints, newEqFlags);
      SmallVector<Value, 2> intSetArgs{herdIds[0], herdIds[1]};
      auto aif = affine::AffineIfOp::create(rewriter, loc,
                                            air::AsyncTokenType::get(ctx),
                                            intSet, intSetArgs, true);
      rewriter.setInsertionPointToStart(aif.getThenBlock());
      cloneAndYield(rewriter, memcpyOp.getOperation(), intSet);
      // Reconnect dependency graph
      auto asyncMemcpyOp =
          dyn_cast<air::AsyncOpInterface>(memcpyOp.getOperation());
      asyncMemcpyOp.getAsyncToken().replaceAllUsesWith(aif.getResult(0));
      rewriter.setInsertionPointToStart(aif.getElseBlock());
      auto waitAllOp =
          air::WaitAllOp::create(rewriter, loc, air::AsyncTokenType::get(ctx),
                                 memcpyOp.getAsyncDependencies());
      affine::AffineYieldOp::create(
          rewriter, loc, SmallVector<Value>{waitAllOp.getAsyncToken()});
      rewriter.eraseOp(memcpyOp);
      return success();
    }

    // Multi-segment case
    for (unsigned i = 0; i < numSegments; ++i) {
      SmallVector<AffineExpr, 2> newConstraints;
      SmallVector<bool, 2> newEqFlags;
      const SmallVector<AffineExpr, 1> iSyms{getAffineConstantExpr(i, ctx)};
      const SmallVector<AffineExpr, 2> syms{getAffineSymbolExpr(0, ctx),
                                            getAffineSymbolExpr(1, ctx)};
      int cIter = 0;
      for (const AffineExpr &c : constraints) {
        if (!c.isSymbolicOrConstant()) {
          auto newC = c.replaceSymbols(iSyms).replaceDims(syms);
          newConstraints.push_back(newC);
          newEqFlags.push_back(eqFlags[cIter]);
        }
        ++cIter;
      }
      auto intSet = IntegerSet::get(0, 2, newConstraints, newEqFlags);
      SmallVector<Value, 2> intSetArgs{herdIds[0], herdIds[1]};
      if (i == 0) {
        auto aif = affine::AffineIfOp::create(rewriter, loc,
                                              air::AsyncTokenType::get(ctx),
                                              intSet, intSetArgs, true);
        rewriter.setInsertionPointToStart(aif.getThenBlock());
        cloneAndYield(rewriter, memcpyOp.getOperation(), intSet);
        auto asyncMemcpyOp =
            dyn_cast<air::AsyncOpInterface>(memcpyOp.getOperation());
        asyncMemcpyOp.getAsyncToken().replaceAllUsesWith(aif.getResult(0));
        rewriter.setInsertionPointToStart(aif.getElseBlock());
      } else if (i < numSegments - 1) {
        auto aif = affine::AffineIfOp::create(rewriter, loc,
                                              air::AsyncTokenType::get(ctx),
                                              intSet, intSetArgs, true);
        rewriter.setInsertionPointToStart(aif.getThenBlock());
        cloneAndYield(rewriter, memcpyOp.getOperation(), intSet);
        rewriter.setInsertionPointAfter(aif);
        SmallVector<Value, 1> parentBlockYieldToken{aif.getResult(0)};
        affine::AffineYieldOp::create(rewriter, loc, parentBlockYieldToken);
        rewriter.setInsertionPointToStart(aif.getElseBlock());
      } else {
        cloneAndYield(rewriter, memcpyOp.getOperation(), intSet);
      }
    }
    rewriter.eraseOp(memcpyOp);
    return success();
  }
};
} // end anonymous namespace

class AIRSpecializeDmaBroadcast
    : public air::impl::AIRSpecializeDmaBroadcastBase<
          AIRSpecializeDmaBroadcast> {

public:
  AIRSpecializeDmaBroadcast() = default;
  AIRSpecializeDmaBroadcast(const AIRSpecializeDmaBroadcast &pass){};

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps) {
      runOnFunction(f);
      // Renumber the air dma op ids
      air::renumberMemcpyIfOps(&f.getRegion());
    }
    {
      RewritePatternSet patterns(module.getContext());
      patterns.add<SpecializeChannelBroadcastPattern>(module.getContext());
      (void)applyPatternsGreedily(module, std::move(patterns));
    }
  }

  void runOnFunction(func::FuncOp f) {
    // Phase 1: Specialize broadcastable air.dma_memcpy_nd and air.channel
    {
      RewritePatternSet patterns(f.getContext());
      patterns.add<SpecializeDmaBroadcastPattern>(f.getContext());
      (void)applyPatternsGreedily(f, std::move(patterns));
    }
    // Phase 2: Simplify DMA indices with affine set
    {
      RewritePatternSet patterns(f.getContext());
      patterns.add<SimplifyDmaIndicesWithAffineSetPattern>(f.getContext());
      (void)applyPatternsGreedily(f, std::move(patterns));
    }
  }

private:
};

class AIRFuseParallelHerdPass
    : public air::impl::AIRFuseParallelHerdPassBase<AIRFuseParallelHerdPass> {

public:
  AIRFuseParallelHerdPass() = default;
  AIRFuseParallelHerdPass(const AIRFuseParallelHerdPass &pass){};

  void runOnOperation() override;

private:
};

void AIRFuseParallelHerdPass::runOnOperation() {

  auto module = getOperation();
  // auto ctx = module.getContext();

  air::HerdOp launchOp = nullptr;
  scf::ParallelOp parOp = nullptr;

  module.walk([&](air::HerdOp launch) {
    // launch must be enclosed by scf.parallel
    parOp = launch->getParentOfType<scf::ParallelOp>();
    if (!parOp)
      return;

    // launch must be at the top level of the scf.parallel
    if (parOp.getBody() != launch->getBlock())
      return;

    launchOp = launch;
  });

  if (!launchOp || !parOp)
    return;

  // if the herd launch is size 1 in one dimension
  // and the herd launch is enclosed by a 1-d scf.parallel
  // then we try to fuse the scf.parallel onto the herd launch
  auto herd_size = launchOp.getSizeOperands();
  int64_t herd_size_x = launchOp.getNumCols();
  int64_t herd_size_y = launchOp.getNumRows();
  if (herd_size_x != 1 && herd_size_y != 1)
    return;

  OpBuilder b(parOp);
  SmallVector<Value, 2> dims;
  if (herd_size_x == 1)
    dims = {parOp.getUpperBound()[0], herd_size[1]};
  else
    dims = {herd_size[0], parOp.getUpperBound()[0]};

  SmallVector<Value, 8> args;
  SmallVector<Value, 4> constants;
  llvm::SetVector<Value> region_args;

  getUsedValuesDefinedAbove(parOp.getRegion(), region_args);
  for (Value v : region_args) {
    if (isa_and_present<arith::ConstantOp, ub::PoisonOp>(v.getDefiningOp()))
      constants.push_back(v);
    else
      args.push_back(v);
  }

  auto newLaunchOp = air::HerdOp::create(
      b, parOp.getLoc(), launchOp.getAsyncDependencies(), dims, args,
      launchOp->getNumResults() > 0, launchOp->getAttrs());

  IRMapping remap;
  remap.map(parOp.getInductionVars()[0], (herd_size_x == 1)
                                             ? newLaunchOp.getIds()[0]
                                             : newLaunchOp.getIds()[1]);

  b.setInsertionPointToStart(&newLaunchOp.getBody().front());

  for (auto &o : *parOp.getBody()) {
    if (isa<air::HerdOp>(o)) {
      int idx = 0;
      remap.map(launchOp.getSize()[0], launchOp.getSizeOperands()[0]);
      remap.map(launchOp.getSize()[1], launchOp.getSizeOperands()[1]);
      if (herd_size_x == 1)
        remap.map(launchOp.getIds()[0], launchOp.getSizeOperands()[0]);
      else
        remap.map(launchOp.getIds()[0], newLaunchOp.getIds()[0]);
      if (herd_size_x == 1)
        remap.map(launchOp.getIds()[1], newLaunchOp.getIds()[1]);
      else
        remap.map(launchOp.getIds()[1], launchOp.getSizeOperands()[1]);
      for (auto &a : launchOp.getKernelArguments()) {
        auto v = launchOp.getKernelOperand(idx++);
        remap.map(a, remap.lookupOrDefault(v));
      }
      for (auto &ho : launchOp.getBody().front()) {
        if (isa<air::HerdTerminatorOp>(ho))
          continue;
        b.clone(ho, remap);
      }
    } else if (isa<scf::YieldOp>(o)) {
      continue;
    } else {
      b.clone(o, remap);
    }
  }

  b.setInsertionPointToStart(&newLaunchOp.getBody().front());
  for (auto c : constants) {
    replaceAllUsesInRegionWith(c, b.clone(*c.getDefiningOp())->getResult(0),
                               newLaunchOp.getRegion());
  }

  int idx = 0;
  auto kernel_args = newLaunchOp.getKernelArguments();
  for (Value v : args)
    replaceAllUsesInRegionWith(v, kernel_args[idx++], newLaunchOp.getRegion());

  parOp.erase();
}

class AIRRenumberDmaIdPass
    : public air::impl::AIRRenumberDmaIdPassBase<AIRRenumberDmaIdPass> {

public:
  AIRRenumberDmaIdPass() = default;
  AIRRenumberDmaIdPass(const AIRRenumberDmaIdPass &pass){};

  void runOnOperation() override;

private:
};

void AIRRenumberDmaIdPass::runOnOperation() {
  auto func = getOperation();
  if (clMode == "global") {
    // Renumber DMA ops in func op.
    air::renumberMemcpyIfOps(&func.getRegion());
  } else if (clMode == "herd") {
    // Renumber DMA ops in herd op.
    func.walk(
        [](air::HerdOp herd) { air::renumberMemcpyIfOps(&herd.getBody()); });
  } else if (clMode == "segment") {
    // Renumber DMA ops in segment op.
    func.walk([](air::SegmentOp segment) {
      air::renumberMemcpyIfOps(&segment.getBody());
    });
  } else if (clMode == "launch") {
    // Renumber DMA ops in launch op.
    func.walk([](air::LaunchOp launch) {
      air::renumberMemcpyIfOps(&launch.getBody());
    });
  } else
    func->emitError("Unknown dma renumber mode. Supported modes: global, herd, "
                    "segment, launch");
}

class ParallelToForConversion : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (!op->getParentOfType<air::HerdOp>())
      return failure();

    IRMapping remap;
    scf::ForOp forOp = nullptr;
    for (unsigned int i = 0; i < op.getNumLoops(); i++) {
      auto lb = op.getLowerBound()[i];
      auto ub = op.getUpperBound()[i];
      auto step = op.getStep()[i];
      forOp = scf::ForOp::create(rewriter, op->getLoc(), lb, ub, step);
      rewriter.setInsertionPointToStart(forOp.getBody());
      remap.map(op.getInductionVars()[i], forOp.getInductionVar());
    }
    for (Operation &o : op.getBody()->getOperations())
      if (!isa<scf::ReduceOp>(o))
        rewriter.clone(o, remap);

    rewriter.eraseOp(op);
    return success();
  }
};

class AIRLowerHerdParallelPass
    : public air::impl::AIRLowerHerdParallelPassBase<AIRLowerHerdParallelPass> {

public:
  AIRLowerHerdParallelPass() = default;
  AIRLowerHerdParallelPass(const AIRLowerHerdParallelPass &pass){};

  void runOnOperation() override;

private:
};

void AIRLowerHerdParallelPass::runOnOperation() {
  auto op = getOperation();
  auto context = op->getContext();
  RewritePatternSet patterns(context);
  patterns.add<ParallelToForConversion>(context);
  (void)applyPatternsGreedily(op, std::move(patterns));
}

class AIRLabelBroadcastChannelWithTilePass
    : public air::impl::AIRLabelBroadcastChannelWithTilePassBase<
          AIRLabelBroadcastChannelWithTilePass> {

public:
  AIRLabelBroadcastChannelWithTilePass() = default;
  AIRLabelBroadcastChannelWithTilePass(
      const AIRLabelBroadcastChannelWithTilePass &pass){};

  void runOnOperation() override;

private:
};

void AIRLabelBroadcastChannelWithTilePass::runOnOperation() {
  // Util. functions for air dependency
  xilinx::air::dependencyCanonicalizer canonicalizer;
  auto func = getOperation();
  auto ctx = func.getContext();
  func.walk([&](air::ChannelInterface op) {
    auto aif = op->getParentOfType<affine::AffineIfOp>();
    auto herd = op->getParentOfType<air::HerdOp>();
    if (aif && herd) {
      // Fast forward through affine.if nest
      std::vector<Operation *> affine_if_nest;
      Operation *spatial_loop = nullptr;
      getAffineIfNestAndSpatialLoopFromOp(op, affine_if_nest, spatial_loop);
      std::vector<int> position;
      std::vector<Attribute> tiles;
      OpBuilder builder(op);
      for (unsigned i = 0; i < canonicalizer.getTripCountInHierarchyOp(herd);
           i++) {
        SmallVector<NamedAttribute, 5> attrs;
        auto current_position = canonicalizer.getPositionFromIterator(i, herd);
        if (positionHitsAffineIfCondition(op, spatial_loop, affine_if_nest,
                                          current_position)) {
          attrs.push_back(
              NamedAttribute(StringAttr::get(ctx, "col"),
                             builder.getI64IntegerAttr(current_position[0])));
          attrs.push_back(
              NamedAttribute(StringAttr::get(ctx, "row"),
                             builder.getI64IntegerAttr(current_position[1])));
          tiles.push_back(DictionaryAttr::get(ctx, attrs));
        }
      }
      op->setAttr("tile", ArrayAttr::get(ctx, tiles));
    }
  });
}

class AIRCollapseHerdPass
    : public air::impl::AIRCollapseHerdPassBase<AIRCollapseHerdPass> {

public:
  AIRCollapseHerdPass() = default;
  AIRCollapseHerdPass(const AIRCollapseHerdPass &pass){};
  AIRCollapseHerdPass(const ::xilinx::air::AIRCollapseHerdPassOptions &options)
      : AIRCollapseHerdPassBase(options) {}

  void runOnOperation() override;

private:
};

void AIRCollapseHerdPass::runOnOperation() {
  SmallVector<air::HerdOp> herds;
  auto func = getOperation();
  int maximumColumnSize = clMaxColSize;
  if (clMaxColSize == -1)
    maximumColumnSize = INT_MAX; // max-col-size disabled.
  func.walk([&](air::HerdOp op) {
    if (op.getNumCols() != 1 && op.getNumDims() == 2 &&
        op.getNumRows() * op.getNumCols() <= (unsigned)maximumColumnSize)
      herds.push_back(op);
  });

  for (auto h : herds) {
    OpBuilder outsideBuilder(h);
    Location loc = h.getLoc();

    // Assumption: herd is two-dimensional, and both of which we collapse into a
    // single dim.
    if (h.getNumDims() != 2)
      continue;
    SmallVector<unsigned> dims = {0, 1};

    // Combine iteration spaces.
    SmallVector<Value, 3> lowerBounds, upperBounds, steps;
    auto cst0 = arith::ConstantIndexOp::create(outsideBuilder, loc, 0);
    auto cst1 = arith::ConstantIndexOp::create(outsideBuilder, loc, 1);
    // First dimension size set to one, i.e. a single column
    lowerBounds.push_back(cst0);
    steps.push_back(cst1);
    upperBounds.push_back(cst1);
    // Second dimension onwards
    Value newUpperBound =
        arith::ConstantIndexOp::create(outsideBuilder, loc, 1);
    for (auto idx : dims) {
      newUpperBound = arith::MulIOp::create(
          outsideBuilder, loc, newUpperBound,
          h->getOperand(h.getAsyncDependencies().size() + idx));
    }
    lowerBounds.push_back(cst0);
    steps.push_back(cst1);
    upperBounds.push_back(newUpperBound);

    OpBuilder insideBuilder(h);
    insideBuilder.setInsertionPointToStart(&h.getBody().front());
    auto old_upper_bound = mlir::getConstantIntValue(
        h.getOperand(h.getAsyncDependencies().size() + dims[1]));
    if (!old_upper_bound)
      return; // Found air.herd with dynamic shape. NYI.
    auto old_upper_b_v =
        arith::ConstantIndexOp::create(insideBuilder, loc, *old_upper_bound);

    // Determine the current induction value's current loop iteration
    Value iv_1 = arith::RemSIOp::create(insideBuilder, loc, h.getIds()[1],
                                        old_upper_b_v);
    llvm::cast<Value>(h.getIds()[1])
        .replaceAllUsesExcept(iv_1, iv_1.getDefiningOp());

    // Remove the effect of the current induction value to prepare for
    // the next value.
    Value iv_0 = arith::DivSIOp::create(insideBuilder, loc, h.getIds()[1],
                                        old_upper_b_v);
    replaceAllUsesInRegionWith(h.getIds()[0], iv_0, h.getBody());

    // Update upper bounds.
    int operandsIdxOffset = h.getAsyncDependencies().size();
    for (unsigned i = operandsIdxOffset; i < operandsIdxOffset + h.getNumDims();
         i++) {
      h->getOpOperand(i).assign(upperBounds[i - operandsIdxOffset]);
    }
  }
}

// Controls the dimension ordering for the fused herd:
// - OuterInner: outer herd's non-unit dimension first, then inner's
// - InnerOuter: inner herd's non-unit dimension first, then outer's
enum class DimOrder { OuterInner, InnerOuter };

class AIRFuseNestedHerdPass
    : public air::impl::AIRFuseNestedHerdPassBase<AIRFuseNestedHerdPass> {

public:
  AIRFuseNestedHerdPass() = default;
  AIRFuseNestedHerdPass(const AIRFuseNestedHerdPass &pass){};
  AIRFuseNestedHerdPass(
      const ::xilinx::air::AIRFuseNestedHerdPassOptions &options)
      : AIRFuseNestedHerdPassBase(options) {}

  void runOnOperation() override;

private:
};

// Pattern that matches an outer air.herd directly containing a single inner
// air.herd, with no intervening side-effecting ops, and collapses them into
// a single fused herd with a 2D tile space.
//
// Preconditions:
//  - The outer and inner herds must each have exactly one non-unit dimension
//    (either rows > 1, cols == 1 OR rows == 1, cols > 1).
//  - No other herds or non-pure ops between outer herd start and inner herd.
//  - The fusion order is determined by the DimOrder enum.
struct NestedHerdCollapsePattern : public OpRewritePattern<air::HerdOp> {
  NestedHerdCollapsePattern(MLIRContext *ctx, DimOrder &order)
      : OpRewritePattern<air::HerdOp>(ctx), order(order) {}

  LogicalResult matchAndRewrite(air::HerdOp outer,
                                PatternRewriter &rewriter) const override {
    Location loc = outer.getLoc();
    //===------------------------------------------------------------------===//
    // Step 1: Identify inner herd and check for "perfect nesting".
    // Perfect nesting here means: exactly one inner herd, and no side-effecting
    // ops between the outer herd entry and that inner herd.
    //===------------------------------------------------------------------===//
    auto &outerBody = outer.getBody().front();
    air::HerdOp inner = nullptr;
    for (Operation &op : outerBody.without_terminator()) {
      if (auto h = dyn_cast<air::HerdOp>(&op)) {
        if (inner)
          return rewriter.notifyMatchFailure(
              outer, "multiple inner herds, not perfect");
        inner = h;
      } else if (!mlir::isMemoryEffectFree(&op))
        return rewriter.notifyMatchFailure(
            outer, "side effects in between outer and inner herds");
    }
    if (!inner)
      return failure();

    //===------------------------------------------------------------------===//
    // Step 2: Validate shapes. Each herd must have exactly one non-unit dim.
    // This constraint ensures that the fused herd is still 2D.
    //===------------------------------------------------------------------===//
    auto nonUnitDims = [&](air::HerdOp h) {
      int n = (h.getNumRows() > 1) + (h.getNumCols() > 1);
      return n;
    };
    if (nonUnitDims(outer) != 1 || nonUnitDims(inner) != 1) {
      return rewriter.notifyMatchFailure(
          outer, "one of the herds to be fused has more than one non-unit "
                 "dimensions, so that they cannot fuse into one 2D herd");
    }

    // Determine extents (number of tiles) for each herd’s non-unit dimension.
    uint64_t oTx =
        outer.getNumCols() > 1 ? outer.getNumCols() : outer.getNumRows();
    uint64_t iTx =
        inner.getNumCols() > 1 ? inner.getNumCols() : inner.getNumRows();

    // Determine which tile-id in each herd is the “target” (non-unit) dimension
    // and which is the dummy (unit) dimension.
    std::pair<Value, Value> outerTargetIVAndDummyIVPair =
        outer.getNumCols() > 1
            ? std::make_pair(outer.getIds()[0], outer.getIds()[1])
            : std::make_pair(outer.getIds()[1], outer.getIds()[0]);
    std::pair<Value, Value> innerTargetIVAndDummyIVPair =
        inner.getNumCols() > 1
            ? std::make_pair(inner.getIds()[0], inner.getIds()[1])
            : std::make_pair(inner.getIds()[1], inner.getIds()[0]);

    // Compute fused herd dimensions based on DimOrder.
    uint64_t newTy = (order == DimOrder::OuterInner) ? oTx : iTx;
    uint64_t newTx = (order == DimOrder::OuterInner) ? iTx : oTx;
    Value newTyVal = getValueOrCreateConstantIndexOp(
        rewriter, loc, rewriter.getIndexAttr(newTy));
    Value newTxVal = getValueOrCreateConstantIndexOp(
        rewriter, loc, rewriter.getIndexAttr(newTx));

    //===------------------------------------------------------------------===//
    // Step 3: Create fused herd op. Inherit operands from outer herd, as herd
    // has `IsolatedFromAbove` trait.
    //===------------------------------------------------------------------===//
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(outer);

    SetVector<Value> fusedOpers;
    fusedOpers.insert(outer.getKernelOperands().begin(),
                      outer.getKernelOperands().end());

    auto fused = air::HerdOp::create(rewriter, loc,
                                     /*grid*/ ValueRange{newTyVal, newTxVal},
                                     /*args*/ fusedOpers.takeVector());

    //===------------------------------------------------------------------===//
    // Step 4: Map tile IDs and tied kernel arguments.
    // Map the "target" IV of each herd to the correct fused herd IV based on
    // DimOrder, and map the "dummy" IVs to constant 0.
    //===------------------------------------------------------------------===//
    Value t0 = fused.getIds()[0];
    Value t1 = fused.getIds()[1];
    Value c0 = getValueOrCreateConstantIndexOp(rewriter, loc,
                                               rewriter.getIndexAttr(0));
    auto [to0, to1] = outerTargetIVAndDummyIVPair;
    auto [ti0, ti1] = innerTargetIVAndDummyIVPair;

    IRMapping map;
    if (order == DimOrder::OuterInner) {
      map.map(to0, t0);
      map.map(to1, c0);
      BlockArgument tiTied0 = inner.getTiedKernelArgument(to0);
      if (tiTied0)
        map.map(tiTied0, t0);
      BlockArgument tiTied1 = inner.getTiedKernelArgument(to1);
      if (tiTied1)
        map.map(tiTied1, c0);

      map.map(ti0, t1);
      map.map(ti1, c0);
    } else { // InnerOuter
      map.map(ti0, t0);
      map.map(ti1, c0);

      map.map(to0, t1);
      map.map(to1, c0);
      BlockArgument tiTied0 = inner.getTiedKernelArgument(to0);
      if (tiTied0)
        map.map(tiTied0, t1);
      BlockArgument tiTied1 = inner.getTiedKernelArgument(to1);
      if (tiTied1)
        map.map(tiTied1, c0);
    }

    //===------------------------------------------------------------------===//
    // Step 5: Map kernel arguments from outer/inner to fused.
    // This loop maps outer herd kernel arguments that were tied to outer's
    // kernel arguments to the corresponding fused arguments.
    //===------------------------------------------------------------------===//
    for (auto bbarg : inner.getKernelArguments()) {
      bbarg.replaceAllUsesWith(inner.getTiedKernelOperand(bbarg));
    }

    auto fusedArgsIt = fused.getKernelArguments().begin();

    for (auto old : outer.getKernelArguments()) {
      map.map(old, *fusedArgsIt++);
    }

    //===------------------------------------------------------------------===//
    // Step 6: Inline bodies into fused herd.
    // Clone outer body ops (except inner herd) and then all inner body ops,
    // applying the mapping from old operands to new fused operands.
    //===------------------------------------------------------------------===//
    rewriter.setInsertionPointToStart(&fused.getBody().front());
    for (Operation &op :
         llvm::make_early_inc_range(outerBody.without_terminator())) {
      if (&op == inner)
        continue;
      rewriter.clone(op, map);
    }
    Block &innerBody = inner.getBody().front();
    for (Operation &op : innerBody.without_terminator())
      rewriter.clone(op, map);

    //===------------------------------------------------------------------===//
    // Step 7: Replace outer herd with fused herd.
    // Copy over discardable attributes and symbol name.
    //===------------------------------------------------------------------===//
    fused->setDiscardableAttrs(outer->getDiscardableAttrDictionary());
    fused->setDiscardableAttrs(inner->getDiscardableAttrDictionary());
    fused.setSymName(outer.getSymName());
    rewriter.replaceOp(outer, fused);

    return success();
  }

private:
  DimOrder &order;
};

/// Lift a 1-symbol, 0-dim IntegerSet to a 2-symbol, 0-dim set:
/// - Original constraints E(s0) are remapped to E(s[newIdx])
/// - The "other" symbol gets full-range constraints [0, extent-1]
static IntegerSet lift1DTo2D(IntegerSet orig, unsigned newIdx,
                             int64_t otherExtent, MLIRContext *ctx) {
  if (orig.getNumDims() != 0)
    llvm::report_fatal_error("lift1DTo2D: expected 0 dims");
  if (orig.getNumSymbols() != 1)
    llvm::report_fatal_error("lift1DTo2D: expected 1 symbol");
  if (otherExtent <= 0)
    llvm::report_fatal_error("lift1DTo2D: otherExtent must be > 0");

  SmallVector<AffineExpr> cons;
  SmallVector<bool> eqs;

  // Remap the single original symbol s0 to s[newIdx], and increase the symbol
  // count from 1 to 2. All original constraints are updated accordingly.
  for (auto [e, isEq] : llvm::zip(orig.getConstraints(), orig.getEqFlags())) {
    AffineExpr mapped = e.replaceSymbols(getAffineSymbolExpr(newIdx, ctx));
    cons.push_back(mapped);
    eqs.push_back(isEq);
  }

  // Add an unconstrained range for the "other" symbol: s_other ∈ [0,
  // otherExtent-1]. This ensures the new symbol is valid for all iterations of
  // its loop.
  unsigned otherIdx = 1u - newIdx;
  int64_t ub = otherExtent - 1;
  cons.push_back(getAffineSymbolExpr(otherIdx, ctx) -
                 getAffineConstantExpr(0, ctx)); // s_other - 0 >= 0
  eqs.push_back(false);
  cons.push_back(getAffineConstantExpr(ub, ctx) -
                 getAffineSymbolExpr(otherIdx, ctx)); // ub - s_other >= 0
  eqs.push_back(false);

  return IntegerSet::get(/*dims=*/0, /*syms=*/2, cons, eqs);
}

/// Pattern to complete herd IV usage in affine.if conditions.
///
/// Matches `affine.if` ops inside a static `air.herd` whose condition depends
/// on exactly one of the herd's induction variables (IVs). Rewrites them to
/// depend on *both* herd IVs by:
///  1. Adding the unused IV as an extra symbol operand.
///  2. Lifting the condition's IntegerSet from 1 symbol to 2 symbols, using
///     `lift1DTo2D` so the new symbol has a full-range constraint.
struct CompleteIfHerdIVsPattern : OpRewritePattern<affine::AffineIfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineIfOp ifOp,
                                PatternRewriter &rewriter) const override {
    auto herd = ifOp->getParentOfType<xilinx::air::HerdOp>();
    if (!herd)
      return rewriter.notifyMatchFailure(ifOp, "not inside air.herd");

    // Require static grid sizes so we can encode full-range constraints.
    int64_t rows = herd.getNumRows();
    int64_t cols = herd.getNumCols();
    if (rows <= 0 || cols <= 0)
      return rewriter.notifyMatchFailure(
          ifOp, "dynamic or non-positive herd extents");

    Value id0 = herd.getIds()[0];
    Value id1 = herd.getIds()[1];

    IntegerSet is = ifOp.getIntegerSet();
    // Only handle simple form: set<()[s0]: ...>
    if (is.getNumDims() != 0 || is.getNumSymbols() != 1)
      return rewriter.notifyMatchFailure(
          ifOp, "expects set<()[s0]:...> (0 dims, 1 sym)");

    // Operand list must contain exactly one symbol operand, which must be one
    // of the herd IVs.
    if (ifOp.getOperands().size() != 1)
      return rewriter.notifyMatchFailure(ifOp,
                                         "expects exactly one symbol operand");
    Value only = ifOp.getOperands().front();

    bool usesId0 = (only == id0);
    bool usesId1 = (only == id1);
    if (!(usesId0 ^ usesId1))
      return rewriter.notifyMatchFailure(
          ifOp, "operand must be exactly one of herd IVs");

    // Build lifted 2D set:
    //   if uses id0 -> original constraints on s0, s1 full range [0, cols-1]
    //   if uses id1 -> original constraints on s1, s0 full range [0, rows-1]
    MLIRContext *ctx = ifOp.getContext();
    IntegerSet newIS = lift1DTo2D(is,
                                  /*newIdxForOrig=*/usesId0 ? 0u : 1u,
                                  /*otherExtent=*/usesId0 ? rows : cols, ctx);

    // New operands map directly to [s0, s1] = [id0, id1].
    SmallVector<Value> newOperands{id0, id1};

    // Create new affine.if op with lifted set and both IVs as operands.
    auto newIf = affine::AffineIfOp::create(
        rewriter, ifOp.getLoc(), ifOp.getResultTypes(), newIS, newOperands,
        /*withElseRegion=*/ifOp.hasElse());

    // Move (not clone) regions to preserve body contents and attributes.
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    if (ifOp.hasElse())
      newIf.getElseRegion().takeBody(ifOp.getElseRegion());

    // Replace the old op with the new one, preserving results if any.
    rewriter.replaceOp(ifOp, newIf);

    return success();
  }
};

void AIRFuseNestedHerdPass::runOnOperation() {
  func::FuncOp func = getOperation();

  DimOrder dimOrder = DimOrder::OuterInner;
  if (clOrder == "inner-outer")
    dimOrder = DimOrder::InnerOuter;
  else if (clOrder == "outer-inner")
    dimOrder = DimOrder::OuterInner;
  else {
    func->emitOpError("-air-fuse-nested-herd pass's 'order' option only "
                      "accepts one of 'inner-outer' and 'outer-inner'.");
    return signalPassFailure();
  }

  MLIRContext *ctx = func.getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<NestedHerdCollapsePattern>(ctx, dimOrder);
  (void)applyPatternsGreedily(func, std::move(patterns));

  RewritePatternSet postProcPatterns(ctx);
  postProcPatterns.add<CompleteIfHerdIVsPattern>(ctx);
  (void)applyPatternsGreedily(func, std::move(postProcPatterns));
}

class AIRUnrollOuterPerfectlyNestedLoopsPass
    : public air::impl::AIRUnrollOuterPerfectlyNestedLoopsPassBase<
          AIRUnrollOuterPerfectlyNestedLoopsPass> {

public:
  AIRUnrollOuterPerfectlyNestedLoopsPass() = default;
  AIRUnrollOuterPerfectlyNestedLoopsPass(
      const AIRUnrollOuterPerfectlyNestedLoopsPass &pass){};
  AIRUnrollOuterPerfectlyNestedLoopsPass(
      const ::xilinx::air::AIRUnrollOuterPerfectlyNestedLoopsPassOptions
          &options)
      : AIRUnrollOuterPerfectlyNestedLoopsPassBase(options) {}

  void runOnOperation() override;

private:
};

void AIRUnrollOuterPerfectlyNestedLoopsPass::runOnOperation() {
  SmallVector<air::HerdOp> herds;
  auto func = getOperation();
  SmallVector<affine::AffineForOp> roots;
  func.walk([&](affine::AffineForOp afo) {
    if (!isa<affine::AffineForOp>(afo->getParentOp())) {
      roots.push_back(afo);
    }
  });
  for (auto root : roots) {
    SmallVector<affine::AffineForOp> perfectly_nested_loops;
    affine::getPerfectlyNestedLoops(perfectly_nested_loops, root);
    if (perfectly_nested_loops.empty()) {
      continue;
    }

    int depth = std::min((int)clDepth, (int)perfectly_nested_loops.size());
    // Unroll from inner to outer, otherwise gathered inner loops are lost when
    // outer loop is unrolled.
    for (int i = depth - 1; i >= 0; i--) {
      (void)loopUnrollFull(perfectly_nested_loops[i]);
    }
  }
}

// <split_dim_on_offsets, split_affine_map, split_offset, split_size,
// split_stride>
typedef std::tuple<int, AffineMap, std::optional<int>, std::optional<int>,
                   std::optional<int>>
    infoEntryTy;
// <split_type, split_factor, map<split_dim, vector<info_entry>>>
typedef std::tuple<std::string, int,
                   llvm::MapVector<int, SmallVector<infoEntryTy>>>
    memrefSplitInfoTy;

class AIRSplitL2MemrefForBufferConstraintPass
    : public air::impl::AIRSplitL2MemrefForBufferConstraintPassBase<
          AIRSplitL2MemrefForBufferConstraintPass> {

public:
  AIRSplitL2MemrefForBufferConstraintPass() = default;
  AIRSplitL2MemrefForBufferConstraintPass(
      const AIRSplitL2MemrefForBufferConstraintPass &pass){};

  void runOnOperation() override;

private:
  void partitionMemref(
      SmallVector<air::ChannelPutOp> &puts,
      SmallVector<air::ChannelGetOp> &gets, int memrefDim, Operation *allocOp,
      llvm::MapVector<air::ChannelInterface, infoEntryTy> &opToSplitInfoMap);
  FailureOr<llvm::DenseMap<memref::AllocOp, memrefSplitInfoTy>>
  getTargetMemrefAllocs(
      func::FuncOp func,
      llvm::MapVector<air::ChannelInterface, infoEntryTy> &opToSplitInfoMap);
  std::optional<int> getMemrefSplitDim(SetVector<air::ChannelInterface> putgets,
                                       SmallVector<int> memrefShape);
};

template <typename T>
void push_back_if_unique(SmallVector<T> &vec, T entry) {
  if (std::find(vec.begin(), vec.end(), entry) == vec.end()) {
    vec.push_back(entry);
  }
}

// Tile air.channel put/get wrt a memref.
FailureOr<Value> tileChannelOpByFactor(
    air::ChannelInterface originalChanOp, int factor, int originalMemrefSize,
    SmallVector<infoEntryTy> &splitInfoVec,
    llvm::MapVector<air::ChannelInterface, infoEntryTy> &opToSplitInfoMap,
    air::ChannelOp newChanOp, Location loc, MLIRContext *ctx) {
  IRRewriter rewriter(ctx);
  rewriter.setInsertionPoint(originalChanOp);
  Value zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
  // Create and apply affine map onto the split channel ops.
  SmallVector<Value> tokens;
  int memorySpace =
      dyn_cast<BaseMemRefType>(originalChanOp.getMemref().getType())
          .getMemorySpaceAsInt();
  for (int i = 0; i < factor; i++) {
    // Get affine map and split size from splitInfo.
    auto &[splitInfoDimOnOffsets, splitInfoAffineMap, splitInfoSplitOffset,
           splitInfoSplitSize, splitInfoSplitStrideFactor] = splitInfoVec[i];

    int splitDimOnOffsets = splitInfoDimOnOffsets;

    Operation *affineApplyOp = nullptr;
    // Get any existing affine map operating on the target split dimension.
    if (!originalChanOp.getOffsets().empty()) {
      auto offsetDefOp =
          originalChanOp.getOffsets()[splitDimOnOffsets].getDefiningOp();
      if (isa_and_present<affine::AffineApplyOp, air::ExecuteOp>(offsetDefOp))
        affineApplyOp = offsetDefOp;
    }

    auto getOriginalApplyOperands =
        [zeroIdx, splitDimOnOffsets](Operation *affineApplyOp,
                                     air::ChannelInterface originalChanOp,
                                     std::optional<int> splitInfoSplitOffset) {
          SmallVector<Value> originalApplyOperands;
          if (auto applyOp =
                  dyn_cast_if_present<affine::AffineApplyOp>(affineApplyOp)) {
            originalApplyOperands = applyOp->getOperands();
          } else if (auto execOp =
                         dyn_cast_if_present<air::ExecuteOp>(affineApplyOp)) {
            SetVector<Value> opers;
            getUsedValuesDefinedAbove(execOp.getRegion(), opers);
            originalApplyOperands = llvm::to_vector(opers);
          } else {
            if (air::isDefaultDataAccessPattern(originalChanOp.getSizes(),
                                                originalChanOp.getStrides()))
              originalApplyOperands.push_back(zeroIdx);
            else
              originalApplyOperands.push_back(
                  originalChanOp.getOffsets()[splitDimOnOffsets]);
          }
          return originalApplyOperands;
        };

    auto getOriginalExpr = [&rewriter](Operation *affineApplyOp,
                                       AffineMap splitInfoAffineMap) {
      AffineExpr originalExpr = nullptr;
      if (auto applyOp =
              dyn_cast_if_present<affine::AffineApplyOp>(affineApplyOp)) {
        originalExpr = applyOp.getAffineMap().getResult(0);
      } else if (auto execOp =
                     dyn_cast_if_present<air::ExecuteOp>(affineApplyOp)) {
        originalExpr =
            dyn_cast<affine::AffineApplyOp>(execOp.getChildOps().front())
                .getAffineMap()
                .getResult(0);
      } else {
        originalExpr = rewriter.getAffineSymbolExpr(0);
      }
      return originalExpr;
    };

    SmallVector<Value> originalApplyOperands = getOriginalApplyOperands(
        affineApplyOp, originalChanOp, splitInfoSplitOffset);
    AffineExpr originalExpr =
        getOriginalExpr(affineApplyOp, splitInfoAffineMap);

    // Preserve original channel indices and prepend the split index
    SmallVector<Value> newIndices;
    newIndices.push_back(
        arith::ConstantIndexOp::create(rewriter, loc, i)); // Split dimension
    for (Value origIdx : originalChanOp.getIndices()) {
      newIndices.push_back(origIdx); // Original indices (e.g., %arg13)
    }
    // If original had no indices, add a zero
    if (originalChanOp.getIndices().empty()) {
      newIndices.push_back(zeroIdx);
    }
    // Create affine.apply on induction variable.
    auto checkpoint = rewriter.saveInsertionPoint();
    if (affineApplyOp)
      rewriter.setInsertionPoint(affineApplyOp);
    // If allocOp has "affine_map" attribute set, then use that map instead
    // (potentially overlapping access pattern).
    affine::AffineApplyOp newApplyOp = nullptr;

    // Methods to compose affine expression for offset at each split.
    auto composeAffineExprWithOffsetAndAffineMap = [](AffineExpr originalExpr,
                                                      AffineMap affineMap,
                                                      std::optional<int> offset,
                                                      MLIRContext *ctx) {
      int const_in = offset ? *offset : 0;
      if (affineMap) {
        auto original_map = affineMap;
        if (original_map.getNumSymbols() > 0) {
          original_map =
              original_map.replace(getAffineSymbolExpr(0, ctx),
                                   getAffineConstantExpr(const_in, ctx), 0, 1);
        } else if (original_map.getNumDims() > 0) {
          original_map =
              original_map.replace(getAffineDimExpr(0, ctx),
                                   getAffineConstantExpr(const_in, ctx), 1, 0);
        }
        AffineExpr add = originalExpr + original_map.getResult(0);
        return AffineMap::get(0, 1, add);
      }
      AffineExpr add = originalExpr + getAffineConstantExpr(const_in, ctx);
      return AffineMap::get(0, 1, add);
    };
    auto composeAffineExprFromSizes = [](AffineExpr originalExpr,
                                         int originalMemrefSize, int factor,
                                         int i) {
      AffineExpr add =
          originalExpr + i * llvm::divideCeilSigned(originalMemrefSize, factor);
      return AffineMap::get(0, 1, add);
    };

    AffineMap map;
    if (splitInfoAffineMap || splitInfoSplitSize ||
        splitInfoSplitStrideFactor) {
      // If any overriding offset affine mapping, size or stride factor is
      // logged, it must be respected throughout the splitting process.
      map = composeAffineExprWithOffsetAndAffineMap(
          originalExpr, splitInfoAffineMap, splitInfoSplitOffset, ctx);
    } else
      map = composeAffineExprFromSizes(originalExpr, originalMemrefSize, factor,
                                       i);
    newApplyOp = affine::AffineApplyOp::create(rewriter, loc, map,
                                               originalApplyOperands);
    if (affineApplyOp)
      rewriter.restoreInsertionPoint(checkpoint);
    SmallVector<Value> newOffsets = originalChanOp.getOffsets();
    SmallVector<Value> newWraps = originalChanOp.getSizes();
    SmallVector<Value> newStrides = originalChanOp.getStrides();
    if (newOffsets.empty() && newWraps.empty())
      air::populateDefaultWrapsAndStrides(rewriter, originalChanOp.getMemref(),
                                          newOffsets, newWraps, newStrides);
    newOffsets[splitDimOnOffsets] = newApplyOp.getResult();
    if (splitInfoSplitSize)
      newWraps[splitDimOnOffsets] =
          arith::ConstantIndexOp::create(rewriter, loc, *splitInfoSplitSize);
    else
      newWraps[splitDimOnOffsets] = arith::ConstantIndexOp::create(
          rewriter, loc, llvm::divideCeilSigned(originalMemrefSize, factor));
    // Stride manipulation is only allowed for L3 memory: we are not splitting
    // the L3 memref; we are only splitting its access pattern.
    // Strategy: add one dimension to wrap-and-stride list. Rationale: (1) the
    // stride factor should apply on existing size, (2) original offset must
    // continue with the original stride.
    if (splitInfoSplitStrideFactor &&
        memorySpace == (int)air::MemorySpace::L3) {
      newStrides.insert(newStrides.begin() + splitDimOnOffsets,
                        newStrides[splitDimOnOffsets]);
      newStrides[splitDimOnOffsets + 1] = arith::ConstantIndexOp::create(
          rewriter, loc,
          *getConstantIntValue(newStrides[splitDimOnOffsets]) *
              (*splitInfoSplitStrideFactor));
      newWraps.insert(newWraps.begin() + splitDimOnOffsets,
                      arith::ConstantIndexOp::create(rewriter, loc, 1));
      newOffsets.insert(newOffsets.begin() + splitDimOnOffsets,
                        newOffsets[splitDimOnOffsets]);
      newOffsets[splitDimOnOffsets + 1] =
          arith::ConstantIndexOp::create(rewriter, loc, 0);
    }
    auto deps = dyn_cast<air::AsyncOpInterface>(originalChanOp.getOperation())
                    .getAsyncDependencies();
    SmallVector<Type, 4> tys = {air::AsyncTokenType::get(ctx)};
    if (isa<air::ChannelGetOp>(originalChanOp)) {
      auto newGetOp = air::ChannelGetOp::create(
          rewriter, loc, tys, deps, newChanOp.getSymName(), newIndices,
          originalChanOp.getMemref(), newOffsets, newWraps, newStrides);
      newGetOp->setAttrs(originalChanOp->getDiscardableAttrDictionary());
      tokens.push_back(newGetOp.getAsyncToken());
      opToSplitInfoMap[newGetOp] = splitInfoVec[i];
    } else {
      auto newPutOp = air::ChannelPutOp::create(
          rewriter, loc, tys, deps, newChanOp.getSymName(), newIndices,
          originalChanOp.getMemref(), newOffsets, newWraps, newStrides);
      newPutOp->setAttrs(originalChanOp->getDiscardableAttrDictionary());
      tokens.push_back(newPutOp.getAsyncToken());
      opToSplitInfoMap[newPutOp] = splitInfoVec[i];
    }
  }
  auto newWaitAll = air::WaitAllOp::create(
      rewriter, loc, air::AsyncTokenType::get(ctx), tokens);
  return newWaitAll.getAsyncToken();
}

// Get scf.for op whose iv (indirectly) produces the val.
scf::ForOp getScfForFromVal(Value val) {
  if (!val)
    return scf::ForOp();
  if (auto res = scf::getForInductionVarOwner(val))
    return res;
  auto defOp = val.getDefiningOp();
  if (!defOp)
    return scf::ForOp();
  SetVector<Value> opers;
  if (auto exec = dyn_cast<air::ExecuteOp>(defOp)) {
    getUsedValuesDefinedAbove(exec.getRegion(), opers);
  } else {
    opers.insert(defOp->getOperands().begin(), defOp->getOperands().end());
  }
  for (auto oper : opers) {
    if (auto res = scf::getForInductionVarOwner(oper))
      return res;
  }
  return scf::ForOp();
}

// Partition L2 memref.
void AIRSplitL2MemrefForBufferConstraintPass::partitionMemref(
    SmallVector<air::ChannelPutOp> &puts, SmallVector<air::ChannelGetOp> &gets,
    int memrefDim, Operation *allocOp,
    llvm::MapVector<air::ChannelInterface, infoEntryTy> &opToSplitInfoMap) {
  auto memref = puts.front().getMemref();
  MemRefType ty = llvm::cast<MemRefType>(memref.getType());
  if (isa<air::ExecuteOp>(allocOp->getParentOp()))
    allocOp = allocOp->getParentOp();
  auto loc = allocOp->getLoc();
  auto ctx = allocOp->getContext();
  Operation *deallocOp = nullptr;
  for (auto user : memref.getUsers()) {
    if (auto execOp = dyn_cast<air::ExecuteOp>(user->getParentOp())) {
      if (llvm::any_of(execOp.getChildOps(), [](Operation &child_op) {
            return isa<memref::DeallocOp>(child_op);
          })) {
        deallocOp = execOp;
        break;
      }
    } else if (isa<memref::DeallocOp>(user)) {
      deallocOp = user;
      break;
    }
  }

  std::map<int, SmallVector<air::ChannelInterface>> chanOpPartitions;
  SmallVector<int> keys;

  // Get map of channel ops
  auto getChanOpPartitionsMap =
      [ctx](std::map<int, SmallVector<air::ChannelInterface>> &chanOpPartitions,
            SmallVector<int> &keys, int offsetDim, air::ChannelInterface op) {
        auto offset = getConstantIntValue(op.getOffsets()[offsetDim]);
        int offset_key = -1;
        if (offset)
          offset_key = *offset; // Const offset.
        else { // Variadic offset (induction variable to an scf.for).
          auto forOp = getScfForFromVal(op.getOffsets()[offsetDim]);
          if (!forOp)
            return;
          auto lb = getConstantIntValue(forOp.getLowerBound());
          if (!lb)
            return;
          // Get any existing affine map operating on the target split
          // dimension.
          auto offsetDefOp = op.getOffsets()[offsetDim].getDefiningOp();
          affine::AffineApplyOp apply;
          if (auto applyOp =
                  dyn_cast_if_present<affine::AffineApplyOp>(offsetDefOp)) {
            apply = applyOp;
          } else if (auto execOp =
                         dyn_cast_if_present<air::ExecuteOp>(offsetDefOp)) {
            apply =
                dyn_cast<affine::AffineApplyOp>(execOp.getChildOps().front());
          }
          if (apply) {
            SmallVector<std::optional<int64_t>> sym_ints;
            SmallVector<std::optional<int64_t>> dim_ints;
            for (auto oper : apply.getSymbolOperands()) {
              if (auto constVal = getConstantIntValue(oper))
                sym_ints.push_back(constVal);
              else
                sym_ints.push_back(lb);
            }
            for (auto oper : apply.getDimOperands()) {
              if (auto constVal = getConstantIntValue(oper))
                dim_ints.push_back(constVal);
              else
                dim_ints.push_back(lb);
            }
            auto key_opt = air::evaluateConstantsInMap(apply.getAffineMap(),
                                                       sym_ints, dim_ints, ctx);
            if (!key_opt)
              return;
            offset_key = *key_opt;
          } else {
            offset_key = *lb;
          }
        }
        if (offset_key < 0)
          return;
        push_back_if_unique<int>(keys, offset_key);
        chanOpPartitions[offset_key].push_back(op);
      };

  for (auto op : puts) {
    if (!opToSplitInfoMap.count(op))
      continue;
    auto &[splitInfoDimOnOffsets, splitAffineMap, splitOffset, splitSize,
           splitStride] = opToSplitInfoMap[op];
    getChanOpPartitionsMap(chanOpPartitions, keys, splitInfoDimOnOffsets, op);
  }
  for (auto op : gets) {
    if (!opToSplitInfoMap.count(op))
      continue;
    auto &[splitInfoDimOnOffsets, splitAffineMap, splitOffset, splitSize,
           splitStride] = opToSplitInfoMap[op];
    getChanOpPartitionsMap(chanOpPartitions, keys, splitInfoDimOnOffsets, op);
  }

  OpBuilder builder(allocOp);
  SmallVector<scf::ForOp> mutatedScfForOps;
  for (auto key : keys) {
    SmallVector<int64_t> newMemrefShape;
    for (unsigned i = 0; i < air::getTensorShape(ty).size(); i++) {
      newMemrefShape.push_back(air::getTensorShape(ty)[i]);
    }
    for (auto op : chanOpPartitions[key]) {
      auto &[splitInfoDimOnOffsets, splitAffineMap, splitOffset, splitSize,
             splitStride] = opToSplitInfoMap[op];
      int offsetDim = splitInfoDimOnOffsets;
      if (op.getSizes().size() != newMemrefShape.size())
        continue;

      // Get post-splitting size at split_dim from allocOp attributes.
      if (splitSize)
        newMemrefShape[memrefDim] = *splitSize;
      else {
        auto offset = getConstantIntValue(op.getOffsets()[offsetDim]);
        if (offset)
          newMemrefShape[memrefDim] =
              *getConstantIntValue(op.getSizes()[offsetDim]);
        else {
          auto forOp = getScfForFromVal(op.getOffsets()[offsetDim]);
          if (!forOp)
            continue;
          auto trip_count = air::getStaticScfForTripCountAsInt(forOp);
          if (!trip_count)
            continue;
          newMemrefShape[memrefDim] =
              *getConstantIntValue(op.getSizes()[offsetDim]) * (*trip_count);
        }
      }
      break;
    }

    auto newMemrefType =
        MemRefType::get(newMemrefShape, ty.getElementType(),
                        ty.getLayout().getAffineMap(), ty.getMemorySpace());
    Value newMemref = nullptr;
    // Create new alloc ops.
    if (isa<air::ExecuteOp>(allocOp)) {
      auto execOp =
          air::ExecuteOp::create(builder, loc, air::AsyncTokenType::get(ctx),
                                 newMemrefType, SmallVector<Value>{});
      Block *async_bb = builder.createBlock(&execOp.getRegion());
      builder.setInsertionPointToStart(async_bb);
      auto childMemAlloc = memref::AllocOp::create(builder, loc, newMemrefType);
      xilinx::air::ExecuteTerminatorOp::create(builder, loc,
                                               childMemAlloc->getResults());
      newMemref = execOp->getResult(1);
      builder.setInsertionPoint(execOp);
    } else
      newMemref = memref::AllocOp::create(builder, loc, newMemrefType);
    // Create new dealloc ops.
    if (deallocOp) {
      builder.setInsertionPoint(deallocOp);
      if (auto execDeallocOp = dyn_cast<air::ExecuteOp>(deallocOp)) {
        auto execOp =
            air::ExecuteOp::create(builder, loc, air::AsyncTokenType::get(ctx),
                                   execDeallocOp.getAsyncDependencies());
        Block *async_bb = builder.createBlock(&execOp.getRegion());
        builder.setInsertionPointToStart(async_bb);
        memref::DeallocOp::create(builder, loc, newMemref);
        xilinx::air::ExecuteTerminatorOp::create(builder, loc);
      } else
        memref::DeallocOp::create(builder, loc, newMemref);
      builder.setInsertionPoint(newMemref.getDefiningOp());
    }
    // Mutate air.channel.put/get memref and async token usage.
    for (auto op : chanOpPartitions[key]) {
      auto &[splitInfoDimOnOffsets, splitAffineMap, splitOffset, splitSize,
             splitStride] = opToSplitInfoMap[op];
      int offsetDim = splitInfoDimOnOffsets;
      int memrefOperandOffset =
          dyn_cast<air::AsyncOpInterface>(op.getOperation())
              .getAsyncDependencies()
              .size() +
          op.getIndices().size();
      auto &memrefOpOper = op->getOpOperand(memrefOperandOffset);
      memrefOpOper.assign(newMemref);
      if (air::getAsyncTokenFromOp(allocOp) &&
          air::getAsyncTokenFromOp(newMemref.getDefiningOp()))
        op->replaceUsesOfWith(
            air::getAsyncTokenFromOp(allocOp),
            air::getAsyncTokenFromOp(newMemref.getDefiningOp()));
      int offsetOperandOffset = memrefOperandOffset + offsetDim + 1;
      auto &offsetOpOper = op->getOpOperand(offsetOperandOffset);

      auto defOp = op.getOffsets()[offsetDim].getDefiningOp();
      if (defOp) {
        // Const offset. Reset offset to 0.
        if (getConstantIntValue(op.getOffsets()[offsetDim]))
          offsetOpOper.assign(arith::ConstantIndexOp::create(builder, loc, 0));
        // Variadic offset. Reset const operands of apply to 0.
        else {
          affine::AffineApplyOp apply =
              dyn_cast_if_present<affine::AffineApplyOp>(defOp);
          air::ExecuteOp exec = dyn_cast_if_present<air::ExecuteOp>(defOp);
          if (exec)
            for (auto &child_op : exec.getChildOps())
              if (auto apply_child_op =
                      dyn_cast<affine::AffineApplyOp>(child_op))
                apply = apply_child_op;
          if (!apply) {
            defOp->emitOpError("Apply op not found. NYI.");
            return;
          }
          // Any const operands to affine map should have been canonicalized
          // away.
          if (llvm::any_of(apply->getOperands(), [](Value oper) {
                return getConstantIntValue(oper);
              })) {
            defOp->emitOpError("found constant operands to affine map, which "
                               "aren't canonicalized away.");
            return;
          }
          // Set map's expressions to cancel out each key's offset
          auto applyExpr = apply.getMap().getResult(0);
          applyExpr = applyExpr - key;
          apply.setMap(AffineMap::get(apply.getDimOperands().size(),
                                      apply.getSymbolOperands().size(),
                                      applyExpr));
        }
      }

      // Update strides (contiguous, row-major) after memref tiling.
      SmallVector<int> newStrides;
      // One dimensional default stride value.
      if (op.getSizes().size() == 1)
        newStrides.push_back(1);
      else
        newStrides = air::getUpdatedStridesAfterShrinkage(
            air::getTensorShape(memref.getType()), newMemrefShape,
            op.getStrides());
      int firstStrideOperandOffset =
          memrefOperandOffset + op.getOffsets().size() * 2 + 1;
      for (unsigned i = 0; i < op.getStrides().size(); i++) {
        auto &strideOpOper = op->getOpOperand(firstStrideOperandOffset + i);
        strideOpOper.assign(
            arith::ConstantIndexOp::create(builder, loc, newStrides[i]));
      }

      // Reconnect async dependency of parent scf.for op, if any.
      if (!isAsyncOp(op))
        continue;
      if (!isa<scf::ForOp>(op->getParentOp()))
        continue;
      auto parentForOp = dyn_cast<scf::ForOp>(op->getParentOp());
      push_back_if_unique<scf::ForOp>(mutatedScfForOps, parentForOp);
    }
  }
  // Reconnect async dependency of parent scf.for op, if any.
  air::dependencyTracer depTracer;
  for (auto mutatedScfForOp : mutatedScfForOps) {
    if (failed(depTracer.traceDependencyFromScfForOp(mutatedScfForOp)))
      signalPassFailure();
  }
  if (deallocOp)
    deallocOp->erase();
}

// Infer the dimension to which the join / distribute pattern happens, as basis
// for memref splitting.
std::optional<int> AIRSplitL2MemrefForBufferConstraintPass::getMemrefSplitDim(
    SetVector<air::ChannelInterface> putgets, SmallVector<int> memrefShape) {
  std::optional<int> memrefDim = std::nullopt;
  for (unsigned i = 0; i < putgets.size() - 1; i++) {
    for (unsigned j = i + 1; j < putgets.size(); j++) {
      air::ChannelInterface ci = putgets[i];
      air::ChannelInterface cj = putgets[j];
      if (ci.getOffsets().size() != cj.getOffsets().size())
        continue;
      auto offsetZip = llvm::zip_equal(ci.getOffsets(), cj.getOffsets());
      auto d =
          llvm::find_if(offsetZip, [](std::tuple<Value, Value> offsetPair) {
            auto [o1, o2] = offsetPair;
            auto defO1 = o1.getDefiningOp();
            auto defO2 = o2.getDefiningOp();
            if (defO1 && defO2) {
              if (air::isEquivalentTo(defO1, defO2))
                return false;
              else
                return true;
            }
            return false;
          });
      if (d != offsetZip.end())
        memrefDim = std::distance(offsetZip.begin(), d);
    }
  }
  // Match offset dims with memref dims.
  if (!memrefDim)
    return std::nullopt;
  air::ChannelInterface c0 = putgets[0];
  return air::getMemrefDimFromOffsetDim(*memrefDim, c0.getOffsets(),
                                        c0.getStrides(), memrefShape);
}

// Get a vector of allocs whose memrefs require splitting; label the single
// split dimension with split factor, split_type and affine_map (if any).
FailureOr<llvm::DenseMap<memref::AllocOp, memrefSplitInfoTy>>
AIRSplitL2MemrefForBufferConstraintPass::getTargetMemrefAllocs(
    func::FuncOp func,
    llvm::MapVector<air::ChannelInterface, infoEntryTy> &opToSplitInfoMap) {
  auto ctx = func.getContext();
  SmallVector<memref::AllocOp> allocOps;
  func.walk([&](memref::AllocOp allocOp) {
    if (allocOp->getParentOfType<air::SegmentOp>() &&
        llvm::cast<MemRefType>(allocOp.getMemref().getType())
                .getMemorySpaceAsInt() == (int)air::MemorySpace::L2) {
      allocOps.push_back(allocOp);
    }
  });

  // Condition to split a memref: detected multiple-in-single-out or
  // single-in-multiple-out channel patterns. Such pattern is represented via
  // the memref being accessed by multiple unique channel puts/gets.
  llvm::DenseMap<memref::AllocOp, memrefSplitInfoTy> targetMemrefsToInfoMap;

  // If there is an affine.apply operating on offsets[offsetDim], then
  // log the affine.map.
  auto getAffineMapOnMemrefSplitDim = [](air::ChannelInterface chanOp,
                                         int offsetDim) {
    auto offsetDefOp = chanOp.getOffsets()[offsetDim].getDefiningOp();
    affine::AffineApplyOp apply =
        dyn_cast_if_present<affine::AffineApplyOp>(offsetDefOp);
    if (auto exec = dyn_cast_if_present<air::ExecuteOp>(offsetDefOp))
      for (auto &child_op : exec.getChildOps())
        if (auto apply_child_op = dyn_cast<affine::AffineApplyOp>(child_op))
          apply = apply_child_op;
    return apply;
  };

  for (auto allocOp : allocOps) {
    Value memref = allocOp.getMemref();
    if (auto exec = dyn_cast<air::ExecuteOp>(allocOp->getParentOp()))
      memref = exec->getResult(1);
    // Maps of MM2S and S2MM channels and their sub-channels.
    llvm::MapVector<air::ChannelOp, SmallVector<SmallVector<Value>>>
        MM2SChannels, S2MMChannels;
    for (auto user : memref.getUsers()) {
      if (auto put = dyn_cast<air::ChannelPutOp>(user)) {
        // Condition 2: accessed by multiple puts with unique names.
        push_back_if_unique<SmallVector<Value>>(
            MM2SChannels[air::getChannelDeclarationThroughSymbol(put)],
            put.getIndices());
      } else if (auto get = dyn_cast<air::ChannelGetOp>(user)) {
        // Condition 3: accessed by multiple gets with unique names.
        push_back_if_unique<SmallVector<Value>>(
            S2MMChannels[air::getChannelDeclarationThroughSymbol(get)],
            get.getIndices());
      }
    }
    auto getChanCount =
        [](llvm::MapVector<air::ChannelOp, SmallVector<SmallVector<Value>>>
               Channels) {
          int count = 0;
          for (auto &[chanOp, indicesVec] : Channels) {
            count += indicesVec.size();
          }
          return count;
        };
    // Single-in-single-out. Skip.
    if (getChanCount(MM2SChannels) <= 1 && getChanCount(S2MMChannels) <= 1)
      continue;
    // Multiple-in-multiple-out (MIMO).
    if (getChanCount(MM2SChannels) > 1 && getChanCount(S2MMChannels) > 1) {
      // MIMO with different number of actors on each side. Skip.
      if (getChanCount(MM2SChannels) != getChanCount(S2MMChannels))
        continue;
    }

    // Get tiling factor.
    int tilingFactor =
        std::max(getChanCount(MM2SChannels), getChanCount(S2MMChannels));

    // Single-channel side: check if all endpoints of this channel are
    // splittable. An endpoint is not splittable if operating on an air.herd.
    auto isSplittingChannelGetsOnHerd = [&MM2SChannels]() {
      return llvm::any_of(
          MM2SChannels,
          [](std::pair<air::ChannelOp, SmallVector<SmallVector<Value>>>
                 mapEntry) {
            return llvm::any_of(
                air::getChannelGetOpThroughSymbol(mapEntry.first),
                [](Operation *ci) {
                  return ci->getParentOfType<air::HerdOp>();
                });
          });
    };
    auto isSplittingChannelPutsOnHerd = [&S2MMChannels]() {
      return llvm::any_of(
          S2MMChannels,
          [](std::pair<air::ChannelOp, SmallVector<SmallVector<Value>>>
                 mapEntry) {
            return llvm::any_of(
                air::getChannelPutOpThroughSymbol(mapEntry.first),
                [](Operation *ci) {
                  return ci->getParentOfType<air::HerdOp>();
                });
          });
    };
    if (getChanCount(MM2SChannels) == 1)
      if (isSplittingChannelGetsOnHerd())
        continue;
    if (getChanCount(S2MMChannels) == 1)
      if (isSplittingChannelPutsOnHerd())
        continue;

    llvm::MapVector<int, SmallVector<infoEntryTy>> infoEntryMap;
    std::optional<int> splitDimOffset = std::nullopt;
    std::optional<int> splitDimSize = std::nullopt;
    std::optional<int> splitDimStrideFactor = std::nullopt;
    std::optional<int> splitDim = std::nullopt;

    // Get all puts and/or gets, whichever direction has multiple operators.
    SetVector<air::ChannelInterface> putgets;
    if (getChanCount(MM2SChannels) > 1) {
      for (auto &[chanOp, __] : MM2SChannels)
        for (auto put : air::getChannelPutOpThroughSymbol(chanOp))
          putgets.insert(put);
    }
    if (getChanCount(S2MMChannels) > 1) {
      for (auto &[chanOp, __] : S2MMChannels)
        for (auto get : air::getChannelGetOpThroughSymbol(chanOp))
          putgets.insert(get);
    }

    splitDim =
        getMemrefSplitDim(putgets, air::getTensorShape(memref.getType()));
    if (!splitDim) {
      allocOp->emitWarning(
          "memref splitting analysis failed to get the split dimension.");
      continue;
    }

    // Methods to get root offset/size/stride from air.channel's operands, where
    // root is either a constant, or a loop's induction variable.
    auto getRootOffset = [&](Value offsetVal) {
      std::optional<int> rootOffset = std::nullopt;
      if (auto constOffset = getConstantIntValue(offsetVal))
        rootOffset = *constOffset;
      else if (auto forOp = getScfForFromVal(offsetVal))
        rootOffset = *getConstantIntValue(forOp.getLowerBound());
      return rootOffset;
    };
    auto getRootSize = [&](Value offsetVal, Value sizeVal) {
      std::optional<int> rootSize = std::nullopt;
      if (auto forOp = getScfForFromVal(offsetVal)) {
        if (auto trip_count = air::getStaticScfForTripCountAsInt(forOp))
          rootSize = *getConstantIntValue(sizeVal) * (*trip_count);
        else
          forOp->emitOpError("has dynamic loop bound. NYI.");
      }
      return rootSize;
    };
    auto getRootStrideFactor = [&](Value offsetVal, Value strideVal) {
      std::optional<int> rootStrideFactor = std::nullopt;
      if (auto forOp = getScfForFromVal(offsetVal))
        rootStrideFactor = *getConstantIntValue(forOp.getStep());
      return rootStrideFactor;
    };

    for (unsigned i = 0; i < putgets.size(); i++) {
      // Infer the size at splitDim for both overlapping and non-overlapping
      // access pattern.
      air::ChannelInterface ci = putgets[i];
      auto offsetDimOpt = air::getOffsetDimFromMemrefDim(
          *splitDim, ci.getStrides(), air::getTensorShape(memref.getType()));
      // Infer offset at splitDim.
      if (auto rootOffset = getRootOffset(ci.getOffsets()[*offsetDimOpt]))
        splitDimOffset = *rootOffset;
      // Infer size at splitDim.
      if (auto rootSize = getRootSize(ci.getOffsets()[*offsetDimOpt],
                                      ci.getSizes()[*offsetDimOpt]))
        splitDimSize = *rootSize;
      // Infer stride (factor) at splitDim. If the root comes from an scf.for
      // loop, and if the loop has non-unit step size, then that multiplier
      // should be applied to other split channe put/get ops.
      // Note: 1d access pattern is disabled (leads to inserting stride!=1
      // dimension at inner-most dimension).
      auto rootStrideFactor = getRootStrideFactor(
          ci.getOffsets()[*offsetDimOpt], ci.getStrides()[*offsetDimOpt]);
      if (rootStrideFactor && ci.getOffsets().size() > 1) {
        splitDimStrideFactor = *rootStrideFactor;
        // Cancel out the non-unit step size on the for loop, to get contiguous
        // access pattern on memrefs after split.
        if (auto forOp = getScfForFromVal(ci.getOffsets()[*offsetDimOpt])) {
          forOp->setAttr("mutate_step_size_to",
                         IntegerAttr::get(IntegerType::get(ctx, 32), 1));
        }
      }
      AffineMap applyMap;
      auto apply = getAffineMapOnMemrefSplitDim(ci, *offsetDimOpt);
      if (apply)
        applyMap = apply.getAffineMap();

      infoEntryTy newEntry = {*offsetDimOpt, applyMap, splitDimOffset,
                              splitDimSize, splitDimStrideFactor};
      infoEntryMap[*splitDim].push_back(newEntry);
      opToSplitInfoMap[putgets[i]] = newEntry;
    }

    // Get output map.
    if (getChanCount(MM2SChannels) > 1 && getChanCount(S2MMChannels) > 1) {
      targetMemrefsToInfoMap[allocOp] = {"MM2SAndS2MMChannels", tilingFactor,
                                         infoEntryMap};
    } else {
      if (getChanCount(MM2SChannels) > 1) {
        targetMemrefsToInfoMap[allocOp] = {"MM2SChannels", tilingFactor,
                                           infoEntryMap};
      } else {
        targetMemrefsToInfoMap[allocOp] = {"S2MMChannels", tilingFactor,
                                           infoEntryMap};
      }
    }
  }
  return targetMemrefsToInfoMap;
}

// Check if each L2 memref shall violate buffer hardware constraint, and if so,
// attempt to split it (in columns, per NPU device layout).
void AIRSplitL2MemrefForBufferConstraintPass::runOnOperation() {
  SmallVector<air::HerdOp> herds;
  auto func = getOperation();
  auto ctx = &getContext();

  // Helper lambda to compute tile count in a segment.
  auto getTileCountInSegment = [](air::SegmentOp seg) {
    DenseMap<StringRef, uint64_t>
        herdNumTiles; // Herds with the same name are assumed to be different
                      // time phases of the same physical herd.
    unsigned tileCount = 0;
    seg.walk([&](air::HerdOp h) {
      if (!h.getSymName()) {
        tileCount += h.getNumCols() * h.getNumRows();
        return;
      }
      StringRef herdSym = *h.getSymName();
      herdNumTiles[herdSym] =
          herdNumTiles.count(herdSym)
              ? std::max(herdNumTiles[herdSym], h.getNumCols() * h.getNumRows())
              : h.getNumCols() * h.getNumRows();
    });
    for (const auto &[herdSym, count] : herdNumTiles)
      tileCount += count;
    return tileCount;
  };

  // First, identify which segments actually need splitting based on tile count.
  // Only process allocOps from segments that exceed the tile threshold.
  // This ensures each launch/segment is processed independently.
  llvm::DenseSet<air::SegmentOp> segmentsNeedingSplit;
  func.walk([&](air::SegmentOp seg) {
    if (getTileCountInSegment(seg) > clNumTilesPerL2Tile) {
      segmentsNeedingSplit.insert(seg);
    }
  });

  // If no segment needs splitting, exit early.
  if (segmentsNeedingSplit.empty())
    return;

  // Collect only allocOps from segments that need splitting.
  SmallVector<memref::AllocOp> allocOps;
  func.walk([&](memref::AllocOp allocOp) {
    auto parentSeg = allocOp->getParentOfType<air::SegmentOp>();
    if (parentSeg && segmentsNeedingSplit.contains(parentSeg) &&
        llvm::cast<MemRefType>(allocOp.getMemref().getType())
                .getMemorySpaceAsInt() == (int)air::MemorySpace::L2) {
      allocOps.push_back(allocOp);
    }
  });

  // If no target allocOps found, exit early.
  if (allocOps.empty())
    return;

  // STEP 1: Unroll scf.parallels only in segments that need splitting
  SmallVector<scf::ParallelOp> parOps;
  func.walk([&](scf::ParallelOp parOp) {
    auto parentSeg = parOp->getParentOfType<air::SegmentOp>();
    if (parentSeg && segmentsNeedingSplit.contains(parentSeg)) {
      parOps.push_back(parOp);
    }
  });

  llvm::SetVector<Operation *> erased;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> partialUnrollMap;
  for (auto par : parOps) {
    IRRewriter rewriter(ctx);
    IRMapping remap, waitAllRemap;
    rewriter.setInsertionPoint(par);
    if (par.getNumLoops() > 1) {
      // NOTE: Splitting along the first dimension of scf.parallel only.
      if (failed(air::unrollScfParallelOnDims(rewriter, par, remap, {0},
                                              partialUnrollMap)))
        signalPassFailure();
    } else {
      if (failed(
              air::unrollScfParallel(rewriter, par, remap, partialUnrollMap)))
        signalPassFailure();
      if (air::isAsyncOp(par)) {
        rewriter.setInsertionPoint(par);
        auto waitAll =
            air::replaceAsyncOpWithWaitAll(rewriter, waitAllRemap, par, false);
        air::getAsyncTokenFromOp(par).replaceAllUsesWith(
            waitAll.getAsyncToken());
        rewriter.eraseOp(par);
      }
    }
  }

  // Fold affine maps and constants after loop unrolling.
  RewritePatternSet cano_affine_map_patterns(ctx);
  mlir::affine::AffineApplyOp::getCanonicalizationPatterns(
      cano_affine_map_patterns, ctx);
  air::ExecuteOp::getCanonicalizationPatterns(cano_affine_map_patterns, ctx);
  (void)applyPatternsGreedily(func, std::move(cano_affine_map_patterns));

  // STEP 2: Map between the target memref alloc ops and all column-wise tiling
  // factors per alloc.
  llvm::MapVector<air::ChannelInterface, infoEntryTy> opToSplitInfoMap;
  auto targetMemrefsToInfoMap = getTargetMemrefAllocs(func, opToSplitInfoMap);
  if (failed(targetMemrefsToInfoMap))
    return;

  // Look up or create a new air.channel name.
  std::map<std::string, std::string> chanNameMap;
  auto lookUpOrCreateChanName = [&](std::string chanName, ModuleOp module) {
    if (chanNameMap.count(chanName))
      return chanNameMap[chanName];
    else {
      auto newChanName = air::createChannelName(module);
      chanNameMap[chanName] = newChanName;
      return newChanName;
    }
  };

  // STEP 3: Tile the side accessed by a single air.channel.
  IRRewriter rewriter(ctx);
  for (auto &[allocOp, splitInfo] : *targetMemrefsToInfoMap) {
    auto &[splitType, splitFactor, infoEntryMap] = splitInfo;
    auto &[splitDim, infoEntryVec] =
        infoEntryMap.front(); // TODO: assuming only one dimension is subject to
                              // splitting for now.
    int targetColTilingFactor = splitFactor;
    allocOp->setAttr("split",
                     mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                            targetColTilingFactor));
    Value memref = isa<air::ExecuteOp>(allocOp->getParentOp())
                       ? allocOp->getParentOp()->getResult(1)
                       : dyn_cast<Value>(allocOp.getMemref());
    for (auto user : memref.getUsers()) {
      if (!isa<air::ChannelInterface>(user))
        continue;
      // Multiple-channel side. Skip.
      if (isa<air::ChannelPutOp>(user) &&
          (splitType == "MM2SChannels" || splitType == "MM2SAndS2MMChannels"))
        continue;
      if (isa<air::ChannelGetOp>(user) &&
          (splitType == "S2MMChannels" || splitType == "MM2SAndS2MMChannels"))
        continue;

      // Single-channel side found. Perform tiling on put/get ops operating on
      // the channel.
      auto chanUserOp = dyn_cast<air::ChannelInterface>(user);
      auto loc = chanUserOp->getLoc();
      auto ctx = chanUserOp->getContext();

      // Creating a new unique channel for tiling.
      Operation *o =
          &chanUserOp->getParentOfType<ModuleOp>().getBody()->front();
      while (dyn_cast_or_null<air::ChannelOp>(o))
        o = o->getNextNode();
      rewriter.setInsertionPoint(o);
      auto cname =
          lookUpOrCreateChanName(chanUserOp.getChanName().str(),
                                 chanUserOp->getParentOfType<ModuleOp>());
      air::ChannelOp new_chan;
      auto new_chan_op = mlir::SymbolTable::lookupSymbolIn(
          chanUserOp->getParentOfType<ModuleOp>(), cname);
      if (new_chan_op) {
        new_chan = dyn_cast<air::ChannelOp>(new_chan_op);
      } else {
        // Get original channel shape and prepend split dimension
        auto origChanOp = air::getChannelDeclarationThroughSymbol(chanUserOp);
        SmallVector<int64_t> channel_sizes;
        channel_sizes.push_back(targetColTilingFactor); // New split dimension

        // Check if original had indices (non-scalar usage)
        if (!chanUserOp.getIndices().empty()) {
          // Preserve original index dimensions
          ArrayAttr origSizeAttr = origChanOp.getSize();
          for (auto sizeAttr : origSizeAttr) {
            channel_sizes.push_back(cast<IntegerAttr>(sizeAttr).getInt());
          }
          // If somehow no size attr but had indices, add 1s
          if (origSizeAttr.empty()) {
            for (unsigned i = 0; i < chanUserOp.getIndices().size(); i++)
              channel_sizes.push_back(1);
          }
        } else {
          // Scalar channel (no indices used), just add one more dimension
          channel_sizes.push_back(1);
        }
        new_chan = air::ChannelOp::create(
            rewriter, loc, cname, rewriter.getI64ArrayAttr(channel_sizes),
            rewriter.getStringAttr("dma_stream"));
      }

      // Perform tiling on these channel put/get ops which are using the memref.
      auto memrefShape = air::getTensorShape(memref.getType());
      int dim = splitDim;
      auto offsetDimOpt = air::getOffsetDimFromMemrefDim(
          dim, chanUserOp.getStrides(), memrefShape);
      int offsetDim = offsetDimOpt ? *offsetDimOpt : dim;
      // Update split dimension index on offsets
      for (auto &[splitInfoDimOnOffsets, splitAffineMap, splitOffset, splitSize,
                  splitStride] : infoEntryVec)
        splitInfoDimOnOffsets = offsetDim;
      auto newWaitAll = tileChannelOpByFactor(
          chanUserOp, targetColTilingFactor, memrefShape[dim], infoEntryVec,
          opToSplitInfoMap, new_chan, loc, ctx);
      if (failed(newWaitAll))
        return;
      rewriter.replaceAllUsesWith(air::getAsyncTokenFromOp(chanUserOp),
                                  *newWaitAll);

      // Now that one side of those channels are tiled, perform tiling on the
      // other side, too. Process ALL channel ops on the other side.
      auto theOtherChanOps = air::getTheOtherChannelOpThroughSymbol(chanUserOp);

      // Process each channel op on the other side
      for (auto &theOtherChanOp : theOtherChanOps) {
        // Account for cases where rank reduction results from at least
        // of the dimensions being equal to one.
        SmallVector<Value> wraps = theOtherChanOp.getSizes();
        SmallVector<Value> offsets = theOtherChanOp.getOffsets();
        SmallVector<Value> strides = theOtherChanOp.getStrides();
        if (wraps.empty()) {
          // Populate default wraps, if wraps is an empty vector.
          rewriter.setInsertionPoint(theOtherChanOp);
          air::populateDefaultWrapsAndStrides(
              rewriter, theOtherChanOp.getMemref(), offsets, wraps, strides);
        }

        // Bump up the offset, wrap and stride list to match both sides.
        SmallVector<Value> refSizes = chanUserOp.getSizes();
        SmallVector<Value> refOffsets = chanUserOp.getOffsets();
        SmallVector<Value> refStrides = chanUserOp.getStrides();
        if (refSizes.empty())
          air::populateDefaultWrapsAndStrides(rewriter, chanUserOp.getMemref(),
                                              refOffsets, refSizes, refStrides);
        SmallVector<int> newSizes, newStrides;
        rewriter.setInsertionPoint(theOtherChanOp);
        auto zeroIdx = arith::ConstantIndexOp::create(rewriter, loc, 0);
        auto oneIdx = arith::ConstantIndexOp::create(rewriter, loc, 1);
        if (wraps.size() < refSizes.size()) {
          int currIdx = offsets.size() - 1;
          for (int i = refSizes.size() - 1; i >= 0; i--) {
            // Ref size one. Insert a size-one dimension.
            if (*getConstantIntValue(refSizes[i]) == 1) {
              offsets.insert(offsets.begin() + currIdx, zeroIdx);
              wraps.insert(wraps.begin() + currIdx, oneIdx);
              auto currStride = *getConstantIntValue(strides[currIdx]);
              strides.insert(
                  strides.begin() + currIdx,
                  arith::ConstantIndexOp::create(rewriter, loc, currStride));
              continue;
            }
            // Ref size equals curr size. Continue.
            if (*getConstantIntValue(wraps[currIdx]) ==
                *getConstantIntValue(refSizes[i])) {
              currIdx = currIdx == 0 ? 0 : currIdx - 1;
              continue;
            }
            if (*getConstantIntValue(wraps[currIdx]) %
                *getConstantIntValue(refSizes[i]))
              break; // encountered size not divisible
            // Ref size neq curr size. Tile curr dimension.
            int factor = *getConstantIntValue(wraps[currIdx]) /
                         *getConstantIntValue(refSizes[i]);
            offsets.insert(offsets.begin() + currIdx, zeroIdx);
            auto newWrapVal =
                arith::ConstantIndexOp::create(rewriter, loc, factor);
            wraps.insert(wraps.begin() + currIdx, newWrapVal);
            auto newStrideVal = arith::ConstantIndexOp::create(
                rewriter, loc,
                *getConstantIntValue(refSizes[i]) *
                    *getConstantIntValue(strides[currIdx]));
            strides.insert(strides.begin() + currIdx, newStrideVal);
            wraps[currIdx + 1] = arith::ConstantIndexOp::create(
                rewriter, loc, *getConstantIntValue(refSizes[i]));
          }
        }
        air::ChannelInterface updatedChanOp = theOtherChanOp;
        if (auto put =
                dyn_cast<air::ChannelPutOp>(theOtherChanOp.getOperation())) {
          auto attrs = put->getDiscardableAttrDictionary();
          erased.insert(put);
          auto newPut = air::ChannelPutOp::create(
              rewriter, loc, put.getResultTypes(), put.getAsyncDependencies(),
              put.getChanName(), put.getIndices(), put.getMemref(), offsets,
              wraps, strides);
          newPut->setAttrs(attrs);
          rewriter.replaceAllUsesWith(put->getResults(), newPut->getResults());
          updatedChanOp = newPut;
        } else if (auto get = dyn_cast<air::ChannelGetOp>(
                       theOtherChanOp.getOperation())) {
          auto attrs = get->getDiscardableAttrDictionary();
          erased.insert(get);
          auto newGet = air::ChannelGetOp::create(
              rewriter, loc, get.getResultTypes(), get.getAsyncDependencies(),
              get.getChanName(), get.getIndices(), get.getMemref(), offsets,
              wraps, strides);
          newGet->setAttrs(attrs);
          rewriter.replaceAllUsesWith(get->getResults(), newGet->getResults());
          updatedChanOp = newGet;
        }

        auto newWaitAll1 = tileChannelOpByFactor(
            updatedChanOp, targetColTilingFactor,
            *getConstantIntValue(wraps[offsetDim]), infoEntryVec,
            opToSplitInfoMap, new_chan, loc, ctx);

        if (failed(newWaitAll1))
          return;

        // Update dependency.
        rewriter.replaceAllUsesWith(air::getAsyncTokenFromOp(updatedChanOp),
                                    *newWaitAll1);
        erased.insert(updatedChanOp);
      }
      erased.insert(chanUserOp);
    }
  }

  // STEP 4: Unroll all remaining scf.parallels only in segments that need
  // splitting.
  parOps.clear();
  func.walk([&](scf::ParallelOp parOp) {
    auto parentSeg = parOp->getParentOfType<air::SegmentOp>();
    if (parentSeg && segmentsNeedingSplit.contains(parentSeg)) {
      parOps.push_back(parOp);
    }
  });
  llvm::DenseMap<Operation *, SmallVector<Operation *>> parUnrollMap;
  for (auto par : parOps) {
    IRRewriter rewriter(ctx);
    IRMapping remap;
    rewriter.setInsertionPoint(par);
    if (failed(air::unrollScfParallel(rewriter, par, remap, parUnrollMap)))
      signalPassFailure();
    erased.insert(par);
  }
  // Update map after loop unrolling.
  for (auto &[oldOp, splitInfo] : opToSplitInfoMap) {
    Operation *o = oldOp;
    infoEntryTy info = splitInfo;
    auto unrollMapEntry = llvm::find_if(
        parUnrollMap,
        [o](std::tuple<Operation *, SmallVector<Operation *>> mapEnry) {
          return std::get<0>(mapEnry)->isAncestor(o);
        });
    if (unrollMapEntry == parUnrollMap.end())
      continue;
    for (auto newOpAncestor : std::get<1>(*unrollMapEntry))
      newOpAncestor->walk(
          [&opToSplitInfoMap, info](air::ChannelInterface newOp) {
            opToSplitInfoMap[newOp] = info;
          });
  }

  auto context = &getContext();
  RewritePatternSet canoPatterns(context);
  // Fold constants.
  mlir::arith::ConstantIndexOp::getCanonicalizationPatterns(canoPatterns,
                                                            context);
  air::ExecuteOp::getCanonicalizationPatterns(canoPatterns, context);
  (void)applyPatternsGreedily(func, std::move(canoPatterns));

  // STEP 5: Split memrefs; mutate all uses of the original memref into the
  // split ones.
  allocOps.clear();
  func.walk([&](memref::AllocOp allocOp) {
    if (allocOp->hasAttr("split")) {
      allocOps.push_back(allocOp);
    }
  });
  for (auto allocOp : allocOps) {
    auto splitInfo = (*targetMemrefsToInfoMap)[allocOp];
    auto &[splitType, splitFactor, infoEntryMap] = splitInfo;
    auto &[splitDim, infoEntryVec] =
        infoEntryMap.front(); // TODO: assuming only one dimension is subject to
                              // splitting for now.
    Value memref = isa<air::ExecuteOp>(allocOp->getParentOp())
                       ? allocOp->getParentOp()->getResult(1)
                       : dyn_cast<Value>(allocOp.getMemref());
    SmallVector<air::ChannelPutOp> puts;
    SmallVector<air::ChannelGetOp> gets;
    for (auto user : memref.getUsers()) {
      if (auto put = dyn_cast<air::ChannelPutOp>(user))
        puts.push_back(put);
      else if (auto get = dyn_cast<air::ChannelGetOp>(user))
        gets.push_back(get);
    }
    int dim = splitDim;
    partitionMemref(puts, gets, dim, allocOp, opToSplitInfoMap);
  }
  for (auto allocOp : allocOps) {
    if (auto execOp = dyn_cast<air::ExecuteOp>(allocOp->getParentOp())) {
      erased.insert(execOp);
    } else {
      erased.insert(allocOp);
    }
  }

  IRMapping waitAllRemap;
  for (auto e : erased) {
    // Replace all remaining uses of erased op's token with a new wait_all.
    if (air::isAsyncOp(e)) {
      rewriter.setInsertionPoint(e);
      auto waitAll =
          air::replaceAsyncOpWithWaitAll(rewriter, waitAllRemap, e, false);
      rewriter.replaceAllUsesWith(air::getAsyncTokenFromOp(e),
                                  waitAll.getAsyncToken());
    }
  }
  for (auto e : erased)
    rewriter.eraseOp(e);
  // Mutate the for loops to get contiguous access pattern on memrefs.
  func.walk([&](scf::ForOp forOp) {
    if (forOp->hasAttr("mutate_step_size_to")) {
      rewriter.setInsertionPoint(forOp);
      int newStep =
          forOp->getAttrOfType<IntegerAttr>("mutate_step_size_to").getInt();
      int oldStep = *getConstantIntValue(forOp.getStep());
      forOp.setStep(
          arith::ConstantIndexOp::create(rewriter, forOp->getLoc(), newStep));
      forOp.setUpperBound(arith::ConstantIndexOp::create(
          rewriter, forOp->getLoc(),
          *getConstantIntValue(forOp.getUpperBound()) / oldStep));
      forOp->removeAttr("mutate_step_size_to");
    }
  });

  air::renumberMemcpyIfOps(&func.getBody());
}

// Experimental pattern to override the memory space of `memref.alloc`
// operations when they appear inside a specified parent scope (e.g. herd,
// segment).
struct OverrideMemorySpacePattern : public OpRewritePattern<memref::AllocOp> {
  OverrideMemorySpacePattern(MLIRContext *ctx, StringRef scope, int memSpace)
      : OpRewritePattern<memref::AllocOp>(ctx), clScope(scope),
        clMemorySpace(memSpace) {}

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rewriter) const override {
    Operation *parent = nullptr;

    if (clScope == "herd")
      parent = alloc->getParentOfType<air::HerdOp>();
    else if (clScope == "segment")
      parent = alloc->getParentOfType<air::SegmentOp>();
    else if (clScope == "launch")
      parent = alloc->getParentOfType<air::LaunchOp>();
    else if (clScope == "func")
      parent = alloc->getParentOfType<func::FuncOp>();
    else
      return alloc->emitOpError(
          "Invalid clScope value: expected one of herd/segment/launch/func");

    if (!parent)
      return failure();

    auto memrefTy = dyn_cast<MemRefType>(alloc.getMemref().getType());
    if (!memrefTy)
      return failure();
    if ((int)memrefTy.getMemorySpaceAsInt() == clMemorySpace)
      return failure();

    auto newMemrefType =
        MemRefType::get(memrefTy.getShape(), memrefTy.getElementType(),
                        memrefTy.getLayout().getAffineMap(),
                        rewriter.getI32IntegerAttr(clMemorySpace));

    rewriter.replaceOpWithNewOp<memref::AllocOp>(alloc, newMemrefType);

    return success();
  }

private:
  StringRef clScope; // Parent operation type to match
  int clMemorySpace; // Target memory space value to assign
};

// Pattern to correct memory spaces of view-like operations within a given
// scope, following the application of OverrideMemorySpacePattern.
template <typename OpTy>
struct correctViewLikeOpIOMemorySpacesInScope : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy IFAOp,
                                PatternRewriter &rewriter) const override {

    if (!IFAOp->template hasTrait<OpTrait::IsIsolatedFromAbove>())
      return failure();
    llvm::DenseMap<ViewLikeOpInterface, SmallVector<OpResult>> viewLikeOpsToRes;
    IFAOp->walk([&](ViewLikeOpInterface viewLike) {
      auto srcTy = dyn_cast<MemRefType>(viewLike.getViewSource().getType());
      if (!srcTy)
        return;
      for (auto res : viewLike->getResults()) {
        auto destTy = dyn_cast<MemRefType>(res.getType());
        if (!destTy)
          return;
        if (srcTy.getMemorySpaceAsInt() == destTy.getMemorySpaceAsInt())
          continue;
        viewLikeOpsToRes[viewLike].push_back(res);
      }
    });
    for (auto [viewLike, results] : viewLikeOpsToRes) {
      for (OpResult res : results) {
        auto srcTy = dyn_cast<MemRefType>(viewLike.getViewSource().getType());
        auto destTy = dyn_cast<MemRefType>(res.getType());
        MemRefType::Builder builder(destTy);
        builder.setMemorySpace(srcTy.getMemorySpace());
        rewriter.modifyOpInPlace(viewLike, [&]() { res.setType(builder); });
      }
    }
    return success();
  }
};

// An experimental pass forcing all memrefs allocated within a specified air
// code region to have the specified memory space.
class AIROverrideMemRefMemorySpacePass
    : public air::impl::AIROverrideMemRefMemorySpaceBase<
          AIROverrideMemRefMemorySpacePass> {

public:
  AIROverrideMemRefMemorySpacePass() = default;
  AIROverrideMemRefMemorySpacePass(
      const AIROverrideMemRefMemorySpacePass &pass){};
  AIROverrideMemRefMemorySpacePass(
      const ::xilinx::air::AIROverrideMemRefMemorySpaceOptions &options)
      : AIROverrideMemRefMemorySpaceBase(options) {}

  void runOnOperation() override;

private:
};

void AIROverrideMemRefMemorySpacePass::runOnOperation() {
  auto moduleOp = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  patterns.add<OverrideMemorySpacePattern>(context, clScope, clMemorySpace);
  (void)applyPatternsGreedily(moduleOp, std::move(patterns));
  RewritePatternSet fixResTypePatterns(context);
  if (clScope == "herd") {
    fixResTypePatterns.add<correctViewLikeOpIOMemorySpacesInScope<air::HerdOp>>(
        context);
  } else if (clScope == "segment") {
    fixResTypePatterns
        .add<correctViewLikeOpIOMemorySpacesInScope<air::SegmentOp>>(context);
  } else if (clScope == "launch") {
    fixResTypePatterns
        .add<correctViewLikeOpIOMemorySpacesInScope<air::LaunchOp>>(context);
  } else if (clScope == "func") {
    fixResTypePatterns
        .add<correctViewLikeOpIOMemorySpacesInScope<func::FuncOp>>(context);
  }
  (void)applyPatternsGreedily(moduleOp, std::move(fixResTypePatterns));
}

} // namespace xilinx

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRExamplePass() {
  return std::make_unique<AIRExamplePass>();
}

std::unique_ptr<Pass> createAIRSpecializeDmaBroadcast() {
  return std::make_unique<AIRSpecializeDmaBroadcast>();
}

std::unique_ptr<Pass> createAIRLinalgNamePass() {
  return std::make_unique<AIRLinalgNamePass>();
}

std::unique_ptr<Pass> createAIRRemoveLinalgNamePass() {
  return std::make_unique<AIRRemoveLinalgNamePass>();
}

std::unique_ptr<Pass> createAIRFuseParallelHerdPass() {
  return std::make_unique<AIRFuseParallelHerdPass>();
}

std::unique_ptr<Pass> createAIRRenumberDmaIdPass() {
  return std::make_unique<AIRRenumberDmaIdPass>();
}

std::unique_ptr<Pass> createAIRLowerHerdParallelPass() {
  return std::make_unique<AIRLowerHerdParallelPass>();
}

std::unique_ptr<Pass> createAIRLabelBroadcastChannelWithTilePass() {
  return std::make_unique<AIRLabelBroadcastChannelWithTilePass>();
}

std::unique_ptr<Pass> createAIRCollapseHerdPass() {
  return std::make_unique<AIRCollapseHerdPass>();
}

std::unique_ptr<Pass>
createAIRCollapseHerdPass(AIRCollapseHerdPassOptions options) {
  return std::make_unique<AIRCollapseHerdPass>(options);
}

std::unique_ptr<Pass> createAIRFuseNestedHerdPass() {
  return std::make_unique<AIRFuseNestedHerdPass>();
}

std::unique_ptr<Pass>
createAIRFuseNestedHerdPass(AIRFuseNestedHerdPassOptions options) {
  return std::make_unique<AIRFuseNestedHerdPass>(options);
}

std::unique_ptr<Pass> createAIRUnrollOuterPerfectlyNestedLoopsPass() {
  return std::make_unique<AIRUnrollOuterPerfectlyNestedLoopsPass>();
}

std::unique_ptr<Pass> createAIRUnrollOuterPerfectlyNestedLoopsPass(
    AIRUnrollOuterPerfectlyNestedLoopsPassOptions options) {
  return std::make_unique<AIRUnrollOuterPerfectlyNestedLoopsPass>(options);
}

std::unique_ptr<Pass> createAIRSplitL2MemrefForBufferConstraintPass() {
  return std::make_unique<AIRSplitL2MemrefForBufferConstraintPass>();
}

std::unique_ptr<Pass> createAIROverrideMemRefMemorySpacePass() {
  return std::make_unique<AIROverrideMemRefMemorySpacePass>();
}
std::unique_ptr<Pass> createAIROverrideMemRefMemorySpacePass(
    AIROverrideMemRefMemorySpaceOptions options) {
  return std::make_unique<AIROverrideMemRefMemorySpacePass>(options);
}

} // namespace air
} // namespace xilinx
