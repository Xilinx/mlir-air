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
using namespace xilinx;

namespace {

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
  }

  void runOnFunction(func::FuncOp f) {
    // Specialize broadcastable DMA into affine.if regions
    specializeDmaBroadcastWithAffineIf(f);
    // Walk the affine.if's affine.set and simplify DMA source indices
    simplifyDmaIndicesWithAffineSet(f);
  }

private:
  void specializeDmaBroadcastWithAffineIf(func::FuncOp f) {
    f.walk([&](air::DmaMemcpyNdOp memcpyOp) {
      auto herdOp = memcpyOp->getParentOfType<air::HerdOp>();
      if (!herdOp)
        return;
      auto herd_id = herdOp.getIds();
      OpBuilder builder(memcpyOp);
      auto loc = memcpyOp->getLoc();
      auto broadcast_pattern =
          memcpyOp->getAttrOfType<mlir::IntegerSetAttr>("broadcast_pattern");
      auto ctx = memcpyOp->getContext();
      if (!broadcast_pattern)
        return;
      auto is = broadcast_pattern.getValue();
      auto constraints = is.getConstraints();
      auto eqFlags = is.getEqFlags();
      unsigned numSegments = 1;
      // Get symbol range (i.e. segment range)
      SmallVector<AffineExpr, 1> zero_syms{
          getAffineConstantExpr(0, ctx),
      };
      for (auto c : constraints) {
        if (c.isSymbolicOrConstant()) {
          auto newC = c.replaceSymbols(zero_syms);
          auto expr =
              dyn_cast<AffineConstantExpr>(simplifyAffineExpr(newC, 0, 1));
          if (!expr) {
            continue;
          }
          if (expr.getValue() != 0) {
            numSegments = expr.getValue() + 1;
          }
        }
      }
      // Walk each set in the patitioning scheme
      // Specialize each affine set
      for (unsigned i = 0; i < numSegments; i++) {
        SmallVector<AffineExpr, 2> newConstraints;
        SmallVector<bool, 2> newEqflags;
        SmallVector<AffineExpr, 1> i_syms{
            getAffineConstantExpr(i, ctx),
        };
        SmallVector<AffineExpr, 2> syms{
            getAffineSymbolExpr(0, ctx),
            getAffineSymbolExpr(1, ctx),
        };
        int c_iter = 0;
        for (auto c : constraints) {
          if (!c.isSymbolicOrConstant()) {
            // Substitute segment id i_syms into inequalities
            auto newC = c.replaceSymbols(i_syms);
            // Replace all dims with symbols
            newC = newC.replaceDims(syms);
            newConstraints.push_back(newC);
            newEqflags.push_back(eqFlags[c_iter]);
          }
          c_iter++;
        }
        auto int_set = IntegerSet::get(0, 2, newConstraints, newEqflags);
        SmallVector<Value, 2> int_set_args{herd_id[0], herd_id[1]};
        // Duplicate dma ops per spatial segment
        if (i == 0) {
          affine::AffineIfOp aif = builder.create<affine::AffineIfOp>(
              loc, air::AsyncTokenType::get(ctx), int_set, int_set_args, true);
          builder.setInsertionPointToStart(aif.getThenBlock());
          auto memcpyOp_cloned = builder.clone(*memcpyOp.getOperation());
          memcpyOp_cloned->removeAttr("broadcast_pattern");
          memcpyOp_cloned->setAttr("broadcast_set",
                                   mlir::IntegerSetAttr::get(int_set));
          SmallVector<Value, 1> yield_token;
          yield_token.push_back(
              dyn_cast<air::AsyncOpInterface>(memcpyOp_cloned).getAsyncToken());
          builder.create<affine::AffineYieldOp>(memcpyOp_cloned->getLoc(),
                                                yield_token);
          // Reconnect dependency graph using the outermost affine.if's
          // token
          auto async_memcpyOp =
              dyn_cast<air::AsyncOpInterface>(memcpyOp.getOperation());
          async_memcpyOp.getAsyncToken().replaceAllUsesWith(aif.getResult(0));
          builder.setInsertionPointToStart(aif.getElseBlock());
          // If single segment, then create an empty else block.
          if (numSegments == 1) {
            auto waitAllOp = builder.create<air::WaitAllOp>(
                memcpyOp_cloned->getLoc(),
                air::AsyncTokenType::get(memcpyOp_cloned->getContext()),
                memcpyOp.getAsyncDependencies());
            builder.create<affine::AffineYieldOp>(
                memcpyOp_cloned->getLoc(),
                SmallVector<Value>{waitAllOp.getAsyncToken()});
          }
        } else if (i < numSegments - 1) {
          affine::AffineIfOp aif = builder.create<affine::AffineIfOp>(
              builder.getUnknownLoc(), air::AsyncTokenType::get(ctx), int_set,
              int_set_args, (i != numSegments - 1));
          builder.setInsertionPointToStart(aif.getThenBlock());
          auto memcpyOp_cloned = builder.clone(*memcpyOp.getOperation());
          memcpyOp_cloned->removeAttr("broadcast_pattern");
          memcpyOp_cloned->setAttr("broadcast_set",
                                   mlir::IntegerSetAttr::get(int_set));
          SmallVector<Value, 1> yield_token;
          yield_token.push_back(
              dyn_cast<air::AsyncOpInterface>(memcpyOp_cloned).getAsyncToken());
          builder.create<affine::AffineYieldOp>(memcpyOp_cloned->getLoc(),
                                                yield_token);
          builder.setInsertionPointAfter(aif);
          SmallVector<Value, 1> parent_block_yield_token = {aif.getResult(0)};
          builder.create<affine::AffineYieldOp>(builder.getUnknownLoc(),
                                                parent_block_yield_token);
          builder.setInsertionPointToStart(aif.getElseBlock());
        } else {
          auto memcpyOp_cloned = builder.clone(*memcpyOp.getOperation());
          memcpyOp_cloned->removeAttr("broadcast_pattern");
          memcpyOp_cloned->setAttr("broadcast_set",
                                   mlir::IntegerSetAttr::get(int_set));
          SmallVector<Value, 1> yield_token;
          yield_token.push_back(
              dyn_cast<air::AsyncOpInterface>(memcpyOp_cloned).getAsyncToken());
          builder.create<affine::AffineYieldOp>(memcpyOp_cloned->getLoc(),
                                                yield_token);
        }
      }
      memcpyOp.erase();
    });
  }

  void simplifyDmaIndicesWithAffineSet(func::FuncOp f) {

    f.walk([&](air::DmaMemcpyNdOp memcpyOp) {
      auto ctx = memcpyOp->getContext();
      if (!memcpyOp->hasAttr("broadcast_set"))
        return;
      auto broadcast_set =
          memcpyOp->getAttrOfType<mlir::IntegerSetAttr>("broadcast_set");
      // Get all ops on the dependency connection between dma and herd launch
      std::vector<Operation *> op_history;
      auto loop_dep_history = traceDependentHerdId(memcpyOp);
      // "loop_dep_history" tuple fields: value, ancestors and producers to
      // those ancestors.

      // Walk constraints in broadcast pattern, and get shape of the broadcast
      // pattern
      auto is = broadcast_set.getValue();
      auto constraints = is.getConstraints();
      auto eqFlags = is.getEqFlags();

      // Check which dimension op operates on; initialize current_shape_expr
      SmallVector<AffineExpr, 2> current_shape_expr = {nullptr, nullptr};
      SmallVector<Value, 2> herdDimToDmaOffsetDimMap = {nullptr, nullptr};
      for (auto &elem : loop_dep_history) {
        for (auto v : std::get<1>(elem)) {
          if (!air::getHerdArgOwner(v))
            continue;
          auto hl_op = air::getHerdArgOwner(v);
          for (unsigned j = 0; j < current_shape_expr.size(); j++) {
            if (v != hl_op.getIds()[j])
              continue;
            for (unsigned i = 0; i < constraints.size(); i++) {
              auto c = constraints[i];
              if (!c.isFunctionOfSymbol(j))
                continue;
              if (!eqFlags[i])
                continue;
              auto eval = air::evaluateSymbolEqualityInSet(c, ctx);
              current_shape_expr[j] = getAffineConstantExpr(eval, ctx);
              herdDimToDmaOffsetDimMap[j] = std::get<0>(elem);
              op_history.insert(op_history.end(), std::get<2>(elem).begin(),
                                std::get<2>(elem).end());
            }
          }
        }
      }

      // Evaluate broadcast pattern by propagating expr through scalar
      // operations in op history, last-in-first-out
      for (std::vector<Operation *>::reverse_iterator i = op_history.rbegin();
           i != op_history.rend(); ++i) {
        if (auto exec_op = dyn_cast<air::ExecuteOp>(*i)) {
          Operation *op = &exec_op.getChildOps().front();
          // If the async op is affine.apply
          if (auto apply_op = dyn_cast<affine::AffineApplyOp>(op)) {
            // Can only propagate affine.apply ops with single operand.
            if (apply_op.getNumOperands() != 1)
              return;
            auto map = apply_op.getAffineMap();
            for (unsigned j = 0; j < current_shape_expr.size(); j++) {
              if (current_shape_expr[j]) {
                replaceSymbolAndEvaluateConstantInMap(
                    map, current_shape_expr[j], ctx);
                // Remove dependence from scalar op to memcpyOp if present
                auto async_memcpyOp =
                    dyn_cast<air::AsyncOpInterface>(memcpyOp.getOperation());
                eraseAsyncDependencyFromAsyncOp(async_memcpyOp,
                                                exec_op.getAsyncToken());
              }
            }
          }

          // If the async op is arith op
          else if (auto arith_op = dyn_cast<arith::AddIOp>(op)) {
            propagateAffineConstantExprThroughArithOp<arith::AddIOp>(
                arith_op, current_shape_expr, memcpyOp.getOperation(), ctx);
          } else if (auto arith_op = dyn_cast<arith::MulIOp>(op)) {
            propagateAffineConstantExprThroughArithOp<arith::MulIOp>(
                arith_op, current_shape_expr, memcpyOp.getOperation(), ctx);
          }
        }
      }

      // Mutate memcpy op.
      (void)replaceMemcpyOpWithSimplifiedOperands(memcpyOp, current_shape_expr,
                                                  herdDimToDmaOffsetDimMap);
    });
  }

  // Evaluate the affine expression of affine map if the only symbolic
  // identifier is replaced with zero
  void replaceSymbolAndEvaluateConstantInMap(AffineMap map, AffineExpr &c,
                                             MLIRContext *ctx) {
    auto newmap = map.replace(getAffineSymbolExpr(0, ctx), c, 0, 1);
    auto const_int = simplifyAffineMap(newmap).getSingleConstantResult();
    c = getAffineConstantExpr(const_int, ctx);
  }

  // AddI for AffineConstantExpr
  void applyArithOpToAffineConstantExpr(arith::AddIOp arith_op, AffineExpr &c,
                                        MLIRContext *ctx) {
    arith::ConstantIndexOp add_operand = nullptr;
    if (arith_op.getLhs().getDefiningOp() &&
        dyn_cast<arith::ConstantIndexOp>(arith_op.getLhs().getDefiningOp())) {
      add_operand =
          dyn_cast<arith::ConstantIndexOp>(arith_op.getLhs().getDefiningOp());
    } else if (arith_op.getRhs().getDefiningOp() &&
               dyn_cast<arith::ConstantIndexOp>(
                   arith_op.getRhs().getDefiningOp())) {
      add_operand =
          dyn_cast<arith::ConstantIndexOp>(arith_op.getRhs().getDefiningOp());
    } else {
      // arith::AddIOp has no arith::ConstantIndexOp operand. Abort trying to
      // specialize the expr
      c = nullptr;
      return;
    }
    auto acc = add_operand.value();
    if (!isa<AffineConstantExpr>(c)) {
      arith_op->emitOpError("non-constant affine expression.");
      return;
    }
    acc += dyn_cast<AffineConstantExpr>(c).getValue();
    c = getAffineConstantExpr(acc, ctx);
  }

  // MulI for AffineConstantExpr
  void applyArithOpToAffineConstantExpr(arith::MulIOp arith_op, AffineExpr &c,
                                        MLIRContext *ctx) {
    arith::ConstantIndexOp mul_operand = nullptr;
    if (arith_op.getLhs().getDefiningOp() &&
        dyn_cast<arith::ConstantIndexOp>(arith_op.getLhs().getDefiningOp())) {
      mul_operand =
          dyn_cast<arith::ConstantIndexOp>(arith_op.getLhs().getDefiningOp());
    } else if (arith_op.getRhs().getDefiningOp() &&
               dyn_cast<arith::ConstantIndexOp>(
                   arith_op.getRhs().getDefiningOp())) {
      mul_operand =
          dyn_cast<arith::ConstantIndexOp>(arith_op.getRhs().getDefiningOp());
    } else {
      // arith::MulIOp has no arith::ConstantIndexOp operand. Abort trying to
      // specialize the expr
      c = nullptr;
      return;
    }
    auto mul = mul_operand.value();
    if (!isa<AffineConstantExpr>(c)) {
      arith_op->emitOpError("non-constant affine expression.");
      return;
    }
    mul *= dyn_cast<AffineConstantExpr>(c).getValue();
    c = getAffineConstantExpr(mul, ctx);
  }

  // Propagate AffineConstantExpr through arith addi/muli op
  template <typename T>
  void propagateAffineConstantExprThroughArithOp(
      T arith_op, SmallVector<AffineExpr, 2> &current_shape_expr,
      Operation *memcpyOp, MLIRContext *ctx) {
    air::ExecuteOp parent_region_op =
        arith_op->template getParentOfType<air::ExecuteOp>();
    for (unsigned j = 0; j < current_shape_expr.size(); j++) {
      if (current_shape_expr[j]) {
        applyArithOpToAffineConstantExpr(arith_op, current_shape_expr[j], ctx);
        // Remove dependence from scalar op to memcpyOp if present
        auto async_memcpyOp = dyn_cast<air::AsyncOpInterface>(memcpyOp);
        eraseAsyncDependencyFromAsyncOp(async_memcpyOp,
                                        parent_region_op.getAsyncToken());
      }
    }
  }

  // Replace memcpyOp's dependent operand with const
  LogicalResult replaceMemcpyOpWithSimplifiedOperands(
      air::DmaMemcpyNdOp &memcpyOp,
      SmallVector<AffineExpr, 2> current_shape_expr,
      SmallVector<Value, 2> herdDimToDmaOffsetDimMap) {
    OpBuilder builder(memcpyOp);
    auto loc = memcpyOp->getLoc();
    bool opIsUpdated = false;
    for (unsigned i = 0; i < current_shape_expr.size(); i++) {
      if (!current_shape_expr[i])
        continue;
      if (!herdDimToDmaOffsetDimMap[i])
        continue;
      int opOperandId = -1;
      for (unsigned j = 0; j < memcpyOp->getNumOperands(); j++)
        if (memcpyOp->getOperand(j) == herdDimToDmaOffsetDimMap[i])
          opOperandId = j;
      if (opOperandId < 0)
        continue;
      auto val = dyn_cast<AffineConstantExpr>(current_shape_expr[i]).getValue();
      auto cop = builder.create<arith::ConstantIndexOp>(loc, val);
      memcpyOp->getOpOperand(opOperandId).assign(cop);
      opIsUpdated = true;
    }
    if (opIsUpdated)
      return success();
    return failure();
  }
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
    if (isa_and_present<arith::ConstantOp>(v.getDefiningOp()))
      constants.push_back(v);
    else
      args.push_back(v);
  }

  auto newLaunchOp = b.create<air::HerdOp>(
      parOp.getLoc(), launchOp.getAsyncDependencies(), dims, args,
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
      forOp = rewriter.create<scf::ForOp>(op->getLoc(), lb, ub, step);
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
    auto cst0 = outsideBuilder.create<arith::ConstantIndexOp>(loc, 0);
    auto cst1 = outsideBuilder.create<arith::ConstantIndexOp>(loc, 1);
    // First dimension size set to one, i.e. a single column
    lowerBounds.push_back(cst0);
    steps.push_back(cst1);
    upperBounds.push_back(cst1);
    // Second dimension onwards
    Value newUpperBound = outsideBuilder.create<arith::ConstantIndexOp>(loc, 1);
    for (auto idx : dims) {
      newUpperBound = outsideBuilder.create<arith::MulIOp>(
          loc, newUpperBound,
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
        insideBuilder.create<arith::ConstantIndexOp>(loc, *old_upper_bound);

    // Determine the current induction value's current loop iteration
    Value iv_1 =
        insideBuilder.create<arith::RemSIOp>(loc, h.getIds()[1], old_upper_b_v);
    llvm::cast<Value>(h.getIds()[1])
        .replaceAllUsesExcept(iv_1, iv_1.getDefiningOp());

    // Remove the effect of the current induction value to prepare for
    // the next value.
    Value iv_0 =
        insideBuilder.create<arith::DivSIOp>(loc, h.getIds()[1], old_upper_b_v);
    replaceAllUsesInRegionWith(h.getIds()[0], iv_0, h.getBody());

    // Update upper bounds.
    int operandsIdxOffset = h.getAsyncDependencies().size();
    for (unsigned i = operandsIdxOffset; i < operandsIdxOffset + h.getNumDims();
         i++) {
      h->getOpOperand(i).assign(upperBounds[i - operandsIdxOffset]);
    }
  }
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
  std::optional<int>
  getMemrefSplitDim(SmallVector<air::ChannelInterface> putgets,
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
  Value zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  // Create and apply affine map onto the split channel ops.
  SmallVector<Value> tokens;
  int memorySpace = dyn_cast<MemRefType>(originalChanOp.getMemref().getType())
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

    SmallVector<Value> newIndices{
        rewriter.create<arith::ConstantIndexOp>(loc, i), zeroIdx};
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
        original_map =
            original_map.replace(getAffineSymbolExpr(0, ctx),
                                 getAffineConstantExpr(const_in, ctx), 0, 1);
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
    newApplyOp =
        rewriter.create<affine::AffineApplyOp>(loc, map, originalApplyOperands);
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
          rewriter.create<arith::ConstantIndexOp>(loc, *splitInfoSplitSize);
    else
      newWraps[splitDimOnOffsets] = rewriter.create<arith::ConstantIndexOp>(
          loc, llvm::divideCeilSigned(originalMemrefSize, factor));
    // Stride manipulation is only allowed for L3 memory: we are not splitting
    // the L3 memref; we are only splitting its access pattern.
    // Strategy: add one dimension to wrap-and-stride list. Rationale: (1) the
    // stride factor should apply on existing size, (2) original offset must
    // continue with the original stride.
    if (splitInfoSplitStrideFactor &&
        memorySpace == (int)air::MemorySpace::L3) {
      newStrides.insert(newStrides.begin() + splitDimOnOffsets,
                        newStrides[splitDimOnOffsets]);
      newStrides[splitDimOnOffsets + 1] =
          rewriter.create<arith::ConstantIndexOp>(
              loc, *getConstantIntValue(newStrides[splitDimOnOffsets]) *
                       (*splitInfoSplitStrideFactor));
      newWraps.insert(newWraps.begin() + splitDimOnOffsets,
                      rewriter.create<arith::ConstantIndexOp>(loc, 1));
      newOffsets.insert(newOffsets.begin() + splitDimOnOffsets,
                        newOffsets[splitDimOnOffsets]);
      newOffsets[splitDimOnOffsets + 1] =
          rewriter.create<arith::ConstantIndexOp>(loc, 0);
    }
    auto deps = dyn_cast<air::AsyncOpInterface>(originalChanOp.getOperation())
                    .getAsyncDependencies();
    SmallVector<Type, 4> tys = {air::AsyncTokenType::get(ctx)};
    if (isa<air::ChannelGetOp>(originalChanOp)) {
      auto newGetOp = rewriter.create<air::ChannelGetOp>(
          loc, tys, deps, newChanOp.getSymName(), newIndices,
          originalChanOp.getMemref(), newOffsets, newWraps, newStrides);
      newGetOp->setAttrs(originalChanOp->getDiscardableAttrDictionary());
      tokens.push_back(newGetOp.getAsyncToken());
      opToSplitInfoMap[newGetOp] = splitInfoVec[i];
    } else {
      auto newPutOp = rewriter.create<air::ChannelPutOp>(
          loc, tys, deps, newChanOp.getSymName(), newIndices,
          originalChanOp.getMemref(), newOffsets, newWraps, newStrides);
      newPutOp->setAttrs(originalChanOp->getDiscardableAttrDictionary());
      tokens.push_back(newPutOp.getAsyncToken());
      opToSplitInfoMap[newPutOp] = splitInfoVec[i];
    }
  }
  auto newWaitAll = rewriter.create<air::WaitAllOp>(
      loc, air::AsyncTokenType::get(ctx), tokens);
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
  if (auto exec = dyn_cast<air::ExecuteOp>(defOp)) {
    SetVector<Value> opers;
    getUsedValuesDefinedAbove(exec.getRegion(), opers);
    for (auto oper : opers)
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
          builder.create<air::ExecuteOp>(loc, air::AsyncTokenType::get(ctx),
                                         newMemrefType, SmallVector<Value>{});
      Block *async_bb = builder.createBlock(&execOp.getRegion());
      builder.setInsertionPointToStart(async_bb);
      auto childMemAlloc = builder.create<memref::AllocOp>(loc, newMemrefType);
      builder.create<xilinx::air::ExecuteTerminatorOp>(
          loc, childMemAlloc->getResults());
      newMemref = execOp->getResult(1);
      builder.setInsertionPoint(execOp);
    } else
      newMemref = builder.create<memref::AllocOp>(loc, newMemrefType);
    // Create new dealloc ops.
    if (deallocOp) {
      builder.setInsertionPoint(deallocOp);
      if (auto execDeallocOp = dyn_cast<air::ExecuteOp>(deallocOp)) {
        auto execOp = builder.create<air::ExecuteOp>(
            loc, air::AsyncTokenType::get(ctx),
            execDeallocOp.getAsyncDependencies());
        Block *async_bb = builder.createBlock(&execOp.getRegion());
        builder.setInsertionPointToStart(async_bb);
        builder.create<memref::DeallocOp>(loc, newMemref);
        builder.create<xilinx::air::ExecuteTerminatorOp>(loc);
      } else
        builder.create<memref::DeallocOp>(loc, newMemref);
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
          offsetOpOper.assign(builder.create<arith::ConstantIndexOp>(loc, 0));
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
            builder.create<arith::ConstantIndexOp>(loc, newStrides[i]));
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
    SmallVector<air::ChannelInterface> putgets, SmallVector<int> memrefShape) {
  std::optional<int> memrefDim = std::nullopt;
  for (unsigned i = 0; i < putgets.size() - 1; i++) {
    for (unsigned j = i + 1; j < putgets.size(); j++) {
      if (putgets[i].getOffsets().size() != putgets[j].getOffsets().size())
        continue;
      auto offsetZip =
          llvm::zip_equal(putgets[i].getOffsets(), putgets[j].getOffsets());
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
  return air::getMemrefDimFromOffsetDim(*memrefDim, putgets[0].getOffsets(),
                                        putgets[0].getStrides(), memrefShape);
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
    // Multiple-in-multiple-out. Skip.
    if (getChanCount(MM2SChannels) > 1 && getChanCount(S2MMChannels) > 1)
      continue;

    // Get tiling factor.
    int tilingFactor =
        std::max(getChanCount(MM2SChannels), getChanCount(S2MMChannels));

    llvm::MapVector<int, SmallVector<infoEntryTy>> infoEntryMap;
    std::optional<int> splitDimOffset = std::nullopt;
    std::optional<int> splitDimSize = std::nullopt;
    std::optional<int> splitDimStrideFactor = std::nullopt;
    std::optional<int> splitDim = std::nullopt;

    // Get all puts or gets, whichever direction has multiple operators.
    SmallVector<air::ChannelInterface> putgets;
    if (getChanCount(MM2SChannels) > 1) {
      for (auto &[chanOp, __] : MM2SChannels)
        for (auto put : air::getChannelPutOpThroughSymbol(chanOp))
          putgets.push_back(put);
    } else {
      for (auto &[chanOp, __] : S2MMChannels)
        for (auto get : air::getChannelGetOpThroughSymbol(chanOp))
          putgets.push_back(get);
    }

    splitDim =
        getMemrefSplitDim(putgets, air::getTensorShape(memref.getType()));
    if (!splitDim) {
      allocOp->emitOpError(
          "memref splitting analysis failed to get the split dimension.");
      return failure();
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
      auto offsetDimOpt =
          air::getOffsetDimFromMemrefDim(*splitDim, putgets[i].getStrides(),
                                         air::getTensorShape(memref.getType()));
      // Infer offset at splitDim.
      if (auto rootOffset =
              getRootOffset(putgets[i].getOffsets()[*offsetDimOpt]))
        splitDimOffset = *rootOffset;
      // Infer size at splitDim.
      if (auto rootSize = getRootSize(putgets[i].getOffsets()[*offsetDimOpt],
                                      putgets[i].getSizes()[*offsetDimOpt]))
        splitDimSize = *rootSize;
      // Infer stride (factor) at splitDim. If the root comes from an scf.for
      // loop, and if the loop has non-unit step size, then that multiplier
      // should be applied to other split channe put/get ops.
      if (auto rootStrideFactor =
              getRootStrideFactor(putgets[i].getOffsets()[*offsetDimOpt],
                                  putgets[i].getStrides()[*offsetDimOpt])) {
        splitDimStrideFactor = *rootStrideFactor;
        // Cancel out the non-unit step size on the for loop, to get contiguous
        // access pattern on memrefs after split.
        if (auto forOp =
                getScfForFromVal(putgets[i].getOffsets()[*offsetDimOpt])) {
          forOp->setAttr("mutate_step_size_to",
                         IntegerAttr::get(IntegerType::get(ctx, 32), 1));
        }
      }
      AffineMap applyMap;
      auto apply = getAffineMapOnMemrefSplitDim(putgets[i], *offsetDimOpt);
      if (apply)
        applyMap = apply.getAffineMap();

      infoEntryTy newEntry = {*offsetDimOpt, applyMap, splitDimOffset,
                              splitDimSize, splitDimStrideFactor};
      infoEntryMap[*splitDim].push_back(newEntry);
      opToSplitInfoMap[putgets[i]] = newEntry;
    }

    // Get output map.
    if (getChanCount(MM2SChannels) > 1) {
      targetMemrefsToInfoMap[allocOp] = {"MM2SChannels", tilingFactor,
                                         infoEntryMap};
    } else {
      targetMemrefsToInfoMap[allocOp] = {"S2MMChannels", tilingFactor,
                                         infoEntryMap};
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
  SmallVector<memref::AllocOp> allocOps;
  func.walk([&](memref::AllocOp allocOp) {
    if (allocOp->getParentOfType<air::SegmentOp>() &&
        llvm::cast<MemRefType>(allocOp.getMemref().getType())
                .getMemorySpaceAsInt() == (int)air::MemorySpace::L2) {
      allocOps.push_back(allocOp);
    }
  });

  // Check if any segment must be allocated to more than one column which
  // implies more than one memtile (assumption is one memtile per column). If
  // none, then memref splitting is not needed, as no routings or channels can
  // be saved if only allocating to a single memtile.
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
  if (llvm::none_of(allocOps, [&](memref::AllocOp a) {
        if (auto s = a->getParentOfType<air::SegmentOp>()) {
          return getTileCountInSegment(s) > clNumTilesPerL2Tile;
        } else
          return false;
      }))
    return;

  // STEP 1: Unroll scf.parallels in segment
  SmallVector<scf::ParallelOp> parOps;
  func.walk([&](scf::ParallelOp parOp) {
    if (parOp->getParentOfType<air::SegmentOp>()) {
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
      if (isa<air::ChannelPutOp>(user) && splitType == "MM2SChannels")
        continue;
      if (isa<air::ChannelGetOp>(user) && splitType == "S2MMChannels")
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
        SmallVector<int64_t, 2> channel_sizes = {targetColTilingFactor, 1};
        new_chan = rewriter.create<air::ChannelOp>(
            loc, cname, rewriter.getI64ArrayAttr(channel_sizes));
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
      // other side, too.
      auto theOtherChanOp = air::getTheOtherChannelOpThroughSymbol(chanUserOp);

      // Account for cases where rank reduction results from at least
      // of the dimensions being equal to one.
      SmallVector<Value> wraps = theOtherChanOp[0].getSizes();
      SmallVector<Value> offsets = theOtherChanOp[0].getOffsets();
      SmallVector<Value> strides = theOtherChanOp[0].getStrides();
      if (wraps.empty()) {
        // Populate default wraps, if wraps is an empty vector.
        rewriter.setInsertionPoint(theOtherChanOp[0]);
        air::populateDefaultWrapsAndStrides(
            rewriter, theOtherChanOp[0].getMemref(), offsets, wraps, strides);
      }

      // Bump up the offset, wrap and stride list to match both sides.
      SmallVector<Value> refSizes = chanUserOp.getSizes();
      SmallVector<Value> refOffsets = chanUserOp.getOffsets();
      SmallVector<Value> refStrides = chanUserOp.getStrides();
      if (refSizes.empty())
        air::populateDefaultWrapsAndStrides(rewriter, chanUserOp.getMemref(),
                                            refOffsets, refSizes, refStrides);
      SmallVector<int> newSizes, newStrides;
      rewriter.setInsertionPoint(theOtherChanOp[0]);
      auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
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
                rewriter.create<arith::ConstantIndexOp>(loc, currStride));
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
              rewriter.create<arith::ConstantIndexOp>(loc, factor);
          wraps.insert(wraps.begin() + currIdx, newWrapVal);
          auto newStrideVal = rewriter.create<arith::ConstantIndexOp>(
              loc, *getConstantIntValue(refSizes[i]) *
                       *getConstantIntValue(strides[currIdx]));
          strides.insert(strides.begin() + currIdx, newStrideVal);
          wraps[currIdx + 1] = rewriter.create<arith::ConstantIndexOp>(
              loc, *getConstantIntValue(refSizes[i]));
        }
      }
      if (auto put =
              dyn_cast<air::ChannelPutOp>(theOtherChanOp[0].getOperation())) {
        auto attrs = put->getDiscardableAttrDictionary();
        erased.insert(put);
        auto newPut = rewriter.create<air::ChannelPutOp>(
            loc, put.getResultTypes(), put.getAsyncDependencies(),
            put.getChanName(), put.getIndices(), put.getMemref(), offsets,
            wraps, strides);
        newPut->setAttrs(attrs);
        rewriter.replaceAllUsesWith(put->getResults(), newPut->getResults());
        theOtherChanOp[0] = newPut;
      } else if (auto get = dyn_cast<air::ChannelGetOp>(
                     theOtherChanOp[0].getOperation())) {
        auto attrs = get->getDiscardableAttrDictionary();
        erased.insert(get);
        auto newGet = rewriter.create<air::ChannelGetOp>(
            loc, get.getResultTypes(), get.getAsyncDependencies(),
            get.getChanName(), get.getIndices(), get.getMemref(), offsets,
            wraps, strides);
        newGet->setAttrs(attrs);
        rewriter.replaceAllUsesWith(get->getResults(), newGet->getResults());
        theOtherChanOp[0] = newGet;
      }

      auto newWaitAll1 = tileChannelOpByFactor(
          theOtherChanOp[0], targetColTilingFactor,
          *getConstantIntValue(wraps[offsetDim]), infoEntryVec,
          opToSplitInfoMap, new_chan, loc, ctx);

      if (failed(newWaitAll1))
        return;

      // Update dependency.
      rewriter.replaceAllUsesWith(air::getAsyncTokenFromOp(theOtherChanOp[0]),
                                  *newWaitAll1);
      erased.insert(theOtherChanOp[0]);
      erased.insert(chanUserOp);
    }
  }

  // STEP 4: Unroll all remaining scf.parallels in segment.
  parOps.clear();
  func.walk([&](scf::ParallelOp parOp) {
    if (parOp->getParentOfType<air::SegmentOp>()) {
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
          rewriter.create<arith::ConstantIndexOp>(forOp->getLoc(), newStep));
      forOp.setUpperBound(rewriter.create<arith::ConstantIndexOp>(
          forOp->getLoc(),
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
    else
      return alloc->emitOpError(
          "Invalid clScope value: expected one of herd/segment/launch");

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
  func::FuncOp funcOp = getOperation();
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  patterns.add<OverrideMemorySpacePattern>(context, clScope, clMemorySpace);
  (void)applyPatternsGreedily(funcOp, std::move(patterns));
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
  }
  (void)applyPatternsGreedily(funcOp, std::move(fixResTypePatterns));
}

} // anonymous namespace

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
