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

// AIRPromoteUniformL1Dma
class AIRPromoteUniformL1Dma
    : public air::impl::AIRPromoteUniformL1DmaBase<AIRPromoteUniformL1Dma> {

public:
  AIRPromoteUniformL1Dma() = default;
  AIRPromoteUniformL1Dma(const AIRPromoteUniformL1Dma &pass){};

  void runOnOperation() override;

private:
};

void do_clone(OpBuilder &builder, Operation *op, IRMapping &mapping) {
  if (!op)
    return;
  for (auto o : op->getOperands()) {
    if (mapping.contains(o))
      continue;
    do_clone(builder, o.getDefiningOp(), mapping);
  }
  builder.clone(*op, mapping);
}

void AIRPromoteUniformL1Dma::runOnOperation() {
  auto module = getOperation();
  // auto ctx = module.getContext();

  std::vector<Operation *> erasedOps;
  int64_t max_id = -1;
  SmallVector<air::DmaMemcpyNdOp, 16> memCopies;
  module.walk([&](air::DmaMemcpyNdOp memcpyOp) {
    memCopies.push_back(memcpyOp);
    IntegerAttr attr = memcpyOp->getAttrOfType<IntegerAttr>("id");
    if (!attr)
      return;
    max_id = std::max(max_id, attr.getInt());
  });

  for (auto memcpyOp : memCopies) {
    auto pipeline = memcpyOp->getParentOfType<air::HerdPipelineOp>();
    auto stage = memcpyOp->getParentOfType<air::PipelineStageOp>();
    auto launch = memcpyOp->getParentOfType<air::HerdOp>();
    if (!pipeline || !stage || !launch)
      continue;

    // auto direction = pipeline->getAttrOfType<StringAttr>("direction");
    auto uniform = stage->getAttrOfType<BoolAttr>("uniform");
    if (!uniform)
      continue;

    auto src_type = memcpyOp.getSrc().getType().cast<MemRefType>();
    auto dst_type = memcpyOp.getDst().getType().cast<MemRefType>();
    auto src_space = src_type.getMemorySpaceAsInt();
    auto dst_space = dst_type.getMemorySpaceAsInt();

    MemRefType ty = nullptr;
    bool to_l1 = (src_space == 0 && dst_space == 2);
    bool from_l1 = (src_space == 2 && dst_space == 0);
    if (to_l1)
      ty = dst_type;
    else if (from_l1)
      ty = src_type;
    else
      continue;

    OpBuilder builder(launch);
    auto loc = memcpyOp->getLoc();
    auto alloc = builder.create<memref::AllocOp>(
        loc, MemRefType::get(ty.getShape(), ty.getElementType(),
                             ty.getLayout().getAffineMap(), 1));
    std::vector<Value> launch_operands;
    IRMapping remap;
    for (unsigned int i = 0; i < launch.getNumKernelOperands(); i++) {
      auto arg = launch.getKernelArguments()[i];
      auto oper = launch.getKernelOperand(i);
      remap.map(arg, oper);
    }
    if (to_l1)
      remap.map(memcpyOp.getDst(), alloc);
    do_clone(builder, memcpyOp.getOperation(), remap);

    launch_operands.insert(launch_operands.begin(),
                           launch->getOperands().begin(),
                           launch->getOperands().end());
    launch_operands.push_back(alloc.getResult());
    launch->setOperands(launch_operands);
    launch.getBody().front().addArgument(alloc.getType(), loc);
    auto sizeAttr = launch->getAttr("operand_segment_sizes")
                        .cast<::mlir::DenseIntElementsAttr>();
    const uint32_t *it = &*sizeAttr.value_begin<uint32_t>();
    auto newAttr = DenseIntElementsAttr::get(sizeAttr.getType(),
                                             {it[0], it[1], it[2], it[3] + 1});
    launch->setAttr("operand_segment_sizes", newAttr);

    builder.setInsertionPoint(memcpyOp);
    SmallVector<Value, 2> opers{};
    SmallVector<Value, 2> mt;
    Value a = launch.getKernelArguments()[it[3]];
    builder.create<air::DmaMemcpyNdOp>(
        loc, SmallVector<Type, 1>{}, mt, to_l1 ? memcpyOp.getDst() : a, mt, mt,
        mt, to_l1 ? a : memcpyOp.getSrc(), mt, mt, mt);
    erasedOps.push_back(memcpyOp);
  }
  for (auto e : erasedOps)
    e->erase();
}

// return true if op is a function of v
bool isFuncOf(Operation *op, Value v, std::vector<Operation *> &ops) {
  bool r = false;
  if (!op)
    return r;

  for (auto o : op->getOperands()) {
    if ((o == v) || (isFuncOf(o.getDefiningOp(), v, ops))) {
      if (std::find(std::begin(ops), std::end(ops), op) == std::end(ops))
        ops.push_back(op);
      r = true;
    }
  }
  return r;
}

// AIRSpecializeDma
class AIRSpecializeDma
    : public air::impl::AIRSpecializeDmaBase<AIRSpecializeDma> {

public:
  AIRSpecializeDma() = default;
  AIRSpecializeDma(const AIRSpecializeDma &pass){};

  void runOnOperation() override;

private:
};

void AIRSpecializeDma::runOnOperation() {
  auto module = getOperation();
  auto ctx = module.getContext();

  module.walk([&](air::HerdOp launch) {
    launch.walk([&](air::DmaMemcpyNdOp memcpyOp) {
      std::vector<Operation *> xOps, yOps;
      bool fn_x = isFuncOf(memcpyOp, launch.getIds()[0], xOps);
      bool fn_y = isFuncOf(memcpyOp, launch.getIds()[1], yOps);
      int64_t herd_size_x = launch.getNumCols();
      int64_t herd_size_y = launch.getNumRows();
      if (fn_x && !fn_y) {
        auto loc = memcpyOp->getLoc();
        OpBuilder builder(memcpyOp);
        auto pipe = builder.create<air::HerdPipelineOp>(loc);
        pipe->setAttr("direction", StringAttr::get(ctx, "horiz"));
        auto pipe_bb = new Block();
        pipe.getBody().push_back(pipe_bb);
        builder.setInsertionPointToEnd(pipe_bb);
        builder.create<air::PipelineTerminatorOp>(loc, SmallVector<Value, 1>{});
        builder.setInsertionPointToStart(pipe_bb);
        for (int x = 0; x < herd_size_x; x++) {
          auto stage = builder.create<air::PipelineStageOp>(
              loc, SmallVector<Type, 1>{}, SmallVector<Value, 1>{});
          stage->setAttr("uniform", BoolAttr::get(ctx, true));
          auto stage_bb = new Block();
          stage.getBody().push_back(stage_bb);
          auto stage_builder = OpBuilder::atBlockEnd(stage_bb);
          auto c_x = stage_builder.create<arith::ConstantIndexOp>(loc, x);
          IRMapping remap;
          remap.map(launch.getIds()[0], c_x);
          for (auto xop : xOps)
            stage_builder.clone(*xop, remap);
          stage_builder.create<air::PipelineYieldOp>(
              loc, SmallVector<Type, 1>{}, SmallVector<Value, 1>{});
        }
        memcpyOp.erase();
      }
      if (fn_y && !fn_x) {
        auto loc = memcpyOp->getLoc();
        OpBuilder builder(memcpyOp);
        auto pipe = builder.create<air::HerdPipelineOp>(loc);
        pipe->setAttr("direction", StringAttr::get(ctx, "vert"));
        auto pipe_bb = new Block();
        pipe.getBody().push_back(pipe_bb);
        builder.setInsertionPointToEnd(pipe_bb);
        builder.create<air::PipelineTerminatorOp>(loc, SmallVector<Value, 1>{});
        builder.setInsertionPointToStart(pipe_bb);
        for (int y = 0; y < herd_size_y; y++) {
          auto stage = builder.create<air::PipelineStageOp>(
              loc, SmallVector<Type, 1>{}, SmallVector<Value, 1>{});
          stage->setAttr("uniform", BoolAttr::get(ctx, true));
          auto stage_bb = new Block();
          stage.getBody().push_back(stage_bb);
          auto stage_builder = OpBuilder::atBlockEnd(stage_bb);
          auto c_y = stage_builder.create<arith::ConstantIndexOp>(loc, y);
          IRMapping remap;
          remap.map(launch.getIds()[1], c_y);
          for (auto yop : yOps)
            stage_builder.clone(*yop, remap);
          stage_builder.create<air::PipelineYieldOp>(
              loc, SmallVector<Type, 1>{}, SmallVector<Value, 1>{});
        }
        memcpyOp.erase();
      }
    });
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
      air::renumberDmaOps(f, "global");
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

    f.walk([&](air::HerdOp launch) {
      launch.walk([&](air::DmaMemcpyNdOp memcpyOp) {
        auto herd_id = launch.getIds();
        OpBuilder builder(memcpyOp);
        auto loc = memcpyOp->getLoc();
        auto broadcast_pattern =
            memcpyOp->getAttrOfType<mlir::IntegerSetAttr>("broadcast_pattern");
        auto ctx = memcpyOp->getContext();
        if (broadcast_pattern) {
          auto is = broadcast_pattern.getValue();
          auto constraints = is.getConstraints();
          auto eqFlags = is.getEqFlags();

          unsigned numSegments = 0;
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
                  loc, air::AsyncTokenType::get(ctx), int_set, int_set_args,
                  (i != numSegments - 1));
              builder.setInsertionPointToStart(aif.getThenBlock());
              auto memcpyOp_cloned = builder.clone(*memcpyOp.getOperation());
              memcpyOp_cloned->removeAttr("broadcast_pattern");
              memcpyOp_cloned->setAttr("broadcast_set",
                                       mlir::IntegerSetAttr::get(int_set));
              SmallVector<Value, 1> yield_token;
              yield_token.push_back(
                  dyn_cast<air::AsyncOpInterface>(memcpyOp_cloned)
                      .getAsyncToken());
              builder.create<affine::AffineYieldOp>(memcpyOp_cloned->getLoc(),
                                                    yield_token);
              if (numSegments != 1) {
                // If more than 1 spatial segments, then move loc to else
                // block
                builder.setInsertionPointToStart(aif.getElseBlock());
              }
              // Reconnect dependency graph using the outermost affine.if's
              // token
              auto async_memcpyOp =
                  dyn_cast<air::AsyncOpInterface>(memcpyOp.getOperation());
              async_memcpyOp.getAsyncToken().replaceAllUsesWith(
                  aif.getResult(0));
            } else if (i < numSegments - 1) {
              affine::AffineIfOp aif = builder.create<affine::AffineIfOp>(
                  builder.getUnknownLoc(), air::AsyncTokenType::get(ctx),
                  int_set, int_set_args, (i != numSegments - 1));
              builder.setInsertionPointToStart(aif.getThenBlock());
              auto memcpyOp_cloned = builder.clone(*memcpyOp.getOperation());
              memcpyOp_cloned->removeAttr("broadcast_pattern");
              memcpyOp_cloned->setAttr("broadcast_set",
                                       mlir::IntegerSetAttr::get(int_set));
              SmallVector<Value, 1> yield_token;
              yield_token.push_back(
                  dyn_cast<air::AsyncOpInterface>(memcpyOp_cloned)
                      .getAsyncToken());
              builder.create<affine::AffineYieldOp>(memcpyOp_cloned->getLoc(),
                                                    yield_token);
              builder.setInsertionPointAfter(aif);
              SmallVector<Value, 1> parent_block_yield_token = {
                  aif.getResult(0)};
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
                  dyn_cast<air::AsyncOpInterface>(memcpyOp_cloned)
                      .getAsyncToken());
              builder.create<affine::AffineYieldOp>(memcpyOp_cloned->getLoc(),
                                                    yield_token);
            }
          }
          memcpyOp.erase();
        }
      });
    });
  }

  void simplifyDmaIndicesWithAffineSet(func::FuncOp f) {

    f.walk([&](air::DmaMemcpyNdOp memcpyOp) {
      auto ctx = memcpyOp->getContext();
      if (auto broadcast_set =
              memcpyOp->getAttrOfType<mlir::IntegerSetAttr>("broadcast_set")) {
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
        for (auto &elem : loop_dep_history) {
          for (auto v : std::get<1>(elem)) {
            if (auto hl_op = air::getHerdArgOwner(v)) {
              for (unsigned j = 0; j < current_shape_expr.size(); j++) {
                if (v == hl_op.getIds()[j]) {
                  for (unsigned i = 0; i < constraints.size(); i++) {
                    auto c = constraints[i];
                    if (c.isFunctionOfSymbol(j) && eqFlags[i]) {
                      auto eval = evaluateSymbolEqualityInSet(c, ctx);
                      current_shape_expr[j] = getAffineConstantExpr(eval, ctx);
                      op_history.insert(op_history.end(),
                                        std::get<2>(elem).begin(),
                                        std::get<2>(elem).end());
                    }
                  }
                }
              }
            }
          }
        }

        // Evaluate broadcast pattern by propagating expr through scalar
        // operations in op history, last-in-first-out
        for (std::vector<Operation *>::reverse_iterator i = op_history.rbegin();
             i != op_history.rend(); ++i) {
          if (auto exec_op = dyn_cast<air::ExecuteOp>(*i)) {
            Operation *op = exec_op.getChildOp();
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
              propagateAFfineConstantExprThroughArithOp<arith::AddIOp>(
                  arith_op, current_shape_expr, memcpyOp.getOperation(), ctx);
            } else if (auto arith_op = dyn_cast<arith::MulIOp>(op)) {
              propagateAFfineConstantExprThroughArithOp<arith::MulIOp>(
                  arith_op, current_shape_expr, memcpyOp.getOperation(), ctx);
            }
          }
        }

        // Replace memcpyOp's dependent operand with const
        auto newMemcpyOp =
            replaceMemcpyOpWithSimplifiedOperands(memcpyOp, current_shape_expr);
        auto asyncMemcpyOp =
            dyn_cast<air::AsyncOpInterface>(memcpyOp.getOperation());
        auto asyncNewMemcpyOp = dyn_cast<air::AsyncOpInterface>(newMemcpyOp);
        newMemcpyOp->setAttr("broadcast_set", broadcast_set);
        asyncMemcpyOp.getAsyncToken().replaceAllUsesWith(
            asyncNewMemcpyOp.getAsyncToken());
        memcpyOp->erase();
      }
    });
  }

  // Evaluate the integer value of affine set expression if the only symbolic
  // identifier is replaced with zero
  int evaluateSymbolEqualityInSet(AffineExpr c, MLIRContext *ctx) {
    assert(c.isSymbolicOrConstant() && "constraint has dimension identifier");
    SmallVector<AffineExpr, 2> zero_syms{
        getAffineConstantExpr(0, ctx),
        getAffineConstantExpr(0, ctx),
    };
    auto newC = c.replaceSymbols(zero_syms);
    auto expr = dyn_cast<AffineConstantExpr>(simplifyAffineExpr(newC, 0, 1));
    assert(expr);
    int result = expr.getValue();
    // Both + and - constant eval are legal for AffineExpr
    return (result >= 0) ? (result) : (-result);
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
    assert(dyn_cast<AffineConstantExpr>(c) && "non-constant affine expression");
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
    assert(dyn_cast<AffineConstantExpr>(c) && "non-constant affine expression");
    mul *= dyn_cast<AffineConstantExpr>(c).getValue();
    c = getAffineConstantExpr(mul, ctx);
  }

  // Propagate AffineConstantExpr through arith addi/muli op
  template <typename T>
  void propagateAFfineConstantExprThroughArithOp(
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
  Operation *replaceMemcpyOpWithSimplifiedOperands(
      air::DmaMemcpyNdOp &memcpyOp,
      SmallVector<AffineExpr, 2> current_shape_expr) {
    OpBuilder builder(memcpyOp);
    builder.setInsertionPoint(memcpyOp);
    auto loc = memcpyOp->getLoc();
    SmallVector<Value, 1> srcMemrefDimsOrOffsets;
    if (auto memcpyNdOp =
            dyn_cast<air::DmaMemcpyNdOp>(memcpyOp.getOperation())) {
      for (unsigned i = 0; i < current_shape_expr.size(); i++) {
        if (current_shape_expr[i]) {
          auto val =
              dyn_cast<AffineConstantExpr>(current_shape_expr[i]).getValue();
          auto cop = builder.create<arith::ConstantIndexOp>(loc, val);
          srcMemrefDimsOrOffsets.push_back(cop);
        } else {
          // Offset taking into account mismatch between current_shape_expr size
          // and offset list size.
          int md_dma_offset =
              memcpyNdOp.getSrcOffsets().size() - current_shape_expr.size();
          srcMemrefDimsOrOffsets.push_back(
              memcpyNdOp.getSrcOffsets()[md_dma_offset + i]);
        }
      }
      // Replace memcpyOp
      return replaceMemcpyOp(memcpyNdOp, builder, srcMemrefDimsOrOffsets);
    } else {
      assert(false && "Unhandled DmaMemcpyNdOp");
      return nullptr;
    }
  }

  // Replace DmaMemcpyNdOp with updated src operands
  Operation *replaceMemcpyOp(air::DmaMemcpyNdOp op, OpBuilder &builder,
                             SmallVector<Value, 1> srcMemrefDimsOrOffsets) {
    auto loc = op->getLoc();
    // Fill higher dims with zeros if offset rank is lower than size rank.
    while (srcMemrefDimsOrOffsets.size() < op.getSrcSizes().size()) {
      srcMemrefDimsOrOffsets.insert(
          srcMemrefDimsOrOffsets.begin(),
          builder.create<arith::ConstantIndexOp>(loc, 0));
    }
    air::DmaMemcpyNdOp newMemcpyOp = builder.create<air::DmaMemcpyNdOp>(
        loc, air::AsyncTokenType::get(op->getContext()),
        op.getAsyncDependencies(), op.getDstMemref(), op.getDstOffsets(),
        op.getDstSizes(), op.getDstStrides(), op.getSrcMemref(),
        srcMemrefDimsOrOffsets, op.getSrcSizes(), op.getSrcStrides());
    return newMemcpyOp.getOperation();
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
    if (v.getDefiningOp() && isa<arith::ConstantOp>(v.getDefiningOp()))
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
  b.create<air::HerdTerminatorOp>(parOp.getLoc());

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
  air::renumberDmaOps(func, clMode);
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
  (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
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
    h.getIds()[1].cast<Value>().replaceAllUsesExcept(iv_1,
                                                     iv_1.getDefiningOp());

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

class AIRSplitL2MemrefForBufferConstraintPass
    : public air::impl::AIRSplitL2MemrefForBufferConstraintPassBase<
          AIRSplitL2MemrefForBufferConstraintPass> {

public:
  AIRSplitL2MemrefForBufferConstraintPass() = default;
  AIRSplitL2MemrefForBufferConstraintPass(
      const AIRSplitL2MemrefForBufferConstraintPass &pass){};

  void runOnOperation() override;

private:
  void partitionMemref(SmallVector<air::ChannelPutOp> &puts,
                       SmallVector<air::ChannelGetOp> &gets, int dim,
                       std::string splitType);
  SmallVector<memref::AllocOp>
  getTargetMemrefAllocs(func::FuncOp func,
                        std::map<memref::AllocOp, SmallVector<int>>
                            &targetMemrefsToColTilingFactors);
};

template <typename T> void push_back_if_unique(SmallVector<T> &vec, T entry) {
  if (std::find(vec.begin(), vec.end(), entry) == vec.end()) {
    vec.push_back(entry);
  }
}

// Find GCD of a vector of ints.
int findGCD(SmallVector<int> vec) {
  int result = vec[0];
  for (unsigned i = 1; i < vec.size(); i++) {
    result = std::gcd(vec[i], result);

    if (result == 1) {
      return 1;
    }
  }
  return result;
}

// Check if an air.channel is single-consumer-single-producer.
bool hasSinglePutAndGet(air::ChannelOp chan) {
  auto puts =
      getChannelPutOpThroughSymbol(chan, chan->getParentOfType<ModuleOp>());
  auto gets =
      getChannelGetOpThroughSymbol(chan, chan->getParentOfType<ModuleOp>());
  return puts.size() == 1 && gets.size() == 1;
}

// Tile air.channel put/get wrt a memref.
Value tileChannelOpByFactor(air::ChannelInterface originalChanOp, int factor,
                            int originalMemrefSize, int dim,
                            air::ChannelOp newChanOp, Location loc,
                            MLIRContext *ctx) {
  OpBuilder builder(originalChanOp);
  SmallVector<Value> originalApplyOperands;
  Operation *affineApplyOp = nullptr;
  if (!originalChanOp.getOffsets().empty())
    affineApplyOp = originalChanOp.getOffsets()[dim].getDefiningOp();
  if (affineApplyOp && isa<affine::AffineApplyOp>(affineApplyOp))
    originalApplyOperands = affineApplyOp->getOperands();
  else if (affineApplyOp && isa<air::ExecuteOp>(affineApplyOp)) {
    auto execOp = dyn_cast<air::ExecuteOp>(affineApplyOp);
    originalApplyOperands = execOp.getChildOp()->getOperands();
  } else
    originalApplyOperands.push_back(
        builder.create<arith::ConstantIndexOp>(loc, 0));
  SmallVector<Value> tokens;
  for (int i = 0; i < factor; i++) {
    SmallVector<Value> newIndices{
        builder.create<arith::ConstantIndexOp>(loc, i),
        builder.create<arith::ConstantIndexOp>(loc, 0)};
    // Update y offset.
    // Create affine.apply on induction variable.
    auto checkpoint = builder.saveInsertionPoint();
    if (affineApplyOp)
      builder.setInsertionPoint(affineApplyOp);
    AffineExpr s0 = builder.getAffineSymbolExpr(0);
    AffineExpr mul = s0 * originalMemrefSize;
    AffineExpr add = mul + i * mlir::ceilDiv(originalMemrefSize, factor);
    auto map = AffineMap::get(0, 1, add);
    auto newApplyOp =
        builder.create<affine::AffineApplyOp>(loc, map, originalApplyOperands);
    if (affineApplyOp)
      builder.restoreInsertionPoint(checkpoint);
    SmallVector<Value> newOffsets = originalChanOp.getOffsets();
    SmallVector<Value> newWraps = originalChanOp.getSizes();
    SmallVector<Value> newStrides = originalChanOp.getStrides();
    if (newOffsets.empty() && newWraps.empty())
      air::populateDefaultWrapsAndStrides(builder, originalChanOp.getMemref(),
                                          newOffsets, newWraps, newStrides);
    newOffsets[dim] = newApplyOp.getResult();
    newWraps[dim] = builder.create<arith::ConstantIndexOp>(
        loc, mlir::ceilDiv(originalMemrefSize, factor));
    auto deps = dyn_cast<air::AsyncOpInterface>(originalChanOp.getOperation())
                    .getAsyncDependencies();
    SmallVector<Type, 4> tys = {air::AsyncTokenType::get(ctx)};
    if (isa<air::ChannelGetOp>(originalChanOp)) {
      auto newGetOp = builder.create<air::ChannelGetOp>(
          loc, tys, deps, newChanOp.getSymName(), newIndices,
          originalChanOp.getMemref(), newOffsets, newWraps, newStrides);
      newGetOp->setAttr("id",
                        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                               originalChanOp.getId()));
      tokens.push_back(newGetOp.getAsyncToken());
    } else {
      auto newPutOp = builder.create<air::ChannelPutOp>(
          loc, tys, deps, newChanOp.getSymName(), newIndices,
          originalChanOp.getMemref(), newOffsets, newWraps, newStrides);
      newPutOp->setAttr("id",
                        mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                               originalChanOp.getId()));
      tokens.push_back(newPutOp.getAsyncToken());
    }
  }
  auto newWaitAll = builder.create<air::WaitAllOp>(
      loc, air::AsyncTokenType::get(ctx), tokens);
  return newWaitAll.getAsyncToken();
}

std::optional<int> getFirstConstantOffsetValue(SmallVector<Value> offsets,
                                               int memrefRank,
                                               int &initialDim) {
  int offsetDim = (int)offsets.size() >= memrefRank
                      ? offsets.size() - memrefRank + initialDim
                      : 0;
  auto offset = getConstantIntValue(offsets[offsetDim]);
  // Find the first constant offset to use as key for memref splitting.
  while (!offset && offsetDim < (int)offsets.size()) {
    offset = getConstantIntValue(offsets[++offsetDim]);
    initialDim++;
  }
  return offset;
}

int getFirstConstantOffsetValueIndex(SmallVector<Value> offsets, int memrefRank,
                                     int initialDim = 0) {
  int offsetDim = (int)offsets.size() >= memrefRank
                      ? offsets.size() - memrefRank + initialDim
                      : 0;
  auto offset = getConstantIntValue(offsets[offsetDim]);
  // Find the first constant offset to use as key for memref splitting.
  while (!offset && offsetDim < (int)offsets.size()) {
    offset = getConstantIntValue(offsets[++offsetDim]);
  }
  return offsetDim;
}

// Partition L2 memref.
void AIRSplitL2MemrefForBufferConstraintPass::partitionMemref(
    SmallVector<air::ChannelPutOp> &puts, SmallVector<air::ChannelGetOp> &gets,
    int dim, std::string splitType = "") {
  // dim // MM2SChannels
  auto memref = puts.front().getMemref();
  MemRefType ty = memref.getType().cast<MemRefType>();
  auto allocOp = memref.getDefiningOp();
  auto loc = allocOp->getLoc();
  Operation *deallocOp = nullptr;
  for (auto user : memref.getUsers()) {
    if (auto execOp = dyn_cast<air::ExecuteOp>(user->getParentOp())) {
      if (isa<memref::DeallocOp>(execOp.getChildOp())) {
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
  for (auto op : puts) {
    auto offset = getFirstConstantOffsetValue(
        op.getOffsets(), air::getTensorShape(ty).size(), dim);
    assert(offset);
    push_back_if_unique<int>(keys, *offset);
    if (!chanOpPartitions.count(*offset))
      chanOpPartitions[*offset] = SmallVector<air::ChannelInterface>{op};
    else
      chanOpPartitions[*offset].push_back(op);
  }
  for (auto op : gets) {
    auto offset = getFirstConstantOffsetValue(
        op.getOffsets(), air::getTensorShape(ty).size(), dim);
    assert(offset);
    push_back_if_unique<int>(keys, *offset);
    if (!chanOpPartitions.count(*offset))
      chanOpPartitions[*offset] = SmallVector<air::ChannelInterface>{op};
    else
      chanOpPartitions[*offset].push_back(op);
  }
  OpBuilder builder(allocOp);
  SmallVector<scf::ForOp> mutatedScfForOps;
  for (auto key : keys) {
    SmallVector<int64_t> newMemrefShape;
    for (unsigned i = 0; i < air::getTensorShape(ty).size(); i++) {
      newMemrefShape.push_back(air::getTensorShape(ty)[i]);
    }
    for (auto op : chanOpPartitions[key]) {
      int offsetDim =
          op.getOffsets().size() >= air::getTensorShape(ty).size()
              ? op.getOffsets().size() - air::getTensorShape(ty).size() + dim
              : 0;
      if (op.getSizes().size() == newMemrefShape.size()) {
        newMemrefShape[dim] = *getConstantIntValue(op.getSizes()[offsetDim]);
        break;
      }
    }

    auto newMemrefType = MemRefType::get(newMemrefShape, ty.getElementType(),
                                         ty.getLayout().getAffineMap(),
                                         ty.getMemorySpaceAsInt());
    Value newMemref = nullptr;
    // Create new alloc ops.
    if (isa<air::ExecuteOp>(allocOp)) {
      auto execOp = builder.create<air::ExecuteOp>(
          loc, air::AsyncTokenType::get(allocOp->getContext()), newMemrefType,
          SmallVector<Value>{});
      Block *async_bb = builder.createBlock(&execOp.getBody());
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
            loc, air::AsyncTokenType::get(deallocOp->getContext()),
            execDeallocOp.getAsyncDependencies());
        Block *async_bb = builder.createBlock(&execOp.getBody());
        builder.setInsertionPointToStart(async_bb);
        builder.create<memref::DeallocOp>(loc, newMemref);
        builder.create<xilinx::air::ExecuteTerminatorOp>(loc);
      } else
        builder.create<memref::DeallocOp>(loc, newMemref);
      builder.setInsertionPoint(newMemref.getDefiningOp());
    }
    // Mutate air.channel.put/get opoperands.
    for (auto op : chanOpPartitions[key]) {
      int memrefOperandOffset =
          dyn_cast<air::AsyncOpInterface>(op.getOperation())
              .getAsyncDependencies()
              .size() +
          op.getIndices().size();
      auto &memrefOpOper = op->getOpOperand(memrefOperandOffset);
      memrefOpOper.assign(newMemref);
      int offsetDim = getFirstConstantOffsetValueIndex(
          op.getOffsets(), air::getTensorShape(ty).size(), dim);
      int offsetOperandOffset = memrefOperandOffset + offsetDim + 1;
      auto &offsetOpOper = op->getOpOperand(offsetOperandOffset);
      offsetOpOper.assign(builder.create<arith::ConstantIndexOp>(loc, 0));
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
    depTracer.traceDependencyFromScfForOp(mutatedScfForOp);
  }
  if (deallocOp)
    deallocOp->erase();
}

SmallVector<memref::AllocOp>
AIRSplitL2MemrefForBufferConstraintPass::getTargetMemrefAllocs(
    func::FuncOp func, std::map<memref::AllocOp, SmallVector<int>>
                           &targetMemrefsToColTilingFactors) {
  auto ctx = func.getContext();
  SmallVector<memref::AllocOp> allocOps;
  func.walk([&](memref::AllocOp allocOp) {
    if (allocOp->getParentOfType<air::SegmentOp>() &&
        allocOp.getMemref()
                .getType()
                .cast<MemRefType>()
                .getMemorySpaceAsInt() == (int)air::MemorySpace::L2) {
      allocOps.push_back(allocOp);
    }
  });

  // Filter out L2 memrefs who shall not fit in memtile due to hw buffer
  // constraints.
  SmallVector<memref::AllocOp> targetMemrefs;
  // Map between the target memref alloc ops and all column-wise tiling factors
  // per alloc.
  for (auto allocOp : allocOps) {
    Value memref = allocOp.getMemref();
    if (auto exec = dyn_cast<air::ExecuteOp>(allocOp->getParentOp()))
      memref = exec->getResult(1);
    SmallVector<std::string> MM2SChannels;
    SmallVector<std::string> S2MMChannels;
    for (auto user : memref.getUsers()) {
      if (isa<air::ChannelInterface>(user) &&
          isa<scf::ParallelOp>(user->getParentOp())) {
        SmallVector<int, 2> lbs_spatial;
        SmallVector<int, 2> ubs_spatial;
        air::getSizesFromSpatialLoop(user->getParentOp(), lbs_spatial,
                                     ubs_spatial);

        if (!targetMemrefsToColTilingFactors.count(allocOp)) {
          targetMemrefsToColTilingFactors[allocOp] = SmallVector<int>{};
          targetMemrefs.push_back(allocOp);
          allocOp->setAttr("split", BoolAttr::get(ctx, true));
          allocOp->setAttr("split_type", StringAttr::get(ctx, "scf.parallel"));
        }
        for (unsigned i = 0; i < ubs_spatial.size(); i++) {
          targetMemrefsToColTilingFactors[allocOp].push_back(
              ubs_spatial[i] - lbs_spatial[i] + 1);
        }
      } else if (auto put = dyn_cast<air::ChannelPutOp>(user)) {
        push_back_if_unique<std::string>(MM2SChannels, put.getChanName().str());
      } else if (auto get = dyn_cast<air::ChannelGetOp>(user)) {
        push_back_if_unique<std::string>(S2MMChannels, get.getChanName().str());
      }
    }
    if (!targetMemrefsToColTilingFactors.count(allocOp)) {
      targetMemrefsToColTilingFactors[allocOp] = SmallVector<int>{};
      targetMemrefs.push_back(allocOp);
      allocOp->setAttr("split", BoolAttr::get(func.getContext(), true));
    }
    if (MM2SChannels.size() > 1) {
      targetMemrefsToColTilingFactors[allocOp].push_back(MM2SChannels.size());
      allocOp->setAttr("split_type", StringAttr::get(ctx, "MM2SChannels"));
    }
    if (S2MMChannels.size() > 1) {
      targetMemrefsToColTilingFactors[allocOp].push_back(S2MMChannels.size());
      allocOp->setAttr("split_type", StringAttr::get(ctx, "S2MMChannels"));
    }
  }
  return targetMemrefs;
}

// Check if each L2 memref shall violate buffer hardware constraint, and if so,
// attempt to split it (in columns, per IPU device layout).
void AIRSplitL2MemrefForBufferConstraintPass::runOnOperation() {
  SmallVector<air::HerdOp> herds;
  auto func = getOperation();
  auto ctx = &getContext();
  SmallVector<memref::AllocOp> allocOps;
  func.walk([&](memref::AllocOp allocOp) {
    if (allocOp->getParentOfType<air::SegmentOp>() &&
        allocOp.getMemref()
                .getType()
                .cast<MemRefType>()
                .getMemorySpaceAsInt() == (int)air::MemorySpace::L2) {
      allocOps.push_back(allocOp);
    }
  });

  // Map between the target memref alloc ops and all column-wise tiling factors
  // per alloc.
  std::map<memref::AllocOp, SmallVector<int>> targetMemrefsToColTilingFactors;

  SmallVector<memref::AllocOp> targetMemrefs =
      getTargetMemrefAllocs(func, targetMemrefsToColTilingFactors);
  if (targetMemrefs.empty())
    return;

  // Tile memrefs.
  for (auto allocOp : targetMemrefs) {
    int targetColTilingFactor =
        findGCD(targetMemrefsToColTilingFactors[allocOp]);
    allocOp->setAttr("split",
                     mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                            targetColTilingFactor));
    auto splitTypeAttr = allocOp->getAttrOfType<StringAttr>("split_type");
    Value memref = isa<air::ExecuteOp>(allocOp->getParentOp())
                       ? allocOp->getParentOp()->getResult(1)
                       : dyn_cast<Value>(allocOp.getMemref());
    for (auto user : memref.getUsers()) {
      if (auto chanUserOp = dyn_cast<air::ChannelInterface>(user)) {
        auto chanUserChannelDeclr =
            air::getChannelDeclarationThroughSymbol(chanUserOp);
        if (!hasSinglePutAndGet(chanUserChannelDeclr)) {
          assert(false && "NYI");
        } else if (auto par = dyn_cast<scf::ParallelOp>(user->getParentOp())) {
          // Case 1: Parallel access to the memref represented with scf.parallel
          // op. Data access specialization method: unroll the scf.parallel
          // loop.
          SmallVector<int, 2> lbs_spatial, ubs_spatial;
          air::getSizesFromSpatialLoop(user->getParentOp(), lbs_spatial,
                                       ubs_spatial);
          // TODO: currently hardcoded tiling dimension to be the last
          // dimension.
          if (ubs_spatial.back() - lbs_spatial.back() + 1 <
              targetColTilingFactor) {
            // Tile the air.channel op by targetColTilingFactor. NYI.
            assert(false && "NYI");
          }
          OpBuilder builder(user->getParentOp());
          IRMapping remap;
          (void)air::unrollAIRChannelPutGetInScfParallel(builder, par, user,
                                                         remap);
          par.erase();
        } else if ((isa<air::ChannelPutOp>(user) &&
                    splitTypeAttr.str() == "MM2SChannels") ||
                   (isa<air::ChannelGetOp>(user) &&
                    splitTypeAttr.str() == "S2MMChannels")) {
          // Case 2: Parallel access to the memref represented with multiple
          // air.channel put/gets. Data access specialization method:
          // specializing memref wrt each unique air.channel access. To be
          // handled below.
        } else {
          // Case 3: A single put/get op with default data access pattern
          // (contiguous, row major) spanning the entire memref. Data access
          // specialization method: tiling the air.channel op by
          // targetColTilingFactor.
          auto loc = chanUserOp->getLoc();
          auto ctx = chanUserOp->getContext();
          OpBuilder builder(chanUserOp);
          builder.setInsertionPointToStart(
              chanUserOp->getParentOfType<ModuleOp>().getBody());
          SmallVector<Type, 4> tys = {
              air::AsyncTokenType::get(chanUserOp->getContext())};
          auto cname =
              air::createChannelName(chanUserOp->getParentOfType<ModuleOp>());
          SmallVector<int64_t, 2> channel_sizes = {targetColTilingFactor, 1};
          auto new_chan = builder.create<air::ChannelOp>(
              loc, cname, builder.getI64ArrayAttr(channel_sizes));
          auto memrefShape = air::getTensorShape(memref.getType());

          int dim = 0;
          for (unsigned i = 0; i < memrefShape.size(); i++) {
            if (chanUserOp.getOffsets().empty())
              break;
            int offsetDim =
                chanUserOp.getOffsets().size() - memrefShape.size() + i;
            if (getConstantIntValue(chanUserOp.getOffsets()[offsetDim])) {
              dim = i;
              break;
            }
          }
          auto newWaitAll =
              tileChannelOpByFactor(chanUserOp, targetColTilingFactor,
                                    memrefShape[dim], dim, new_chan, loc, ctx);

          // Update async dependency.
          auto old_token =
              dyn_cast<air::AsyncOpInterface>(chanUserOp.getOperation())
                  .getAsyncToken();
          old_token.replaceAllUsesWith(newWaitAll);

          // Update the other channel op of the chanUserChannelDeclr.
          auto theOtherChanOp =
              air::getTheOtherChannelOpThroughSymbol(chanUserOp);
          Value newWaitAll1 =
              tileChannelOpByFactor(theOtherChanOp[0], targetColTilingFactor,
                                    memrefShape[dim], dim, new_chan, loc, ctx);

          // Update dependency.
          auto oldToken =
              dyn_cast<air::AsyncOpInterface>(theOtherChanOp[0].getOperation())
                  .getAsyncToken();
          oldToken.replaceAllUsesWith(newWaitAll1);
          theOtherChanOp[0].erase();
          chanUserOp.erase();
        }
      }
    }
  }

  auto context = &getContext();
  RewritePatternSet canoPatterns(context);
  // Fold constants.
  mlir::arith::ConstantIndexOp::getCanonicalizationPatterns(canoPatterns,
                                                            context);
  (void)applyPatternsAndFoldGreedily(func, std::move(canoPatterns));

  // Split memrefs.
  allocOps.clear();
  func.walk([&](memref::AllocOp allocOp) {
    if (allocOp->hasAttr("split")) {
      allocOps.push_back(allocOp);
    }
  });
  for (auto allocOp : allocOps) {
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
    partitionMemref(puts, gets, 0,
                    allocOp->getAttrOfType<StringAttr>("split_type").str());
  }
  for (auto allocOp : allocOps) {
    if (auto execOp = dyn_cast<air::ExecuteOp>(allocOp->getParentOp())) {
      OpBuilder builder(execOp);
      auto waitAllOp = builder.create<air::WaitAllOp>(
          allocOp->getLoc(), air::AsyncTokenType::get(allocOp->getContext()),
          execOp.getAsyncDependencies());
      execOp.getAsyncToken().replaceAllUsesWith(waitAllOp.getAsyncToken());
      execOp->erase();
    } else
      allocOp->erase();
  }

  air::renumberChannelOps(&func.getBody().front());
}

} // anonymous namespace

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRExamplePass() {
  return std::make_unique<AIRExamplePass>();
}

std::unique_ptr<Pass> createAIRSpecializeDma() {
  return std::make_unique<AIRSpecializeDma>();
}

std::unique_ptr<Pass> createAIRSpecializeDmaBroadcast() {
  return std::make_unique<AIRSpecializeDmaBroadcast>();
}

std::unique_ptr<Pass> createAIRPromoteUniformL1Dma() {
  return std::make_unique<AIRPromoteUniformL1Dma>();
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

} // namespace air
} // namespace xilinx
