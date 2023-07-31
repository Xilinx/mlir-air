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
#include "PassDetail.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Transform/AIRTilingUtils.h"
#include "air/Util/Dependency.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

class AIRExamplePass : public air::AIRExamplePassBase<AIRExamplePass> {

public:
  AIRExamplePass() = default;
  AIRExamplePass(const AIRExamplePass &pass){};

  void runOnOperation() override;

private:
};

void AIRExamplePass::runOnOperation() {}

class AIRLinalgNamePass : public air::AIRLinalgNamePassBase<AIRLinalgNamePass> {

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
    : public air::AIRRemoveLinalgNamePassBase<AIRRemoveLinalgNamePass> {

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
    : public air::AIRPromoteUniformL1DmaBase<AIRPromoteUniformL1Dma> {

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
class AIRSpecializeDma : public air::AIRSpecializeDmaBase<AIRSpecializeDma> {

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
    : public air::AIRSpecializeDmaBroadcastBase<AIRSpecializeDmaBroadcast> {

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
                  simplifyAffineExpr(newC, 0, 1).dyn_cast<AffineConstantExpr>();
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
              AffineIfOp aif = builder.create<AffineIfOp>(
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
              builder.create<AffineYieldOp>(memcpyOp_cloned->getLoc(),
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
              AffineIfOp aif = builder.create<AffineIfOp>(
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
              builder.create<AffineYieldOp>(memcpyOp_cloned->getLoc(),
                                            yield_token);
              builder.setInsertionPointAfter(aif);
              SmallVector<Value, 1> parent_block_yield_token = {
                  aif.getResult(0)};
              builder.create<AffineYieldOp>(builder.getUnknownLoc(),
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
              builder.create<AffineYieldOp>(memcpyOp_cloned->getLoc(),
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
        SmallVector<Value, 1> loop_dep_history;
        std::vector<Operation *> op_history;
        traceDependentInductionVar(memcpyOp, loop_dep_history, op_history);

        // Walk constraints in broadcast pattern, and get shape of the broadcast
        // pattern
        auto is = broadcast_set.getValue();
        auto constraints = is.getConstraints();
        auto eqFlags = is.getEqFlags();

        // Check which dimension op operates on; initialize current_shape_expr
        SmallVector<AffineExpr, 2> current_shape_expr = {nullptr, nullptr};
        for (auto v : loop_dep_history) {
          if (auto hl_op = air::getHerdArgOwner(v)) {
            for (unsigned j = 0; j < current_shape_expr.size(); j++) {
              if (v == hl_op.getIds()[j]) {
                for (unsigned i = 0; i < constraints.size(); i++) {
                  auto c = constraints[i];
                  if (c.isFunctionOfSymbol(j) && eqFlags[i]) {
                    auto eval = evaluateSymbolEqualityInSet(c, ctx);
                    current_shape_expr[j] = getAffineConstantExpr(eval, ctx);
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
          if (auto air_region_op = dyn_cast<air::ExecuteOp>(*i)) {
            assert(air_region_op.getBody().front().getOperations().size() ==
                       2 &&
                   "air::ExecuteOp should have only one child operation beside "
                   "the terminator");
            // Get current scalar op
            Operation *op = nullptr;
            for (auto &child_op :
                 air_region_op.getBody().front().getOperations()) {
              if (!dyn_cast<air::ExecuteTerminatorOp>(child_op))
                op = &child_op;
            }
            // If the async op is affine.apply
            if (auto apply_op = dyn_cast<AffineApplyOp>(op)) {
              auto map = apply_op.getAffineMap();
              for (unsigned j = 0; j < current_shape_expr.size(); j++) {
                if (current_shape_expr[j]) {
                  replaceSymbolAndEvaluateConstantInMap(
                      map, current_shape_expr[j], ctx);
                  // Remove dependence from scalar op to memcpyOp if present
                  auto async_memcpyOp =
                      dyn_cast<air::AsyncOpInterface>(memcpyOp.getOperation());
                  eraseAsyncDependencyFromAsyncOp(
                      async_memcpyOp, air_region_op.getAsyncToken());
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
  int evaluateSymbolEqualityInSet(mlir::AffineExpr c, MLIRContext *ctx) {
    assert(c.isSymbolicOrConstant() && "constraint has dimension identifier");
    SmallVector<AffineExpr, 2> zero_syms{
        getAffineConstantExpr(0, ctx),
        getAffineConstantExpr(0, ctx),
    };
    auto newC = c.replaceSymbols(zero_syms);
    auto expr = simplifyAffineExpr(newC, 0, 1).dyn_cast<AffineConstantExpr>();
    assert(expr);
    int result = expr.getValue();
    // Both + and - constant eval are legal for AffineExpr
    return (result >= 0) ? (result) : (-result);
  }

  // Evaluate the affine expression of affine map if the only symbolic
  // identifier is replaced with zero
  void replaceSymbolAndEvaluateConstantInMap(mlir::AffineMap map,
                                             mlir::AffineExpr &c,
                                             MLIRContext *ctx) {
    auto newmap = map.replace(getAffineSymbolExpr(0, ctx), c, 0, 1);
    auto const_int = simplifyAffineMap(newmap).getSingleConstantResult();
    c = getAffineConstantExpr(const_int, ctx);
  }

  // AddI for AffineConstantExpr
  void applyArithOpToAffineConstantExpr(arith::AddIOp arith_op,
                                        mlir::AffineExpr &c, MLIRContext *ctx) {
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
    assert(c.dyn_cast<mlir::AffineConstantExpr>() &&
           "non-constant affine expression");
    acc += c.dyn_cast<mlir::AffineConstantExpr>().getValue();
    c = getAffineConstantExpr(acc, ctx);
  }

  // MulI for AffineConstantExpr
  void applyArithOpToAffineConstantExpr(arith::MulIOp arith_op,
                                        mlir::AffineExpr &c, MLIRContext *ctx) {
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
    assert(c.dyn_cast<mlir::AffineConstantExpr>() &&
           "non-constant affine expression");
    mul *= c.dyn_cast<mlir::AffineConstantExpr>().getValue();
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
          auto val = current_shape_expr[i]
                         .template dyn_cast<AffineConstantExpr>()
                         .getValue();
          auto cop = builder.create<arith::ConstantIndexOp>(loc, val);
          srcMemrefDimsOrOffsets.push_back(cop);
        } else {
          srcMemrefDimsOrOffsets.push_back(memcpyNdOp.getSrcOffsets()[i]);
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
    air::DmaMemcpyNdOp newMemcpyOp = builder.create<air::DmaMemcpyNdOp>(
        loc, air::AsyncTokenType::get(op->getContext()),
        op.getAsyncDependencies(), op.getDstMemref(), op.getDstOffsets(),
        op.getDstSizes(), op.getDstStrides(), op.getSrcMemref(),
        srcMemrefDimsOrOffsets, op.getSrcSizes(), op.getSrcStrides());
    return newMemcpyOp.getOperation();
  }
};

class AIRFuseParallelHerdPass
    : public air::AIRFuseParallelHerdPassBase<AIRFuseParallelHerdPass> {

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
    : public air::AIRRenumberDmaIdPassBase<AIRRenumberDmaIdPass> {

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
      if (!isa<scf::YieldOp>(o))
        rewriter.clone(o, remap);

    rewriter.eraseOp(op);
    return success();
  }
};

class AIRLowerHerdParallelPass
    : public air::AIRLowerHerdParallelPassBase<AIRLowerHerdParallelPass> {

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

class HoistScfChannelGetPutPass : public RewritePattern {
public:
  HoistScfChannelGetPutPass(MLIRContext *context, PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto chanOp = dyn_cast<air::ChannelInterface>(op);
    if (!chanOp)
      return failure();
    MemRefType memrefTy = chanOp.getMemref().getType().cast<MemRefType>();

    // channel op must be in an scf.for loop
    auto forOp = op->getParentOfType<scf::ForOp>();
    if (!forOp)
      return failure();

    // channel op must be the only op in the loop
    if (forOp.getBody()->getOperations().size() != 2)
      return failure();

    // scf.for must have constant bounds and step
    auto lb = forOp.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto ub = forOp.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    auto step = forOp.getStep().getDefiningOp<arith::ConstantIndexOp>();
    if (!lb || !ub || !step)
      return failure();

    auto trip_count = (ub.value() - lb.value()) / step.value();

    rewriter.setInsertionPoint(forOp);
    auto loc = op->getLoc();

    // look for an offset that is a loop iv
    SmallVector<Value> offsets;
    Value iv = nullptr;
    int ivOffsetIdx = 0;
    for (auto v : chanOp.getOffsets()) {
      if (v == forOp.getInductionVar()) {
        if (iv) // there can be only one
          return failure();
        iv = v;
        v = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      }
      offsets.push_back(v);
      if (!iv)
        ivOffsetIdx++;
    }

    SmallVector<Value> sizes;
    sizes.push_back(rewriter.create<arith::ConstantIndexOp>(loc, trip_count));
    for (auto v : chanOp.getSizes()) {
      auto c = dyn_cast<arith::ConstantIndexOp>(v.getDefiningOp());
      if (!c)
        return failure();
      sizes.push_back(c);
    }

    // compute a new stride equal to the stride of the repeated dimension
    // multiplied by the number of scf.loop iterations we're replacing
    SmallVector<Value> strides;
    auto lastStrideOffset =
        chanOp.getStrides().size() - memrefTy.getRank() + ivOffsetIdx;
    auto lastStride = dyn_cast<arith::ConstantIndexOp>(
        chanOp.getStrides()[lastStrideOffset].getDefiningOp());
    auto foldedStride = step.value() * lastStride.value();

    strides.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, foldedStride));
    for (auto v : chanOp.getStrides()) {
      auto c = dyn_cast<arith::ConstantIndexOp>(v.getDefiningOp());
      if (!c)
        return failure();
      strides.push_back(c);
    }

    SmallVector<Value> deps;
    SmallVector<Type> tys;
    auto ctx = op->getContext();
    auto channel = FlatSymbolRefAttr::get(ctx, chanOp.getChanName());
    if (isa<air::ChannelPutOp>(op))
      rewriter.create<air::ChannelPutOp>(
          loc, tys, deps, channel, chanOp.getIndices(), chanOp.getMemref(),
          offsets, sizes, strides);
    else
      rewriter.create<air::ChannelGetOp>(
          loc, tys, deps, channel, chanOp.getIndices(), chanOp.getMemref(),
          offsets, sizes, strides);
    rewriter.eraseOp(op);
    rewriter.eraseOp(forOp);
    return success();
  }
};

class AIRHoistScfChannelGetPutPass
    : public air::AIRHoistScfChannelGetPutPassBase<
          AIRHoistScfChannelGetPutPass> {

public:
  AIRHoistScfChannelGetPutPass() = default;
  AIRHoistScfChannelGetPutPass(const AIRHoistScfChannelGetPutPass &pass){};

  void runOnOperation() override;

private:
};

void AIRHoistScfChannelGetPutPass::runOnOperation() {
  auto op = getOperation();
  auto context = op->getContext();
  RewritePatternSet patterns(context);
  patterns.add<HoistScfChannelGetPutPass>(context);
  (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
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

std::unique_ptr<Pass> createAIRHoistScfChannelGetPutPass() {
  return std::make_unique<AIRHoistScfChannelGetPutPass>();
}

} // namespace air
} // namespace xilinx