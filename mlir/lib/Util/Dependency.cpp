//===- Dependency.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Util/Dependency.h"
#include "air/Util/Util.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/SmallSet.h"
#include <sys/stat.h>

#define DEBUG_TYPE "air-dependency-util"

using namespace mlir;

namespace xilinx {
namespace air {

using Graph = dependencyGraph::Graph;

bool areEqualIndices(mlir::Value index_0, mlir::Value index_1) {
  if (index_0 == nullptr || index_1 == nullptr) {
    // Note: memref with index is subset to memref without index (i.e. the
    // entire memref)
    return true;
  } else {
    if (index_0 == index_1)
      return true;
    else if (!index_0.getDefiningOp())
      return false;
    else if (!index_1.getDefiningOp())
      return false;
    else {
      if (auto index_0_const_op =
              dyn_cast<arith::ConstantOp>(index_0.getDefiningOp())) {
        if (auto index_1_const_op =
                dyn_cast<arith::ConstantOp>(index_1.getDefiningOp())) {
          if (index_0_const_op.getValue() == index_1_const_op.getValue())
            return true;
        }
      }
      return false;
    }
  }
}

void traceDependentInductionVar(SmallVector<Value, 1> candidate_scalar_operands,
                                SmallVector<Value, 1> &loop_dep_history,
                                std::vector<Operation *> &op_history) {
  for (auto operand : candidate_scalar_operands) {
    if (!llvm::isa<IndexType, IntegerType, FloatType>(operand.getType()))
      continue; // Only tracing scalar operands
    // If parent loop op is an scf.for
    if (auto for_op = mlir::scf::getForInductionVarOwner(operand)) {
      loop_dep_history.push_back(for_op.getInductionVar());
    }
    // If parent loop op is an scf.parallel
    if (auto par_op = mlir::scf::getParallelForInductionVarOwner(operand)) {
      for (auto ind_var : par_op.getInductionVars())
        if (ind_var == operand)
          loop_dep_history.push_back(ind_var);
    }

    // If parent loop op is an air.herd
    if (auto hl_op = getHerdArgOwner(operand)) {
      for (auto id : hl_op.getIds()) {
        if (operand == id) {
          loop_dep_history.push_back(id);
        }
      }
    }
  }

  // Recursively trace dependency to loop induction vars
  for (auto operand : candidate_scalar_operands) {
    if (!llvm::isa<IndexType, IntegerType, FloatType>(operand.getType()))
      continue; // Only tracing scalar operands
    auto defOp = operand.getDefiningOp();
    if (defOp) {
      op_history.push_back(defOp);
      traceDependentInductionVar(defOp, loop_dep_history, op_history);
    } else {
      // Trace dependency through a for loop
      if (auto for_op = getForRegionIterArgsOwner(operand)) {
        for (auto iter_arg : for_op.getInitArgs()) {
          if (operand == iter_arg) {
            loop_dep_history.push_back(iter_arg);
          }
        }
      }
      // Trace dependency through a parallel loop
      // TODO: decide if parallel should exist in herd launch
    }
  }
}

// Recursively check for dependency to loop induction vars arising from dma
void traceDependentInductionVar(air::MemcpyInterface memcpyif_op,
                                SmallVector<Value, 1> &loop_dep_history,
                                std::vector<Operation *> &op_history) {
  // Check for immediate dependency to loop induction vars
  SmallVector<Value, 1> candidate_scalar_operands;
  if (memcpyif_op.getSrcMemref()) {
    for (unsigned i = 0; i < memcpyif_op.getSrcOffsets().size(); i++) {
      candidate_scalar_operands.push_back(memcpyif_op.getSrcOffsets()[i]);
      candidate_scalar_operands.push_back(memcpyif_op.getSrcSizes()[i]);
      candidate_scalar_operands.push_back(memcpyif_op.getSrcStrides()[i]);
    }
  }
  if (memcpyif_op.getDstMemref()) {
    for (unsigned i = 0; i < memcpyif_op.getDstOffsets().size(); i++) {
      candidate_scalar_operands.push_back(memcpyif_op.getDstOffsets()[i]);
      candidate_scalar_operands.push_back(memcpyif_op.getDstSizes()[i]);
      candidate_scalar_operands.push_back(memcpyif_op.getDstStrides()[i]);
    }
  }

  // Check for dependency through any parent affine if guards
  if (auto parentAffineIf =
          memcpyif_op->getParentOfType<affine::AffineIfOp>()) {
    if (parentAffineIf->getParentOfType<air::HerdOp>()) {
      candidate_scalar_operands.insert(candidate_scalar_operands.end(),
                                       parentAffineIf.getOperands().begin(),
                                       parentAffineIf.getOperands().end());
    }
  }

  // Start recursion.
  traceDependentInductionVar(candidate_scalar_operands, loop_dep_history,
                             op_history);
}

// Recursively check for dependency to any loop induction vars
void traceDependentInductionVar(Operation *op,
                                SmallVector<Value, 1> &loop_dep_history,
                                std::vector<Operation *> &op_history) {
  SmallVector<Value, 1> candidate_scalar_operands;
  // Get child op if op is air.execute
  if (auto air_region_op = dyn_cast<air::ExecuteOp>(op)) {
    if (air_region_op.getRegion().front().getOperations().size() != 2) {
      air_region_op->emitOpError("air::ExecuteOp should have only one child "
                                 "operation beside the terminator");
      return;
    }
    for (auto &child_op : air_region_op.getRegion().front().getOperations()) {
      if (!dyn_cast<air::ExecuteTerminatorOp>(child_op))
        op = &child_op;
    }
  }
  candidate_scalar_operands.insert(candidate_scalar_operands.end(),
                                   op->getOperands().begin(),
                                   op->getOperands().end());

  // Check for dependency through any parent affine if guards
  if (auto parentAffineIf = op->getParentOfType<affine::AffineIfOp>()) {
    if (parentAffineIf->getParentOfType<air::HerdOp>()) {
      candidate_scalar_operands.insert(candidate_scalar_operands.end(),
                                       parentAffineIf.getOperands().begin(),
                                       parentAffineIf.getOperands().end());
    }
  }

  // Start recursion.
  traceDependentInductionVar(op->getOperands(), loop_dep_history, op_history);
}

// Recursively check for dependency to any air.herd induction variables.
void traceDependentHerdId(Operation *async_op,
                          SmallVector<Value> &loop_dep_history,
                          SmallVector<Operation *> &op_history) {
  if (!isAsyncOp(async_op))
    return;
  // Get child op if async_op is air.execute
  Operation *op = nullptr;
  if (auto air_execute_op = dyn_cast<air::ExecuteOp>(async_op)) {
    op = &air_execute_op.getChildOps().front();
  } else {
    op = async_op;
  }

  // Check for immediate dependency to loop induction vars
  for (auto operand : op->getOperands()) {
    // If parent loop op is an air.launch_herd
    if (auto hl_op = air::getHerdArgOwner(operand)) {
      for (auto id : hl_op.getIds()) {
        if (operand == id) {
          loop_dep_history.push_back(id);
        }
      }
    }
  }

  // Recursively trace dependency to loop induction vars
  for (auto operand : op->getOperands()) {
    if (operand && llvm::isa<IndexType>(
                       operand.getType())) { // Only tracing scalar operands
      if (operand.getDefiningOp() &&
          mlir::dyn_cast<air::AsyncOpInterface>(operand.getDefiningOp())) {
        op_history.push_back(operand.getDefiningOp());
        traceDependentHerdId(operand.getDefiningOp(), loop_dep_history,
                             op_history);
      } else if (auto scf_for = scf::getForInductionVarOwner(operand)) {
        op_history.push_back(scf_for);
        traceDependentHerdId(scf_for, loop_dep_history, op_history);
      }
    }
  }
}

// Recursively check for dependency to air herd op induction vars.
std::vector<std::tuple<Value, SmallVector<Value>, SmallVector<Operation *>>>
traceDependentHerdId(air::DmaMemcpyNdOp dmaNd_op) {
  // Tuple fields: value, ancestors and producers to those ancestors.
  std::vector<std::tuple<Value, SmallVector<Value>, SmallVector<Operation *>>>
      loop_dep_history;
  for (unsigned i = 0; i < dmaNd_op.getSrcOffsets().size(); i++) {
    loop_dep_history.push_back(std::make_tuple(dmaNd_op.getSrcOffsets()[i],
                                               SmallVector<Value>{},
                                               SmallVector<Operation *>{}));
    loop_dep_history.push_back(std::make_tuple(dmaNd_op.getSrcSizes()[i],
                                               SmallVector<Value>{},
                                               SmallVector<Operation *>{}));
    loop_dep_history.push_back(std::make_tuple(dmaNd_op.getSrcStrides()[i],
                                               SmallVector<Value>{},
                                               SmallVector<Operation *>{}));
  }
  for (unsigned i = 0; i < dmaNd_op.getDstOffsets().size(); i++) {
    loop_dep_history.push_back(std::make_tuple(dmaNd_op.getDstOffsets()[i],
                                               SmallVector<Value>{},
                                               SmallVector<Operation *>{}));
    loop_dep_history.push_back(std::make_tuple(dmaNd_op.getDstSizes()[i],
                                               SmallVector<Value>{},
                                               SmallVector<Operation *>{}));
    loop_dep_history.push_back(std::make_tuple(dmaNd_op.getDstStrides()[i],
                                               SmallVector<Value>{},
                                               SmallVector<Operation *>{}));
  }
  for (auto &elem : loop_dep_history) {
    // If parent loop op is an air.launch_herd
    if (auto hl_op = air::getHerdArgOwner(std::get<0>(elem))) {
      for (auto id : hl_op.getIds()) {
        if (std::get<0>(elem) == id) {
          std::get<1>(elem).push_back(id);
        }
      }
    }
  }

  // Recursively trace dependency to loop induction vars
  for (auto &elem : loop_dep_history) {
    if (std::get<0>(elem) &&
        llvm::isa<IndexType>(
            std::get<0>(elem).getType())) { // Only tracing scalar operands
      if (std::get<0>(elem).getDefiningOp() &&
          mlir::dyn_cast<air::AsyncOpInterface>(
              std::get<0>(elem).getDefiningOp())) {
        auto ancestor_async_op = std::get<0>(elem).getDefiningOp();
        std::get<2>(elem).push_back(ancestor_async_op);
        traceDependentHerdId(ancestor_async_op, std::get<1>(elem),
                             std::get<2>(elem));
      }
    }
  }

  return loop_dep_history;
}

void eraseAsyncDependencyFromAsyncOp(xilinx::air::AsyncOpInterface op,
                                     Value token) {
  if (!token)
    return;
  if (!llvm::isa<air::AsyncTokenType>(token.getType()))
    return;
  auto dependency_list = op.getAsyncDependencies();
  if (!dependency_list.size())
    return;
  for (int i = dependency_list.size() - 1; i >= 0; i--) {
    if (dependency_list[i] == token) {
      op.eraseAsyncDependency(i);
    }
  }
}

void clearAsyncDependenciesOfAsyncOpImpl(xilinx::air::AsyncOpInterface op) {
  auto dependency_list = op.getAsyncDependencies();
  if (!dependency_list.size())
    return;
  for (int i = dependency_list.size() - 1; i >= 0; i--) {
    op.eraseAsyncDependency(i);
  }
}
void clearAsyncDependenciesOfAsyncOpImpl(scf::ForOp op) {
  SmallVector<Value> operands_without_wait_all;
  for (auto iter_oper : op.getInitArgs()) {
    // Push to vec if unique
    if (std::find(operands_without_wait_all.begin(),
                  operands_without_wait_all.end(),
                  iter_oper) == operands_without_wait_all.end()) {
      operands_without_wait_all.push_back(iter_oper);
    }
  }
  for (auto v : operands_without_wait_all) {
    OpBuilder builder(op);
    SmallVector<Value> dep_list = {};
    air::WaitAllOp wait_all_op_before_loop = xilinx::air::WaitAllOp::create(
        builder, builder.getUnknownLoc(),
        air::AsyncTokenType::get(op->getContext()), dep_list);
    op->replaceUsesOfWith(v, wait_all_op_before_loop.getAsyncToken());
  }
}
void clearAsyncDependenciesOfAsyncOpImpl(scf::ParallelOp op) {
  SmallVector<Value> operands_without_wait_all;
  for (auto init_val : op.getInitVals()) {
    if (auto wa_op = dyn_cast<air::WaitAllOp>(init_val.getDefiningOp())) {
      clearAsyncDependenciesOfAsyncOpImpl(wa_op);
    } else {
      // Push to vec if unique
      if (std::find(operands_without_wait_all.begin(),
                    operands_without_wait_all.end(),
                    init_val) == operands_without_wait_all.end()) {
        operands_without_wait_all.push_back(init_val);
      }
    }
  }
  for (auto v : operands_without_wait_all) {
    OpBuilder builder(op);
    SmallVector<Value> dep_list = {};
    air::WaitAllOp wait_all_op_before_loop = xilinx::air::WaitAllOp::create(
        builder, builder.getUnknownLoc(),
        air::AsyncTokenType::get(op->getContext()), dep_list);
    op->replaceUsesOfWith(v, wait_all_op_before_loop.getAsyncToken());
  }
}
void clearAsyncDependenciesOfAsyncOp(Operation *op) {
  if (!isAsyncOp(op)) {
    return;
  }
  if (auto async_op = dyn_cast<air::AsyncOpInterface>(op)) {
    clearAsyncDependenciesOfAsyncOpImpl(async_op);
  } else if (auto for_op = dyn_cast<scf::ForOp>(op)) {
    clearAsyncDependenciesOfAsyncOpImpl(for_op);
  } else if (auto parallel_op = dyn_cast<scf::ParallelOp>(op)) {
    clearAsyncDependenciesOfAsyncOpImpl(parallel_op);
  } else
    op->emitOpError("unknown async op");
}

// Get loop-carried dependency token from scf loop op
Value getLoopCarriedTokenFromScfOp(scf::ParallelOp op) {
  if (!op.getInitVals().size()) {
    op->emitOpError("has no init_val");
    return nullptr;
  }
  auto token = op.getInitVals()[0];
  if (!llvm::isa<air::AsyncTokenType>(token.getType())) {
    op->emitOpError("init_val is not an async token");
    return nullptr;
  }
  return token;
}
Value getLoopCarriedTokenFromScfOp(scf::ForOp op,
                                   std::string operand_or_argument) {
  if (operand_or_argument == "operand") {
    if (!op.getInitArgs().size()) {
      return nullptr;
    }
    for (auto initArg : op.getInitArgs()) {
      if (isa<air::AsyncTokenType>(initArg.getType()))
        return initArg;
    }
    // No async token found - return nullptr without error
    return nullptr;
  } else if (operand_or_argument == "argument") {
    if (!op.getRegionIterArgs().size()) {
      return nullptr;
    }
    for (auto iterArg : op.getRegionIterArgs()) {
      if (isa<air::AsyncTokenType>(iterArg.getType())) {
        return iterArg;
      }
    }
    // No async token found - return nullptr without error
    return nullptr;
  } else {
    op->emitOpError("unknown string in operand_or_argument");
    return nullptr;
  }
}
air::WaitAllOp assignEmptyWaitAllAtScfForIterArg(OpBuilder builder,
                                                 scf::ForOp &op) {
  auto checkpoint = builder.saveInsertionPoint();
  builder.setInsertionPoint(op);
  air::WaitAllOp sink_wait_all_op = air::WaitAllOp::create(
      builder, op->getLoc(), air::AsyncTokenType::get(builder.getContext()),
      SmallVector<Value>{});
  op->getOpOperand(op.getNumControlOperands())
      .assign(sink_wait_all_op.getAsyncToken());
  builder.restoreInsertionPoint(checkpoint);
  return sink_wait_all_op;
}

// Create scf.reduce op to reduce all async tokens in an scf.parallel
scf::ReduceOp createSCFReduceForAsyncSCFParallel(OpBuilder builder,
                                                 Location loc, Value token,
                                                 MLIRContext *ctx) {
  auto reduce_op = scf::ReduceOp::create(builder, loc, token);
  builder.setInsertionPointToStart(&reduce_op.getRegion(0).front());
  SmallVector<Value, 4> reduce_tokens;
  reduce_tokens.push_back(reduce_op.getRegion(0).front().getArgument(0));
  reduce_tokens.push_back(reduce_op.getRegion(0).front().getArgument(1));
  auto reduce_res = xilinx::air::WaitAllOp::create(
      builder, builder.getUnknownLoc(), air::AsyncTokenType::get(ctx),
      reduce_tokens);
  scf::ReduceReturnOp::create(builder, builder.getUnknownLoc(),
                              reduce_res.getResult(0));
  return reduce_op;
}

// Get dependency list of op
SmallVector<Value> getAsyncDependenciesFromOpImpl(air::AsyncOpInterface op) {
  return op.getAsyncDependencies();
}
SmallVector<Value> getAsyncDependenciesFromOpImpl(scf::ForOp op) {
  return op.getInitArgs();
}
SmallVector<Value> getAsyncDependenciesFromOpImpl(scf::ParallelOp op) {
  return op.getInitVals();
}
SmallVector<Value> getAsyncDependenciesFromOpImpl(affine::AffineIfOp op) {
  SmallVector<Value> depList;
  for (auto operand : op->getOperands()) {
    if (!isa<air::AsyncTokenType>(operand.getType()))
      continue;
    depList.push_back(operand);
  }
  return depList;
}
SmallVector<Value> getAsyncDependenciesFromOp(Operation *op) {
  if (auto async_op = dyn_cast<air::AsyncOpInterface>(op))
    return getAsyncDependenciesFromOpImpl(async_op);
  else if (auto for_op = dyn_cast<scf::ForOp>(op))
    return getAsyncDependenciesFromOpImpl(for_op);
  else if (auto par_op = dyn_cast<scf::ParallelOp>(op))
    return getAsyncDependenciesFromOpImpl(par_op);
  else if (auto aif_op = dyn_cast<affine::AffineIfOp>(op))
    return getAsyncDependenciesFromOpImpl(aif_op);
  else
    return SmallVector<Value>();
}

// Get returned async token from op
Value getAsyncTokenFromOpImpl(air::AsyncOpInterface op) {
  return op.getAsyncToken();
}
Value getAsyncTokenFromOpImpl(Operation *op) {
  for (auto res : op->getResults()) {
    if (isa<air::AsyncTokenType>(res.getType()))
      return res;
  }
  return nullptr;
}
Value getAsyncTokenFromOp(Operation *op) {
  if (auto async_op = dyn_cast<air::AsyncOpInterface>(op))
    return getAsyncTokenFromOpImpl(async_op);
  else
    return getAsyncTokenFromOpImpl(op);
}

// Convert scf.for op to async, by adding an async token iter arg.
struct MakeAsyncScfForPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  MakeAsyncScfForPattern(MLIRContext *ctx, Value token)
      : OpRewritePattern(ctx), token(token) {}

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    if (!air::getAsyncDependenciesFromOp(forOp).empty())
      return failure();
    if (!isa<air::AsyncTokenType>(token.getType()))
      return failure();
    if (failed(forOp.replaceWithAdditionalIterOperands(
            rewriter, SmallVector<Value>{token}, true)))
      return failure();
    return success();
  }

private:
  Value token;
};

// Add async dependency to op if unique
void addAsyncDependencyIfNewImpl(air::AsyncOpInterface op, Value token) {
  if (!llvm::isa<air::AsyncTokenType>(token.getType())) {
    op->emitOpError("value is not an async token");
    return;
  }
  bool foundTokenInDepList = false;
  if (op.getAsyncDependencies().size()) {
    for (auto old_dep : op.getAsyncDependencies())
      if (old_dep == token)
        foundTokenInDepList = true;
    if (!foundTokenInDepList) {
      op.addAsyncDependency(token);
    }
  } else {
    op.addAsyncDependency(token);
  }
}
void addAsyncDependencyIfNewImpl(scf::ForOp op, Value token) {
  auto ctx = op->getContext();
  llvm::SetVector<Value> operands_without_wait_all;
  for (auto iter_oper : op.getInitArgs()) {
    if (!isa_and_present<air::AsyncTokenType>(iter_oper.getType()))
      continue;
    auto wa_op = dyn_cast_if_present<air::WaitAllOp>(iter_oper.getDefiningOp());
    if (wa_op && wa_op.getAsyncToken().hasOneUse())
      addAsyncDependencyIfNewImpl(wa_op, token);
    else
      operands_without_wait_all.insert(iter_oper);
  }
  for (auto v : operands_without_wait_all) {
    OpBuilder builder(op);
    SmallVector<Value> dep_list = {token, v};
    air::WaitAllOp wait_all_op_before_loop =
        xilinx::air::WaitAllOp::create(builder, builder.getUnknownLoc(),
                                       air::AsyncTokenType::get(ctx), dep_list);
    op->replaceUsesOfWith(v, wait_all_op_before_loop.getAsyncToken());
  }
  // If scf.for loop isn't async, then make it async.
  if (!isAsyncOp(op)) {
    RewritePatternSet patterns(ctx);
    patterns.insert<MakeAsyncScfForPattern>(ctx, token);
    (void)applyOpPatternsGreedily(ArrayRef<Operation *>{op},
                                  std::move(patterns));
  }
}
void addAsyncDependencyIfNewImpl(scf::ParallelOp op, Value token) {
  SmallVector<Value> operands_without_wait_all;
  for (auto init_val : op.getInitVals()) {
    if (init_val.getDefiningOp() &&
        isa<air::WaitAllOp>(init_val.getDefiningOp())) {
      auto wa_op = dyn_cast<air::WaitAllOp>(init_val.getDefiningOp());
      addAsyncDependencyIfNewImpl(wa_op, token);
    } else {
      // Push to vec if unique
      if (std::find(operands_without_wait_all.begin(),
                    operands_without_wait_all.end(),
                    init_val) == operands_without_wait_all.end()) {
        operands_without_wait_all.push_back(init_val);
      }
    }
  }
  for (auto v : operands_without_wait_all) {
    OpBuilder builder(op);
    SmallVector<Value> dep_list = {};
    air::WaitAllOp wait_all_op_before_loop = xilinx::air::WaitAllOp::create(
        builder, builder.getUnknownLoc(),
        air::AsyncTokenType::get(op->getContext()), dep_list);
    op->replaceUsesOfWith(v, wait_all_op_before_loop.getAsyncToken());
    replaceAllUsesInRegionWith(v, wait_all_op_before_loop.getAsyncToken(),
                               op.getRegion());
    addAsyncDependencyIfNewImpl(wait_all_op_before_loop, v);
    addAsyncDependencyIfNewImpl(wait_all_op_before_loop, token);
  }
}
void addAsyncDependencyIfNewImpl(affine::AffineIfOp op, Value token) {
  // Process operations in the then block
  for (auto &nested_op : op.getThenBlock()->getOperations()) {
    // Skip affine.yield terminators
    if (isa<affine::AffineYieldOp>(nested_op))
      continue;

    // Recursively process nested affine.if operations
    if (auto nested_affine_if = dyn_cast<affine::AffineIfOp>(nested_op)) {
      addAsyncDependencyIfNewImpl(nested_affine_if, token);
    } else {
      // For non-affine.if operations, call addAsyncDependencyIfNew
      addAsyncDependencyIfNew(&nested_op, token);
    }
  }

  // Process operations in the else block if it exists
  if (op.hasElse()) {
    for (auto &nested_op : op.getElseBlock()->getOperations()) {
      // Skip affine.yield terminators
      if (isa<affine::AffineYieldOp>(nested_op))
        continue;

      // Recursively process nested affine.if operations
      if (auto nested_affine_if = dyn_cast<affine::AffineIfOp>(nested_op)) {
        addAsyncDependencyIfNewImpl(nested_affine_if, token);
      } else {
        // For non-affine.if operations, call addAsyncDependencyIfNew
        addAsyncDependencyIfNew(&nested_op, token);
      }
    }
  }
}
void addAsyncDependencyIfNew(Operation *op, Value token) {
  if (!op)
    return;
  if (!isAsyncOp(op))
    return;
  if (!token)
    return;
  if (token.getDefiningOp() && token.getDefiningOp() == op)
    return;
  if (auto async_op = dyn_cast<air::AsyncOpInterface>(op)) {
    addAsyncDependencyIfNewImpl(async_op, token);
  } else if (auto for_op = dyn_cast<scf::ForOp>(op)) {
    addAsyncDependencyIfNewImpl(for_op, token);
  } else if (auto parallel_op = dyn_cast<scf::ParallelOp>(op)) {
    addAsyncDependencyIfNewImpl(parallel_op, token);
  } else if (auto affine_if_op = dyn_cast<affine::AffineIfOp>(op)) {
    addAsyncDependencyIfNewImpl(affine_if_op, token);
  } else
    op->emitOpError("unknown async op");
}

bool isAsyncOp(Operation *op) {
  if (!op)
    return false;
  if (llvm::any_of(op->getResults(), [](Value r) {
        return isa<air::AsyncTokenType>(r.getType());
      }))
    return true;
  return false;
}

// Air dependency comes in two forms: production and consumption of the same
// async token, and usage of the same air.channel.
bool areAsyncDependent(Operation *a, Operation *b) {
  SmallVector<Value> dep_a = getAsyncDependenciesFromOp(a);
  Value token_a = getAsyncTokenFromOp(a);
  SmallVector<Value> dep_b = getAsyncDependenciesFromOp(b);
  Value token_b = getAsyncTokenFromOp(b);
  if (!token_a)
    return false;
  if (!token_b)
    return false;
  for (auto dep : dep_a)
    if (dep == token_b)
      return true;
  for (auto dep : dep_b)
    if (dep == token_a)
      return true;
  // Deep async dependency tracing through air.wait_all.
  if (isAsyncDependent(a, b))
    return true;
  if (isAsyncDependent(b, a))
    return true;

  auto chanA = dyn_cast<air::ChannelInterface>(a);
  auto chanB = dyn_cast<air::ChannelInterface>(b);
  if (chanA && chanB)
    if (chanA.getChanName() == chanB.getChanName()) {
      if (chanA.getIndices().size() != chanB.getIndices().size())
        return true;
      // Check all index positions. If ANY position has two different constant
      // values, we can prove independence regardless of other positions.
      for (unsigned i = 0; i < chanA.getIndices().size(); i++) {
        auto constIdxA = getConstantIntValue(chanA.getIndices()[i]);
        auto constIdxB = getConstantIntValue(chanB.getIndices()[i]);
        // If BOTH are constants AND they differ → INDEPENDENT
        if (constIdxA && constIdxB && (*constIdxA != *constIdxB))
          return false;
      }
      // After checking all indices, if none were provably different →
      // DEPENDENT
      return true;
    }
  return false;
}

// Returns true if b is asynchronously dependent on a. This function performs a
// deep dependency tracing that propagates through air.wait_all ops.
bool isAsyncDependent(Operation *a, Operation *b) {
  if (a == b)
    return true;
  Value token_a = getAsyncTokenFromOp(a);
  SmallVector<Value> dep_b = getAsyncDependenciesFromOp(b);
  if (!token_a)
    return false;
  if (dep_b.empty())
    return false;
  for (auto dep : dep_b) {
    if (dep == token_a)
      return true;
    else if (dep.getDefiningOp() && air::isAsyncOp(dep.getDefiningOp())) {
      if (isAsyncDependent(a, dep.getDefiningOp()))
        return true;
    }
  }
  return false;
}

// Generate a wait_all at the end of block, which gathers all dangling async
// tokens.
air::WaitAllOp generateWaitAllToTerminateBlock(Block &block, OpBuilder &b,
                                               bool isBlocking) {
  llvm::SetVector<Value> blockTokens, danglingTokens;
  blockTokens.insert(block.getArguments().begin(), block.getArguments().end());
  for (auto &o : block.getOperations())
    blockTokens.insert(o.getResults().begin(), o.getResults().end());
  for (auto t : blockTokens) {
    if (!t.use_empty())
      continue;
    if (!isa<air::AsyncTokenType>(t.getType()))
      continue;
    danglingTokens.insert(t);
  }
  if (block.mightHaveTerminator())
    b.setInsertionPoint(block.getTerminator());
  else
    b.setInsertionPointToEnd(&block);
  if (isBlocking)
    return air::WaitAllOp::create(b, b.getUnknownLoc(),
                                  /*result_type*/ Type(),
                                  danglingTokens.takeVector());
  else
    return air::WaitAllOp::create(b, b.getUnknownLoc(),
                                  air::AsyncTokenType::get(b.getContext()),
                                  danglingTokens.takeVector());
}

// Splits an SCF for loop into two for loops, by hoisting target operations in
// for loop to a new for loop located at the same scope.
scf::ForOp hoistTargetOpsToNewSCFFor(PatternRewriter &rewriter,
                                     scf::ForOp for_op,
                                     SmallVector<Operation *> target_ops) {
  auto loc = for_op->getLoc();
  // If target ops are already perfectly nested, then skip
  auto hasNChannelOps = [target_ops](Block *block, unsigned N) {
    unsigned counter = 0;
    block->walk<WalkOrder::PreOrder, ForwardDominanceIterator<>>(
        [target_ops, &counter](Operation *op) {
          if (op->hasTrait<OpTrait::IsIsolatedFromAbove>())
            return WalkResult::skip();
          if (llvm::is_contained(target_ops, op)) {
            counter++;
            return WalkResult::skip();
          }
          if (isa<air::ChannelInterface>(op))
            counter++;
          counter++;
          return WalkResult::advance();
        });
    return counter == N;
  };
  if (hasNChannelOps(for_op.getBody(), 1))
    return for_op;

  rewriter.setInsertionPoint(for_op);
  IRMapping remap;
  auto new_for_op = scf::ForOp::create(rewriter, loc, for_op.getLowerBound(),
                                       for_op.getUpperBound(), for_op.getStep(),
                                       for_op.getInitArgs());
  remap.map(for_op.getInductionVar(), new_for_op.getInductionVar());
  remap.map(getLoopCarriedTokenFromScfOp(for_op, "argument"),
            getLoopCarriedTokenFromScfOp(new_for_op, "argument"));
  rewriter.setInsertionPointToStart(new_for_op.getBody());
  // Build up a log of ops to be cloned; using SetVector to avoid repetition.
  llvm::SetVector<Operation *> ops_to_be_cloned;
  for (auto op : target_ops) {
    if (op->getParentOp() != for_op.getOperation())
      continue;
    // Clone defining ops of both the target_op's operands, and any used values
    // within its regions.
    llvm::SetVector<Value> region_opers;
    for (auto &region : op->getRegions())
      getUsedValuesDefinedAbove(region, region_opers);
    region_opers.insert(op->getOperands().begin(), op->getOperands().end());
    SmallVector<Value> region_opers_vec = region_opers.takeVector();
    llvm::SetVector<Operation *> backwardSlices;
    air::getBackwardSliceInRegion(rewriter, &for_op.getRegion(),
                                  region_opers_vec, backwardSlices);
    ops_to_be_cloned.insert(backwardSlices.begin(), backwardSlices.end());
    ops_to_be_cloned.insert(op);
  }

  // Clone all collected operations into the new for loop body
  for (auto o : ops_to_be_cloned)
    rewriter.clone(*o, remap);

  SmallVector<Value> yield_operands;
  // If the new for loop is async, we need to properly terminate it with async
  // tokens
  if (air::isAsyncOp(new_for_op)) {
    // Generate a wait_all op that collects all dangling async tokens in the
    // loop body. This ensures all async operations within the loop are properly
    // synchronized.
    auto waitAllOp = generateWaitAllToTerminateBlock(
        *new_for_op.getBody(), rewriter, /*isBlocking*/ false);
    yield_operands.push_back(getAsyncTokenFromOp(waitAllOp));

    // Create the scf.yield operation for the loop, yielding a wait_all token.
    // The yielded wait_all synchronizes on all operations collected by
    // waitAllOp, allowing the for loop to properly propagate async dependencies
    // to subsequent iterations.
    scf::YieldOp::create(
        rewriter, loc,
        SmallVector<Value>{air::WaitAllOp::create(
                               rewriter, loc,
                               air::AsyncTokenType::get(rewriter.getContext()),
                               yield_operands)
                               ->getResult(0)});
  }

  return new_for_op;
}

// Walk the body of an scf.for loop and attaches an attribute (e.g.,
// "scf.for_result_id") to each operation that defines a value yielded by
// scf.yield. Each label includes the index of the result it corresponds to
// (e.g., 0 for the first result, etc.).
void labelYieldDefiningOpsOfForLoop(scf::ForOp forOp, StringRef attrName) {
  // Get the yield operation in the loop body.
  auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yieldOp)
    return;

  for (auto it : llvm::enumerate(yieldOp.getOperands())) {
    Value yieldedValue = it.value();
    Operation *defOp = yieldedValue.getDefiningOp();

    // Only label ops that are defined within the loop body
    if (defOp && forOp.getRegion().isAncestor(defOp->getParentRegion())) {
      defOp->setAttr(attrName,
                     IntegerAttr::get(IntegerType::get(forOp->getContext(), 32),
                                      static_cast<int32_t>(it.index())));
    }
  }
}

// Collect operations with the integer attribute "scf.for_result_id" and
// groups them into vectors by the attribute value. Result is a vector of
// vectors Operation*, where each sub-vector corresponds to a specific result
// index.
SmallVector<SmallVector<Operation *>>
collectOpsByScfForResultId(Block &block, StringRef attrName) {
  llvm::DenseMap<int64_t, SmallVector<Operation *>> resultAsMap;
  block.walk([&](Operation *op) {
    auto attr = op->getAttrOfType<IntegerAttr>(attrName);
    if (!attr)
      return;
    int64_t resultId = attr.getInt();
    resultAsMap[resultId].push_back(op);
    op->removeAttr(attrName);
  });

  // Convert to a vector of vectors.
  SmallVector<SmallVector<Operation *>> groupedOps;
  for (auto &entry : resultAsMap) {
    groupedOps.push_back(entry.second);
  }

  return groupedOps;
}

llvm::SmallDenseSet<OpOperand *>
getUsesOfAsyncTokens(const SmallVector<Operation *> &ops) {
  llvm::SmallDenseSet<OpOperand *> uses;

  for (Operation *op : ops) {
    if (!air::isAsyncOp(op))
      continue;
    Value token = air::getAsyncTokenFromOp(op);
    for (OpOperand &use : token.getUses()) {
      uses.insert(&use);
    }
  }

  return uses;
}

// Maintain async token dependencies for unrolled loop ops.
void preserveAsyncDependenciesAfterUnroll(Block &parentBlock) {
  // Collect unrolled ops corresponding to each original loop result
  SmallVector<SmallVector<Operation *>> unrolledOps =
      collectOpsByScfForResultId(parentBlock, "scf.for_result_id");

  for (auto &vec : unrolledOps) {
    auto tokenUses = getUsesOfAsyncTokens(vec);

    for (OpOperand *use : tokenUses) {
      Operation *user = use->getOwner();
      if (auto yieldUser = dyn_cast<scf::YieldOp>(user)) {
        OpBuilder builder(yieldUser);
        user = air::WaitAllOp::create(
            builder, yieldUser->getLoc(),
            air::AsyncTokenType::get(yieldUser->getContext()),
            SmallVector<Value>{use->get()});
        use->assign(user->getResult(0));
      }

      air::AsyncOpInterface asyncUser = dyn_cast<air::AsyncOpInterface>(user);
      if (!asyncUser) {
        user->emitWarning(
            "An async token returned by an unrolled scf.for loop is used by "
            "an op not in AsyncOpInterface type. Only the last unrolled "
            "iteration's dependency is preserved.");
        continue;
      }

      for (Operation *op : vec) {
        if (!air::isAsyncOp(op))
          continue;

        // Only add dependency if SSA dominance is preserved
        DominanceInfo domInfo(op);
        if (!domInfo.properlyDominates(op, asyncUser))
          continue;

        asyncUser.addAsyncDependency(air::getAsyncTokenFromOp(op));
      }
    }
  }
}

// Fully unrolls an `scf.for` loop while preserving async token dependencies.
//
// This function labels the operations that define the values yielded by
// `scf.yield`, then performs full unrolling of the loop. After unrolling,
// it identifies async-producing operations corresponding to each yielded value
// and ensures that any users of the original loop results (which consume async
// tokens) are updated to depend on the corresponding unrolled ops that dominate
// them.
//
// If `annotateFn` is provided, it is passed to `loopUnrollByFactor` for result
// tagging.
LogicalResult loopUnrollFullWithAsyncTokenPreserved(
    scf::ForOp forOp,
    function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn) {
  // Label the ops that define values yielded by scf.yield
  labelYieldDefiningOpsOfForLoop(forOp, "scf.for_result_id");

  Block *parentBlock = forOp->getBlock();

  // Fully unroll the loop
  if (annotateFn) {
    auto unroll_factor = air::getStaticScfForTripCountAsInt(forOp);
    if (!unroll_factor) {
      forOp->emitOpError("failed to fully unroll: dynamic loop bound.");
      return failure();
    }
    if (failed(loopUnrollByFactor(forOp, *unroll_factor, annotateFn))) {
      forOp->emitOpError("failed to fully unroll.");
      return failure();
    }
  } else {
    if (failed(loopUnrollFull(forOp))) {
      forOp->emitOpError("failed to fully unroll.");
      return failure();
    }
  }

  preserveAsyncDependenciesAfterUnroll(*parentBlock);
  return success();
}

// Unrolls an `scf.for` loop by a given factor while preserving async token
// dependencies.
//
// This function first labels operations that define yielded values, then
// performs unrolling by the specified factor using `loopUnrollByFactor`.
// After unrolling, it ensures that users of async tokens returned by the
// original loop are properly connected to async-producing ops from unrolled
// iterations, based on SSA dominance rules.
LogicalResult loopUnrollByFactorWithAsyncTokenPreserved(
    scf::ForOp forOp, uint64_t unrollFactor,
    function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn) {
  // Label the ops that define values yielded by scf.yield
  labelYieldDefiningOpsOfForLoop(forOp, "scf.for_result_id");

  Block *parentBlock = forOp->getBlock();

  // Unroll the loop by factor
  if (failed(loopUnrollByFactor(forOp, unrollFactor, annotateFn))) {
    forOp->emitOpError("failed to fully unroll.");
    return failure();
  }

  preserveAsyncDependenciesAfterUnroll(*parentBlock);
  return success();
}

// Unroll scf.parallel ops.
LogicalResult unrollScfParallel(
    OpBuilder builder, scf::ParallelOp par, IRMapping remap,
    llvm::DenseMap<Operation *, SmallVector<Operation *>> &opMap) {
  auto loc = par->getLoc();
  auto ctx = par->getContext();
  SmallVector<int, 2> lbs_spatial, ubs_spatial;
  air::getSizesFromSpatialLoop(par.getOperation(), lbs_spatial, ubs_spatial);
  std::vector<unsigned> par_size;
  unsigned par_vol = 1;
  for (unsigned i = 0; i < lbs_spatial.size(); i++) {
    par_size.push_back(ubs_spatial[i] - lbs_spatial[i] + 1);
    par_vol *= ubs_spatial[i] - lbs_spatial[i] + 1;
  }
  SmallVector<Value> yieldedTokens;
  Operation *curr_new_op = nullptr;
  for (unsigned iter = 0; iter < par_vol; iter++) {
    IRMapping localRemap = remap;
    std::vector<unsigned> position =
        air::getMDVectorFromIterator(par_size, iter);
    std::reverse(
        position.begin(),
        position
            .end()); // scf.parallel induction vars. have LSD at highest index.
    for (unsigned i = 0; i < position.size(); i++) {
      localRemap.map(par.getInductionVars()[i],
                     arith::ConstantIndexOp::create(
                         builder, builder.getUnknownLoc(),
                         position[i] * *getConstantIntValue(par.getStep()[i]) +
                             *getConstantIntValue(par.getLowerBound()[i])));
    }
    SmallVector<Value> yieldedTokensInIter;
    // Clone ops
    for (auto &op : par.getBody()->without_terminator()) {
      curr_new_op = builder.clone(op, localRemap);
      if (air::getAsyncTokenFromOp(curr_new_op))
        yieldedTokensInIter.push_back(air::getAsyncTokenFromOp(curr_new_op));
      opMap[&op].push_back(curr_new_op);
    }
    // Unroll air.wait_all token reduction
    if (getAsyncTokenFromOp(par)) {
      auto yieldedWaitAll = air::WaitAllOp::create(
          builder, loc, air::AsyncTokenType::get(builder.getContext()),
          yieldedTokensInIter);
      yieldedTokens.push_back(yieldedWaitAll.getAsyncToken());
    }
  }

  if (auto parToken = getAsyncTokenFromOp(par)) {
    parToken.replaceAllUsesWith(
        air::WaitAllOp::create(builder, loc, air::AsyncTokenType::get(ctx),
                               yieldedTokens)
            .getAsyncToken());
  }
  return success();
}

// Separate one scf.parallel with multiple loops into two scf.parallels, with
// the outer parallel containing the specified dimensions.
FailureOr<std::pair<scf::ParallelOp, scf::ParallelOp>>
separateScfParallelByDims(RewriterBase &rewriter, scf::ParallelOp par,
                          IRMapping remap, SmallVector<int> dims) {
  if (par.getNumLoops() < dims.size())
    return failure();
  auto loc = par->getLoc();
  // Separate scf.parallel into multiple scf.parallel loops
  SmallVector<Value> lbs = par.getLowerBound();
  SmallVector<Value> ubs = par.getUpperBound();
  SmallVector<Value> steps = par.getStep();
  SmallVector<Value> inits = par.getInitVals();

  SmallVector<Value> outerLbs, outerUbs, outerSteps, innerLbs, innerUbs,
      innerSteps;
  for (unsigned i = 0; i < par.getNumLoops(); i++) {
    if (llvm::is_contained(dims, i)) {
      outerLbs.push_back(lbs[i]);
      outerUbs.push_back(ubs[i]);
      outerSteps.push_back(steps[i]);
    } else {
      innerLbs.push_back(lbs[i]);
      innerUbs.push_back(ubs[i]);
      innerSteps.push_back(steps[i]);
    }
  }

  auto outerPar = scf::ParallelOp::create(rewriter, loc, outerLbs, outerUbs,
                                          outerSteps, inits);
  rewriter.setInsertionPointToStart(outerPar.getBody());
  auto innerPar = scf::ParallelOp::create(rewriter, loc, innerLbs, innerUbs,
                                          innerSteps, inits);
  int innerIdx = 0;
  int outerIdx = 0;
  for (unsigned i = 0; i < par.getNumLoops(); i++) {
    if (llvm::is_contained(dims, i)) {
      remap.map(par.getInductionVars()[i],
                outerPar.getInductionVars()[outerIdx]);
      outerIdx++;
    } else {
      remap.map(par.getInductionVars()[i],
                innerPar.getInductionVars()[innerIdx]);
      innerIdx++;
    }
  }

  // Clone body with remap.
  rewriter.setInsertionPointToStart(innerPar.getBody());
  for (auto &o : par.getOps()) {
    rewriter.clone(o, remap);
  }

  // Clone the terminator.
  auto term = innerPar.getBody()->getTerminator();
  rewriter.setInsertionPointToEnd(&outerPar.getRegion().front());
  remap.map(term->getOperands(), innerPar->getResults());
  rewriter.clone(*term, remap);

  rewriter.replaceOp(par, outerPar);

  return std::make_pair(outerPar, innerPar);
}

// Unroll scf.parallel ops along a specific set of dimensions.
LogicalResult unrollScfParallelOnDims(
    RewriterBase &rewriter, scf::ParallelOp par, IRMapping remap,
    SmallVector<int> dims,
    llvm::DenseMap<Operation *, SmallVector<Operation *>> &opMap) {

  // Separate the parallel op into two parallel ops, with the outer loop
  // containing dimensions specified in dims.
  auto sepRes = separateScfParallelByDims(rewriter, par, remap, dims);
  if (failed(sepRes))
    return failure();
  auto [outerPar, innerPar] = *sepRes;

  // Unroll outer parallel.
  rewriter.setInsertionPoint(outerPar);
  IRMapping unrollRemap;
  if (failed(unrollScfParallel(rewriter, outerPar, unrollRemap, opMap)))
    return failure();

  if (air::isAsyncOp(outerPar)) {
    rewriter.setInsertionPoint(outerPar);
    auto waitAll =
        air::replaceAsyncOpWithWaitAll(rewriter, unrollRemap, outerPar, false);
    air::getAsyncTokenFromOp(outerPar).replaceAllUsesWith(
        waitAll.getAsyncToken());
  }
  rewriter.eraseOp(outerPar);
  return success();
}

// Unroll air.channel.put/get in scf.parallel.
struct unrollAIRChannelPutInScfParallelPattern
    : public OpRewritePattern<air::ChannelPutOp> {
  using OpRewritePattern<air::ChannelPutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(air::ChannelPutOp put,
                                PatternRewriter &rewriter) const override {
    scf::ParallelOp parParentOp = put->getParentOfType<scf::ParallelOp>();
    SmallVector<scf::ParallelOp> parOps;
    while (parParentOp) {
      parOps.push_back(parParentOp);
      parParentOp = parParentOp->getParentOfType<scf::ParallelOp>();
    }
    if (parOps.empty())
      return failure();
    IRMapping remap;
    llvm::DenseMap<Operation *, SmallVector<Operation *>> opMap;
    for (auto par : parOps) {
      rewriter.setInsertionPoint(par);
      auto res = unrollScfParallel(rewriter, par, remap, opMap);
      if (res.failed())
        return failure();
      rewriter.eraseOp(par);
    }
    return success();
  }

private:
};
struct unrollAIRChannelGetInScfParallelPattern
    : public OpRewritePattern<air::ChannelGetOp> {
  using OpRewritePattern<air::ChannelGetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(air::ChannelGetOp get,
                                PatternRewriter &rewriter) const override {
    scf::ParallelOp parParentOp = get->getParentOfType<scf::ParallelOp>();
    SmallVector<scf::ParallelOp> parOps;
    while (parParentOp) {
      parOps.push_back(parParentOp);
      parParentOp = parParentOp->getParentOfType<scf::ParallelOp>();
    }
    if (parOps.empty())
      return failure();
    IRMapping remap;
    llvm::DenseMap<Operation *, SmallVector<Operation *>> opMap;
    for (auto par : parOps) {
      rewriter.setInsertionPoint(par);
      auto res = unrollScfParallel(rewriter, par, remap, opMap);
      if (res.failed())
        return failure();
      rewriter.eraseOp(par);
    }
    return success();
  }

private:
};

// Erase empty async scf.parallel ops. Non-empty reduce op region, if filled
// with air.wait_all, doesn't get automatically canonicalized.
struct EmptyAIRAsyncScfParallelRemovalPattern
    : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp par,
                                PatternRewriter &rewriter) const override {
    if (llvm::all_of(par.getBody()->without_terminator(), [](Operation &op) {
          return air::isPure(&op) || isa<air::WaitAllOp>(op);
        })) {
      rewriter.replaceOpWithNewOp<air::WaitAllOp>(
          par, air::AsyncTokenType::get(par->getContext()),
          getAsyncDependenciesFromOp(par));
      return success();
    }
    return failure();
  }

private:
};

void populateAIRunrollAIRChannelPutGetInScfParallelPatterns(
    RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.insert<unrollAIRChannelPutInScfParallelPattern,
                  unrollAIRChannelGetInScfParallelPattern,
                  EmptyAIRAsyncScfParallelRemovalPattern>(ctx);
  air::WaitAllOp::getCanonicalizationPatterns(patterns, ctx);
  air::ExecuteOp::getCanonicalizationPatterns(patterns, ctx);
  affine::AffineApplyOp::getCanonicalizationPatterns(patterns, ctx);
}

// Replace async op with wait_all op
air::WaitAllOp replaceAsyncOpWithWaitAll(OpBuilder builder, IRMapping &remap,
                                         Operation *op, bool cloneDepList) {
  if (!air::isAsyncOp(op)) {
    op->emitOpError("op isn't an async op");
    return air::WaitAllOp();
  }
  SmallVector<Value> dep_list_remap;
  if (cloneDepList) {
    for (auto dep : air::getAsyncDependenciesFromOp(op)) {
      dep_list_remap.push_back(remap.lookupOrDefault(dep));
    }
  }
  auto wa_op = air::WaitAllOp::create(
      builder, builder.getUnknownLoc(),
      air::AsyncTokenType::get(op->getContext()), dep_list_remap);
  remap.map(air::getAsyncTokenFromOp(op), wa_op.getAsyncToken());
  return wa_op;
}

// Get memref operands which are read accessed by op. Each entry has the
// following format: pair<memref, tuple<offsets, sizes, strides>>.
FailureOr<SmallVector<
    std::pair<Value, std::tuple<SmallVector<Value>, SmallVector<Value>,
                                SmallVector<Value>>>>>
getAllReadAccessedMemrefOperandsFromOp(Operation *op) {
  SmallVector<
      std::pair<Value, std::tuple<SmallVector<Value>, SmallVector<Value>,
                                  SmallVector<Value>>>>
      operands;
  if (!op)
    return failure();
  auto getMemrefEntry = [](Value memref) {
    std::pair<Value, std::tuple<SmallVector<Value>, SmallVector<Value>,
                                SmallVector<Value>>>
        entry;
    entry.first = memref;
    return entry;
  };
  auto getMemrefAndAccessPatternEntry =
      [](Value memref, SmallVector<Value> offsets, SmallVector<Value> sizes,
         SmallVector<Value> strides) {
        std::pair<Value, std::tuple<SmallVector<Value>, SmallVector<Value>,
                                    SmallVector<Value>>>
            entry;
        entry.first = memref;
        std::get<0>(entry.second) = offsets;
        std::get<1>(entry.second) = sizes;
        std::get<2>(entry.second) = strides;
        return entry;
      };
  auto pushMemrefEntryToVector = [](auto entry, auto &vector) {
    if (!isa<BaseMemRefType>(entry.first.getType()))
      return;
    vector.push_back(entry);
  };
  // Below is an incomplete list of common mlir ops that provide interfaces
  // allowing for separating read and write accesses in its operands.
  if (auto linalgop = dyn_cast<linalg::LinalgOp>(op)) {
    for (auto oper : linalgop.getDpsInputs())
      pushMemrefEntryToVector(getMemrefEntry(oper), operands);
  } else if (auto memref_copy = dyn_cast<memref::CopyOp>(op)) {
    pushMemrefEntryToVector(getMemrefEntry(memref_copy.getSource()), operands);
  } else if (auto memcpy = mlir::dyn_cast<xilinx::air::MemcpyInterface>(op)) {
    if (memcpy.getSrcMemref())
      pushMemrefEntryToVector(getMemrefAndAccessPatternEntry(
                                  memcpy.getSrcMemref(), memcpy.getSrcOffsets(),
                                  memcpy.getSrcSizes(), memcpy.getSrcStrides()),
                              operands);
  } else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
    // memref.load reads from the memref
    pushMemrefEntryToVector(getMemrefEntry(loadOp.getMemRef()), operands);
  } else if (isa<memref::StoreOp>(op)) {
    // memref.store writes to the memref -- no read of the memref itself
  } else { // If unknown op, then assume all operands are read.
    for (auto oper : op->getOperands())
      pushMemrefEntryToVector(getMemrefEntry(oper), operands);
  }

  // if operand is defined by a memref reshape op
  // TODO: fix me
  for (auto &oper : operands) {
    if (isa_and_present<memref::ReshapeOp, memref::ExpandShapeOp,
                        memref::CollapseShapeOp>(oper.first.getDefiningOp())) {
      oper.first = oper.first.getDefiningOp()->getOperand(0);
    }
  }
  return operands;
}

// Get memref operands which are write accessed by op. Each entry has the
// following format: pair<memref, tuple<offsets, sizes, strides>>.
FailureOr<SmallVector<
    std::pair<Value, std::tuple<SmallVector<Value>, SmallVector<Value>,
                                SmallVector<Value>>>>>
getAllWriteAccessedMemrefOperandsFromOp(Operation *op) {
  SmallVector<
      std::pair<Value, std::tuple<SmallVector<Value>, SmallVector<Value>,
                                  SmallVector<Value>>>>
      operands;
  if (!op)
    return failure();
  auto getMemrefEntry = [](Value memref) {
    std::pair<Value, std::tuple<SmallVector<Value>, SmallVector<Value>,
                                SmallVector<Value>>>
        entry;
    entry.first = memref;
    return entry;
  };
  auto getMemrefAndAccessPatternEntry =
      [](Value memref, SmallVector<Value> offsets, SmallVector<Value> sizes,
         SmallVector<Value> strides) {
        std::pair<Value, std::tuple<SmallVector<Value>, SmallVector<Value>,
                                    SmallVector<Value>>>
            entry;
        entry.first = memref;
        std::get<0>(entry.second) = offsets;
        std::get<1>(entry.second) = sizes;
        std::get<2>(entry.second) = strides;
        return entry;
      };
  auto pushMemrefEntryToVector = [](auto entry, auto &vector) {
    if (!isa<BaseMemRefType>(entry.first.getType()))
      return;
    vector.push_back(entry);
  };
  // Below is an incomplete list of common mlir ops that provide interfaces
  // allowing for separating read and write accesses in its operands.
  if (auto linalgop = dyn_cast<linalg::LinalgOp>(op)) {
    for (auto oper :
         llvm::concat<Value>(linalgop.getDpsInits(), linalgop->getResults()))
      pushMemrefEntryToVector(getMemrefEntry(oper), operands);
  } else if (auto memref_copy = dyn_cast<memref::CopyOp>(op)) {
    pushMemrefEntryToVector(getMemrefEntry(memref_copy.getTarget()), operands);
  } else if (auto memcpy = mlir::dyn_cast<xilinx::air::MemcpyInterface>(op)) {
    if (memcpy.getDstMemref())
      pushMemrefEntryToVector(getMemrefAndAccessPatternEntry(
                                  memcpy.getDstMemref(), memcpy.getDstOffsets(),
                                  memcpy.getDstSizes(), memcpy.getDstStrides()),
                              operands);
  } else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
    // memref.store writes to the memref destination only
    pushMemrefEntryToVector(getMemrefEntry(storeOp.getMemRef()), operands);
  } else if (isa<memref::LoadOp>(op)) {
    // memref.load reads from the memref -- no write access
  } else { // If unknown op, then assume all operands and results are written
           // to.
    for (auto oper : llvm::concat<Value>(op->getOperands(), op->getResults()))
      pushMemrefEntryToVector(getMemrefEntry(oper), operands);
  }
  return operands;
}

// Get index or scalar operands accessed by op.
FailureOr<SmallVector<Value>>
getAllAccessedScalarOperandsFromOp(Operation *op) {
  SmallVector<Value> operands;
  if (!op)
    return failure();
  for (Value oper : op->getOperands()) {
    Type type = oper.getType();
    if (isa<IndexType>(type) || isa<IntegerType>(type) || isa<FloatType>(type))
      operands.push_back(oper);
  }
  return operands;
}

//===----------------------------------------------------------------------===//
// Dependency graph
//===----------------------------------------------------------------------===//

// Recursively parse all dependency edges in the given graph and its subgraphs.
// This function walks the hierarchy of the dependencyGraph and applies
// parseDependencyEdgesInGraph to each level.
void dependencyCanonicalizer::parseAllDependencyEdges(
    dependencyGraph &graph, dependencyContext &dep_ctx) {
  parseDependencyEdgesInGraph(graph.g, dep_ctx);
  for (auto &subgraph : graph.subgraphs) {
    parseAllDependencyEdges(subgraph, dep_ctx);
  }
}

// Recursively connect terminator nodes to all leaf operations in the graph
// and its subgraphs. This ensures that each command graph is properly
// terminated.
void dependencyCanonicalizer::connectAllTerminators(dependencyGraph &graph) {
  connectTerminatorInGraph(graph.g);
  for (auto &subgraph : graph.subgraphs) {
    connectAllTerminators(subgraph);
  }
}

// Recursively connect the start node and set up pointer relationships between
// each graph and its associated hierarchy operation and terminator.
// If a parent graph is provided, link the child's terminator to the parent's
// graph.
void dependencyCanonicalizer::connectAndUpdateGraphPointers(
    dependencyGraph &graph, dependencyGraph *parent) {
  connectStartNodeInCommandGraph(graph);
  updatePointerFromGraphToHierarchyTerminator(graph);

  if (parent)
    updatePointerFromHierarchyTerminatorToGraph(*parent, graph);

  updatePointerFromHierarchyOpToGraph(graph);

  for (auto &subgraph : graph.subgraphs) {
    connectAndUpdateGraphPointers(subgraph, &graph);
  }
}

void dependencyCanonicalizer::parseCommandGraphs(func::FuncOp &toplevel,
                                                 dependencyGraph &global_graph,
                                                 dependencyContext &dep_ctx,
                                                 std::string granularity,
                                                 bool dump_dot,
                                                 std::string dump_dir) {
  // Graph parsing granularity. Tuple format: <expandLaunch, expandSegment,
  // expandHerd, expandCore>
  graphGranularityProperties expandHier = {true, true, true, false};
  if (granularity == "herd") {
    std::get<2>(expandHier) = true;
    std::get<3>(expandHier) = false;
  } else if (granularity == "core") {
    std::get<2>(expandHier) = true;
    std::get<3>(expandHier) = true;
  } else {
    toplevel->emitOpError("unknown graph parsing granularity");
    return;
  }

  // Create vertices for graphs
  // Build up host graph
  toplevel.walk([&](Operation *op) {
    if (!op->getParentOfType<air::HierarchyInterface>()) {
      addVertexFromOpImpls(op, &global_graph, dep_ctx);
      if (auto launch = dyn_cast<air::LaunchOp>(op)) {
        addVerticesInLaunch(global_graph.subgraphs, launch, dep_ctx,
                            expandHier);
      } else if (dyn_cast<air::SegmentOp>(op) &&
                 (!op->getParentOfType<air::LaunchOp>())) {
        auto segment = dyn_cast<air::SegmentOp>(op);
        addVerticesInSegment(global_graph.subgraphs, segment, dep_ctx,
                             expandHier);
      } else if (dyn_cast<air::HerdOp>(op) &&
                 (!op->getParentOfType<air::LaunchOp>()) &&
                 (!op->getParentOfType<air::SegmentOp>())) {
        auto herd = dyn_cast<air::HerdOp>(op);
        addVerticesInHerd(global_graph.subgraphs, herd, dep_ctx, expandHier);
      }
    }
  });

  // Adds edges between async ops.
  parseAllDependencyEdges(global_graph, dep_ctx);

  // Connect leaf vertices to launch, segment and herd terminators.
  for (auto &G_l : global_graph.subgraphs) {
    connectAllTerminators(G_l);
  }

  // Connect the start node per graph as graph inception point;
  // update pointer from graph to air.hierarchy op terminators.
  connectStartNodeInCommandGraph(global_graph);
  updatePointerFromGraphToHierarchyTerminator(global_graph);
  updatePointerFromHierarchyOpToGraph(global_graph);

  for (auto &launchGraph : global_graph.subgraphs) {
    connectAndUpdateGraphPointers(launchGraph, &global_graph);
  }
}

void dependencyCanonicalizer::addVerticesInHerd(
    std::vector<dependencyGraph> &herd_subgraphs, air::HerdOp herd,
    dependencyContext &dep_ctx, graphGranularityProperties expandHier) {
  // Build up herd graph
  bool showCores = std::get<3>(expandHier);
  if (showCores) {
    auto hier = dyn_cast<air::HierarchyInterface>(herd.getOperation());
    for (unsigned i = 0; i < getTripCountInHierarchyOp(hier); i++) {
      herd_subgraphs.push_back(dependencyGraph(herd.getOperation(), true));
      dependencyGraph *current_herd_graph = &(herd_subgraphs.back());

      // Core id
      auto current_position = getPositionFromIterator(i, herd);
      auto &position_ref = current_herd_graph->position;
      // auto check_prod = 1;
      for (auto id : current_position) {
        position_ref.push_back(id);
      }
      // Write core position at "start" node
      std::string position_str = "core position: ";
      position_str += toPositionString(current_position);
      current_herd_graph->g[current_herd_graph->start_vertex]
          .detailed_description = position_str;

      herd.walk([&](Operation *herd_childop) {
        if (!dyn_cast<air::HerdOp>(herd_childop)) {
          addVertexFromOpImpls(herd_childop, current_herd_graph, dep_ctx);
        }
      });
    }
  } else {
    herd_subgraphs.push_back(dependencyGraph(herd.getOperation(), true));
    dependencyGraph *current_herd_graph = &(herd_subgraphs.back());

    herd.walk([&](Operation *herd_childop) {
      if (!dyn_cast<air::HerdOp>(herd_childop)) {
        addVertexFromOpImpls(herd_childop, current_herd_graph, dep_ctx);
      }
    });
  }
}

void dependencyCanonicalizer::addVerticesInSegment(
    std::vector<dependencyGraph> &part_subgraphs, air::SegmentOp segment,
    dependencyContext &dep_ctx, graphGranularityProperties expandHier) {
  // Build up segment graph
  part_subgraphs.push_back(dependencyGraph(segment.getOperation(), true));
  dependencyGraph *current_part_graph = &(part_subgraphs.back());

  segment.walk([&](Operation *part_childop) {
    if (!part_childop->getParentOfType<air::HerdOp>() &&
        !dyn_cast<air::SegmentOp>(part_childop)) {
      addVertexFromOpImpls(part_childop, current_part_graph, dep_ctx);
      if (auto herd = dyn_cast<air::HerdOp>(part_childop)) {
        addVerticesInHerd(current_part_graph->subgraphs, herd, dep_ctx,
                          expandHier);
      }
    }
  });
}

void dependencyCanonicalizer::addVerticesInLaunch(
    std::vector<dependencyGraph> &launch_subgraphs, air::LaunchOp launch,
    dependencyContext &dep_ctx, graphGranularityProperties expandHier) {
  // Build up launch graph
  launch_subgraphs.push_back(dependencyGraph(launch.getOperation(), true));
  dependencyGraph *current_launch_graph = &(launch_subgraphs.back());

  launch.walk([&](Operation *launch_childop) {
    if (!launch_childop->getParentOfType<air::SegmentOp>() &&
        !launch_childop->getParentOfType<air::HerdOp>() &&
        !dyn_cast<air::LaunchOp>(launch_childop)) {
      addVertexFromOpImpls(launch_childop, current_launch_graph, dep_ctx);
      if (auto segment = dyn_cast<air::SegmentOp>(launch_childop)) {
        addVerticesInSegment(current_launch_graph->subgraphs, segment, dep_ctx,
                             expandHier);
      } else if (dyn_cast<air::HerdOp>(launch_childop) &&
                 (!launch_childop->getParentOfType<air::SegmentOp>())) {
        auto herd = dyn_cast<air::HerdOp>(launch_childop);
        addVerticesInHerd(current_launch_graph->subgraphs, herd, dep_ctx,
                          expandHier);
      }
    }
  });
}

Graph::VertexId
dependencyCanonicalizer::addVertexFromOpImpls(Operation *op, dependencyGraph *G,
                                              dependencyContext &dep_ctx) {
  if (auto dma_op = mlir::dyn_cast<xilinx::air::DmaMemcpyNdOp>(op)) {
    return addVertexFromDmaOp(dma_op, G, dep_ctx);
  } else if (auto channel_op =
                 mlir::dyn_cast<xilinx::air::ChannelInterface>(op)) {
    return addVertexFromChannelOp(channel_op, G, dep_ctx);
  } else if (auto execute_op = dyn_cast<xilinx::air::ExecuteOp>(op)) {
    return addVertexFromExecuteOp(execute_op, G, dep_ctx);
  } else if (auto wa_op = dyn_cast<xilinx::air::WaitAllOp>(op)) {
    return addVertexFromWaitAllOp(wa_op, G, dep_ctx);
  } else if (auto forop = dyn_cast<scf::ForOp>(op)) {
    return addVertexFromOp(op, dep_ctx.ForOpID, "for_loop", "ScfForOp",
                           graphNodeProperties("control"), G, dep_ctx);
  } else if (auto parallelop = dyn_cast<scf::ParallelOp>(op)) {
    return addVertexFromOp(op, dep_ctx.ParallelOpID, "parallel_loop",
                           "ScfParallelOp", graphNodeProperties("control"), G,
                           dep_ctx);
  } else if (auto hier_op =
                 mlir::dyn_cast<xilinx::air::HierarchyInterface>(op)) {
    return addVertexFromHierarchyOp(hier_op, G, dep_ctx);
  } else if (auto reduce_op = dyn_cast<scf::ReduceOp>(op)) {
    return addVertexFromReduceOp(reduce_op, G, dep_ctx);
  } else if (op->mightHaveTrait<OpTrait::IsTerminator>()) {
    return addVertexFromTerminatorOp(op, G, dep_ctx);
  } else
    return 0;
}

// Create graph vertex from op
Graph::VertexId dependencyCanonicalizer::addVertexFromOp(
    Operation *op, uint64_t &id, std::string event_type, std::string event_name,
    graphNodeProperties properties, dependencyGraph *G,
    dependencyContext &dep_ctx, Operation *pointer_op) {
  op->setAttr("id", mlir::IntegerAttr::get(
                        mlir::IntegerType::get(op->getContext(), 32), ++id));
  auto v = G->g.addVertex();
  G->g[v].asyncEventName = event_name;
  G->g[v].asyncEventType = event_type;
  G->g[v].color = properties.color;
  G->g[v].shape = properties.shape;
  G->g[v].detailed_description = properties.detailed_description;
  G->g[v].operationId = id;
  if (pointer_op)
    G->g[v].op = pointer_op;
  else
    G->g[v].op = op;
  // Update op-to-graph mapping
  auto entry = make_pair(event_type, id);
  dep_ctx.op_to_v.insert(make_pair(entry, v));
  dep_ctx.op_to_g.insert(make_pair(entry, G));
  return v;
}

Graph::VertexId
dependencyCanonicalizer::addVertexFromDmaOp(xilinx::air::DmaMemcpyNdOp op,
                                            dependencyGraph *G,
                                            dependencyContext &dep_ctx) {
  if (dyn_cast<xilinx::air::DmaMemcpyNdOp>(op.getOperation())) {
    return addVertexFromOp(op, dep_ctx.DmaOpID, "dma", "DmaMemcpyNdOp",
                           graphNodeProperties("data"), G, dep_ctx);
  } else {
    op->emitOpError("unknown dma op");
    return 0;
  }
}

Graph::VertexId dependencyCanonicalizer::addVertexFromChannelOp(
    xilinx::air::ChannelInterface op, dependencyGraph *G,
    dependencyContext &dep_ctx) {
  if (auto channel_put =
          dyn_cast<xilinx::air::ChannelPutOp>(op.getOperation())) {
    std::string memorySpaceSrcStr =
        getMemorySpaceAsString(channel_put.getSrc());
    std::vector<air::ChannelGetOp> channel_gets =
        getTheOtherChannelOpThroughSymbol(channel_put);
    if (!channel_gets.size())
      op->emitOpError("found channel op not in pairs");
    std::string memorySpaceDstStr =
        getMemorySpaceAsString(channel_gets[0].getDst());
    std::string event_name = "ChannelPutOp@" + channel_put.getChanName().str() +
                             "(" + memorySpaceSrcStr + "-->" +
                             memorySpaceDstStr + ")";
    auto channel_op = getChannelDeclarationThroughSymbol(op);
    std::string detailed_description = "";
    if (channel_op->hasAttr("broadcast_shape")) {
      auto size = extractFromIntegerArrayAttr<int64_t>(channel_op.getSize());
      detailed_description += "(broadcast[";
      for (auto &s : size) {
        detailed_description += std::to_string(s);
        if (&s != &size.back())
          detailed_description += ",";
      }
      detailed_description += "]-->[";
      auto bsize = extractFromIntegerArrayAttr<int64_t>(
          channel_op->getAttrOfType<mlir::ArrayAttr>("broadcast_shape"));
      for (auto &s : bsize) {
        detailed_description += std::to_string(s);
        if (&s != &bsize.back())
          detailed_description += ",";
      }
      detailed_description += "])";
    }
    return addVertexFromOp(op, dep_ctx.ChannelOpID, "channel", event_name,
                           graphNodeProperties("data", detailed_description), G,
                           dep_ctx);
  } else if (auto channel_get =
                 dyn_cast<xilinx::air::ChannelGetOp>(op.getOperation())) {
    std::string memorySpaceDstStr =
        getMemorySpaceAsString(channel_get.getDst());
    std::vector<air::ChannelPutOp> channel_puts =
        getTheOtherChannelOpThroughSymbol(channel_get);
    if (!channel_puts.size())
      op->emitOpError("found channel op not in pairs");
    std::string memorySpaceSrcStr =
        getMemorySpaceAsString(channel_puts[0].getSrc());
    std::string event_name = "ChannelGetOp@" + channel_get.getChanName().str() +
                             "(" + memorySpaceDstStr + "<--" +
                             memorySpaceSrcStr + ")";
    auto channel_op = getChannelDeclarationThroughSymbol(op);
    std::string detailed_description = "";
    if (channel_op->hasAttr("broadcast_shape")) {
      auto size = extractFromIntegerArrayAttr<int64_t>(channel_op.getSize());
      detailed_description += "(broadcast[";
      for (auto &s : size) {
        detailed_description += std::to_string(s);
        if (&s != &size.back())
          detailed_description += ",";
      }
      detailed_description += "]-->[";
      auto bsize = extractFromIntegerArrayAttr<int64_t>(
          channel_op->getAttrOfType<mlir::ArrayAttr>("broadcast_shape"));
      for (auto &s : bsize) {
        detailed_description += std::to_string(s);
        if (&s != &bsize.back())
          detailed_description += ",";
      }
      detailed_description += "])";
    }
    return addVertexFromOp(op, dep_ctx.ChannelOpID, "channel", event_name,
                           graphNodeProperties("data", detailed_description), G,
                           dep_ctx);
  } else {
    op->emitOpError("unknown channel op");
    return 0;
  }
}

Graph::VertexId dependencyCanonicalizer::addVertexFromHierarchyOp(
    xilinx::air::HierarchyInterface op, dependencyGraph *G,
    dependencyContext &dep_ctx) {
  std::string detailed_description = "";
  auto nameAttr =
      op->getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName());
  // Annotate hierarchy op's symbolic name
  if (nameAttr)
    detailed_description += "(" + nameAttr.str() + ")";
  if (dyn_cast<xilinx::air::LaunchOp>(op.getOperation())) {
    return addVertexFromOp(
        op, dep_ctx.HierarchyOpID, "hierarchy", "LaunchOp",
        graphNodeProperties("hierarchy", detailed_description), G, dep_ctx);
  } else if (auto seg = dyn_cast<xilinx::air::SegmentOp>(op.getOperation())) {
    // Annotate with physical address, if already placed
    if (seg.getNumRows() && seg.getNumCols()) {
      detailed_description += "[" + std::to_string(*seg.getNumCols()) + ", " +
                              std::to_string(*seg.getNumRows()) + "]";
    }
    return addVertexFromOp(
        op, dep_ctx.HierarchyOpID, "hierarchy", "SegmentOp",
        graphNodeProperties("hierarchy", detailed_description), G, dep_ctx);
  } else if (auto herd = dyn_cast<xilinx::air::HerdOp>(op.getOperation())) {
    // Annotate with physical address, if already placed
    if (herd.getNumRows() && herd.getNumCols()) {
      detailed_description += "[" + std::to_string(herd.getNumCols()) + ", " +
                              std::to_string(herd.getNumRows()) + "]";
    }
    return addVertexFromOp(
        op, dep_ctx.HierarchyOpID, "hierarchy", "HerdOp",
        graphNodeProperties("hierarchy", detailed_description), G, dep_ctx);
  } else {
    op->emitOpError("unknown hierarchy op");
    return 0;
  }
}

Graph::VertexId dependencyCanonicalizer::addVertexFromTerminatorOp(
    Operation *op, dependencyGraph *G, dependencyContext &dep_ctx) {
  std::string detailed_description = "";
  if (dyn_cast<xilinx::air::LaunchTerminatorOp>(op)) {
    return addVertexFromOp(
        op, dep_ctx.TerminatorID, "hierarchy_terminator", "LaunchTerminator",
        graphNodeProperties("hierarchy", detailed_description), G, dep_ctx);
  } else if (dyn_cast<xilinx::air::SegmentTerminatorOp>(op)) {
    return addVertexFromOp(
        op, dep_ctx.TerminatorID, "hierarchy_terminator", "SegmentTerminator",
        graphNodeProperties("hierarchy", detailed_description), G, dep_ctx);
  } else if (dyn_cast<xilinx::air::HerdTerminatorOp>(op)) {
    // Annotate core id, if showing cores
    if (G->position.size()) {
      detailed_description += "core id: ";
      detailed_description += toPositionString(G->position);
    }
    return addVertexFromOp(
        op, dep_ctx.TerminatorID, "hierarchy_terminator", "HerdTerminator",
        graphNodeProperties("hierarchy", detailed_description), G, dep_ctx);
  } else if (isa<scf::YieldOp>(op)) {
    if (getScfParentOpFromYieldOp<scf::ParallelOp>(op)) {
      // Note: disabled parsing scf parallel yield op since it currently acts as
      // a no-op return addVertexFromOp(op, dep_ctx.TerminatorID, "terminator",
      //                        "ScfParallelYieldOp", "crimson", "box", G,
      //                        dep_ctx);
    } else if (getScfParentOpFromYieldOp<scf::ForOp>(op)) {
      return addVertexFromOp(
          op, dep_ctx.TerminatorID, "terminator", "ScfForYieldOp",
          graphNodeProperties("control", detailed_description), G, dep_ctx);
    }
  }
  return 0;
}

// Note: in the current scf parallel spec, reduce op takes the role of yielding
// the ssa value. Hence, here we parse reduce op as a terminator.
Graph::VertexId dependencyCanonicalizer::addVertexFromReduceOp(
    Operation *op, dependencyGraph *G, dependencyContext &dep_ctx) {
  return addVertexFromOp(op, dep_ctx.TerminatorID, "terminator", "ScfReduceOp",
                         graphNodeProperties("control"), G, dep_ctx);
}

Graph::VertexId dependencyCanonicalizer::addVertexFromExecuteOp(
    xilinx::air::ExecuteOp op, dependencyGraph *G, dependencyContext &dep_ctx) {
  int iter_count = 0;
  Graph::VertexId v_prev = 0;
  Graph::VertexId v = 0;
  Operation *pointer_op = op;
  int num_non_shape_alt_ops = 0;
  for (auto &child_op : op->getRegions().front().getOps()) {
    if (auto linalg_child_op = dyn_cast<linalg::LinalgOp>(child_op)) {
      std::string detailed_description = "";
      // Annotate linalg op's type
      if (auto broadcast_pattern = linalg_child_op->getAttrOfType<StringAttr>(
              "__internal_linalg_transform__")) {
        detailed_description += "(" + broadcast_pattern.str() + ")";
      } else {
        detailed_description += "(" + to_string(&child_op) + ")";
      }
      v = addVertexFromOp(&child_op, dep_ctx.ExecuteOpID, "execute", "LinalgOp",
                          graphNodeProperties("compute", detailed_description),
                          G, dep_ctx, pointer_op);
    } else if (auto alloc_child_op = dyn_cast<memref::AllocOp>(child_op)) {
      std::string detailed_description = "";
      // Annotate memref's memory space
      std::string memorySpaceStr =
          getMemorySpaceAsString(alloc_child_op.getMemref());
      auto ty =
          llvm::cast<BaseMemRefType>(alloc_child_op.getMemref().getType());
      detailed_description += "(" + memorySpaceStr + ", " +
                              std::to_string(getTensorVolume(ty)) + ", " +
                              getElementTypeAsString(ty) + ")";
      v = addVertexFromOp(&child_op, dep_ctx.ExecuteOpID, "execute", "AllocOp",
                          graphNodeProperties("compute", detailed_description),
                          G, dep_ctx, pointer_op);
    } else if (auto dealloc_child_op = dyn_cast<memref::DeallocOp>(child_op)) {
      std::string detailed_description = "";
      // Annotate memref's memory space
      std::string memorySpaceStr =
          getMemorySpaceAsString(dealloc_child_op.getMemref());
      auto ty =
          llvm::cast<BaseMemRefType>(dealloc_child_op.getMemref().getType());
      detailed_description += "(" + memorySpaceStr + ", " +
                              std::to_string(getTensorVolume(ty)) + ", " +
                              getElementTypeAsString(ty) + ")";
      v = addVertexFromOp(&child_op, dep_ctx.ExecuteOpID, "execute",
                          "DeallocOp",
                          graphNodeProperties("compute", detailed_description),
                          G, dep_ctx, pointer_op);
    } else if (dyn_cast<memref::CopyOp>(child_op)) {
      v = addVertexFromOp(&child_op, dep_ctx.ExecuteOpID, "execute", "CopyOp",
                          graphNodeProperties("data"), G, dep_ctx, pointer_op);
    } else if (dyn_cast<xilinx::air::ExecuteTerminatorOp>(child_op)) {
      v = addVertexFromOp(&child_op, dep_ctx.ExecuteOpID, "execute",
                          "ExecuteTerminatorOp", graphNodeProperties("compute"),
                          G, dep_ctx, pointer_op);
    } else if (isa_and_present<memref::ReshapeOp, memref::ExpandShapeOp,
                               memref::CollapseShapeOp>(child_op)) {
      v = addVertexFromOp(&child_op, dep_ctx.ExecuteOpID, "execute",
                          "ReshapeOp", graphNodeProperties("data"), G, dep_ctx,
                          pointer_op);
      num_non_shape_alt_ops--;
    } else {
      v = addVertexFromOp(
          &child_op, dep_ctx.ExecuteOpID, "execute", air::to_string(&child_op),
          graphNodeProperties("compute"), G, dep_ctx, pointer_op);
    }
    // Make connections within execute
    if (iter_count > 0)
      G->g.addEdge(v_prev, v);
    if (num_non_shape_alt_ops > 0)
      pointer_op = nullptr;
    v_prev = v;
    iter_count++;
    num_non_shape_alt_ops++;
  }
  return v;
}

Graph::VertexId dependencyCanonicalizer::addVertexFromWaitAllOp(
    xilinx::air::WaitAllOp op, dependencyGraph *G, dependencyContext &dep_ctx) {
  // Note: disabled parsing wait_all op inside of reduce op
  if (op->getParentOfType<scf::ReduceOp>())
    return 0;
  return addVertexFromOp(op, dep_ctx.WaitAllOpID, "wait_all", "WaitAllOp",
                         graphNodeProperties("control"), G, dep_ctx);
}

// Get type-id pair from op, which will be used to look up vertex in op_to_v
std::pair<std::string, unsigned>
dependencyCanonicalizer::getTypeIdPairFromOp(Operation *op) {
  std::pair<std::string, unsigned> output;
  std::string type = getOpTypeFromOpImpls(op);
  output.first = type;
  output.second = xilinx::air::getIdAttr(op);
  return output;
}

std::string dependencyCanonicalizer::getOpTypeFromOpImpls(Operation *op) {
  if (isa<air::DmaMemcpyNdOp>(op)) {
    return "dma";
  } else if (isa<air::ChannelInterface>(op)) {
    return "channel";
  } else if (isa<air::WaitAllOp>(op)) {
    return "wait_all";
  } else if (isa<xilinx::air::HierarchyInterface>(op)) {
    return "hierarchy";
  } else if (isa<scf::ForOp>(op)) {
    return "for_loop";
  } else if (isa<scf::ParallelOp>(op)) {
    return "parallel_loop";
  } else if (isa<xilinx::air::LaunchTerminatorOp>(op)) {
    return "hierarchy_terminator";
  } else if (isa<xilinx::air::SegmentTerminatorOp>(op)) {
    return "hierarchy_terminator";
  } else if (isa<xilinx::air::HerdTerminatorOp>(op)) {
    return "hierarchy_terminator";
  } else if (isa<scf::YieldOp>(op)) {
    return "terminator";
  } else if (isa<scf::ReduceOp>(op)) {
    return "terminator";
  } else {
    if (isa<xilinx::air::ExecuteOp>(op->getParentOp())) {
      return "execute";
    } else {
      op->emitOpError("unknown op type");
      return "";
    }
  }
}

// Get vertex descriptor from op
// "front_or_back": if op is an air.execute, then "front" returns the first op
// in region, while "back" returns the terminator in region.
std::pair<Graph::VertexId, dependencyGraph *>
dependencyCanonicalizer::getVertexFromOp(Operation *op,
                                         dependencyContext dep_ctx,
                                         std::string front_or_back) {
  std::pair<Graph::VertexId, dependencyGraph *> output;
  if (auto execute_op = dyn_cast<xilinx::air::ExecuteOp>(op)) {
    if (front_or_back == "front") {
      auto execute_front_op = &execute_op.getChildOps().front();
      std::pair<std::string, unsigned> entry_pair =
          getTypeIdPairFromOp(execute_front_op);
      output.first = dep_ctx.op_to_v[entry_pair];
      output.second = dep_ctx.op_to_g[entry_pair];
    } else if (front_or_back == "back") {
      auto execute_end_op =
          execute_op.getRegion().getBlocks().front().getTerminator();
      std::pair<std::string, unsigned> entry_pair =
          getTypeIdPairFromOp(execute_end_op);
      output.first = dep_ctx.op_to_v[entry_pair];
      output.second = dep_ctx.op_to_g[entry_pair];
    } else {
      op->emitOpError(
          "unknown string operand (only accepts 'front' or 'back')");
    }
  } else {
    std::pair<std::string, unsigned> entry_pair = getTypeIdPairFromOp(op);
    output.first = dep_ctx.op_to_v[entry_pair];
    output.second = dep_ctx.op_to_g[entry_pair];
  }
  return output;
}

// Trace dependency of every op in a graph
void dependencyCanonicalizer::parseDependencyEdgesInGraph(
    Graph &g, dependencyContext dep_ctx) {
  auto vp = g.getVertices();
  for (auto vit : vp) {
    auto op = g[vit].op;
    if (!op)
      continue;
    connectOpToItsDepListImpls(op, g, dep_ctx);
  }
}

void dependencyCanonicalizer::connectOpToItsDepListImpls(
    Operation *op, Graph &g, dependencyContext dep_ctx) {
  SmallVector<Value, 1> dep_list;
  // air.asyncopinterface
  if (auto async_op = mlir::dyn_cast<xilinx::air::AsyncOpInterface>(op)) {
    for (auto dep_token : async_op.getAsyncDependencies()) {
      dep_list.push_back(dep_token);
    }
  }
  // scf.for
  else if (auto forop = dyn_cast<scf::ForOp>(op)) {
    for (auto iter_operand : forop.getInitArgs()) {
      dep_list.push_back(iter_operand);
    }
  }
  // scf.parallel
  else if (auto parallelop = dyn_cast<scf::ParallelOp>(op)) {
    for (auto operand : parallelop->getOperands()) {
      dep_list.push_back(operand);
    }
  }
  // scf.yield
  else if (auto yieldop = dyn_cast<scf::YieldOp>(op)) {
    for (auto operand : yieldop->getOperands()) {
      dep_list.push_back(operand);
    }
  }
  // scf.reduce
  else if (auto reduceop = dyn_cast<scf::ReduceOp>(op)) {
    for (auto operand : reduceop->getOperands()) {
      dep_list.push_back(operand);
    }
  }
  if (dep_list.size()) {
    connectOpToItsDepList(op, dep_list, g, dep_ctx);
  }
}

// Connect an async op to ops in its dependency list
void dependencyCanonicalizer::connectOpToItsDepList(
    Operation *op, SmallVector<Value, 1> dep_list, Graph &g,
    dependencyContext dep_ctx) {
  auto dst_v = getVertexFromOp(op, dep_ctx, "front").first;
  if (dep_list.size()) {
    for (auto dep_token : dep_list) {
      auto src_vector = traceOpFromToken(op, dep_token);
      if (src_vector.size()) {
        for (auto src_op : src_vector) {
          auto src_v = getVertexFromOp(src_op, dep_ctx, "back").first;
          if (!g.hasEdge(src_v, dst_v)) {
            g.addEdge(src_v, dst_v);
          }
        }
      }
    }
  }
}

// Trace op from a token in dependency list
std::vector<Operation *>
dependencyCanonicalizer::traceOpFromToken(Operation *op, Value dep_token) {
  std::vector<Operation *> output;
  // If dependency token is the init arg of an scf parallel loop
  // Note: checking for scf parallel first here, because its init_val is not its
  // block argument
  if (auto parallelop = getParallelRegionInitValsOwner(op, dep_token)) {
    output.push_back(parallelop);
    return output;
  }
  // Else if dependency token is the iter arg of an scf for loop
  else if (auto forop = getForRegionIterArgsOwner(dep_token)) {
    output.push_back(forop);
    return output;
  }
  // Else if dependency token originates from async op
  else if (dep_token.getDefiningOp() &&
           mlir::dyn_cast<xilinx::air::AsyncOpInterface>(
               dep_token.getDefiningOp())) {
    output.push_back(dep_token.getDefiningOp());
    return output;
  }
  // Else if dependency token is yielded from scf.for
  else if (dep_token.getDefiningOp() &&
           dyn_cast<scf::ForOp>(dep_token.getDefiningOp())) {
    auto forop = dyn_cast<scf::ForOp>(dep_token.getDefiningOp());
    auto forop_terminator = forop.getBody()->getTerminator();
    output.push_back(forop_terminator);
    return output;
  }
  // Else if dependency token is yielded from scf.parallel
  else if (dep_token.getDefiningOp() &&
           dyn_cast<scf::ParallelOp>(dep_token.getDefiningOp())) {
    auto parallelop = dyn_cast<scf::ParallelOp>(dep_token.getDefiningOp());
    for (auto parallelop_reduceop : parallelop.getOps<scf::ReduceOp>()) {
      output.push_back(parallelop_reduceop);
      return output;
    }
  }
  // Else if dependency token is from affine if (joint token from multiple ops)
  else if (dep_token.getDefiningOp() &&
           dyn_cast<affine::AffineIfOp>(dep_token.getDefiningOp())) {
    auto aifop = dyn_cast<affine::AffineIfOp>(dep_token.getDefiningOp());
    // The first then block
    auto then_terminator = aifop.getThenBlock()->getTerminator();
    for (auto operand : then_terminator->getOperands()) {
      if (auto op = operand.getDefiningOp()) {
        output.push_back(op);
      }
    }
    // Recursion
    affine::AffineIfOp current_aif = aifop;
    while (getAffineIfInBlock(current_aif.getElseBlock())) {
      auto child_aif_op = getAffineIfInBlock(current_aif.getElseBlock());
      auto child_aif_terminator = child_aif_op.getThenBlock()->getTerminator();
      for (auto operand : child_aif_terminator->getOperands()) {
        if (auto op = operand.getDefiningOp()) {
          output.push_back(op);
        }
      }
      current_aif = child_aif_op;
    }
    // The last else block
    auto last_else_terminator = current_aif.getElseBlock()->getTerminator();
    for (auto operand : last_else_terminator->getOperands()) {
      if (auto op = operand.getDefiningOp()) {
        output.push_back(op);
      }
    }
    return output;
  }
  return output;
}

// Connects launch, segment and herd terminators
void dependencyCanonicalizer::connectTerminatorInGraph(Graph &g) {
  Graph::VertexId terminator_v = 0;
  for (auto vit : g.getVertices()) {
    if (g[vit].asyncEventType == "hierarchy_terminator") {
      terminator_v = vit;
    }
  }
  if (terminator_v == 0)
    return;

  for (auto vit : g.getVertices()) {
    if ((terminator_v != vit) && !g.outDegree(vit) &&
        (g[vit].asyncEventType != "start")) {
      g.addEdge(vit, terminator_v);
    }
  }
}

// Create start node for graph
void dependencyCanonicalizer::connectStartNodeInCommandGraph(
    dependencyGraph &G) {
  auto v = G.start_vertex;
  auto vp = G.g.getVertices();
  for (auto vit : vp) {
    if ((v != vit) && !G.g.inDegree(vit)) {
      G.g.addEdge(v, vit);
    }
  }
}

// Adds pointer from command graph to launch, segment and herd terminators
void dependencyCanonicalizer::updatePointerFromGraphToHierarchyTerminator(
    dependencyGraph &G) {
  auto vp = G.g.getVertices();
  for (auto v : vp) {
    if (G.g[v].asyncEventType == "hierarchy_terminator") {
      G.terminator_vertex = v;
      return;
    }
  }
}

// Adds pointer from hierarchy terminator to parent command graph
void dependencyCanonicalizer::updatePointerFromHierarchyTerminatorToGraph(
    dependencyGraph &G, dependencyGraph &subG) {
  for (auto v : subG.g.getVertices()) {
    if (subG.g[v].asyncEventType == "hierarchy_terminator") {
      subG.g[v].nextDependencyGraphs.push_back(&G);
      return;
    }
  }
}

// Adds pointer from hierarchy op to sub command graph
void dependencyCanonicalizer::updatePointerFromHierarchyOpToGraph(
    dependencyGraph &G) {
  unsigned idx = 0;
  std::vector<Graph::VertexId> hier_vs;
  auto vp = G.g.getVertices();
  for (auto v : vp) {
    if (G.g[v].asyncEventType == "hierarchy") {
      hier_vs.push_back(v);
    }
  }

  for (auto v : hier_vs) {

    // If expand cores in herd
    if (G.subgraphs[idx].position.size()) {
      if (!isa<air::HerdOp>(G.g[v].op))
        G.g[v].op->emitOpError("found non-herd op with core id");
      auto hier = dyn_cast<air::HierarchyInterface>(G.g[v].op);
      for (unsigned i = 0; i < getTripCountInHierarchyOp(hier); i++) {
        G.g[v].nextDependencyGraphs.push_back(&(G.subgraphs[idx]));
        if (G.g[v].op != G.subgraphs[idx].hierarchyOp)
          G.g[v].op->emitOpError("mismatch between graph and hierarchy op");
        idx++;
      }
    } else {
      G.g[v].nextDependencyGraphs.push_back(&(G.subgraphs[idx]));
      if (G.g[v].op != G.subgraphs[idx].hierarchyOp)
        G.g[v].op->emitOpError("mismatch between graph and hierarchy op");
      idx++;
    }
  }
}

// Perform transitive reduction to canonicalize the dependency graph.

// Recursive canonicalization.
void dependencyCanonicalizer::canonicalizeRecursive(const dependencyGraph &src,
                                                    dependencyGraph &dst) {

  // Check for subgraph count.
  if (src.subgraphs.size() != dst.subgraphs.size())
    src.hierarchyOp->emitOpError("graph tree size mismatch");
  transitiveReductionImpl(src.g, dst.g);

  // Recurse into subgraphs.
  for (unsigned i = 0; i < src.subgraphs.size(); i++) {
    canonicalizeRecursive(src.subgraphs[i], dst.subgraphs[i]);
  }
}

// Recursively copy the structure (hierarchyOp) of the dependencyGraph.
void dependencyCanonicalizer::buildEmptyGraphStructure(
    const dependencyGraph &src, dependencyGraph &dst) {
  for (const auto &childSrc : src.subgraphs) {
    dst.subgraphs.push_back(dependencyGraph(childSrc.hierarchyOp));
    buildEmptyGraphStructure(childSrc, dst.subgraphs.back());
  }
}

// Entry point.
void dependencyCanonicalizer::canonicalizeGraphs(
    const dependencyGraph &global_graph, dependencyGraph &tr_graph) {

  // Construct empty post-canonicalization dependency graph.
  buildEmptyGraphStructure(global_graph, tr_graph);
  // Perform canonicalization recursively.
  canonicalizeRecursive(global_graph, tr_graph);
}

void dependencyCanonicalizer::transitiveReductionImpl(
    const Graph &asyncExecuteGraph, Graph &asyncExecuteGraphTR) {
  asyncExecuteGraphTR = asyncExecuteGraph;
  asyncExecuteGraphTR.applyTransitiveReduction();
}

// Recursively purge dependency list from the graph and its subgraphs.
void dependencyCanonicalizer::purgeAllDependencyLists(dependencyGraph &graph) {
  purgeAIRDepList(graph);
  for (auto &subgraph : graph.subgraphs) {
    purgeAllDependencyLists(subgraph);
  }
}

// Recursively fill dependency list using the transitive-reduced graph.
void dependencyCanonicalizer::fillAllDependencyListsFromTR(
    dependencyGraph &graph) {
  fillAIRDepListUsingGraphTR(graph);
  for (auto &subgraph : graph.subgraphs) {
    fillAllDependencyListsFromTR(subgraph);
  }
}

// Update dependency list based on transformed graph.
void dependencyCanonicalizer::updateDepList(func::FuncOp func,
                                            dependencyGraph &global_graph) {

  // Purge dependency list.
  purgeAllDependencyLists(global_graph);

  // Rewrite dependency list.
  fillAllDependencyListsFromTR(global_graph);

  // Cleanup op ids. Only leave dma, execute and hierarchy ids.
  func.walk([&](Operation *op) {
    if (isa<air::DmaMemcpyNdOp>(op)) {
    } else if (isa<air::ChannelInterface>(op)) {
    } else if (isa<air::HierarchyInterface>(op)) {
    } else {
      op->removeAttr("id");
    }
  });
}

void dependencyCanonicalizer::purgeAIRDepList(dependencyGraph &graph) {
  auto vp = graph.g.getVertices();
  for (auto dstTRVertex : vp) {
    auto op = graph.g[dstTRVertex].op;
    if (!op)
      continue;
    auto async_op = mlir::dyn_cast<xilinx::air::AsyncOpInterface>(op);
    if (!async_op)
      continue;
    clearAsyncDependenciesOfAsyncOp(async_op);
  }
}

void dependencyCanonicalizer::fillAIRDepListUsingGraphTR(
    dependencyGraph &graph) {
  auto vp = graph.g.getVertices();
  for (auto dstTRVertex : vp) {
    auto op = graph.g[dstTRVertex].op;
    if (!op)
      continue;
    auto async_op = mlir::dyn_cast<xilinx::air::AsyncOpInterface>(op);
    if (!async_op)
      continue;
    auto incoming_deps = graph.g.inverseAdjacentVertices(dstTRVertex);
    for (auto TRVertex : incoming_deps) {
      auto src_op = graph.g[TRVertex].op;
      if (!src_op)
        continue;
      if (op == src_op)
        continue; // Avoid dep to itself
      if (graph.g[TRVertex].asyncEventType == "for_loop") {
        auto value = getLoopCarriedTokenFromScfOp(dyn_cast<scf::ForOp>(src_op),
                                                  "argument");
        if (value)
          async_op.addAsyncDependency(value);
      } else if (graph.g[TRVertex].asyncEventType == "parallel_loop") {
        auto value =
            getLoopCarriedTokenFromScfOp(dyn_cast<scf::ParallelOp>(src_op));
        if (value)
          async_op.addAsyncDependency(value);
      } else if (graph.g[TRVertex].asyncEventType == "terminator") {
        auto parent_op = src_op->getParentOp();
        auto value = getAsyncTokenFromOp(parent_op);
        if (value)
          async_op.addAsyncDependency(value);
      } else if (auto async_src_op =
                     dyn_cast<xilinx::air::AsyncOpInterface>(src_op)) {
        // Elevate src token if src op is in affine if
        while (auto parent_affine_if_op =
                   dyn_cast<affine::AffineIfOp>(src_op->getParentOp())) {
          DominanceInfo domInfo(src_op);
          if (domInfo.properlyDominates(src_op, async_op)) {
            // SSA dominance check passed. Jump to adding dependency edge.
            break;
          }
          src_op = parent_affine_if_op.getOperation();
        }
        if (src_op->isAncestor(async_op))
          continue; // Avoid depending on its ancestor.
        async_op.addAsyncDependency(src_op->getResult(0));
      }
    }
  }
}

// Get number of cores in herd
unsigned dependencyCanonicalizer::getTripCountInHierarchyOp(
    air::HierarchyInterface hier) {
  auto sizes = hier.getSizeOperands();
  unsigned output = 1;
  for (unsigned i = 0; i < sizes.size(); i++) {
    output *= sizes[i].getDefiningOp<arith::ConstantIndexOp>().value();
  }
  return output;
}

// Write position of each core to dependency graph, if showing cores
std::vector<unsigned>
dependencyCanonicalizer::getPositionFromIterator(unsigned iter,
                                                 air::HerdOp herd) {
  auto herd_size = herd.getSizeOperands();
  auto herd_size_uint = convertVecOfConstIndexToVecOfUInt(herd_size);
  if (herd_size_uint.empty())
    herd->emitOpError("hierarchy op has empty iteration size");
  return getMDVectorFromIterator(herd_size_uint, iter);
}

// Write position of each core to dependency graph, if showing cores
unsigned
dependencyCanonicalizer::getIteratorFromPosition(std::vector<unsigned> position,
                                                 Operation *hier_op) {
  auto herd = dyn_cast<air::HerdOp>(hier_op);
  if (!herd) {
    return 0;
  }
  if (!position.size()) {
    return 0;
  }
  auto herd_size = herd.getSizeOperands();
  auto herd_size_uint = convertVecOfConstIndexToVecOfUInt(herd_size);
  if (herd_size_uint.empty())
    hier_op->emitOpError("hierarchy op has empty iteration size");
  return getIteratorFromMDVector(herd_size_uint, position);
}

// Write position in string
std::string
dependencyCanonicalizer::toPositionString(std::vector<unsigned> position) {
  std::string output = "[";
  for (unsigned i = 0; i < position.size(); i++) {
    output += std::to_string(position[i]);
    if (i < position.size() - 1) {
      output += ",";
    }
  }
  output += "]";
  return output;
}

// Re-trace ops which depend on air.hierarchy
LogicalResult
dependencyCanonicalizer::redoDepTraceIfDepOnHier(func::FuncOp func) {
  air::dependencyTracer depTracer;
  SmallVector<air::ExecuteOp> exec_ops;
  func.walk([&](air::ExecuteOp exec_op) { exec_ops.push_back(exec_op); });
  for (auto exec_op : exec_ops) {
    // Get partial memref reads/writes
    SmallVector<air::partialMemref, 1> sink_op_memref_reads;
    SmallVector<air::partialMemref, 1> sink_op_memref_writes;
    SmallVector<Value, 1> sink_op_scalar_ins;
    SmallVector<Value, 1> sink_op_scalar_outs;
    // Pick the first op that is not a shape altering op.
    Operation *child_op = nullptr;
    for (auto &op : exec_op.getRegion().front().getOperations()) {
      child_op = &op;
      if (!isa_and_present<memref::ReshapeOp, memref::ExpandShapeOp,
                           memref::CollapseShapeOp>(child_op)) {
        break;
      }
    }
    depTracer.getPartialMemrefFromOp(child_op, sink_op_memref_reads,
                                     sink_op_memref_writes, sink_op_scalar_ins,
                                     sink_op_scalar_outs);
    if (sink_op_memref_reads.empty() && sink_op_memref_writes.empty()) {
      continue;
    }
    // Update dependency to air.hierarchy ops
    bool hasDepToHier = false;
    auto dep_list = exec_op.getAsyncDependencies();
    for (auto dep : dep_list) {
      if (dep.getDefiningOp() &&
          isa<air::HierarchyInterface>(dep.getDefiningOp())) {
        eraseAsyncDependencyFromAsyncOp(exec_op, dep);
        hasDepToHier = true;
      }
    }
    if (hasDepToHier) {
      // Trace dependency of op again
      if (failed(
              depTracer.template traceDependencyFromOp<air::AsyncOpInterface>(
                  sink_op_memref_reads, exec_op, "RAW")))
        return failure();
      if (failed(
              depTracer.template traceDependencyFromOp<air::AsyncOpInterface>(
                  sink_op_memref_writes, exec_op, "WAW/WAR")))
        return failure();
      // Detect tile index deps
      depTracer.traceTileIndices(sink_op_memref_reads, sink_op_memref_writes,
                                 sink_op_scalar_ins, sink_op_scalar_outs,
                                 exec_op);
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Dependency tracing
//===----------------------------------------------------------------------===//

// Trace operand's uses at current scope
void dependencyTracer::pushDepsAtCurrentScope(mlir::Value operand,
                                              air::AsyncOpInterface op, char rw,
                                              partialMemref *tile) {
  if (!llvm::isa<BaseMemRefType>(operand.getType()))
    op->emitOpError("operand being traced is not a memref");
  for (auto &u : operand.getUses()) {
    // If used in MemcpyInterface Op
    if (auto memcpy = dyn_cast<xilinx::air::MemcpyInterface>(u.getOwner())) {
      partialMemref memcpy_src, memcpy_dst;
      if (memcpy.getSrcMemref()) {
        memcpy_src =
            partialMemref(memcpy.getSrcMemref(), memcpy.getSrcOffsets(),
                          memcpy.getSrcSizes(), memcpy.getSrcStrides());
      }
      if (memcpy.getDstMemref()) {
        memcpy_dst =
            partialMemref(memcpy.getDstMemref(), memcpy.getDstOffsets(),
                          memcpy.getDstSizes(), memcpy.getDstStrides());
      }

      if (rw == 'r') {
        if (u.is(memcpy.getSrcMemref())) {
          if (tile == nullptr) {
            addDependencyBetweenOps(memcpy.getOperation(), op.getOperation());
          } else if (areOverlappingPartialMemrefs(tile, &memcpy_src)) {
            addDependencyBetweenOps(memcpy.getOperation(), op.getOperation());
          }
        }
      } else if (rw == 'w') {
        if (u.is(memcpy.getDstMemref())) {
          if (tile == nullptr) {
            addDependencyBetweenOps(memcpy.getOperation(), op.getOperation());
          } else if (areOverlappingPartialMemrefs(tile, &memcpy_dst)) {
            addDependencyBetweenOps(memcpy.getOperation(), op.getOperation());
          }
        }
      } else {
        if (tile == nullptr) {
          addDependencyBetweenOps(memcpy.getOperation(), op.getOperation());
        } else if (u.is(memcpy.getDstMemref())) {
          if (areOverlappingPartialMemrefs(tile, &memcpy_dst)) {
            addDependencyBetweenOps(memcpy.getOperation(), op.getOperation());
          }
        } else if (u.is(memcpy.getSrcMemref())) {
          if (areOverlappingPartialMemrefs(tile, &memcpy_src)) {
            addDependencyBetweenOps(memcpy.getOperation(), op.getOperation());
          }
        }
      }
    }

    // If used in a linalg op
    else if (auto linalgop = mlir::dyn_cast<linalg::LinalgOp>(u.getOwner())) {
      if (auto ar = dyn_cast<xilinx::air::ExecuteOp>(linalgop->getParentOp())) {
        if (rw == 'r') {
          if (u.getOperandNumber() <
              linalgop.getNumDpsInputs() + linalgop.getNumDpsInits()) {
            addDependencyBetweenOps(ar.getOperation(), op.getOperation());
          }
        } else if (rw == 'w') {
          if (u.getOperandNumber() >= linalgop.getNumDpsInputs() &&
              u.getOperandNumber() - linalgop.getNumDpsInputs() <
                  linalgop.getNumDpsInits()) {
            addDependencyBetweenOps(ar.getOperation(), op.getOperation());
          }
        } else {
          addDependencyBetweenOps(ar.getOperation(), op.getOperation());
        }
      }
    }

    // If used in hierarchy op
    else if (auto hier =
                 dyn_cast<xilinx::air::HierarchyInterface>(u.getOwner())) {
      // check if the use inside hierarchy op matches with the tracing mode
      // (r or w)
      for (unsigned hier_argument_id = 0;
           hier_argument_id < hier.getNumKernelOperands(); hier_argument_id++) {
        if (u.is(hier.getKernelOperand(hier_argument_id))) {
          auto child_op = hier.getKernelArgument(hier_argument_id);
          char rw_check = checkOperandReadOrWrite(child_op);
          if (rw == 'n' || rw_check == rw) {
            addDependencyBetweenOps(hier.getOperation(), op.getOperation());
          }
        }
      }
    }

    // If used in an unknown op
    else {
      auto unknownop = u.getOwner();
      if (auto ar =
              dyn_cast<xilinx::air::ExecuteOp>(unknownop->getParentOp())) {
        addDependencyBetweenOps(ar.getOperation(), op.getOperation());
      }
    }
  }
}

// Get partial memref tiles from op
void dependencyTracer::getPartialMemrefFromOp(
    Operation *sink_op, SmallVector<partialMemref, 1> &sink_op_memref_reads,
    SmallVector<partialMemref, 1> &sink_op_memref_writes,
    SmallVector<Value, 1> &sink_op_scalar_ins,
    SmallVector<Value, 1> &sink_op_scalar_outs) {
  auto readAccessedMemrefs =
      air::getAllReadAccessedMemrefOperandsFromOp(sink_op);
  auto writeAccessedMemrefs =
      air::getAllWriteAccessedMemrefOperandsFromOp(sink_op);
  auto readAccessedScalars = air::getAllAccessedScalarOperandsFromOp(sink_op);
  auto writeAccessedScalars = air::getAllAccessedScalarOperandsFromOp(sink_op);
  if (failed(readAccessedMemrefs) || failed(writeAccessedMemrefs) ||
      failed(readAccessedScalars) || failed(writeAccessedScalars)) {
    sink_op->emitOpError("failed to get read-accessed operands.");
    return;
  }
  auto getPartialMemrefsFromMemrefAccessPatterns =
      [](SmallVector<
             std::pair<Value, std::tuple<SmallVector<Value>, SmallVector<Value>,
                                         SmallVector<Value>>>> &accessPattern,
         SmallVector<partialMemref, 1> &partialMemrefs) {
        for (auto &entry : accessPattern) {
          if (std::get<0>(entry.second).empty()) {
            partialMemref tile(entry.first);
            partialMemrefs.push_back(tile);
          } else {
            partialMemref tile(entry.first, std::get<0>(entry.second),
                               std::get<1>(entry.second),
                               std::get<2>(entry.second));
            partialMemrefs.push_back(tile);
          }
        }
        return;
      };
  getPartialMemrefsFromMemrefAccessPatterns(*readAccessedMemrefs,
                                            sink_op_memref_reads);
  getPartialMemrefsFromMemrefAccessPatterns(*writeAccessedMemrefs,
                                            sink_op_memref_writes);
  llvm::append_range(sink_op_scalar_ins, *readAccessedScalars);
  llvm::append_range(sink_op_scalar_outs, *writeAccessedScalars);
}

// Add dependency edge
void dependencyTracer::addDependencyBetweenOps(Operation *source,
                                               Operation *sink) {
  auto async_sink = dyn_cast<air::AsyncOpInterface>(sink);
  if (!async_sink)
    sink->emitOpError("dependency sink op has no async interface");
  if (source->getBlock() == sink->getBlock() && source->isBeforeInBlock(sink)) {
    if (auto async_source = dyn_cast<air::AsyncOpInterface>(source)) {
      addAsyncDependencyIfNew(async_sink, async_source.getAsyncToken());
      return;
    }
  }
  for (auto parent = source->getParentOp(); !llvm::isa<mlir::ModuleOp>(parent);
       parent = parent->getParentOp()) {
    if (parent->getBlock() == sink->getBlock() &&
        parent->isBeforeInBlock(sink)) {
      if (auto async_source = dyn_cast<air::AsyncOpInterface>(parent)) {
        addAsyncDependencyIfNew(async_sink, async_source.getAsyncToken());
        return;
      }
    }
  }
}

// Check if two partial memref tiles have overlapping access ranges.
// Returns true if they overlap or if overlap cannot be determined.
// Only returns false when provably disjoint.
bool dependencyTracer::areOverlappingPartialMemrefs(partialMemref *tile_0,
                                                    partialMemref *tile_1) {
  // If any of the two partialMemrefs have empty offsets list, then that
  // partialMemref represents the entire memref, and therefore guarantees to
  // conflict with any other accesses.
  if (tile_0->offsets.empty() || tile_1->offsets.empty())
    return true;

  // Compute the linear range [start, end) for an access pattern.
  // Returns std::nullopt if the range cannot be statically determined.
  auto getAccessRange =
      [](partialMemref *tile) -> std::optional<std::pair<int64_t, int64_t>> {
    int64_t minOffset = 0;
    int64_t maxOffset = 0;

    for (unsigned i = 0; i < tile->offsets.size(); i++) {
      auto constOffset = getConstantIntValue(tile->offsets[i]);
      auto constSize = getConstantIntValue(tile->sizes[i]);
      auto constStride = getConstantIntValue(tile->strides[i]);

      // If any dimension has non-constant offset, size, or stride,
      // we cannot determine the exact range - conservatively assume overlap
      if (!constOffset || !constSize || !constStride)
        return std::nullopt;

      // For this dimension, compute contribution to min and max offsets
      // Min is at index 0: offset * stride
      // Max is at index (size-1): (offset + size - 1) * stride
      int64_t dimMin = (*constOffset) * (*constStride);
      int64_t dimMax = (*constOffset + *constSize - 1) * (*constStride);

      // Handle negative strides
      if (dimMin > dimMax)
        std::swap(dimMin, dimMax);

      minOffset += dimMin;
      maxOffset += dimMax;
    }

    // Return the range [minOffset, maxOffset + 1) - the +1 because end is
    // exclusive
    return std::make_pair(minOffset, maxOffset + 1);
  };

  auto range_0 = getAccessRange(tile_0);
  auto range_1 = getAccessRange(tile_1);

  // If either range cannot be determined, conservatively assume overlap
  if (!range_0 || !range_1)
    return true;

  auto [start_0, end_0] = *range_0;
  auto [start_1, end_1] = *range_1;

  // Check if ranges are disjoint: [start_0, end_0) and [start_1, end_1)
  // Disjoint if: end_0 <= start_1 OR end_1 <= start_0
  if (end_0 <= start_1 || end_1 <= start_0)
    return false; // Provably no overlap

  // Ranges intersect - there is overlap
  return true;
}

char dependencyTracer::checkOperandReadOrWrite(mlir::Value operand) {
  if (!llvm::isa<BaseMemRefType>(operand.getType()))
    operand.getDefiningOp()->emitOpError(
        "operand being traced is not a memref");
  bool foundWriteAccess = false;
  bool foundReadAccess = false;
  for (auto &u : operand.getUses()) {
    char rw_code = checkOpOperandReadOrWrite(u);
    if (rw_code == 'w')
      foundWriteAccess = true;
    else if (rw_code == 'r')
      foundReadAccess = true;
    // If unknown op, then assume write access for safety
    else if (rw_code == 'u')
      foundWriteAccess = true;
  }
  if (foundWriteAccess)
    return 'w';
  else if (foundReadAccess)
    return 'r';
  else
    return 'w';
}

LogicalResult dependencyTracer::traceDependencyFromScfForOp(scf::ForOp &forOp) {
  OpBuilder builder(forOp);
  air::WaitAllOp sink_wait_all_op =
      assignEmptyWaitAllAtScfForIterArg(builder, forOp);
  SmallVector<air::AsyncOpInterface> asyncChildOps;
  forOp.walk<WalkOrder::PreOrder, ForwardDominanceIterator<>>(
      [&](air::AsyncOpInterface op) {
        if (op->hasTrait<OpTrait::IsIsolatedFromAbove>())
          return WalkResult::skip();
        asyncChildOps.push_back(op);
        return WalkResult::advance();
      });
  for (auto op : asyncChildOps) {
    SmallVector<air::partialMemref, 1> sink_op_memref_reads;
    SmallVector<air::partialMemref, 1> sink_op_memref_writes;
    SmallVector<Value, 1> sink_op_scalar_ins;
    SmallVector<Value, 1> sink_op_scalar_outs;
    getPartialMemrefFromOp(op.getOperation(), sink_op_memref_reads,
                           sink_op_memref_writes, sink_op_scalar_ins,
                           sink_op_scalar_outs);
    if (failed(traceDependencyFromOp<air::WaitAllOp>(sink_op_memref_reads,
                                                     sink_wait_all_op, "RAW")))
      return failure();
    if (failed(traceDependencyFromOp<air::WaitAllOp>(
            sink_op_memref_writes, sink_wait_all_op, "WAW/WAR")))
      return failure();
  }
  return success();
}

void dependencyTracer::reconnectLoopCarriedDependencyFromOp(Operation *op) {
  // Get async sink op wrt op
  air::AsyncOpInterface async_op = nullptr;
  if (dyn_cast<air::AsyncOpInterface>(op)) {
    async_op = dyn_cast<air::AsyncOpInterface>(op);
  } else if (auto scf_par = dyn_cast<scf::ParallelOp>(op)) {
    auto token = getLoopCarriedTokenFromScfOp(scf_par);
    if (!token.getDefiningOp()) {
      // loop carried token has no defining op
      return;
    }
    async_op = dyn_cast<air::AsyncOpInterface>(token.getDefiningOp());
  } else if (auto scf_for = dyn_cast<scf::ForOp>(op)) {
    auto token = getLoopCarriedTokenFromScfOp(scf_for, "operand");
    if (!token.getDefiningOp()) {
      // loop carried token has no defining op
      return;
    }
    async_op = dyn_cast<air::AsyncOpInterface>(token.getDefiningOp());
  } else {
    op->emitOpError("unsupported op for loop-carried dependency");
  }

  if (!async_op)
    op->emitOpError("is not an async op");

  // Get parent scf loop op
  auto parent = op->getParentOp();
  if (auto scf_par = dyn_cast<scf::ParallelOp>(parent)) {
    // Get scf parallel's loop-carried token
    auto token = getLoopCarriedTokenFromScfOp(scf_par);

    // Connect dependency between op and token
    addAsyncDependencyIfNew(async_op, token);

    // Get scf parallel's wait_all op before reduce
    SmallVector<scf::ReduceOp, 1> reduce_ops;
    for (auto scf_par_reduce : scf_par.getOps<scf::ReduceOp>()) {
      reduce_ops.push_back(scf_par_reduce);
    }
    if (reduce_ops.size() != 1)
      scf_par->emitOpError("number of reduce ops is not one");
    if (reduce_ops[0].getNumOperands() != 1)
      scf_par->emitOpError("number of reduce operands is not one");
    auto reduce_wait_all =
        dyn_cast<air::WaitAllOp>(reduce_ops[0].getOperand(0).getDefiningOp());
    if (!reduce_wait_all)
      scf_par->emitOpError("reduce op is not dependent on any air::WaitAllOp");

    // Connect op's async token to scf reduce
    auto opToken = getAsyncTokenFromOp(op);
    if (opToken)
      addAsyncDependencyIfNew(reduce_wait_all, opToken);

    // Recurse with parent
    reconnectLoopCarriedDependencyFromOp(parent);
  } else if (auto scf_for = dyn_cast<scf::ForOp>(parent)) {
    // Get scf for's loop-carried token
    auto token = getLoopCarriedTokenFromScfOp(scf_for, "argument");

    // Connect dependency between op and token
    addAsyncDependencyIfNew(async_op, token);

    // Get scf for's wait_all op before yield
    auto scf_for_yield =
        dyn_cast<scf::YieldOp>(scf_for.getBody()->getTerminator());

    // The async token is the last operand in the yield
    Value tokenOperand = scf_for_yield.getOperands().back();
    auto yield_wait_all =
        dyn_cast<air::WaitAllOp>(tokenOperand.getDefiningOp());
    if (!yield_wait_all) {
      OpBuilder b_yield(scf_for_yield);

      // Preserve all existing yield operands except the last one (token)
      SmallVector<Value> yieldOperands;
      auto operands = scf_for_yield.getOperands();
      yieldOperands.append(operands.begin(), std::prev(operands.end()));

      yield_wait_all = air::WaitAllOp::create(
          b_yield, scf_for_yield->getLoc(),
          air::AsyncTokenType::get(scf_for_yield->getContext()),
          SmallVector<Value>{tokenOperand});

      // Append the new wait_all token to the preserved operands
      yieldOperands.push_back(yield_wait_all.getAsyncToken());

      scf::YieldOp::create(b_yield, scf_for_yield->getLoc(), yieldOperands);
      scf_for_yield->erase();
    }

    // Connect op's async token to scf yield
    auto opToken = getAsyncTokenFromOp(op);
    if (opToken)
      addAsyncDependencyIfNew(yield_wait_all, opToken);

    // Recurse with parent
    reconnectLoopCarriedDependencyFromOp(parent);
  } else
    return;
}

// Trace tile index deps
void dependencyTracer::traceTileIndices(
    SmallVector<partialMemref, 1> read_operands,
    SmallVector<partialMemref, 1> write_operands,
    SmallVector<Value, 1> in_scalars, SmallVector<Value, 1> out_scalars,
    air::AsyncOpInterface sink_air_op) {
  for (auto operand : read_operands) {
    for (auto v :
         llvm::concat<Value>(operand.offsets, operand.sizes, operand.strides))
      pushTileIndexAsDep(v, sink_air_op);
  }
  for (auto operand : write_operands) {
    for (auto v :
         llvm::concat<Value>(operand.offsets, operand.sizes, operand.strides))
      pushTileIndexAsDep(v, sink_air_op);
  }
  for (auto scalar : in_scalars) {
    pushTileIndexAsDep(scalar, sink_air_op);
  }
  for (auto scalar : out_scalars) {
    pushTileIndexAsDep(scalar, sink_air_op);
  }
}

// Add tile index deps to op
void dependencyTracer::pushTileIndexAsDep(mlir::Value tile_index,
                                          air::AsyncOpInterface op) {
  if (tile_index != nullptr) {
    // If tile_index is not a nullptr
    // If created by async_region
    if (auto defop = tile_index.getDefiningOp<air::ExecuteOp>()) {
      addAsyncDependencyIfNew(op, defop.getResult(0));
    }
  }
}

} // namespace air
} // namespace xilinx
