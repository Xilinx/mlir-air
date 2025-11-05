//===- AIRDependencyScheduleOpt.cpp -----------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRDependencyScheduleOpt.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Dependency.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Iterators.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

using namespace mlir;

#define DEBUG_TYPE "air-dependency-schedule-opt"

namespace xilinx {
namespace air {

//===----------------------------------------------------------------------===//
// Utility methods for op hoisting
//===----------------------------------------------------------------------===//

// Return the air.execute op in the op's dep list which contains an op of type
// OpTy.
template <typename OpTy>
air::ExecuteOp getDependentExecuteContainingOpTy(Operation *op) {
  auto it = llvm::find_if(air::getAsyncDependenciesFromOp(op), [](Value tok) {
    if (auto execOp = dyn_cast_if_present<air::ExecuteOp>(tok.getDefiningOp()))
      if (llvm::any_of(execOp.getChildOps(),
                       [](Operation &o) { return isa<OpTy>(o); }))
        return true;
    return false;
  });
  if (it == air::getAsyncDependenciesFromOp(op).end())
    return air::ExecuteOp();
  return dyn_cast<air::ExecuteOp>(it->getDefiningOp());
}

// Return the air.execute op which depends on this op and contains an op of type
// OpTy.
template <typename OpTy>
air::ExecuteOp getUserExecuteOpContainingOpTy(Operation *op) {
  auto opTok = air::getAsyncTokenFromOp(op);
  if (!opTok)
    return air::ExecuteOp();
  auto it = llvm::find_if(opTok.getUsers(), [](Operation *usr) {
    if (auto execOp = dyn_cast<air::ExecuteOp>(usr))
      if (llvm::any_of(execOp.getChildOps(),
                       [](Operation &o) { return isa<OpTy>(o); }))
        return true;
    return false;
  });
  if (it == opTok.getUsers().end())
    return air::ExecuteOp();
  return dyn_cast<air::ExecuteOp>(*it);
}

// Reconnect incoming DMA event in the dependency graph
void reconnectIncomingDataMovements(Operation *op, scf::ForOp for_op) {
  air::AsyncOpInterface async_op = dyn_cast<air::AsyncOpInterface>(op);
  auto dependency_list = async_op.getAsyncDependencies();
  if (dependency_list.size()) {
    // Erase dependence to upstream scf.forOp
    eraseAsyncDependencyFromAsyncOp(
        dyn_cast<air::AsyncOpInterface>(async_op.getOperation()),
        for_op.getRegionIterArgs()[0]);
    auto for_op_iter_operand = for_op.getInitArgs()[0];
    op->getResult(0).replaceAllUsesWith(for_op.getRegionIterArgs()[0]);

    replaceAllUsesInRegionWith(for_op_iter_operand, op->getResult(0),
                               *for_op->getParentRegion());
    async_op.addAsyncDependency(for_op_iter_operand);
  }
}

// Reconnect outgoing channel and dealloc events in the dependency graph
void reconnectOutgoingEvents(Operation *op, air::ExecuteOp dealloc_op,
                             scf::ForOp for_op,
                             air::WaitAllOp wait_all_after_for) {
  air::AsyncOpInterface async_op = dyn_cast<air::AsyncOpInterface>(op);
  auto dependency_list = async_op.getAsyncDependencies();
  if (dependency_list.size()) {
    for (unsigned i = 0; i < dependency_list.size(); i++) {
      wait_all_after_for.addAsyncDependency(dependency_list[i]);
    }
    clearAsyncDependenciesOfAsyncOp(async_op);
  }
  eraseAsyncDependencyFromAsyncOp(
      dyn_cast<air::AsyncOpInterface>(wait_all_after_for.getOperation()),
      dealloc_op.getAsyncToken());
  for_op.getResult(0).replaceAllUsesWith(dealloc_op.getResult(0));
  async_op.addAsyncDependency(for_op.getResult(0));
}

//===----------------------------------------------------------------------===//
// Passes
//===----------------------------------------------------------------------===//

struct HoistDmaInAccumPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp for_op,
                                PatternRewriter &rewriter) const override {

    SmallVector<air::DmaMemcpyNdOp, 1> dmamemcpy_incoming_history;
    SmallVector<air::DmaMemcpyNdOp, 1> dmamemcpy_outgoing_history;
    for (auto dma_op : for_op.getOps<air::DmaMemcpyNdOp>()) {
      if (isIncomingDmaOp(dma_op)) {
        dmamemcpy_incoming_history.push_back(dma_op);
      }
      if (isOutgoingDmaOp(dma_op)) {
        dmamemcpy_outgoing_history.push_back(dma_op);
      }
    }
    auto isInvariantWRTForLoop = [](Operation *op, scf::ForOp for_op) {
      return llvm::all_of(op->getOperands(), [&for_op](Value oper) {
        if (isa<air::AsyncTokenType>(oper.getType()))
          return true;
        if (isa_and_present<memref::AllocOp>(oper.getDefiningOp()))
          return true; // HoistDmaInAccumPattern also hoists
                       // memref.alloc/dealloc.
        if (auto execOp =
                dyn_cast_if_present<air::ExecuteOp>(oper.getDefiningOp()))
          if (isa<memref::AllocOp>(execOp.getChildOps().front()))
            return true;
        if (getConstantIntValue(oper))
          return true;
        if (for_op.isDefinedOutsideOfLoop(oper))
          return true;
        return false;
      });
    };
    bool foundDmaPairToHoist = false;
    for (auto op_2 : dmamemcpy_outgoing_history) {
      bool foundDmaPairForThisOp2 = false;
      for (auto op_1 : dmamemcpy_incoming_history) {
        bool areInvariantWRTForLoop = true;
        // Check if the pair of dmas form symmetry in their src and dst
        bool areSymmetric = areSymmetricDmaOps(op_1, op_2);
        // Check if the pair of dmas are invariant with respect to for loop
        // iterations
        areInvariantWRTForLoop &=
            isInvariantWRTForLoop(op_1.getOperation(), for_op);
        areInvariantWRTForLoop &=
            isInvariantWRTForLoop(op_2.getOperation(), for_op);
        if (areSymmetric & areInvariantWRTForLoop) {
          foundDmaPairToHoist = true;
          foundDmaPairForThisOp2 = true;
          // Found a pair of dmas which cancel out each other
          air::ExecuteOp alloc_exec_op =
              getDependentExecuteContainingOpTy<memref::AllocOp>(op_1);
          air::ExecuteOp dealloc_exec_op =
              getUserExecuteOpContainingOpTy<memref::DeallocOp>(op_2);
          if (alloc_exec_op.getAsyncDependencies().size() > 1)
            alloc_exec_op->emitOpError(
                "alloc event should have only one dependent");

          // Reconnect incoming alloc event
          if (alloc_exec_op.getAsyncDependencies().size()) {
            alloc_exec_op.eraseAsyncDependency(0);
          }
          // Reconnect incoming dma event
          reconnectIncomingDataMovements(op_1, for_op);
          // Move ops to before the for loop
          alloc_exec_op->moveBefore(for_op);
          op_1->moveBefore(for_op);

          // Reconnect outgoing dealloc event
          // Reconnect outgoing dma event
          scf::YieldOp yield_op =
              dyn_cast<scf::YieldOp>(for_op.getBody()->getTerminator());
          air::WaitAllOp wait_all_after_for =
              dyn_cast<air::WaitAllOp>(yield_op->getOperand(0).getDefiningOp());
          reconnectOutgoingEvents(op_2, dealloc_exec_op, for_op,
                                  wait_all_after_for);
          // If wait_all depends on outgoing dma, then erase this dependency
          eraseAsyncDependencyFromAsyncOp(
              dyn_cast<air::AsyncOpInterface>(
                  wait_all_after_for.getOperation()),
              dyn_cast<air::AsyncOpInterface>(op_2.getOperation())
                  .getAsyncToken());
          // Move ops to after the for loop
          dealloc_exec_op->moveAfter(for_op);
          op_2->moveAfter(for_op);

          // Move const ops which produce op_2 operands
          // Note: moving consts of which op_1 depends on AFTER op_2 to maintain
          // dominance if consts are shared by both
          for (auto op_2_operand : op_2->getOperands()) {
            if (op_2_operand.getDefiningOp() &&
                isa<arith::ConstantOp>(op_2_operand.getDefiningOp())) {
              rewriter.setInsertionPoint(op_2);
              rewriter.clone(*op_2_operand.getDefiningOp());
            }
          }
          // Move const ops which produce op_1 operands
          for (auto op_1_operand : op_1->getOperands()) {
            if (op_1_operand.getDefiningOp() &&
                isa<arith::ConstantOp>(op_1_operand.getDefiningOp())) {
              rewriter.setInsertionPoint(op_1);
              rewriter.clone(*op_1_operand.getDefiningOp());
            }
          }
        }
      }
      if (foundDmaPairForThisOp2)
        continue; // Ensure unique pairing
    }
    if (foundDmaPairToHoist)
      return success();
    return failure();
  }

private:
  // Check if the dma performs memory copy inbound to the for loop with help
  // from dependency graph
  bool isIncomingDmaOp(air::DmaMemcpyNdOp dma_op) const {
    bool foundScfForDep = false;
    bool foundMemrefAllocDep = false;
    Operation *current_op = dma_op.getOperation();
    air::AsyncOpInterface current_async_op =
        dyn_cast<air::AsyncOpInterface>(current_op);
    auto dependency_list = current_async_op.getAsyncDependencies();
    if (dependency_list.size()) {
      for (auto dep_op : dependency_list) {
        if (dep_op ==
            current_op->getParentOfType<scf::ForOp>().getRegionIterArgs()[0]) {
          // Found scf.forOp in upstream dependency
          foundScfForDep = true;
        } else if (auto exec_op =
                       dyn_cast<air::ExecuteOp>(dep_op.getDefiningOp())) {
          // Found air.ExecuteOp in upstream dependency
          if (llvm::any_of(exec_op.getChildOps(), [](Operation &child_op) {
                return isa<memref::AllocOp>(child_op);
              }))
            foundMemrefAllocDep =
                true; // Found memref.allocOp inside air.ExecuteOp
        }
      }
    }
    return foundScfForDep & foundMemrefAllocDep;
  }

  // Check if the dma performs memory copy outbound to the for loop with help
  // from dependency graph
  bool isOutgoingDmaOp(air::DmaMemcpyNdOp dma_op) const {
    bool foundDepToWaitall = false;
    bool foundDepToMemrefDealloc = false;
    Operation *current_op = dma_op.getOperation();
    air::AsyncOpInterface current_async_op =
        dyn_cast<air::AsyncOpInterface>(current_op);
    auto dependency_token = current_async_op.getAsyncToken();
    for (auto user : dependency_token.getUsers()) {
      auto exec_op = dyn_cast<air::ExecuteOp>(user);
      if (!exec_op)
        continue;
      // Found air.ExecuteOp in downstream dependency
      if (llvm::any_of(exec_op.getChildOps(), [](Operation &child_op) {
            return isa<memref::DeallocOp>(child_op);
          }))
        foundDepToMemrefDealloc = true;
    }
    if (llvm::any_of(dependency_token.getUsers(),
                     [](Operation *user) { return isa<air::WaitAllOp>(user); }))
      foundDepToWaitall = true;
    return foundDepToWaitall & foundDepToMemrefDealloc;
  }

  // Check if two dma ops are symmetric
  bool areSymmetricDmaOps(air::DmaMemcpyNdOp op_1,
                          air::DmaMemcpyNdOp op_2) const {
    bool areSymmetric = op_1.getSrcMemref() == op_2.getDstMemref();
    areSymmetric &= op_2.getSrcMemref() == op_1.getDstMemref();
    // Check offsets, sizes and strides
    auto op_1_dmaNd = dyn_cast<air::DmaMemcpyNdOp>(op_1.getOperation());
    auto op_2_dmaNd = dyn_cast<air::DmaMemcpyNdOp>(op_2.getOperation());
    unsigned op_1_dst_num_entries = op_1_dmaNd.getDstOffsets().size();
    unsigned op_1_src_num_entries = op_1_dmaNd.getSrcOffsets().size();
    unsigned op_2_dst_num_entries = op_2_dmaNd.getDstOffsets().size();
    unsigned op_2_src_num_entries = op_2_dmaNd.getSrcOffsets().size();
    if (areSymmetric && (op_1_dst_num_entries == op_2_src_num_entries) &&
        (op_1_src_num_entries == op_2_dst_num_entries)) {
      for (unsigned i = 0; i < op_1_dst_num_entries; i++) {
        areSymmetric &= areEqualIndices(op_1_dmaNd.getDstOffsets()[i],
                                        op_2_dmaNd.getSrcOffsets()[i]);
        areSymmetric &= areEqualIndices(op_1_dmaNd.getDstSizes()[i],
                                        op_2_dmaNd.getSrcSizes()[i]);
        areSymmetric &= areEqualIndices(op_1_dmaNd.getDstStrides()[i],
                                        op_2_dmaNd.getSrcStrides()[i]);
      }
      for (unsigned i = 0; i < op_1_src_num_entries; i++) {
        areSymmetric &= areEqualIndices(op_1_dmaNd.getSrcOffsets()[i],
                                        op_2_dmaNd.getDstOffsets()[i]);
        areSymmetric &= areEqualIndices(op_1_dmaNd.getSrcSizes()[i],
                                        op_2_dmaNd.getDstSizes()[i]);
        areSymmetric &= areEqualIndices(op_1_dmaNd.getSrcStrides()[i],
                                        op_2_dmaNd.getDstStrides()[i]);
      }
    } else {
      areSymmetric = false;
    }

    return areSymmetric;
  }
};

struct AnnotateFrontAndBackOpsInForPattern
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp for_op,
                                PatternRewriter &rewriter) const override {

    // Check if the for loop is targetted for unrolling
    if (!for_op->hasAttr("unroll"))
      return failure();

    // Check if the for loop is async
    SmallVector<Value> iterTokens;
    for (auto iter_arg : for_op.getRegionIterArgs()) {
      if (llvm::isa<air::AsyncTokenType>(iter_arg.getType())) {
        iterTokens.push_back(iter_arg);
      }
    }
    if (!iterTokens.size())
      return failure();

    // Get async alloc ops
    SmallVector<Operation *> allocOps;
    for (auto &op : for_op.getOps()) {
      if (auto exec_op = dyn_cast<air::ExecuteOp>(op)) {
        bool isFrontCandidate = false;
        if (!exec_op.getAsyncDependencies().size())
          isFrontCandidate = true;
        for (auto d : exec_op.getAsyncDependencies()) {
          for (auto t : iterTokens) {
            if (d == t) {
              isFrontCandidate = true;
            }
          }
        }
        auto child_op = &exec_op.getChildOps().front();
        if (isa<memref::AllocOp>(child_op) && isFrontCandidate) {
          iterTokens.push_back(op.getResult(0));
          // Note: skipping over alloc ops, since they will get hoisted out of
          // loop
          allocOps.push_back(&op);
        }
      }
    }
    if (!allocOps.size())
      return failure();

    // Get ops which are at the front of the loop body's dependency graph
    for (auto &op : for_op.getOps()) {
      bool skip_this_op = false;
      for (auto allocOp : allocOps) {
        if (&op == allocOp)
          skip_this_op = true;
      }

      SmallVector<Value> dep_list;
      skip_this_op |= !getAsyncDependenciesFromOp(&op, dep_list);
      if (skip_this_op)
        continue;

      if (!dep_list.size())
        setBoolAttrForAsyncOp(rewriter, &op, "async_front");
      for (auto dep : dep_list) {
        // Token is in iter_args
        if (llvm::any_of(iterTokens,
                         [dep](Value token) { return token == dep; }))
          setBoolAttrForAsyncOp(rewriter, &op, "async_front");
      }
      // Token is declared outside of for loop
      if (llvm::any_of(dep_list, [for_op](Value token) {
            auto tokenDefOp = token.getDefiningOp();
            if (!tokenDefOp)
              return false;
            return !for_op->isProperAncestor(tokenDefOp);
          })) {
        setBoolAttrForAsyncOp(rewriter, &op, "async_front");
      }
    }

    // Get ops which are at the back of the loop body's dependency graph
    auto yield = for_op.getBody()->getTerminator();
    SmallVector<Value> yielded_tokens;
    for (auto operand : yield->getOperands()) {
      if (llvm::isa<air::AsyncTokenType>(operand.getType())) {
        yielded_tokens.push_back(operand);
      }
    }
    SmallVector<Operation *> back_candidates;
    for (auto token : yielded_tokens) {
      auto back_candidate = token.getDefiningOp();
      if (auto exec_op = dyn_cast<air::ExecuteOp>(back_candidate)) {
        auto child_op = &exec_op.getChildOps().front();
        if (isa<memref::DeallocOp>(child_op)) {
          for (auto d : exec_op.getAsyncDependencies()) {
            back_candidates.push_back(
                getOpAsBackOpCandidate(rewriter, d.getDefiningOp()));
          }
        } else {
          back_candidates.push_back(
              getOpAsBackOpCandidate(rewriter, back_candidate));
        }
      } else {
        back_candidates.push_back(
            getOpAsBackOpCandidate(rewriter, back_candidate));
      }
    }
    for (auto op : back_candidates) {
      setBoolAttrForAsyncOp(rewriter, op, "async_back");
      if (op->hasAttr("async_front"))
        // An op cannot be both "async_back" and "async_front".
        op->removeAttr("async_front");
    }

    return success();
  }

private:
  bool getAsyncDependenciesFromOp(Operation *op,
                                  SmallVector<Value> &dep_list) const {
    bool result = true;
    if (auto async_op = dyn_cast<air::AsyncOpInterface>(op)) {
      dep_list = async_op.getAsyncDependencies();
    } else if (auto async_for_op = dyn_cast<scf::ForOp>(op)) {
      dep_list = async_for_op.getInitArgs();
    } else if (auto async_parallel_op = dyn_cast<scf::ParallelOp>(op)) {
      dep_list = async_parallel_op.getInitVals();
    } else if (auto affine_if_op = dyn_cast<mlir::affine::AffineIfOp>(op)) {
      auto &first_child_op_in_then_block =
          affine_if_op.getThenBlock()->getOperations().front();
      return getAsyncDependenciesFromOp(&first_child_op_in_then_block,
                                        dep_list);
    } else
      result = false;
    return result;
  }

  void setBoolAttrForAsyncOp(OpBuilder builder, Operation *op,
                             std::string attr) const {
    if (auto aif = dyn_cast<affine::AffineIfOp>(op)) {
      aif.getThenBlock()->walk([&](Operation *child_op) {
        child_op->setAttr(attr, builder.getBoolAttr(true));
      });
      aif.getElseBlock()->walk([&](Operation *child_op) {
        child_op->setAttr(attr, builder.getBoolAttr(true));
      });
    } else
      op->setAttr(attr, builder.getBoolAttr(true));
  }

  Operation *getOpAsBackOpCandidate(OpBuilder builder, Operation *op) const {
    if (auto for_candidate = dyn_cast<scf::ForOp>(op)) {
      // Note: if back candidate is scf.for, then since scf.yield is non
      // blocking, an air.wait_all barrier needs to be inserted here
      builder.setInsertionPointAfter(for_candidate);
      SmallVector<Value> dep_list = {};
      air::WaitAllOp wa_op = builder.create<xilinx::air::WaitAllOp>(
          builder.getUnknownLoc(),
          air::AsyncTokenType::get(for_candidate->getContext()), dep_list);
      replaceAllUsesInRegionWith(for_candidate->getResult(0),
                                 wa_op.getAsyncToken(), *op->getParentRegion());
      wa_op.addAsyncDependency(for_candidate->getResult(0));
      return wa_op.getOperation();
    } else
      return op;
  }
};

struct HoistMemallocInForPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  HoistMemallocInForPattern(MLIRContext *ctx, bool keepMemrefDealloc)
      : OpRewritePattern(ctx), keepMemrefDealloc(keepMemrefDealloc) {}

  LogicalResult matchAndRewrite(memref::AllocOp alloc_op,
                                PatternRewriter &rewriter) const override {

    // Find parent air.execute
    auto alloc_exec = alloc_op->getParentOfType<air::ExecuteOp>();
    if (!alloc_exec)
      return failure();

    // Find dealloc
    air::ExecuteOp dealloc_exec = nullptr;
    auto alloc_exec_memref = alloc_exec->getResults()[1];
    if (!llvm::isa<MemRefType>(alloc_exec_memref.getType()))
      alloc_op->emitOpError("the ssa value yielded from execute is not memref");
    auto optDealloc = memref::findDealloc(alloc_exec_memref);
    if (optDealloc.has_value() && optDealloc.value()) {
      dealloc_exec = optDealloc.value()->getParentOfType<air::ExecuteOp>();
    }

    // Check if alloc is the target
    if (!alloc_op->hasAttr("hoist_alloc") &&
        !alloc_exec->hasAttr("hoist_alloc"))
      return failure();

    // Get parent for loop
    auto for_op = alloc_exec->getParentOfType<scf::ForOp>();
    if (!for_op)
      return failure();
    if (for_op.getOperation() != alloc_exec->getParentOp())
      return failure();

    // Reconnect alloc dependency
    skipOverOpInDependencyGraph(rewriter, alloc_exec.getOperation(),
                                for_op.getRegion());

    for (auto ia : for_op.getInitArgs()) {
      alloc_exec.addAsyncDependency(ia);
      for_op->replaceUsesOfWith(ia, alloc_exec.getAsyncToken());
    }

    // Reconnect dealloc dependency
    if (dealloc_exec)
      skipOverOpInDependencyGraph(rewriter, dealloc_exec.getOperation(),
                                  for_op.getRegion());

    if (dealloc_exec && !keepMemrefDealloc) {
      for (auto for_op_token : for_op->getResults()) {
        dealloc_exec.addAsyncDependency(for_op_token);
      }
    }

    // Hoist alloc and dealloc out of for loop
    alloc_exec->moveBefore(for_op);
    if (dealloc_exec && !keepMemrefDealloc)
      dealloc_exec->moveAfter(for_op);

    // Erase alloc hoisting attr
    alloc_op->removeAttr("hoist_alloc");

    return success();
  }

private:
  bool keepMemrefDealloc;

  void skipOverOpInDependencyGraph(OpBuilder &builder, Operation *op,
                                   mlir::Region &region) const {

    auto async_op = dyn_cast<air::AsyncOpInterface>(op);
    if (!async_op)
      return;

    SmallVector<Value> deps;
    for (Value dep : async_op.getAsyncDependencies())
      deps.push_back(dep);
    Value asyncToken = async_op.getAsyncToken();

    if (deps.size() == 1) {
      replaceAllUsesInRegionWith(asyncToken, deps[0], region);
    } else if (deps.size() > 1) {
      builder.setInsertionPoint(async_op);
      air::WaitAllOp wa = builder.create<xilinx::air::WaitAllOp>(
          async_op->getLoc(), air::AsyncTokenType::get(async_op->getContext()),
          deps);
      replaceAllUsesInRegionWith(asyncToken, wa.getAsyncToken(), region);
    }

    air::clearAsyncDependenciesOfAsyncOp(async_op);
  }
};

// Hoist the herd out of a parent scf.for loop.
FailureOr<air::HerdOp> hoistAIRHerdInForImpl(air::HerdOp herdOp,
                                             Region *destRegion,
                                             OpBuilder &builder) {
  auto insertionCheckpoint = builder.saveInsertionPoint();
  auto loc = herdOp->getLoc();
  auto ctx = herdOp->getContext();
  if (!destRegion->isAncestor(herdOp->getParentRegion()))
    return failure();
  SmallVector<scf::ForOp> for_loop_nest;
  Operation *parent = herdOp->getParentOp();
  while (parent && parent != destRegion->getParentOp()) {
    if (auto fop = dyn_cast<scf::ForOp>(parent)) {
      if (hasNImpureOps(fop.getBody(), 1))
        for_loop_nest.push_back(fop);
      else
        return failure(); // Herd is not in perfectly nested loop.
    }
    parent = parent->getParentOp();
  }

  if (for_loop_nest.empty())
    return failure();

  scf::ForOp outerMostLoop = for_loop_nest.back();

  // Create new herd op as parent to the scf.for loop nest.
  builder.setInsertionPoint(outerMostLoop);
  auto originalHerdOperands = herdOp.getOperands().drop_front(
      herdOp.getAsyncDependencies().size() + herdOp.getNumDims());
  SmallVector<Value> herdOperands;
  SmallVector<unsigned> sparseKernelArgIndices;
  IRMapping remap;
  for (unsigned i = 0; i < originalHerdOperands.size(); i++) {
    if (originalHerdOperands[i].getParentRegion()->isAncestor(
            outerMostLoop->getParentRegion())) {
      herdOperands.push_back(originalHerdOperands[i]);
      sparseKernelArgIndices.push_back(i);
    } else
      remap.map(herdOp.getKernelArgument(i), herdOp.getKernelOperand(i));
  }
  auto forRegionIterOperands = outerMostLoop.getOperands().drop_front(
      outerMostLoop.getNumControlOperands());
  auto newHerdOp = builder.create<air::HerdOp>(
      loc, forRegionIterOperands, herdOp.getSizeOperands(), herdOperands, true,
      herdOp->getAttrs());
  outerMostLoop->moveBefore(newHerdOp.getBody().front().getTerminator());
  builder.setInsertionPoint(outerMostLoop);
  for (unsigned i = 0; i < sparseKernelArgIndices.size(); i++)
    remap.map(herdOp.getKernelArgument(sparseKernelArgIndices[i]),
              newHerdOp.getKernelArgument(i));

  // Replace uses of tokens in for loop nest with air.wait_all located at the
  // start of air.herd body.
  for (auto val : forRegionIterOperands) {
    if (!isa<air::AsyncTokenType>(val.getType())) {
      outerMostLoop->emitOpError(
          "loop op's iter_args contain non-async-token-type block arguments, "
          "NYI.");
      return failure();
    }
    auto newAsyncToken =
        builder
            .create<air::WaitAllOp>(loc, air::AsyncTokenType::get(ctx),
                                    SmallVector<Value>{})
            .getAsyncToken();
    remap.map(val, newAsyncToken);
  }
  for (auto loop : llvm::reverse(for_loop_nest)) {
    if (auto definingOp = loop.getUpperBound().getDefiningOp())
      builder.clone(*definingOp, remap);
    if (auto definingOp = loop.getLowerBound().getDefiningOp())
      builder.clone(*definingOp, remap);
    if (auto definingOp = loop.getStep().getDefiningOp())
      builder.clone(*definingOp, remap);
  }

  // Splice herd block into inner-most for loop.
  scf::ForOp innerMostLoop = for_loop_nest.front();
  auto &bb = innerMostLoop.getBody()->getOperations();
  auto &body = herdOp.getBody().front().getOperations();
  bb.splice(bb.begin(), body, body.begin(), --body.end());

  // Connect spliced body ops with the inner-most loop's loop-carried token, to
  // synchronize the herd body operations with the inner-most loop's
  // loop-carried dependency.
  if (air::isAsyncOp(innerMostLoop)) {
    for (auto &bodyOp : innerMostLoop.getBody()->without_terminator()) {
      if (!air::getAsyncDependenciesFromOp(&bodyOp).empty())
        continue;
      addAsyncDependencyIfNew(&bodyOp, air::getLoopCarriedTokenFromScfOp(
                                           innerMostLoop, "argument"));
    }
    // Generate a wait_all op that collects all dangling async tokens in the
    // loop body.
    auto waitAllOp = generateWaitAllToTerminateBlock(
        *innerMostLoop.getBody(), builder, /*isBlocking*/ false);
    // Assign the wait_all token to the loop's terminator (scf.yield).
    innerMostLoop.getBody()->getTerminator()->getOpOperand(0).assign(
        waitAllOp.getAsyncToken());
  }

  builder.setInsertionPoint(herdOp);
  for (auto res : herdOp->getResults())
    remap.map(res,
              builder
                  .create<air::WaitAllOp>(loc, air::AsyncTokenType::get(ctx),
                                          SmallVector<Value>{})
                  .getAsyncToken());
  for (unsigned i = 0; i < herdOp.getNumDims(); i++) {
    remap.map(herdOp.getIds()[i], newHerdOp.getIds()[i]);
    remap.map(herdOp.getSize()[i], newHerdOp.getSize()[i]);
  }
  for (auto [key, val] : remap.getValueMap())
    replaceAllUsesInRegionWith(key, val, newHerdOp.getBody());
  for (auto res : outerMostLoop->getResults())
    res.replaceAllUsesWith(newHerdOp.getAsyncToken());

  builder.restoreInsertionPoint(insertionCheckpoint);
  return newHerdOp;
}

// Merge all herds which are (1) sharing one common parent region and (2) using
// the same symname into one herd.
struct MergeAIRHerdsPattern : public OpRewritePattern<air::HerdOp> {
  using OpRewritePattern<air::HerdOp>::OpRewritePattern;

  MergeAIRHerdsPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(air::HerdOp herdOp,
                                PatternRewriter &rewriter) const override {
    auto parentRegion = herdOp->getParentRegion();

    auto symName = herdOp.getSymName();
    SmallVector<air::HerdOp> herdsWithSameName;
    parentRegion->walk<WalkOrder::PreOrder, ForwardDominanceIterator<>>(
        [symName, &herdsWithSameName](air::HerdOp herd) {
          if (herd.getSymName() != symName)
            return WalkResult::skip();
          herdsWithSameName.push_back(herd);
          return WalkResult::advance();
        });
    // Pattern is only applied to the last herd with the same herd name.
    if (herdOp != herdsWithSameName.back())
      return failure();
    // Failure if there's only one herd with the same herd name.
    if (herdsWithSameName.size() == 1)
      return failure();
    // Get the innermost region that is ancestor to all herds sharing the same
    // herd name.
    Region *region = herdOp->getParentRegion();
    // Failure if not all herds with the same symname are under one single
    // region.
    if (!llvm::all_of(herdsWithSameName, [region](air::HerdOp h) {
          return h->getParentRegion() == region;
        }))
      return failure();

    // Verify that all specified herds are mergeable.
    if (!llvm::all_of(herdsWithSameName, [&herdOp](air::HerdOp h) {
          if (herdOp.getNumDims() != h.getNumDims())
            return false;
          for (unsigned i = 0; i < herdOp.getNumDims(); i++) {
            auto herdOpS = getConstantIntValue(herdOp.getSizeOperands()[i]);
            auto hS = getConstantIntValue(h.getSizeOperands()[i]);
            if (herdOpS && hS && *herdOpS != *hS)
              return false;
          }
          return true;
        }))
      return failure();

    // Merge all herds with the same herd name into one.
    llvm::SetVector<Value> kernelOperands;
    for (auto h : herdsWithSameName)
      kernelOperands.insert(h.getKernelOperands().begin(),
                            h.getKernelOperands().end());
    rewriter.setInsertionPointAfter(herdsWithSameName.back());
    auto newMergedHerd = rewriter.create<air::HerdOp>(
        herdOp->getLoc(), herdsWithSameName.back().getAsyncDependencies(),
        herdOp.getSizeOperands(), kernelOperands.takeVector(),
        (bool)herdOp.getAsyncToken(), herdOp->getAttrs());
    rewriter.setInsertionPointToStart(&newMergedHerd.getBody().front());
    IRMapping remap;
    for (auto h : herdsWithSameName) {
      // Update "link_with" attr
      if (h.getLinkWith())
        newMergedHerd.setLinkWith(h.getLinkWith());
      // Remap kernel arguments
      for (unsigned i = 0; i < h.getNumDims() * 2; i++)
        remap.map(h.getBody().getArgument(i),
                  newMergedHerd.getBody().getArgument(i));
      for (unsigned i = 0; i < h.getNumKernelOperands(); i++)
        remap.map(h.getKernelArgument(i),
                  newMergedHerd.getTiedKernelArgument(
                      h.getTiedKernelOperand(h.getKernelArgument(i))));
      // Clone ops
      for (auto &o : h.getBody().front().without_terminator())
        rewriter.clone(o, remap);
    }

    // Erase original herds; replace async token uses if async.
    for (auto it = herdsWithSameName.begin(); it != herdsWithSameName.end();
         it++) {
      if (!it->getAsyncToken()) {
        rewriter.eraseOp(*it);
        continue;
      }
      rewriter.setInsertionPoint(*it);
      if (it == herdsWithSameName.end() - 1) {
        rewriter.replaceOp(*it, newMergedHerd.getAsyncToken());
        continue;
      }
      auto newWaitAll = air::replaceAsyncOpWithWaitAll(rewriter, remap, *it);
      rewriter.replaceOp(*it, newWaitAll.getAsyncToken());
    }

    return success();
  }

private:
};

struct HoistAIRHerdsToSharedRegionPattern
    : public OpRewritePattern<air::HerdOp> {
  using OpRewritePattern<air::HerdOp>::OpRewritePattern;

  HoistAIRHerdsToSharedRegionPattern(MLIRContext *ctx)
      : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(air::HerdOp herdOp,
                                PatternRewriter &rewriter) const override {
    auto parentRegion = herdOp->getParentRegion();
    auto symName = herdOp.getSymName();
    SmallVector<Operation *> herdsWithSameName;
    parentRegion->walk<WalkOrder::PreOrder, ForwardDominanceIterator<>>(
        [symName, &herdsWithSameName](air::HerdOp herd) {
          if (herd.getSymName() != symName)
            return WalkResult::skip();
          herdsWithSameName.push_back(herd.getOperation());
          return WalkResult::advance();
        });
    if (herdOp != herdsWithSameName.back())
      return failure(); // Apply pattern at the last herd

    // Get the innermost region that is ancestor to all herds sharing the same
    // name
    Region *region = air::findCommonRegionContainingAllAncestors(
        herdsWithSameName,
        herdsWithSameName.front()
            ->getParentWithTrait<OpTrait::IsIsolatedFromAbove>());
    if (!region)
      return failure();
    // If none of the herds are directly contained in region, then abort herd
    // hoisting; otherwise the loop hoisting breaks the IR functionality.
    if (llvm::none_of(herdsWithSameName, [region](Operation *h) {
          return h->getParentRegion() == region;
        }))
      return failure();
    // If only one herd uses the same symname, then hoist it all the way to the
    // body of any parent op with trait "IsolatedFromAbove".
    if (herdsWithSameName.size() == 1)
      while (region->getParentOp() &&
             !region->getParentOp()
                  ->mightHaveTrait<OpTrait::IsIsolatedFromAbove>())
        region = region->getParentRegion();

    // Hoist herds to the shared parent region
    SmallVector<Operation *> processed, unprocessed;
    for (auto h : herdsWithSameName) {
      auto newHerd =
          hoistAIRHerdInForImpl(dyn_cast<air::HerdOp>(h), region, rewriter);
      if (succeeded(newHerd)) {
        rewriter.eraseOp(h);
        processed.push_back(*newHerd);
        continue;
      }
      unprocessed.push_back(h);
    }
    if (processed.empty())
      return failure();
    processed.insert(processed.end(), unprocessed.begin(), unprocessed.end());
    return success();
  }

private:
};

struct ConstructPingPongDependencyPattern
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp for_op,
                                PatternRewriter &rewriter) const override {

    // Check if the loop has been unrolled by factor 2
    if (!for_op->hasAttr("unroll"))
      return failure();
    uint64_t unroll_factor =
        for_op->getAttrOfType<IntegerAttr>("unroll").getInt();
    if (unroll_factor != 2)
      return failure();

    // Find ping and pong allocs and deallocs
    SmallVector<Operation *> alloc_execs;
    for (auto ia : for_op.getInitArgs()) {
      pushToAllocExecsIfHoistedFromLoop(ia, alloc_execs);
    }
    if (!alloc_execs.size())
      return failure();

    SmallVector<Operation *> dealloc_execs;
    for (auto alloc_exec : alloc_execs) {
      for (auto user : alloc_exec->getResult(1).getUsers()) {
        if (isa<memref::DeallocOp>(user) &&
            isa<air::ExecuteOp>(user->getParentOp())) {
          dealloc_execs.push_back(user->getParentOp());
        }
      }
    }

    // Find producer and consumer ops that use ping and pong buffers
    SmallVector<std::pair<SmallVector<Operation *>, SmallVector<Operation *>>>
        all_buffers_user_ops;
    for (auto alloc_exec : alloc_execs) {
      auto buffer_memref = alloc_exec->getResult(1);
      SmallVector<Operation *> producer_ops;
      SmallVector<Operation *> consumer_ops;
      SmallVector<Operation *> candidate_ops;
      for_op.getBody()->walk([&](Operation *child_op) {
        for (auto operand : child_op->getOperands()) {
          if (operand == buffer_memref) {
            push_back_if_unique<Operation *>(candidate_ops, child_op);
          }
        }
      });
      // Check if producer or consumer
      for (auto candidate_op : candidate_ops) {
        if (checkOpOperandReadOrWrite(buffer_memref, candidate_op) == 'w') {
          if (auto candidate_for_op =
                  getOutermostForOpInForOpNest(candidate_op, for_op)) {
            push_back_if_unique<Operation *>(producer_ops,
                                             candidate_for_op.getOperation());
          } else if (auto candidate_parallel_op =
                         getOutermostParallelOpInForOpNest(candidate_op,
                                                           for_op)) {
            push_back_if_unique<Operation *>(
                producer_ops, candidate_parallel_op.getOperation());
          } else if (isa<air::ExecuteOp>(candidate_op->getParentOp())) {
            push_back_if_unique<Operation *>(producer_ops,
                                             candidate_op->getParentOp());
          } else if (isa<affine::AffineIfOp>(candidate_op->getParentOp())) {
            push_back_if_unique<Operation *>(producer_ops, candidate_op);
          } else if (isa<air::AsyncOpInterface>(candidate_op)) {
            push_back_if_unique<Operation *>(producer_ops, candidate_op);
          } else {
            push_back_if_unique<Operation *>(producer_ops, candidate_op);
          }
        } else if (checkOpOperandReadOrWrite(buffer_memref, candidate_op) ==
                   'r') {
          if (auto candidate_for_op =
                  getOutermostForOpInForOpNest(candidate_op, for_op)) {
            push_back_if_unique<Operation *>(consumer_ops,
                                             candidate_for_op.getOperation());
          } else if (auto candidate_parallel_op =
                         getOutermostParallelOpInForOpNest(candidate_op,
                                                           for_op)) {
            push_back_if_unique<Operation *>(
                consumer_ops, candidate_parallel_op.getOperation());
          } else if (isa<air::ExecuteOp>(candidate_op->getParentOp())) {
            push_back_if_unique<Operation *>(consumer_ops,
                                             candidate_op->getParentOp());
          } else if (isa<affine::AffineIfOp>(candidate_op->getParentOp())) {
            push_back_if_unique<Operation *>(consumer_ops, candidate_op);
          } else if (isa<air::AsyncOpInterface>(candidate_op)) {
            push_back_if_unique<Operation *>(consumer_ops, candidate_op);
          } else
            push_back_if_unique<Operation *>(consumer_ops, candidate_op);
        }
      }
      std::pair<SmallVector<Operation *>, SmallVector<Operation *>> vec_pair =
          std::make_pair(producer_ops, consumer_ops);
      all_buffers_user_ops.push_back(vec_pair);
    }

    // Annotate ping and pong
    annotatePingPong(rewriter, alloc_execs);
    annotatePingPong(rewriter, dealloc_execs);
    for (auto pair : all_buffers_user_ops) {
      annotatePingPong(rewriter, pair.first, "producer");
      annotatePingPong(rewriter, pair.second, "consumer");
    }

    // Construct essential dep edges

    // Part 1: alloc to for

    auto alloc_ping_exec = dyn_cast<air::ExecuteOp>(alloc_execs[0]);
    auto alloc_pong_exec = dyn_cast<air::ExecuteOp>(alloc_execs[1]);
    auto alloc_ping_token = alloc_ping_exec.getAsyncToken();
    auto alloc_pong_token = alloc_pong_exec.getAsyncToken();
    air::WaitAllOp ping_pong_token_wait = rewriter.create<air::WaitAllOp>(
        rewriter.getUnknownLoc(),
        air::AsyncTokenType::get(rewriter.getContext()),
        SmallVector<Value>{alloc_ping_token, alloc_pong_token});
    SmallVector<Value> upstream_tokens = alloc_pong_exec.getAsyncDependencies();
    clearAsyncDependenciesOfAsyncOp(alloc_ping_exec);
    for (auto t : upstream_tokens) {
      alloc_ping_exec.addAsyncDependency(t);
    }
    alloc_ping_exec->moveBefore(alloc_pong_exec);

    SmallVector<Value, 1> iter_operands = {
        alloc_ping_token, alloc_pong_token,
        ping_pong_token_wait.getAsyncToken(),
        ping_pong_token_wait.getAsyncToken()};
    scf::ForOp new_loop_op =
        replaceForLoopAndAddIterArgs(rewriter, for_op, iter_operands);
    for_op.getResult(0).replaceAllUsesWith(new_loop_op.getResult(1));

    // Collect ping/pong producer/consumer fronts and backs for ping pong
    // dependency edge connection
    SmallVector<Operation *> ping_producer_fronts;
    SmallVector<Operation *> ping_producer_backs;
    SmallVector<Operation *> ping_consumer_fronts;
    SmallVector<Operation *> ping_consumer_backs;
    SmallVector<Operation *> pong_producer_fronts;
    SmallVector<Operation *> pong_producer_backs;
    SmallVector<Operation *> pong_consumer_fronts;
    SmallVector<Operation *> pong_consumer_backs;

    new_loop_op.getBody()->walk([&](Operation *op) {
      if (op->hasAttr("ping_pong") || op->hasAttr("unrolled_iteration")) {
        auto ping_pong_id =
            op->hasAttr("ping_pong")
                ? (op->getAttrOfType<IntegerAttr>("ping_pong").getUInt())
                : (op->getAttrOfType<IntegerAttr>("unrolled_iteration")
                       .getInt());
        // "Ping" producer fronts
        if (op->hasAttr("async_front") && ping_pong_id == 0) {
          ping_producer_fronts.push_back(op);
        }
        // "Pong" producer fronts
        else if (op->hasAttr("async_front") && ping_pong_id == 1) {
          pong_producer_fronts.push_back(op);
        }
        // "Ping" consumer backs
        else if (op->hasAttr("async_back") && ping_pong_id == 0) {
          ping_consumer_backs.push_back(op);
        }
        // "Pong" consumer backs
        else if (op->hasAttr("async_back") && ping_pong_id == 1) {
          pong_consumer_backs.push_back(op);
        }
        // "Ping" producer backs
        if (op->hasAttr("producer") && ping_pong_id == 0) {
          ping_producer_backs.push_back(op);
        }
        // "Pong" producer backs
        if (op->hasAttr("producer") && ping_pong_id == 1) {
          pong_producer_backs.push_back(op);
        }
        // "Ping" consumer fronts
        if (op->hasAttr("consumer") && ping_pong_id == 0) {
          ping_consumer_fronts.push_back(op);
        }
        // "Pong" consumer fronts
        if (op->hasAttr("consumer") && ping_pong_id == 1) {
          pong_consumer_fronts.push_back(op);
        }
      }
    });

    // Part 2: Connect producers
    for (auto sink : ping_producer_fronts) {
      // "Ping" producers
      addAsyncDependencyIfNew(sink, new_loop_op.getRegionIterArgs()[0]);
      addAsyncDependencyIfNew(sink, new_loop_op.getRegionIterArgs()[3]);
    }
    for (auto sink : pong_producer_fronts) {
      // "Pong" producers
      addAsyncDependencyIfNew(sink, new_loop_op.getRegionIterArgs()[1]);
      for (auto source : ping_producer_backs) {
        Value token = getTokenFromOutermostParentAffineIfOp(source);
        addAsyncDependencyIfNew(sink, token);
      }
    }

    // Part 3: Connect consumers
    for (auto sink : ping_consumer_fronts) {
      // "Ping" consumers
      addAsyncDependencyIfNew(sink, new_loop_op.getRegionIterArgs()[2]);
    }
    for (auto sink : pong_consumer_fronts) {
      // "Pong" consumers
      for (auto source : ping_consumer_backs) {
        Value token = getTokenFromOutermostParentAffineIfOp(source);
        addAsyncDependencyIfNew(sink, token);
      }
    }

    // Part 4: Connect yield
    // Note: currently only supports producer and consumer dep graphs with
    // single back
    rewriter.setInsertionPointToEnd(new_loop_op.getBody());
    SmallVector<Value, 1> yield_operands = {
        getJointTokenFromOps(rewriter, ping_consumer_backs),
        getJointTokenFromOps(rewriter, pong_consumer_backs),
        getJointTokenFromOps(rewriter, pong_consumer_backs),
        getJointTokenFromOps(rewriter, pong_producer_backs)};
    for (auto v : yield_operands) {
      if (!v)
        return failure();
    }
    rewriter.create<scf::YieldOp>(new_loop_op.getLoc(), yield_operands);

    for_op.erase();

    return success();
  }

private:
  void annotatePingPong(OpBuilder builder, SmallVector<Operation *> ops,
                        std::string flag = "") const {
    uint64_t unroll_iter = 0;
    for (auto op : ops) {
      bool hit = false;
      if (op->hasAttr("unrolled_iteration")) {
        hit = true;
        unroll_iter =
            op->getAttrOfType<IntegerAttr>("unrolled_iteration").getInt();
      }
      // If op is in region of an unrolled affine if
      else if (isa<affine::AffineIfOp>(op->getParentOp())) {
        Operation *parent = op->getParentOp();
        while (isa<affine::AffineIfOp>(parent)) {
          if (parent->hasAttr("unrolled_iteration")) {
            unroll_iter =
                parent->getAttrOfType<IntegerAttr>("unrolled_iteration")
                    .getInt();
            hit = true;
          }
          parent = parent->getParentOp();
        }
      }

      if (hit) {
        op->setAttr("ping_pong", builder.getUI32IntegerAttr(unroll_iter));
        op->removeAttr("unrolled_iteration");
        if (flag != "") {
          op->setAttr(flag, builder.getUI32IntegerAttr(unroll_iter));
        }
      }
    }
  }

  scf::ForOp
  replaceForLoopAndAddIterArgs(OpBuilder &builder, scf::ForOp loop_op,
                               SmallVector<Value, 1> iter_operands) const {

    builder.setInsertionPoint(loop_op);
    scf::ForOp new_loop_op = builder.create<scf::ForOp>(
        loop_op.getLoc(), loop_op.getLowerBound(), loop_op.getUpperBound(),
        loop_op.getStep(), iter_operands);

    if (auto attr = loop_op->getAttrOfType<StringAttr>(
            SymbolTable::getSymbolAttrName()))
      new_loop_op->setAttr(SymbolTable::getSymbolAttrName(), attr);

    // Splice the operations inside loop op
    auto &bb = new_loop_op.getBody()->getOperations();
    auto &body = loop_op.getBody()->getOperations();
    bb.splice(bb.begin(), body, body.begin(), --body.end());

    auto iv = loop_op.getInductionVar();
    iv.replaceAllUsesWith(new_loop_op.getInductionVar());
    builder.setInsertionPointToStart(new_loop_op.getBody());

    for (unsigned i = 0; i < loop_op.getRegionIterArgs().size(); i++) {
      loop_op.getRegionIterArgs()[i].replaceAllUsesWith(
          new_loop_op.getRegionIterArgs()[i]);
    }

    return new_loop_op;
  }

  template <typename T>
  void push_back_if_unique(SmallVector<T> &vec, T entry) const {
    if (std::find(vec.begin(), vec.end(), entry) == vec.end()) {
      vec.push_back(entry);
    }
  }

  Value getTokenFromOutermostParentAffineIfOp(Operation *op) const {
    Value token = nullptr;
    Operation *parent = op->getParentOp();
    if (isa<affine::AffineIfOp>(parent)) {
      while (isa<affine::AffineIfOp>(parent)) {
        token = getAsyncTokenFromValues(parent->getResults());
        parent = parent->getParentOp();
      }
      return token;
    } else if (auto async_op = dyn_cast<air::AsyncOpInterface>(op)) {
      return async_op.getAsyncToken();
    } else if (auto for_op = dyn_cast<scf::ForOp>(op)) {
      token = getAsyncTokenFromValues(for_op->getResults());
      return token;
    } else if (auto parallel_op = dyn_cast<scf::ParallelOp>(op)) {
      token = getAsyncTokenFromValues(parallel_op->getResults());
      return token;
    } else
      return nullptr;
  }

  Value getAsyncTokenFromValues(SmallVector<Value> vec) const {
    for (auto v : vec) {
      if (llvm::isa<air::AsyncTokenType>(v.getType())) {
        return v;
      }
    }
    return nullptr;
  }

  Value getJointTokenFromOps(OpBuilder builder,
                             SmallVector<Operation *> vec) const {
    Value token = nullptr;
    if (vec.size() > 1) {
      SmallVector<Value> wa_operands;
      for (auto op : vec) {
        token = getTokenFromOutermostParentAffineIfOp(op);
        push_back_if_unique<Value>(wa_operands, token);
      }
      auto wa = builder.create<air::WaitAllOp>(
          builder.getUnknownLoc(),
          air::AsyncTokenType::get(builder.getContext()), wa_operands);
      token = wa.getAsyncToken();
    } else if (vec.size() == 1) {
      token = getTokenFromOutermostParentAffineIfOp(vec[0]);
    }
    return token;
  }

  scf::ForOp getOutermostForOpInForOpNest(Operation *op,
                                          scf::ForOp ancestor_for) const {
    if (!ancestor_for->isProperAncestor(op)) {
      return scf::ForOp();
    }
    if (op->getParentOfType<scf::ForOp>() == ancestor_for) {
      return scf::ForOp();
    }
    Operation *parent = op->getParentOp();
    scf::ForOp output = nullptr;
    while (parent != ancestor_for.getOperation()) {
      if (auto parent_for = dyn_cast<scf::ForOp>(parent)) {
        output = parent_for;
      }
      parent = parent->getParentOp();
    }
    return output;
  }

  scf::ParallelOp
  getOutermostParallelOpInForOpNest(Operation *op,
                                    scf::ForOp ancestor_for) const {
    if (!ancestor_for->isProperAncestor(op)) {
      return scf::ParallelOp();
    }
    Operation *parent = op->getParentOp();
    scf::ParallelOp output = nullptr;
    while (parent != ancestor_for.getOperation()) {
      if (auto parent_parallel = dyn_cast<scf::ParallelOp>(parent)) {
        output = parent_parallel;
      }
      parent = parent->getParentOp();
    }
    return output;
  }

  // Recursively trace dependency of scf.for's iter args, and search for hoisted
  // allocs
  void pushToAllocExecsIfHoistedFromLoop(
      Value v, SmallVector<Operation *> &alloc_execs) const {
    if (auto exec = v.getDefiningOp<air::ExecuteOp>()) {
      if (exec->hasAttr("unrolled_iteration") && exec->getNumResults() == 2 &&
          llvm::isa<MemRefType>(exec->getResult(1).getType())) {
        alloc_execs.push_back(exec.getOperation());
        for (auto dep : exec.getAsyncDependencies()) {
          pushToAllocExecsIfHoistedFromLoop(dep, alloc_execs);
        }
      }
    }
  }
};

struct HoistOpsNotUsingPingPongPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp for_op,
                                PatternRewriter &rewriter) const override {

    // Check if the loop has already been processed
    if (for_op->hasAttr("isolated"))
      return failure();

    // Check if the loop has been unrolled by factor 2
    if (!for_op->hasAttr("unroll"))
      return failure();
    uint64_t unroll_factor =
        for_op->getAttrOfType<IntegerAttr>("unroll").getInt();
    if (unroll_factor != 2)
      return failure();
    if (for_op.getInitArgs().size() != 1)
      return failure();

    // Check if any alloc.op in the scf.for loop is targetted for ping-pong
    // transform
    bool foundHoistAlloc = false;
    for (auto child_exec : for_op.getOps<air::ExecuteOp>()) {
      for (auto child_alloc : child_exec.getOps<memref::AllocOp>()) {
        if (child_alloc->hasAttr("hoist_alloc")) {
          foundHoistAlloc = true;
        }
      }
    }
    if (!foundHoistAlloc)
      return failure();

    // Find ops which are to be hoisted
    SmallVector<Operation *> target_ops;
    // Find air.herd in scf.for
    for (auto child_herd : for_op.getOps<air::HerdOp>()) {
      target_ops.push_back(child_herd.getOperation());
    }
    // Find scf.parallel only dependent on scf.for's (one) loop-carried
    // dependency token
    for (auto child_par : for_op.getOps<scf::ParallelOp>()) {
      if (child_par.getInitVals().size() == 1 &&
          child_par.getInitVals()[0] == for_op.getRegionIterArgs()[0]) {
        target_ops.push_back(child_par.getOperation());
      }
    }
    if (target_ops.empty()) {
      // Loop is already in isolation
      for_op->setAttr("isolated", rewriter.getBoolAttr(true));
      return failure();
    }

    // Hoist ops out to a new scf.for loop
    hoistTargetOpsToNewSCFFor(rewriter, for_op, target_ops);
    // Erasing the original ops backwards, to avoid erasing op that still has
    // valid uses.
    for (auto o : llvm::reverse(target_ops))
      rewriter.eraseOp(o);

    for_op->setAttr("isolated", rewriter.getBoolAttr(true));

    return success();
  }

private:
};

struct LabelScfForLoopForPingPongPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp for_op,
                                PatternRewriter &rewriter) const override {

    // Check if the loop has been labelled
    if (for_op->hasAttr("unroll"))
      return failure();

    // Check if the loop has child air.execute event containing memref.alloc
    SmallVector<Operation *> alloc_ops;
    for (auto child_exec : for_op.getOps<air::ExecuteOp>()) {
      for (auto child_alloc : child_exec.getOps<memref::AllocOp>()) {
        alloc_ops.push_back(child_alloc.getOperation());
      }
    }
    if (alloc_ops.empty())
      return failure();

    // Label the scf.for loop and all its child memref.allocs
    int unroll_factor = 2; // Unroll factor hardened as 2. TODO: add support for
                           // an arbitrary factor.
    for_op->setAttr("unroll", rewriter.getI32IntegerAttr(unroll_factor));
    for (auto op : alloc_ops) {
      op->setAttr("hoist_alloc", rewriter.getBoolAttr(true));
    }

    return success();
  }

private:
};

struct LabelScfForLoopInAIRSegment : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp for_op,
                                PatternRewriter &rewriter) const override {

    // Check if the loop has been labelled
    if (for_op->hasAttr("unroll"))
      return failure();

    // Avoid unrolling for loop which contains air.herd
    bool hasHerdInBody = false;
    for_op.getBody()->walk([&](air::HerdOp herd) { hasHerdInBody = true; });
    if (hasHerdInBody)
      return failure();

    if (for_op->getParentOfType<air::SegmentOp>() &&
        !for_op->getParentOfType<air::HerdOp>()) {
      // Get for loop trip count
      if (auto tripCount = getStaticScfForTripCountAsInt(for_op))
        for_op->setAttr("unroll", rewriter.getI32IntegerAttr(*tripCount));
    }

    return success();
  }

private:
};

struct UnrollScfParallel : public OpRewritePattern<scf::ParallelOp> {
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ParallelOp par,
                                PatternRewriter &rewriter) const override {

    auto loc = rewriter.getUnknownLoc();

    if (!llvm::all_of(par.getLowerBound(),
                      [](Value lb) { return getConstantIntValue(lb); })) {
      par->emitOpError("non-static scf.parallel lb, NYI");
      return failure();
    }
    if (!llvm::all_of(par.getLowerBound(),
                      [](Value lb) { return *getConstantIntValue(lb) == 0; })) {
      par->emitOpError("non-zero scf.parallel lb, NYI");
      return failure();
    }

    // Get parallel loop trip count.
    SmallVector<int, 2> lbs_spatial, ubs_spatial;
    air::getSizesFromSpatialLoop(par.getOperation(), lbs_spatial, ubs_spatial);
    std::vector<unsigned> par_size;
    unsigned par_vol = 1;
    for (unsigned i = 0; i < lbs_spatial.size(); i++) {
      par_size.push_back(ubs_spatial[i] - lbs_spatial[i] + 1);
      par_vol *= ubs_spatial[i] - lbs_spatial[i] + 1;
    }

    // Collect yielded tokens.
    SmallVector<Value> yieldedTokens;

    // Walk all iterations. Assumption: LB starts from 0.
    for (unsigned iter = 0; iter < par_vol; iter++) {
      IRMapping remap;
      std::vector<unsigned> position =
          air::getMDVectorFromIterator(par_size, iter);
      std::reverse(position.begin(),
                   position.end()); // scf.parallel induction vars. have LSD at
                                    // highest index.
      // Create arith.const ops per position
      SmallVector<Value> positionVals;
      for (unsigned i = 0; i < position.size(); i++) {
        positionVals.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, position[i]));
        remap.map(par.getInductionVars()[i], positionVals[i]);
      }

      // Splice
      for (auto &op : par.getBody()->without_terminator()) {
        rewriter.clone(op, remap);
      }
      if (!par.getBody()->mightHaveTerminator())
        continue;
      auto terminator = par.getBody()->getTerminator();
      SmallVector<Value> tokens;
      for (auto reducedOper : terminator->getOperands())
        if (isa<air::AsyncTokenType>(reducedOper.getType()))
          tokens.push_back(reducedOper);
      auto newWaitAll = rewriter.create<air::WaitAllOp>(
          loc, air::AsyncTokenType::get(rewriter.getContext()),
          lookupOrDefaultRange(tokens, remap));
      yieldedTokens.push_back(newWaitAll.getAsyncToken());
    }

    // Scf.parallel returned token
    if (par->getNumResults()) {
      auto newWaitAll = rewriter.create<air::WaitAllOp>(
          loc, air::AsyncTokenType::get(rewriter.getContext()), yieldedTokens);
      par->getResult(0).replaceAllUsesWith(newWaitAll.getAsyncToken());
    }

    rewriter.eraseOp(par);
    return success();
  }

private:
};

void updateAffineForBounds(affine::AffineForOp loop_op, int lb, int ub,
                           int step) {
  loop_op.setConstantLowerBound(lb);
  loop_op.setConstantUpperBound(ub);
  loop_op.setStep(step);
}

// Update scf.for with specified loop bounds, steps, and induction variable
// type.
FailureOr<scf::ForOp> updateScfForBounds(RewriterBase &rewriter,
                                         scf::ForOp loopOp, int64_t lb,
                                         int64_t ub, int64_t step, Type type) {
  auto loc = loopOp.getLoc();
  rewriter.setInsertionPoint(loopOp);

  Value lbVal, ubVal, stepVal;
  if (isa<IntegerType>(type)) {
    lbVal = rewriter.create<arith::ConstantOp>(
        loc, type, rewriter.getIntegerAttr(type, lb));
    ubVal = rewriter.create<arith::ConstantOp>(
        loc, type, rewriter.getIntegerAttr(type, ub));
    stepVal = rewriter.create<arith::ConstantOp>(
        loc, type, rewriter.getIntegerAttr(type, step));
  } else if (isa<IndexType>(type)) {
    lbVal = rewriter.create<arith::ConstantIndexOp>(loc, lb);
    ubVal = rewriter.create<arith::ConstantIndexOp>(loc, ub);
    stepVal = rewriter.create<arith::ConstantIndexOp>(loc, step);
  } else {
    loopOp->emitOpError("Expected index or integer type, but got: ") << type;
    return failure();
  }

  auto newFor = rewriter.create<scf::ForOp>(loc, lbVal, ubVal, stepVal,
                                            loopOp.getInitArgs());
  auto &bb = newFor.getBody()->getOperations();
  auto &body = loopOp.getBody()->getOperations();
  bb.splice(bb.begin(), body, body.begin(), body.end());
  rewriter.replaceAllUsesWith(loopOp.getInductionVar(),
                              newFor.getInductionVar());
  rewriter.replaceAllUsesWith(loopOp.getRegionIterArgs(),
                              newFor.getRegionIterArgs());
  rewriter.replaceOp(loopOp, newFor);
  return newFor;
}

// Erase op from within an scf.for loop, and reconstruct ssa value usage in the
// process.
LogicalResult eraseOpFromScfFor(RewriterBase &rewriter, scf::ForOp sfo,
                                Operation *op) {
  OpBuilder::InsertionGuard guard(rewriter);
  IRMapping remap;
  if (auto exec = dyn_cast<air::ExecuteOp>(op->getParentOp())) {
    rewriter.replaceAllUsesWith(exec.getResult(1), sfo.getInductionVar());
    if (sfo.getNumRegionIterArgs())
      rewriter.replaceAllUsesWith(exec.getAsyncToken(),
                                  sfo.getRegionIterArgs()[0]);
    else if (exec.getAsyncDependencies().size() == 1)
      rewriter.replaceAllUsesWith(exec.getAsyncToken(),
                                  exec.getAsyncDependencies()[0]);
    else {
      rewriter.setInsertionPoint(exec);
      auto newWaitAll = air::replaceAsyncOpWithWaitAll(rewriter, remap, exec);
      rewriter.replaceAllUsesWith(exec.getAsyncToken(),
                                  newWaitAll.getAsyncToken());
    }
    rewriter.eraseOp(exec);
  } else {
    op->getResult(0).replaceAllUsesWith(sfo.getInductionVar());
    rewriter.eraseOp(op);
  }
  return success();
}

// Fold affine.apply op operating on loop induction variable into loop bounds.
struct CanonicalizeAffineApplyOnLoopInductionVar
    : public OpRewritePattern<affine::AffineApplyOp> {
  using OpRewritePattern<affine::AffineApplyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineApplyOp apply,
                                PatternRewriter &rewriter) const override {

    auto ctx = apply->getContext();

    if (apply.getAffineMap().getNumInputs() != 1)
      return failure();
    auto val = apply->getOperand(0);
    auto ivArg = llvm::dyn_cast<BlockArgument>(val);
    if (!ivArg)
      return failure();
    if (!ivArg.getOwner())
      return failure();
    if (!val.hasOneUse())
      return failure();
    if (apply.getResult().use_empty())
      return failure();
    if (auto exec_apply = dyn_cast<air::ExecuteOp>(apply->getParentOp()))
      if (exec_apply->getResult(1).use_empty())
        return failure();
    auto *containingOp = ivArg.getOwner()->getParentOp();

    // Apply affine map to loop step and bound
    if (auto sfo = dyn_cast<scf::ForOp>(containingOp)) {
      if (!getStaticScfForTripCountAsInt(sfo))
        return failure();
      int tripCount = *getStaticScfForTripCountAsInt(sfo);
      auto new_ub = air::evaluateConstantsInMap(
          apply.getAffineMap(),
          SmallVector<std::optional<int64_t>>{
              *mlir::getConstantIntValue(sfo.getUpperBound())},
          ctx);
      auto new_lb = air::evaluateConstantsInMap(
          apply.getAffineMap(),
          SmallVector<std::optional<int64_t>>{
              *mlir::getConstantIntValue(sfo.getLowerBound())},
          ctx);
      if (!new_lb) {
        apply->emitOpError("failed to evaluate lower bound.");
        return failure();
      }
      if (!new_ub) {
        apply->emitOpError("failed to evaluate upper bound.");
        return failure();
      }
      int newStepInInt = llvm::divideCeilSigned(*new_ub - *new_lb, tripCount);
      auto valueType = apply.getResult().getType();
      if (failed(eraseOpFromScfFor(rewriter, sfo, apply)))
        return failure();
      auto res = updateScfForBounds(rewriter, sfo, *new_lb, *new_ub,
                                    newStepInInt, valueType);
      if (failed(res))
        return failure();
    } else if (auto afo = dyn_cast<affine::AffineForOp>(containingOp)) {
      if (!afo.hasConstantBounds())
        return failure();
      int tripCount = *getStaticAffineForTripCountAsInt(afo);
      auto new_ub = air::evaluateConstantsInMap(
          apply.getAffineMap(),
          SmallVector<std::optional<int64_t>>{afo.getConstantUpperBound()},
          ctx);
      auto new_lb = air::evaluateConstantsInMap(
          apply.getAffineMap(),
          SmallVector<std::optional<int64_t>>{afo.getConstantLowerBound()},
          ctx);
      if (!new_lb) {
        apply->emitOpError("failed to evaluate lower bound.");
        return failure();
      }
      if (!new_ub) {
        apply->emitOpError("failed to evaluate upper bound.");
        return failure();
      }
      int newStepInInt = llvm::divideCeilSigned(*new_ub - *new_lb, tripCount);
      IRMapping remap;
      apply.getResult().replaceAllUsesWith(afo.getInductionVar());
      rewriter.eraseOp(apply);
      updateAffineForBounds(afo, *new_lb, *new_ub, newStepInInt);
    } else
      return failure();

    return success();
  }

private:
};

// Fold arith.muli op operating on loop induction variable into loop bounds.
struct CanonicalizeArithMuliOpOnLoopInductionVar
    : public OpRewritePattern<arith::MulIOp> {
  using OpRewritePattern<arith::MulIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::MulIOp op,
                                PatternRewriter &rewriter) const override {
    Operation *containingOp = nullptr;
    Value const_val = nullptr;
    Value var_val = nullptr;
    for (auto val : SmallVector<Value>{op.getLhs(), op.getRhs()}) {
      if (getConstantIntValue(val)) {
        const_val = val;
        continue;
      }
      auto ivArg = llvm::dyn_cast<BlockArgument>(val);
      if (!ivArg)
        continue;
      if (!ivArg.getOwner())
        continue;
      if (!val.hasOneUse())
        continue;
      if (op.getResult().use_empty())
        continue;
      if (auto exec_muli = dyn_cast<air::ExecuteOp>(op->getParentOp()))
        if (exec_muli->getResult(1).use_empty())
          continue;
      if (isa<scf::ForOp>(ivArg.getOwner()->getParentOp())) {
        containingOp = ivArg.getOwner()->getParentOp();
        var_val = val;
      } else if (isa<affine::AffineForOp>(ivArg.getOwner()->getParentOp())) {
        containingOp = ivArg.getOwner()->getParentOp();
        var_val = val;
      }
    }
    if (!containingOp)
      return failure();
    if (!const_val)
      return failure();
    if (!var_val)
      return failure();

    // Apply arith muli to loop step and bound
    int muli_factor = *mlir::getConstantIntValue(const_val);
    if (auto sfo = dyn_cast<scf::ForOp>(containingOp)) {
      if (!getStaticScfForTripCountAsInt(sfo))
        return failure();
      int tripCount = *getStaticScfForTripCountAsInt(sfo);
      int new_ub =
          *mlir::getConstantIntValue(sfo.getUpperBound()) * muli_factor;
      int new_lb =
          *mlir::getConstantIntValue(sfo.getLowerBound()) * muli_factor;
      int newStepInInt = llvm::divideCeilSigned(new_ub - new_lb, tripCount);
      auto valueType = op.getResult().getType();
      if (failed(eraseOpFromScfFor(rewriter, sfo, op)))
        return failure();
      auto res = updateScfForBounds(rewriter, sfo, new_lb, new_ub, newStepInInt,
                                    valueType);
      if (failed(res))
        return failure();
    } else if (auto afo = dyn_cast<affine::AffineForOp>(containingOp)) {
      if (!afo.hasConstantBounds())
        return failure();
      int tripCount = *getStaticAffineForTripCountAsInt(afo);
      int new_ub = afo.getConstantUpperBound() * muli_factor;
      int new_lb = afo.getConstantLowerBound() * muli_factor;
      int newStepInInt = llvm::divideCeilSigned(new_ub - new_lb, tripCount);
      IRMapping remap;
      op.getResult().replaceAllUsesWith(afo.getInductionVar());
      rewriter.eraseOp(op);
      updateAffineForBounds(afo, new_lb, new_ub, newStepInInt);
    } else
      return failure();

    return success();
  }

private:
};

// Fold arith.addi op operating on loop induction variable into loop bounds.
struct CanonicalizeArithAddiOpOnLoopInductionVar
    : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern<arith::AddIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    Operation *containingOp = nullptr;
    Value const_val = nullptr;
    Value var_val = nullptr;
    for (auto val : SmallVector<Value>{op.getLhs(), op.getRhs()}) {
      if (getConstantIntValue(val)) {
        const_val = val;
        continue;
      }
      auto ivArg = llvm::dyn_cast<BlockArgument>(val);
      if (!ivArg)
        continue;
      if (!ivArg.getOwner())
        continue;
      if (!val.hasOneUse())
        continue;
      if (op.getResult().use_empty())
        continue;
      if (auto exec_addi = dyn_cast<air::ExecuteOp>(op->getParentOp()))
        if (exec_addi->getResult(1).use_empty())
          continue;
      if (isa<scf::ForOp>(ivArg.getOwner()->getParentOp())) {
        containingOp = ivArg.getOwner()->getParentOp();
        var_val = val;
      } else if (isa<affine::AffineForOp>(ivArg.getOwner()->getParentOp())) {
        containingOp = ivArg.getOwner()->getParentOp();
        var_val = val;
      }
    }
    if (!containingOp)
      return failure();
    if (!const_val)
      return failure();
    if (!var_val)
      return failure();

    // Apply arith muli to loop step and bound
    int addi_operand = *mlir::getConstantIntValue(const_val);
    if (auto sfo = dyn_cast<scf::ForOp>(containingOp)) {
      if (!getStaticScfForTripCountAsInt(sfo))
        return failure();
      int tripCount = *getStaticScfForTripCountAsInt(sfo);
      int new_ub =
          *mlir::getConstantIntValue(sfo.getUpperBound()) + addi_operand;
      int new_lb =
          *mlir::getConstantIntValue(sfo.getLowerBound()) + addi_operand;
      int newStepInInt = llvm::divideCeilSigned(new_ub - new_lb, tripCount);
      auto valueType = op.getResult().getType();
      if (failed(eraseOpFromScfFor(rewriter, sfo, op)))
        return failure();
      auto res = updateScfForBounds(rewriter, sfo, new_lb, new_ub, newStepInInt,
                                    valueType);
      if (failed(res))
        return failure();
    } else if (auto afo = dyn_cast<affine::AffineForOp>(containingOp)) {
      if (!afo.hasConstantBounds())
        return failure();
      int tripCount = *getStaticAffineForTripCountAsInt(afo);
      int new_ub = afo.getConstantUpperBound() + addi_operand;
      int new_lb = afo.getConstantLowerBound() + addi_operand;
      int newStepInInt = llvm::divideCeilSigned(new_ub - new_lb, tripCount);
      IRMapping remap;
      op.getResult().replaceAllUsesWith(afo.getInductionVar());
      rewriter.eraseOp(op);
      updateAffineForBounds(afo, new_lb, new_ub, newStepInInt);
    } else
      return failure();

    return success();
  }

private:
};

// Fold arith.index_cast op operating on loop induction variable into loop
// bounds.
struct CanonicalizeArithIndexCastOpOnLoopInductionVar
    : public OpRewritePattern<arith::IndexCastOp> {
  using OpRewritePattern<arith::IndexCastOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::IndexCastOp op,
                                PatternRewriter &rewriter) const override {
    Operation *containingOp = nullptr;
    Value var_val = nullptr;

    Value val = op.getOperand();
    if (!val)
      return failure();
    auto ivArg = llvm::dyn_cast<BlockArgument>(val);
    if (!ivArg)
      return failure();
    if (!ivArg.getOwner())
      return failure();
    if (!val.hasOneUse())
      return failure();
    if (op.getResult().use_empty())
      return failure();
    if (auto exec = dyn_cast<air::ExecuteOp>(op->getParentOp()))
      if (exec->getResult(1).use_empty())
        return failure();
    if (auto containingScfFor =
            dyn_cast<scf::ForOp>(ivArg.getOwner()->getParentOp())) {
      if (containingScfFor.getInductionVar() != ivArg)
        return failure();
      containingOp = containingScfFor;
      var_val = val;
    } else if (isa<affine::AffineForOp>(ivArg.getOwner()->getParentOp())) {
      // Affine for op only operates on index type.
      return failure();
    } else
      return failure();

    // Cast back all of the loop's bounds to index type.
    auto sfo = dyn_cast<scf::ForOp>(containingOp);
    if (!getStaticScfForTripCountAsInt(sfo))
      return failure();
    int new_ub = *mlir::getConstantIntValue(sfo.getUpperBound());
    int new_lb = *mlir::getConstantIntValue(sfo.getLowerBound());
    int new_step = *mlir::getConstantIntValue(sfo.getStep());
    auto valueType = op.getResult().getType();
    if (failed(eraseOpFromScfFor(rewriter, sfo, op)))
      return failure();
    auto res =
        updateScfForBounds(rewriter, sfo, new_lb, new_ub, new_step, valueType);
    if (failed(res))
      return failure();

    return success();
  }

private:
};

struct AIRSpecializeChannelWrapAndStrideInScfFor
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  AIRSpecializeChannelWrapAndStrideInScfFor(MLIRContext *ctx, int &maxNumDims,
                                            int &maxSize,
                                            bool &enableRepeatAtHighestDim)
      : OpRewritePattern(ctx), maxNumDims(maxNumDims), maxSize(maxSize),
        enableRepeatAtHighestDim(enableRepeatAtHighestDim) {}

  LogicalResult matchAndRewrite(scf::ForOp for_op,
                                PatternRewriter &rewriter) const override {
    auto loc = for_op->getLoc();
    auto ctx = for_op->getContext();

    // Check if the loop is the outermost loop in a perfect loop nest
    if (!hasNImpureOps(for_op.getBody(), 1))
      return failure();

    // Check if the loop contains exactly one channel op
    if (llvm::range_size(for_op.getBody()->getOps<air::ChannelInterface>()) !=
        1)
      return failure();
    air::ChannelInterface channel_op =
        *(for_op.getBody()->getOps<air::ChannelInterface>().begin());

    // Fold for loops into channel op's wrap and stride fields
    SmallVector<Value> offsets = channel_op.getOffsets();
    SmallVector<Value> wraps = channel_op.getSizes();
    SmallVector<Value> strides = channel_op.getStrides();

    OpBuilder b(channel_op);
    (void)canonicalizeWrapAndStrideList(
        b, offsets, wraps, strides,
        air::getTensorVolume(channel_op.getMemref().getType()), maxSize);

    // If empty offsets/sizes/strides, then populate the lists with default
    // values.
    if (offsets.empty() && wraps.empty() && strides.empty())
      populateDefaultWrapsAndStrides(b, channel_op.getMemref(), offsets, wraps,
                                     strides);

    // Check if the number of wrap-and-stride dims exceed maxNumDims. TODO:
    // expand this to take into account more wrap-and-stride constraints.
    int numActualWrapDims = 0; // Count the number of actual hardware wrap
                               // dimensions, with wrap value greater than one.
    for (auto v : wraps) {
      if (*getConstantIntValue(v) > 1)
        numActualWrapDims++;
    }
    if (maxNumDims >= 0 && numActualWrapDims > maxNumDims - 1)
      return failure();

    auto res = foldForLoopNestAsExtendedSizesAndStrides(
        rewriter, for_op.getOperation(), channel_op.getOperation(), offsets,
        wraps, strides, channel_op.getMemref());
    if (res.failed())
      return failure();

    (void)canonicalizeWrapAndStrideList(
        rewriter, offsets, wraps, strides,
        air::getTensorVolume(channel_op.getMemref().getType()), maxSize);

    // Whether repeat (i.e. stride = 0) is supported at highest dimension.
    if (enableRepeatAtHighestDim && !wraps.empty()) {
      // Force bump up number of dims to maxNumDims.
      auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      auto oneIdx = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      while ((int)offsets.size() < maxNumDims) {
        offsets.insert(offsets.begin(), zeroIdx);
      }
      while ((int)wraps.size() < maxNumDims) {
        wraps.insert(wraps.begin(), oneIdx);
      }
      while ((int)strides.size() < maxNumDims) {
        strides.insert(strides.begin(), zeroIdx);
      }
      // Stride = 0 means repeat that dimension. If highest dimension (dim 0) is
      // not used, then move the repeat dimension to dim 0, which is the only
      // dim with repeat capability. Else, fall back to unrolling BDs.
      unsigned activeDimsInBetween = 0;
      for (unsigned i = 1; i < strides.size(); i++) {
        auto constWrap = mlir::getConstantIntValue(wraps[i]);
        auto constStride = mlir::getConstantIntValue(strides[i]);
        if (!constWrap)
          continue;
        if (!constStride)
          continue;
        if (*constWrap <= 1)
          continue; // Inactive dimension. Continue.
        if (*constStride) {
          // Found active dimension after dim 0. Any subsequent repeat dimension
          // shall not bump to dim 0 anymore.
          activeDimsInBetween++;
          continue;
        }
        // This is a repeat dimension. Start converting offsets, wraps and
        // strides...
        if (mlir::getConstantIntValue(wraps[0]) &&
            *mlir::getConstantIntValue(wraps[0]) == 1 && !activeDimsInBetween) {
          // Dimension 0 is available. Move the repeat dimension i to dimension
          // 0.
          auto tmp = wraps[0];
          wraps[0] = wraps[i];
          wraps[i] = tmp;
          tmp = strides[0];
          strides[0] = strides[i];
          strides[i] = tmp;
          tmp = offsets[0];
          offsets[0] = offsets[i];
          offsets[i] = tmp;
        } else {
          return air::loopUnrollFullWithAsyncTokenPreserved(for_op);
        }
      }
    }

    air::ChannelInterface new_chan_op = nullptr;
    SmallVector<Type, 1> tys;
    if (isAsyncOp(channel_op.getOperation())) {
      tys.push_back(air::AsyncTokenType::get(ctx));
    }
    SmallVector<Value, 1> deps =
        for_op.getOperands().drop_front(for_op.getNumControlOperands());

    // Hoist any pure ops that the new channel op depends on.
    SmallVector<Value> new_opers = llvm::to_vector(
        llvm::concat<Value>(SmallVector<Value>{channel_op.getMemref()},
                            channel_op.getIndices(), offsets, wraps, strides));
    IRMapping remap;
    llvm::SetVector<Operation *> backwardSlices;
    air::getBackwardSliceInRegion(rewriter, &for_op.getRegion(), new_opers,
                                  backwardSlices);
    for (auto o : backwardSlices) {
      auto cloned = rewriter.clone(*o, remap);
      clearAsyncDependenciesOfAsyncOp(cloned);
      for (auto token : deps)
        addAsyncDependencyIfNew(cloned, token);
      if (auto token = getAsyncTokenFromOp(cloned))
        deps.push_back(token);
    }

    // Create specialized air.channel.put/get.
    if (isa<air::ChannelPutOp>(channel_op))
      new_chan_op = rewriter.create<air::ChannelPutOp>(
          loc, tys, deps, channel_op.getChanName(),
          air::lookupOrDefaultRange(channel_op.getIndices(), remap),
          air::lookupOrDefaultRange(channel_op.getMemref(), remap),
          air::lookupOrDefaultRange(offsets, remap),
          air::lookupOrDefaultRange(wraps, remap),
          air::lookupOrDefaultRange(strides, remap));
    else if (isa<air::ChannelGetOp>(channel_op))
      new_chan_op = rewriter.create<air::ChannelGetOp>(
          loc, tys, deps, channel_op.getChanName(),
          air::lookupOrDefaultRange(channel_op.getIndices(), remap),
          air::lookupOrDefaultRange(channel_op.getMemref(), remap),
          air::lookupOrDefaultRange(offsets, remap),
          air::lookupOrDefaultRange(wraps, remap),
          air::lookupOrDefaultRange(strides, remap));
    new_chan_op->setAttrs(channel_op->getDiscardableAttrDictionary());

    // Clear all external uses of for_op before erasing it.
    for (auto res : for_op.getResults()) {
      if (isa<air::AsyncTokenType>(res.getType())) {
        res.replaceAllUsesWith(
            dyn_cast<air::AsyncOpInterface>(new_chan_op.getOperation())
                .getAsyncToken());
      }
    }
    rewriter.replaceAllUsesWith(for_op.getInductionVar(),
                                for_op.getLowerBound());
    rewriter.eraseOp(for_op.getOperation());

    return success();
  }

private:
  int &maxNumDims;
  int &maxSize;
  bool &enableRepeatAtHighestDim;
};

// This pattern should be executed after
// AIRSpecializeChannelWrapAndStrideInScfFor. The pattern unrolls any remaining
// scf.for loops that iterates over air.channel.put/get but cannot be converted
// directly to wraps and strides. The unrolled air.channel.put/get ops form a bd
// chain.
struct AIRUnrollScfForIntoBDChain : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp for_op,
                                PatternRewriter &rewriter) const override {
    // Check if the loop contains only air.channel.put/get ops, or pure ops.
    auto containsOnlyAIRChannels = [](Block *block) {
      if (block->getOperations().empty())
        return false;
      for (auto &o : block->getOperations()) {
        if (isa<air::ChannelInterface>(o))
          continue;
        else if (isa<air::WaitAllOp>(o))
          continue;
        else if (air::isPure(&o))
          continue;
        return false;
      }
      return true;
    };

    if (!containsOnlyAIRChannels(for_op.getBody()))
      return failure();

    // Check if the loop is ping-pong buffered. Ping-pong buffered loop's body
    // already forms a bd chain.
    int resAsyncTokenCount = 0;
    for (auto resTy : for_op->getResultTypes())
      if (isa<air::AsyncTokenType>(resTy))
        resAsyncTokenCount++;
    if (resAsyncTokenCount > 1)
      return failure();

    // Unroll loop; preserve async tokens after unroll.
    return air::loopUnrollFullWithAsyncTokenPreserved(for_op);
  }

private:
};

// Affine for version of the `AIRUnrollScfForIntoBDChain` pattern above.
struct AIRUnrollAffineForIntoBDChain
    : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp for_op,
                                PatternRewriter &rewriter) const override {
    // Check if the loop contains only air.channel.put/get ops, or pure ops.
    auto containsOnlyAIRChannels = [](Block *block) {
      if (block->getOperations().empty())
        return false;
      for (auto &o : block->getOperations()) {
        if (isa<air::ChannelInterface>(o))
          continue;
        else if (isa<air::WaitAllOp>(o))
          continue;
        else if (air::isPure(&o))
          continue;
        return false;
      }
      return true;
    };

    if (!containsOnlyAIRChannels(for_op.getBody()))
      return failure();

    auto unroll_factor = air::getStaticAffineForTripCountAsInt(for_op);
    if (!unroll_factor)
      return failure(); // Dynamic loop bound.
    (void)loopUnrollFull(for_op);

    return success();
  }

private:
};

struct AIRSpecializeChannelWrapAndStrideInAffineFor
    : public OpRewritePattern<affine::AffineForOp> {
  using OpRewritePattern<affine::AffineForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(affine::AffineForOp for_op,
                                PatternRewriter &rewriter) const override {
    auto loc = for_op->getLoc();
    auto ctx = for_op->getContext();

    // Check if the loop is the outermost loop in a perfect loop nest
    if (!hasNImpureOps(for_op.getBody(), 1))
      return failure();

    // Check if the loop contains exactly one channel op
    if (llvm::range_size(for_op.getBody()->getOps<air::ChannelInterface>()) !=
        1)
      return failure();
    air::ChannelInterface channel_op =
        *(for_op.getBody()->getOps<air::ChannelInterface>().begin());

    // // Fold for loops int channel op's wrap and stride fields
    SmallVector<Value> offsets = channel_op.getOffsets();
    SmallVector<Value> wraps = channel_op.getSizes();
    SmallVector<Value> strides = channel_op.getStrides();

    (void)canonicalizeWrapAndStrideList(
        rewriter, offsets, wraps, strides,
        air::getTensorVolume(channel_op.getMemref().getType()));

    // If empty offsets/sizes/strides, then populate the lists with default
    // values.
    if (offsets.empty() && wraps.empty() && strides.empty())
      populateDefaultWrapsAndStrides(rewriter, channel_op.getMemref(), offsets,
                                     wraps, strides);

    auto res = foldForLoopNestAsExtendedSizesAndStrides(
        rewriter, for_op.getOperation(), channel_op.getOperation(), offsets,
        wraps, strides, channel_op.getMemref());
    if (res.failed())
      return failure();

    (void)canonicalizeWrapAndStrideList(
        rewriter, offsets, wraps, strides,
        air::getTensorVolume(channel_op.getMemref().getType()));

    air::ChannelInterface new_chan_op = nullptr;
    SmallVector<Type, 1> tys;
    if (isAsyncOp(channel_op.getOperation())) {
      tys.push_back(air::AsyncTokenType::get(ctx));
    }
    SmallVector<Value, 1> deps =
        for_op.getOperands().drop_front(for_op.getNumControlOperands());

    // Hoist any pure ops that the new channel op depends on.
    SmallVector<Value> new_opers = llvm::to_vector(
        llvm::concat<Value>(SmallVector<Value>{channel_op.getMemref()},
                            channel_op.getIndices(), offsets, wraps, strides));
    IRMapping remap;
    llvm::SetVector<Operation *> backwardSlices;
    air::getBackwardSliceInRegion(rewriter, &for_op.getRegion(), new_opers,
                                  backwardSlices);
    for (auto o : backwardSlices) {
      auto cloned = rewriter.clone(*o, remap);
      clearAsyncDependenciesOfAsyncOp(cloned);
      for (auto token : deps)
        addAsyncDependencyIfNew(cloned, token);
      if (auto token = getAsyncTokenFromOp(cloned))
        deps.push_back(token);
    }

    // Create specialized air.channel.put/get.
    if (isa<air::ChannelPutOp>(channel_op))
      new_chan_op = rewriter.create<air::ChannelPutOp>(
          loc, tys, deps, channel_op.getChanName(),
          air::lookupOrDefaultRange(channel_op.getIndices(), remap),
          air::lookupOrDefaultRange(channel_op.getMemref(), remap),
          air::lookupOrDefaultRange(offsets, remap),
          air::lookupOrDefaultRange(wraps, remap),
          air::lookupOrDefaultRange(strides, remap));
    else if (isa<air::ChannelGetOp>(channel_op))
      new_chan_op = rewriter.create<air::ChannelGetOp>(
          loc, tys, deps, channel_op.getChanName(),
          air::lookupOrDefaultRange(channel_op.getIndices(), remap),
          air::lookupOrDefaultRange(channel_op.getMemref(), remap),
          air::lookupOrDefaultRange(offsets, remap),
          air::lookupOrDefaultRange(wraps, remap),
          air::lookupOrDefaultRange(strides, remap));
    new_chan_op->setAttrs(channel_op->getDiscardableAttrDictionary());

    for (auto res : for_op.getResults()) {
      if (isa<air::AsyncTokenType>(res.getType())) {
        res.replaceAllUsesWith(
            dyn_cast<air::AsyncOpInterface>(new_chan_op.getOperation())
                .getAsyncToken());
      }
    }
    rewriter.replaceAllUsesWith(for_op.getInductionVar(),
                                rewriter.create<arith::ConstantIndexOp>(
                                    loc, for_op.getConstantLowerBound()));
    rewriter.eraseOp(for_op.getOperation());

    return success();
  }

private:
};

/// This pattern canonicalizes the offset/size/stride lists of `OpT` channel
/// put/get operations.
///
/// **Main transformations**:
/// 1. Detect whether a "highest-dimension repeat" pattern is active (special
///    case where the highest dimension repeats and requires padding to
///    `maxNumDims`).
/// 2. Canonicalize the wrap-and-stride list by invoking
///    `canonicalizeWrapAndStrideList()`, which normalizes
///    offsets/sizes/strides.
/// 3. If highest-dimension repeat is active, extend the rank of
///    offsets/sizes/strides to `maxNumDims` by inserting zeros/ones.
/// 4. Recreate the `OpT` operation with the canonicalized parameters and
///    replace the original operation.
template <typename OpT>
struct AIRCanonicalizeChannelPutGetOpWrapAndStrideList
    : public OpRewritePattern<OpT> {
  using OpRewritePattern<OpT>::OpRewritePattern;

  AIRCanonicalizeChannelPutGetOpWrapAndStrideList(
      MLIRContext *ctx, int &maxSize, int &maxNumDims,
      bool &enableRepeatAtHighestDim)
      : OpRewritePattern<OpT>(ctx), maxSize(maxSize), maxNumDims(maxNumDims),
        enableRepeatAtHighestDim(enableRepeatAtHighestDim) {}

  LogicalResult matchAndRewrite(OpT op,
                                PatternRewriter &rewriter) const override {
    // Collect async token types and dependencies if the op is asynchronous.
    SmallVector<Value, 1> deps;
    SmallVector<Type, 1> tys;
    if (isAsyncOp(op)) {
      tys.push_back(air::AsyncTokenType::get(op->getContext()));
      deps = op.getAsyncDependencies();
    }

    // Extract offsets, sizes, and strides from the op.
    SmallVector<Value> offsets = op.getOffsets();
    SmallVector<Value> sizes = op.getSizes();
    SmallVector<Value> strides = op.getStrides();

    // Detect if highest-dimension repeat logic should be applied.
    // This is true when:
    //  (1) The option enableRepeatAtHighestDim is set,
    //  (2) The stride list is not empty,
    //  (3) The highest (first) stride is 0, indicating repeat dimension,
    //  (4) The highest (first) size is not 1, indicating non-zero repetition.
    bool highestDimRepeatActive = enableRepeatAtHighestDim &&
                                  !strides.empty() &&
                                  *getConstantIntValue(strides.front()) == 0 &&
                                  *getConstantIntValue(sizes.front()) != 1;
    // If highest-dimension repeat is active but the op already has the maximum
    // number of dimensions, no rewrite is needed.
    if (highestDimRepeatActive && (int)offsets.size() == maxNumDims) {
      return failure();
    } else {
      // Canonicalize offsets/sizes/strides using a helper function.
      if (failed(canonicalizeWrapAndStrideList(
              rewriter, offsets, sizes, strides,
              air::getTensorVolume(op.getMemref().getType()), maxSize)))
        return failure();

      // When highest-dimension repeat is active, pad offsets/sizes/strides to
      // match maxNumDims by inserting:
      //  - offset = 0
      //  - size   = 1
      //  - stride = 0
      if (highestDimRepeatActive) {
        auto zeroIdx = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 0);
        auto oneIdx = rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1);
        while ((int)offsets.size() < maxNumDims) {
          offsets.insert(offsets.begin() + 1, zeroIdx);
        }
        while ((int)sizes.size() < maxNumDims) {
          sizes.insert(sizes.begin() + 1, oneIdx);
        }
        while ((int)strides.size() < maxNumDims) {
          strides.insert(strides.begin() + 1, zeroIdx);
        }
      }
    }

    // Create a new op with the canonicalized attributes and operands.
    auto attrs = op->getDiscardableAttrDictionary();
    auto new_op = rewriter.replaceOpWithNewOp<OpT>(
        op, tys, deps, op.getChanName(), op.getIndices(), op.getMemref(),
        offsets, sizes, strides);
    new_op->setAttrs(attrs);

    return success();
  }

private:
  int &maxSize;
  int &maxNumDims;
  bool &enableRepeatAtHighestDim;
};

struct UnrollChannelByFactorPattern {

public:
  void runUnrollChannelByFactorPattern(func::FuncOp funcOp, Operation *op,
                                       int chanDim, int factor) {
    air::ChannelOp chan_op = dyn_cast<air::ChannelOp>(op);
    OpBuilder builder(op);
    SmallVector<int64_t, 2> sizes =
        extractFromIntegerArrayAttr<int64_t>(chan_op.getSize());
    if ((unsigned)chanDim >= sizes.size())
      return;
    if (sizes[chanDim] != 1) {
      op->emitOpError("unrolling channel in a dimension with a size not equal "
                      "to one. Currently unsupported");
      return;
    }

    dim = chanDim;

    // Update channel declaration
    sizes[chanDim] *= factor;
    builder.create<air::ChannelOp>(op->getLoc(), chan_op.getSymName().str(),
                                   builder.getI64ArrayAttr(sizes),
                                   builder.getStringAttr("dma_stream"));

    // Add scf.parallel to unroll channel puts and gets
    auto puts = air::getChannelPutOpThroughSymbol(chan_op);
    auto gets = air::getChannelGetOpThroughSymbol(chan_op);
    for (auto put : puts) {
      builder.setInsertionPoint(put);
      auto init_val =
          createWaitAllToCollectIncomingTokens(builder, put.getOperation());
      if (init_val)
        createSCFParallelToUnroll<air::ChannelPutOp>(builder, put, init_val);
    }
    for (auto get : gets) {
      builder.setInsertionPoint(get);
      auto init_val =
          createWaitAllToCollectIncomingTokens(builder, get.getOperation());
      if (init_val)
        createSCFParallelToUnroll<air::ChannelGetOp>(builder, get, init_val);
    }

    for (auto put : puts) {
      put->erase();
    }
    for (auto get : gets) {
      get->erase();
    }
    op->erase();
  }

private:
  int dim = 0;
  int factor = 1;

  Value createWaitAllToCollectIncomingTokens(OpBuilder builder, Operation *op) {
    auto async_op = dyn_cast<air::AsyncOpInterface>(op);
    if (!async_op) {
      op->emitOpError("is air.channel_put but not async");
      return nullptr;
    } else if (async_op.getAsyncDependencies().empty()) {
      SmallVector<Value> dep_list = {};
      return builder
          .create<air::WaitAllOp>(
              op->getLoc(), air::AsyncTokenType::get(builder.getContext()),
              dep_list)
          .getAsyncToken();
    } else if (async_op.getAsyncDependencies().size() == 1)
      return async_op.getAsyncDependencies().front();
    return builder
        .create<air::WaitAllOp>(op->getLoc(),
                                air::AsyncTokenType::get(builder.getContext()),
                                async_op.getAsyncDependencies())
        .getAsyncToken();
  }

  template <typename T>
  scf::ParallelOp createSCFParallelToUnroll(OpBuilder builder, T op,
                                            Value init_val) {
    if (!isa<air::AsyncOpInterface>(op.getOperation())) {
      op->emitOpError("is air.channel op but not async");
      return nullptr;
    }

    auto loc = op->getLoc();
    SmallVector<Value, 1> merged_incoming_token = {init_val};
    SmallVector<Value, 1> LBs, UBs, Steps;

    LBs.push_back(builder.create<arith::ConstantIndexOp>(loc, 0));
    UBs.push_back(builder.create<arith::ConstantIndexOp>(loc, factor));
    Steps.push_back(builder.create<arith::ConstantIndexOp>(loc, 1));

    auto par = builder.create<scf::ParallelOp>(loc, LBs, UBs, Steps,
                                               merged_incoming_token);

    builder.setInsertionPointToStart(par.getBody());

    // Update channel indices
    SmallVector<Value, 1> new_channel_idx = {};
    if (op.getIndices().empty()) {
      auto const_0 = builder.create<arith::ConstantIndexOp>(par->getLoc(), 0);
      new_channel_idx = {const_0, const_0};
      new_channel_idx[dim] = par.getInductionVars()[0];
    } else
      op->emitOpError(
          "unrolling a sub-channel in a channel bundle currently unsupported");
    // Update memref size (divide by factor)
    SmallVector<Value, 1> new_sizes = op.getSizes();
    if (new_sizes.empty()) {
      auto memTy = llvm::cast<BaseMemRefType>(op.getMemref().getType());
      for (auto d : getTensorShape(memTy)) {
        new_sizes.push_back(
            builder.create<arith::ConstantIndexOp>(par->getLoc(), d));
      }
    }
    auto size_op = new_sizes[dim].getDefiningOp();
    if (auto cIOp = dyn_cast_if_present<arith::ConstantIndexOp>(size_op)) {
      auto val = cIOp.value();
      val = llvm::divideCeilSigned(val, factor);
      new_sizes[dim] =
          builder.create<arith::ConstantIndexOp>(par->getLoc(), val);
    } else {
      new_sizes[dim] = builder.create<arith::FloorDivSIOp>(
          par->getLoc(), new_sizes[dim],
          builder.create<arith::ConstantIndexOp>(par->getLoc(), factor));
    }
    // Update offset (+ induction var. x size)
    SmallVector<Value, 1> new_offsets = op.getOffsets();
    if (new_offsets.empty()) {
      auto const_0 = builder.create<arith::ConstantIndexOp>(par->getLoc(), 0);
      new_offsets = {const_0, const_0};
    }
    auto prod = builder.create<arith::MulIOp>(
        par->getLoc(), new_channel_idx[dim], new_sizes[dim]);
    new_offsets[dim] =
        builder.create<arith::AddIOp>(par->getLoc(), new_offsets[dim], prod);
    // Update strides
    SmallVector<Value, 1> new_strides = op.getStrides();
    if (new_strides.empty()) {
      auto const_1 = builder.create<arith::ConstantIndexOp>(par->getLoc(), 1);
      new_strides = {const_1, const_1};
    }

    // Create new channel op
    SmallVector<Type, 1> tys = {air::AsyncTokenType::get(builder.getContext())};
    auto new_op = builder.create<T>(
        par.getLoc(), tys, merged_incoming_token, op.getChanName(),
        new_channel_idx, op.getMemref(), new_offsets, new_sizes, new_strides);

    // Create scf::ReduceOp
    air::createSCFReduceForAsyncSCFParallel(
        builder, par.getLoc(), new_op.getAsyncToken(), op->getContext());

    // Create air.wait_all as a barrier to protect the non-blocking scf.yield
    // event
    builder.setInsertionPointAfter(par);
    SmallVector<Value, 1> barrier_deps = {par.getResults().front()};
    auto barrier = builder.create<air::WaitAllOp>(
        builder.getUnknownLoc(), air::AsyncTokenType::get(builder.getContext()),
        barrier_deps);

    // Reconnect dependencies
    op.getAsyncToken().replaceAllUsesWith(barrier.getAsyncToken());

    return par;
  }
};

struct BroadcastDetection {

public:
  // Trace dma ops' dependency to loop induction variables
  void getDmaOpLoopDependency(func::FuncOp f) {
    f.walk([&](air::DmaMemcpyNdOp dma_op) {
      int src_memspace = -1;
      int dst_memspace = -1;
      if (dma_op.getSrcMemref())
        src_memspace =
            llvm::cast<BaseMemRefType>(dma_op.getSrcMemref().getType())
                .getMemorySpaceAsInt();
      if (dma_op.getDstMemref())
        dst_memspace =
            llvm::cast<BaseMemRefType>(dma_op.getDstMemref().getType())
                .getMemorySpaceAsInt();
      bool isL1Memcpy = (src_memspace == (int)air::MemorySpace::L1) ||
                        (dst_memspace == (int)air::MemorySpace::L1);
      if (dma_op->getParentOfType<xilinx::air::HerdOp>() && isL1Memcpy) {
        // Start recursively tracing for loop induction variables
        dma_op_history.push_back(dma_op);
        SmallVector<Value, 1> loop_dep_history;
        std::vector<Operation *> op_history;
        auto memcpyif_op = dyn_cast<MemcpyInterface>(dma_op.getOperation());
        traceDependentInductionVar(memcpyif_op, loop_dep_history, op_history);
        dma_op_loop_dep_history.push_back(loop_dep_history);
      }
    });
  }

  // Detect boradcast opportunity based on dependency to loops
  void broadcastDetection() {
    for (unsigned i = 0; i < dma_op_history.size(); i++) {
      auto dma_op = dma_op_history[i];
      SmallVector<Value, 1> loop_dep_history = dma_op_loop_dep_history[i];
      air::HerdOp hl_op = nullptr;
      bool isVariantWrtHerdRows = false;
      bool isVariantWrtHerdCols = false;
      // Create an affine set to represent the broadcast pattern
      auto ctx = dma_op->getContext();
      for (auto v : loop_dep_history) {
        // Check row-wise or col-wise broadcastable based on variance wrt herd
        // dimensions.
        if (getHerdArgOwner(v)) {
          hl_op = getHerdArgOwner(v);
          if (v == hl_op.getIds()[0]) {
            isVariantWrtHerdRows = true;
          }
          if (v == hl_op.getIds()[1]) {
            isVariantWrtHerdCols = true;
          }
        }
      }

      if (!hl_op) {
        // If dma op is completely independent of the parent herd induction
        // vars, then it is broadcastable either row or column wise.
        hl_op = dma_op->getParentOfType<air::HerdOp>();
        isVariantWrtHerdRows = false;
        isVariantWrtHerdCols = false;
      }

      auto numColsOp = dyn_cast<arith::ConstantIndexOp>(
          hl_op.getSizeOperands()[1].getDefiningOp());
      auto numCols = numColsOp.value();
      auto numRowsOp = dyn_cast<arith::ConstantIndexOp>(
          hl_op.getSizeOperands()[0].getDefiningOp());
      auto numRows = numRowsOp.value();
      if (isVariantWrtHerdRows && !isVariantWrtHerdCols) {
        if (numCols > 1) {
          SmallVector<AffineExpr, 5> constraints{
              getAffineDimExpr(0, ctx) - getAffineSymbolExpr(0, ctx),
              getAffineDimExpr(1, ctx), numCols - 1 - getAffineDimExpr(1, ctx),
              getAffineSymbolExpr(0, ctx),
              numRows - 1 - getAffineSymbolExpr(0, ctx)};
          SmallVector<bool, 5> eqflags{true, false, false, false, false};
          auto int_set = IntegerSet::get(2, 1, constraints, eqflags);
          dma_op->setAttr("broadcast_pattern",
                          mlir::IntegerSetAttr::get(int_set));
        }
      } else if (!isVariantWrtHerdRows && isVariantWrtHerdCols) {
        if (numRows > 1) {
          SmallVector<AffineExpr, 5> constraints{
              getAffineDimExpr(0, ctx), numRows - 1 - getAffineDimExpr(0, ctx),
              getAffineDimExpr(1, ctx) - getAffineSymbolExpr(0, ctx),
              getAffineSymbolExpr(0, ctx),
              numCols - 1 - getAffineSymbolExpr(0, ctx)};
          SmallVector<bool, 5> eqflags{false, false, true, false, false};
          auto int_set = IntegerSet::get(2, 1, constraints, eqflags);
          dma_op->setAttr("broadcast_pattern",
                          mlir::IntegerSetAttr::get(int_set));
        }
      } else if (!isVariantWrtHerdRows && !isVariantWrtHerdCols) {
        // If a dma op is independent of herd induction vars, then we broadcast
        // it to every core in the herd.
        if (numRows == 1 && numCols == 1)
          continue;
        else if (numRows > 1 && numCols == 1) {
          SmallVector<AffineExpr, 5> constraints{
              getAffineDimExpr(0, ctx), numRows - 1 - getAffineDimExpr(0, ctx),
              getAffineDimExpr(1, ctx) - getAffineSymbolExpr(0, ctx),
              getAffineSymbolExpr(0, ctx),
              numCols - 1 - getAffineSymbolExpr(0, ctx)};
          SmallVector<bool, 5> eqflags{false, false, true, false, false};
          auto int_set = IntegerSet::get(2, 1, constraints, eqflags);
          dma_op->setAttr("broadcast_pattern",
                          mlir::IntegerSetAttr::get(int_set));
        } else if (numRows == 1 && numCols > 1) {
          SmallVector<AffineExpr, 5> constraints{
              getAffineDimExpr(0, ctx) - getAffineSymbolExpr(0, ctx),
              getAffineDimExpr(1, ctx), numCols - 1 - getAffineDimExpr(1, ctx),
              getAffineSymbolExpr(0, ctx),
              numRows - 1 - getAffineSymbolExpr(0, ctx)};
          SmallVector<bool, 5> eqflags{true, false, false, false, false};
          auto int_set = IntegerSet::get(2, 1, constraints, eqflags);
          dma_op->setAttr("broadcast_pattern",
                          mlir::IntegerSetAttr::get(int_set));
        } else {
          // Broadcast to a 2d array of cores
          SmallVector<AffineExpr, 6> constraints{
              getAffineDimExpr(0, ctx),
              numRows - 1 - getAffineDimExpr(0, ctx),
              getAffineDimExpr(1, ctx),
              numCols - 1 - getAffineDimExpr(1, ctx),
              getAffineSymbolExpr(0, ctx),
              -getAffineSymbolExpr(0, ctx)};
          SmallVector<bool, 5> eqflags{false, false, false,
                                       false, false, false};
          auto int_set = IntegerSet::get(2, 1, constraints, eqflags);
          dma_op->setAttr("broadcast_pattern",
                          mlir::IntegerSetAttr::get(int_set));
        }
      }
    }
  }

  void runBroadcastPattern(func::FuncOp funcOp) {
    // Trace dma ops' dependency to loop induction variables
    // This info will be used for broadcast detection
    getDmaOpLoopDependency(funcOp);
    broadcastDetection();
  }

private:
  // DMA dependency to loop induction variables
  std::vector<MemcpyInterface> dma_op_history;
  SmallVector<SmallVector<Value, 1>, 1> dma_op_loop_dep_history;
};

struct PruneLinalgGenericInputDma {

public:
  void runLinalgGenericPattern(func::FuncOp funcOp) {
    // Detect linalg.GenericOps with redundant input ports
    std::vector<OpOperand *> non_input_generic_operands;
    funcOp.walk([&](linalg::GenericOp generic_op) {
      getNonInputOperands(generic_op, non_input_generic_operands);
      // DMAs copying into these linalg.GenericOp input ports are redundant
      for (auto opoperand : non_input_generic_operands) {
        findAndPruneRedundantDma(opoperand);
      }
    });
  }

  void getNonInputOperands(linalg::GenericOp generic_op,
                           std::vector<OpOperand *> &operands_history) {
    for (auto &g_opoperand : generic_op->getOpOperands()) {
      if (!generic_op.payloadUsesValueFromOperand(&g_opoperand))
        operands_history.push_back(&g_opoperand);
    }
  }

  void findAndPruneRedundantDma(mlir::OpOperand *opoperand) {
    // Elevate to operand of herd launch
    unsigned operand_id = opoperand->getOperandNumber();
    auto op = opoperand->getOwner();
    auto v = op->getOperand(operand_id);

    air::AsyncOpInterface async_op;
    if (air::ExecuteOp exec_op = op->getParentOfType<air::ExecuteOp>()) {
      async_op = dyn_cast<air::AsyncOpInterface>(exec_op.getOperation());
    } else if (auto hl_op = dyn_cast<air::HerdOp>(op)) {
      async_op = dyn_cast<air::AsyncOpInterface>(hl_op.getOperation());
    } else if (auto hier_op = dyn_cast<air::HierarchyInterface>(op)) {
      async_op = dyn_cast<air::AsyncOpInterface>(hier_op.getOperation());
    } else {
      return;
    }
    auto dep_list = async_op.getAsyncDependencies();
    for (int i = dep_list.size() - 1; i >= 0; i--) {
      auto upstream_op = dep_list[i].getDefiningOp();
      if (auto upstream_dma =
              dyn_cast_if_present<air::DmaMemcpyNdOp>(upstream_op)) {
        if (v == upstream_dma.getDstMemref()) {
          // Disconnect dependency between async op and upstream dma
          async_op.eraseAsyncDependency(i);
          // Reconnect upstream dma's dep list to async op
          auto upstream_dma_async =
              dyn_cast<air::AsyncOpInterface>(upstream_dma.getOperation());
          for (auto token : upstream_dma_async.getAsyncDependencies()) {
            async_op.addAsyncDependency(token);
          }
          Value srcMemref = upstream_dma.getSrcMemref();
          // Recursively trace upstream dma
          for (unsigned j = 0; j < upstream_op->getNumOperands(); j++) {
            if (srcMemref == upstream_op->getOperand(j)) {
              findAndPruneRedundantDma(&upstream_op->getOpOperand(j));
            }
          }
          // Elevate from argument to operand of herd launch
          if (auto hl_op = getHerdArgOwner(srcMemref)) {
            for (unsigned i = 0; i < hl_op.getNumKernelOperands(); i++) {
              if (hl_op.getKernelArgument(i) == srcMemref) {
                auto &hl_opoperand = hl_op->getOpOperand(
                    i + hl_op.getAsyncDependencies().size() + 2);
                findAndPruneRedundantDma(&hl_opoperand);
              }
            }
          }
          // Elevate from argument to operand of hierarchy op
          if (auto hier_op = getHierarchyArgOwner(srcMemref)) {
            auto dep_list =
                dyn_cast<air::AsyncOpInterface>(hier_op.getOperation())
                    .getAsyncDependencies();
            for (unsigned i = 0; i < hier_op.getNumKernelOperands(); i++) {
              if (hier_op.getKernelArgument(i) == srcMemref) {
                auto &hier_opoperand = hier_op->getOpOperand(
                    i + dep_list.size() + hier_op.getNumDims());
                findAndPruneRedundantDma(&hier_opoperand);
              }
            }
          }
          upstream_dma->erase();
        }
      }
    }
  }

private:
};

class AIRHoistDmaInAccumPattern
    : public xilinx::air::impl::AIRHoistDmaInAccumPatternBase<
          AIRHoistDmaInAccumPattern> {

public:
  AIRHoistDmaInAccumPattern() = default;
  AIRHoistDmaInAccumPattern(const AIRHoistDmaInAccumPattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<HoistDmaInAccumPattern>(ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps)
      runOptPatterns(f);
  }

private:
};

class AIRBroadcastDetection
    : public xilinx::air::impl::AIRBroadcastDetectionBase<
          AIRBroadcastDetection> {

public:
  AIRBroadcastDetection() = default;
  AIRBroadcastDetection(const AIRBroadcastDetection &pass){};

  void runOnOperation() override {
    BroadcastDetection proc;
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps)
      proc.runBroadcastPattern(f);
  }

private:
};

class AIRPruneLinalgGenericInputDma
    : public xilinx::air::impl::AIRPruneLinalgGenericInputDmaBase<
          AIRPruneLinalgGenericInputDma> {

public:
  AIRPruneLinalgGenericInputDma() = default;
  AIRPruneLinalgGenericInputDma(const AIRPruneLinalgGenericInputDma &pass){};

  void runOnOperation() override {
    PruneLinalgGenericInputDma proc;
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps)
      proc.runLinalgGenericPattern(f);
  }

private:
};

class AIRAnnotateFrontAndBackOpsInForPattern
    : public xilinx::air::impl::AIRAnnotateFrontAndBackOpsInForPatternBase<
          AIRAnnotateFrontAndBackOpsInForPattern> {

public:
  AIRAnnotateFrontAndBackOpsInForPattern() = default;
  AIRAnnotateFrontAndBackOpsInForPattern(
      const AIRAnnotateFrontAndBackOpsInForPattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<AnnotateFrontAndBackOpsInForPattern>(ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps)
      runOptPatterns(f);
  }

private:
};

class AIRHoistMemallocInForPattern
    : public xilinx::air::impl::AIRHoistMemallocInForPatternBase<
          AIRHoistMemallocInForPattern> {

public:
  AIRHoistMemallocInForPattern() = default;
  AIRHoistMemallocInForPattern(const AIRHoistMemallocInForPattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<HoistMemallocInForPattern>(ctx, clKeepMemrefDealloc);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps)
      runOptPatterns(f);
  }

private:
};

class AIRConstructPingPongDependencyPattern
    : public xilinx::air::impl::AIRConstructPingPongDependencyPatternBase<
          AIRConstructPingPongDependencyPattern> {

public:
  AIRConstructPingPongDependencyPattern() = default;
  AIRConstructPingPongDependencyPattern(
      const AIRConstructPingPongDependencyPattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConstructPingPongDependencyPattern>(ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps)
      runOptPatterns(f);
  }

private:
};

class AIRUnrollLoopForPipeliningPattern
    : public xilinx::air::impl::AIRUnrollLoopForPipeliningPatternBase<
          AIRUnrollLoopForPipeliningPattern> {

public:
  AIRUnrollLoopForPipeliningPattern() = default;
  AIRUnrollLoopForPipeliningPattern(
      const AIRUnrollLoopForPipeliningPattern &pass){};

  void runOnOperation() override {
    auto module = getOperation();
    module.walk([&](scf::ForOp for_op) {
      // Check if loop is the target
      if (for_op->hasAttr("unroll")) {
        uint64_t unroll_factor =
            for_op->getAttrOfType<IntegerAttr>("unroll").getInt();
        auto annotateFn = [](unsigned i, Operation *op, OpBuilder b) {
          op->setAttr("unrolled_iteration", b.getI32IntegerAttr(i));
        };
        OpBuilder builder(for_op);
        if (failed(air::loopUnrollByFactorWithAsyncTokenPreserved(
                for_op, unroll_factor, annotateFn)))
          signalPassFailure();
      }
    });
  }

private:
};

// Convert any air.herd op within an scf.for loop body to be strictly async wrt
// other async ops in the loop body.
struct MakeHerdOpAsyncInScfForLoopPattern
    : public OpRewritePattern<air::HerdOp> {
  using OpRewritePattern<air::HerdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(air::HerdOp herdOp,
                                PatternRewriter &rewriter) const override {
    auto forOp = dyn_cast_if_present<scf::ForOp>(herdOp->getParentOp());
    if (!forOp)
      return failure();
    auto loopCarriedToken = getLoopCarriedTokenFromScfOp(forOp, "argument");
    if (herdOp.getAsyncDependencies().size() == 1 &&
        herdOp.getAsyncDependencies().front() == loopCarriedToken &&
        herdOp.getAsyncToken().use_empty())
      return failure();
    clearAsyncDependenciesOfAsyncOp(herdOp);
    herdOp.addAsyncDependency(loopCarriedToken);
    if (!herdOp.getAsyncToken().use_empty())
      herdOp.getAsyncToken().replaceAllUsesWith(loopCarriedToken);
    return success();
  }

private:
};

class AIRHoistOpsNotUsingPingPongPattern
    : public xilinx::air::impl::AIRHoistOpsNotUsingPingPongPatternBase<
          AIRHoistOpsNotUsingPingPongPattern> {

public:
  AIRHoistOpsNotUsingPingPongPattern() = default;
  AIRHoistOpsNotUsingPingPongPattern(
      const AIRHoistOpsNotUsingPingPongPattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<HoistOpsNotUsingPingPongPattern,
                    MakeHerdOpAsyncInScfForLoopPattern>(ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps)
      runOptPatterns(f);
  }

private:
};

class AIRPingPongTransformationPattern
    : public xilinx::air::impl::AIRPingPongTransformationPatternBase<
          AIRPingPongTransformationPattern> {

public:
  AIRPingPongTransformationPattern() = default;
  AIRPingPongTransformationPattern(
      const AIRPingPongTransformationPattern &pass){};
  AIRPingPongTransformationPattern(
      const AIRPingPongTransformationPatternOptions &options)
      : AIRPingPongTransformationPatternBase(options) {}

  void runIsolateScfForOpForPingPong(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<HoistOpsNotUsingPingPongPattern,
                    MakeHerdOpAsyncInScfForLoopPattern>(ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }

  void runOpAnnotationPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<AnnotateFrontAndBackOpsInForPattern>(ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }

  void runHoistMemallocPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<HoistMemallocInForPattern>(ctx, clKeepMemrefDealloc);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }

  void runConstructPingPongDependencyPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConstructPingPongDependencyPattern>(ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }

  void runLoopUnroll(func::FuncOp funcOp) {
    funcOp.walk([&](scf::ForOp for_op) {
      // Check if loop is the target
      if (for_op->hasAttr("unroll")) {
        uint64_t unroll_factor =
            for_op->getAttrOfType<IntegerAttr>("unroll").getInt();
        auto annotateFn = [](unsigned i, Operation *op, OpBuilder b) {
          op->setAttr("unrolled_iteration", b.getI32IntegerAttr(i));
        };
        (void)loopUnrollByFactor(for_op, unroll_factor, annotateFn);
      }
    });
  }

  void runCleanUpAttrs(func::FuncOp funcOp) {
    funcOp.walk([&](Operation *op) {
      // Check if loop is the target
      op->removeAttr("unroll");
      op->removeAttr("hoist_alloc");
      op->removeAttr("unrolled_iteration");
      op->removeAttr("isolated");
      op->removeAttr("ping_pong");
      op->removeAttr("producer");
      op->removeAttr("consumer");
      op->removeAttr("async_front");
      op->removeAttr("async_back");
    });
  }

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps)
      runIsolateScfForOpForPingPong(f);
    for (auto f : funcOps)
      runOpAnnotationPatterns(f);
    for (auto f : funcOps)
      runLoopUnroll(f);
    for (auto f : funcOps)
      runHoistMemallocPatterns(f);
    for (auto f : funcOps)
      runConstructPingPongDependencyPatterns(f);
    for (auto f : funcOps)
      runCleanUpAttrs(f);
  }

private:
};

class AIRLabelScfForLoopForPingPongPattern
    : public xilinx::air::impl::AIRLabelScfForLoopForPingPongPatternBase<
          AIRLabelScfForLoopForPingPongPattern> {

public:
  AIRLabelScfForLoopForPingPongPattern() = default;
  AIRLabelScfForLoopForPingPongPattern(
      const AIRLabelScfForLoopForPingPongPattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<LabelScfForLoopForPingPongPattern>(ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps)
      runOptPatterns(f);
  }

private:
};

class AIRLabelScfForLoopInAIRSegmentPattern
    : public xilinx::air::impl::AIRLabelScfForLoopInAIRSegmentPatternBase<
          AIRLabelScfForLoopInAIRSegmentPattern> {

public:
  AIRLabelScfForLoopInAIRSegmentPattern() = default;
  AIRLabelScfForLoopInAIRSegmentPattern(
      const AIRLabelScfForLoopInAIRSegmentPattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<LabelScfForLoopInAIRSegment>(ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps)
      runOptPatterns(f);
  }

private:
};

LogicalResult AIRSpecializeChannelWrapAndStrideImpl(
    Region *region, int maxNumDims = -1, int maxSize = -1,
    bool enableForLoopUnrolling = true, bool enableRepeatAtHighestDim = false) {
  MLIRContext *ctx = region->getContext();
  RewritePatternSet preproc_patterns(ctx);
  preproc_patterns
      .insert<UnrollScfParallel, CanonicalizeAffineApplyOnLoopInductionVar,
              CanonicalizeArithMuliOpOnLoopInductionVar,
              CanonicalizeArithAddiOpOnLoopInductionVar,
              CanonicalizeArithIndexCastOpOnLoopInductionVar>(ctx);
  // Canonicalize constant operands in affine.apply.
  mlir::affine::AffineApplyOp::getCanonicalizationPatterns(preproc_patterns,
                                                           ctx);
  air::WaitAllOp::getCanonicalizationPatterns(preproc_patterns, ctx);
  air::ExecuteOp::getCanonicalizationPatterns(preproc_patterns, ctx);
  (void)applyPatternsGreedily(*region, std::move(preproc_patterns));

  // Canonicalize wrap and stride list to remove redundant dimensions
  RewritePatternSet preproc_wns_patterns(ctx);
  populateAIRCanonicalizeChannelWrapAndStridePatterns(
      preproc_wns_patterns, maxSize, maxNumDims, enableRepeatAtHighestDim);
  (void)applyPatternsGreedily(*region, std::move(preproc_wns_patterns));

  RewritePatternSet patterns(ctx);
  patterns.insert<CanonicalizeAffineApplyOnLoopInductionVar,
                  CanonicalizeArithMuliOpOnLoopInductionVar,
                  CanonicalizeArithAddiOpOnLoopInductionVar,
                  CanonicalizeArithIndexCastOpOnLoopInductionVar,
                  AIRSpecializeChannelWrapAndStrideInAffineFor>(ctx);
  patterns.insert<AIRSpecializeChannelWrapAndStrideInScfFor>(
      ctx, maxNumDims, maxSize, enableRepeatAtHighestDim);
  air::ExecuteOp::getCanonicalizationPatterns(patterns, ctx);
  affine::AffineApplyOp::getCanonicalizationPatterns(patterns, ctx);
  (void)applyPatternsGreedily(*region, std::move(patterns));

  // Unroll any remaining loops which contain only data movements.
  if (enableForLoopUnrolling) {
    RewritePatternSet unroll_patterns(ctx);
    unroll_patterns
        .insert<AIRUnrollScfForIntoBDChain, AIRUnrollAffineForIntoBDChain>(ctx);
    (void)applyPatternsGreedily(*region, std::move(unroll_patterns));
  }

  // Canonicalize wrap and stride list to remove redundant dimensions
  RewritePatternSet cano_patterns(ctx);
  populateAIRCanonicalizeChannelWrapAndStridePatterns(
      cano_patterns, maxSize, maxNumDims, enableRepeatAtHighestDim);
  ExecuteOp::getCanonicalizationPatterns(cano_patterns, ctx);
  (void)applyPatternsGreedily(*region, std::move(cano_patterns));

  return success();
}

class AIRSpecializeChannelWrapAndStridePattern
    : public xilinx::air::impl::AIRSpecializeChannelWrapAndStridePatternBase<
          AIRSpecializeChannelWrapAndStridePattern> {

public:
  AIRSpecializeChannelWrapAndStridePattern() = default;
  AIRSpecializeChannelWrapAndStridePattern(
      const AIRSpecializeChannelWrapAndStridePattern &pass){};

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<Region *, 4> regions;
    if (clScope == "segment")
      module.walk(
          [&](air::SegmentOp op) { regions.push_back(&op.getRegion()); });
    else if (clScope == "launch")
      module.walk(
          [&](air::LaunchOp op) { regions.push_back(&op.getRegion()); });
    else if (clScope == "func")
      module.walk([&](func::FuncOp op) { regions.push_back(&op.getRegion()); });
    else {
      emitError(module.getLoc(),
                "Unknown scope for -air-specialize-channel-wrap-and-stride. "
                "Must be one of [segment, launch, func].");
      signalPassFailure();
    }
    for (auto region : regions)
      if (AIRSpecializeChannelWrapAndStrideImpl(region).failed())
        signalPassFailure();
  }

private:
};

class AIRUnrollChannelByFactorPattern
    : public xilinx::air::impl::AIRUnrollChannelByFactorPatternBase<
          AIRUnrollChannelByFactorPattern> {

public:
  AIRUnrollChannelByFactorPattern() = default;
  AIRUnrollChannelByFactorPattern(
      const AIRUnrollChannelByFactorPattern &pass){};

  void runOnOperation() override {
    auto module = getOperation();
    UnrollChannelByFactorPattern proc;
    SmallVector<air::ChannelOp> chanOps;
    module.walk([&](air::ChannelOp op) {
      if (op.getSymName().str() == clChanName)
        chanOps.push_back(op);
    });
    if (chanOps.empty())
      return;
    if (chanOps.size() > 1) {
      chanOps.back()->emitOpError(
          "found multiple channel declarations with channel name ")
          << clChanName;
    }
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps)
      proc.runUnrollChannelByFactorPattern(f, chanOps.front(), clUnrollDim,
                                           clUnrollFactor);
  }

private:
};

// A pass which transform multiple channel ops into one, where the data movement
// is time-multiplexed.
class AIRFuseChannels
    : public xilinx::air::impl::AIRFuseChannelsBase<AIRFuseChannels> {

public:
  AIRFuseChannels() = default;
  AIRFuseChannels(const AIRFuseChannels &pass) {}
  AIRFuseChannels(const AIRFuseChannelsOptions &options)
      : AIRFuseChannelsBase(options) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect>();
  }

  void runOnFunction(func::FuncOp f, std::vector<air::ChannelOp> channelOps) {
    init_options();
    if (channelOps.empty())
      return;
    // Rename symbols
    // TODO: make this greedy
    auto renameSymbols =
        [](std::vector<air::ChannelOp> &channelOps,
           std::map<air::ChannelOp, air::ChannelOp> chan_merge_map) {
          for (unsigned i = 0; i < channelOps.size(); i++) {
            for (auto chanKey : channelOps) {
              if (!chan_merge_map.count(chanKey))
                continue;
              auto error = mlir::SymbolTable::replaceAllSymbolUses(
                  chanKey.getOperation(),
                  mlir::SymbolTable::getSymbolName(chan_merge_map[chanKey]),
                  chanKey->getParentOfType<ModuleOp>());
              // FIXME: what if this fails?
              (void)error;
            }
          }
        };
    std::map<air::ChannelOp, air::ChannelOp> chan_merge_map;
    std::vector<std::pair<air::ChannelOp, air::ChannelOp>> nfl_merge_pairs;

    // Identify mergeable channel pairs and classify fusion type.
    for (unsigned i = 0; i < channelOps.size() - 1; i++) {
      for (unsigned j = i + 1; j < channelOps.size(); j++) {
        // Check if channels [i] and [j] are temporally mergeable, and get the
        // merge type.
        auto [mergeable, mergeType] =
            checkIfTemporalMergeable(channelOps[i], channelOps[j]);
        if (!mergeable)
          continue;
        air::ChannelOp chanA = channelOps[i];
        air::ChannelOp chanB = channelOps[j];
        if (mergeType == "LB" || mergeType == "UB") {
          // Case 1: Merge via loop unpeeling (recover missing first/last
          // iterations).
          sortChannelsByLoopNests(chanA, chanB);
          mergeChannelOpsTemporally(chanA, chanB, mergeType);
          chan_merge_map[chanB] = chanA;
        } else if (mergeType == "NFL") {
          // Case 2: Fuse channels into a new loop (requires creating an
          // scf.for).
          chan_merge_map[chanB] = chanA;
          nfl_merge_pairs.push_back(std::make_pair(chanA, chanB));
        }
      }
    }

    // Collect channel interface ops to fuse and erase for NFL (loop-based)
    // merges.
    llvm::SetVector<Operation *> nfl_merge_destinations, nfl_erased_ops;
    for (auto &[destChan, srcChan] : nfl_merge_pairs) {
      auto [toFuse, toErase] = getChannelIfOpsFusableByFor(destChan, srcChan);
      for (auto chanIf : toFuse) {
        nfl_merge_destinations.insert(chanIf);
      }
      for (auto chanIf : toErase) {
        nfl_erased_ops.insert(chanIf);
      }
    }
    // Find minimal enclosing regions for the destinations that need wrapping.
    auto nfl_merge_regions =
        findMinimalChannelIfOpContainingRegions(nfl_merge_destinations);
    IRRewriter rewriter(f.getContext());

    // Apply transformations: create dummy loops or wrap existing regions.
    if (nfl_merge_regions.empty()) {
      // No enclosing regions -> create dummy scf.for loops around fusable ops.
      for (auto &[destChan, srcChan] : nfl_merge_pairs) {
        createDummyForOpsAroundOps<air::ChannelPutOp>(
            getChannelPutsFusableByFor(destChan, srcChan));
        createDummyForOpsAroundOps<air::ChannelGetOp>(
            getChannelGetsFusableByFor(destChan, srcChan));
        // Merge temporally after wrapping with dummy loops.
        mergeChannelOpsTemporally(destChan, srcChan, "UB");
      }
    } else {
      // Found enclosing regions → wrap them with scf.for loops.
      wrapRegionsWithForLoops(rewriter, nfl_merge_regions);
      // Erase obsolete ops (nfl_erased_ops) and replace async semantics.
      for (auto e : nfl_erased_ops) {
        if (air::isAsyncOp(e)) {
          IRMapping remap;
          rewriter.setInsertionPoint(e);
          auto waitAll = air::replaceAsyncOpWithWaitAll(rewriter, remap, e);
          air::getAsyncTokenFromOp(e).replaceAllUsesWith(
              waitAll.getAsyncToken());
        }
        rewriter.eraseOp(e);
      }
    }

    renameSymbols(channelOps, chan_merge_map);
    if (!targetMemorySpaces.empty()) {
      for (unsigned i = 0; i < channelOps.size() - 1; i++) {
        for (unsigned j = i + 1; j < channelOps.size(); j++) {
          if (!checkIfMergeable(channelOps[i], channelOps[j]))
            continue;
          // Aggressively fuse air.channels by time multiplexing.
          mergeChannels(rewriter, channelOps[i], channelOps[j]);
          chan_merge_map[channelOps[j]] = channelOps[i];
        }
      }
    }
    renameSymbols(channelOps, chan_merge_map);
    // Walk the region and mutate scf.for loop bounds based on "setLB" and
    // "setUB" attributes;
    auto getNewBoundValue = [](scf::ForOp fop, std::string attrNameInStr) {
      std::optional<int> setBound;
      SmallVector<Operation *> chanOpRange;
      fop.walk<WalkOrder::PreOrder, ForwardDominanceIterator<>>(
          [fop, &chanOpRange](Operation *o) {
            if (isa<scf::ForOp>(o) && o != fop)
              WalkResult::skip();
            if (isa<air::ChannelInterface>(o))
              chanOpRange.push_back(o);
            WalkResult::advance();
          });
      if (chanOpRange.empty())
        return setBound;
      if (llvm::any_of(chanOpRange, [attrNameInStr](Operation *o) {
            return !o->hasAttr(attrNameInStr);
          }))
        return setBound;
      setBound = (*chanOpRange.begin())
                     ->getAttrOfType<IntegerAttr>(attrNameInStr)
                     .getInt();
      if (llvm::any_of(chanOpRange, [&setBound, attrNameInStr](Operation *o) {
            return o->getAttrOfType<IntegerAttr>(attrNameInStr).getInt() !=
                   *setBound;
          })) {
        fop->emitOpError("contains op marked to set its lower bound, but the "
                         "set value isn't consistent");
        return setBound;
      }
      return setBound;
    };
    auto clearAttrsInBlk = [](Block *blk, std::string attrNameInStr) {
      blk->walk([attrNameInStr](air::ChannelInterface chanOp) {
        chanOp->removeAttr(attrNameInStr);
      });
    };
    f.walk([getNewBoundValue, clearAttrsInBlk](scf::ForOp fop) {
      auto setLB = getNewBoundValue(fop, "setLB");
      if (!setLB)
        return;
      OpBuilder builder(fop);
      fop->getOpOperand(0).assign(
          builder.create<arith::ConstantIndexOp>(fop->getLoc(), *setLB));
      clearAttrsInBlk(fop.getBody(), "setLB");
    });
    f.walk([getNewBoundValue, clearAttrsInBlk](scf::ForOp fop) {
      auto setUB = getNewBoundValue(fop, "setUB");
      if (!setUB)
        return;
      OpBuilder builder(fop);
      fop->getOpOperand(1).assign(
          builder.create<arith::ConstantIndexOp>(fop->getLoc(), *setUB));
      clearAttrsInBlk(fop.getBody(), "setUB");
    });
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto ctx = &getContext();

    SmallVector<func::FuncOp, 4> funcOps;
    std::vector<air::ChannelOp> channelOps;
    module.walk([&](air::ChannelOp op) { channelOps.push_back(op); });
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps) {
      runOnFunction(f, channelOps);
      // Canonicalization patterns.
      RewritePatternSet patterns(ctx);
      air::WaitAllOp::getCanonicalizationPatterns(patterns, ctx);
      scf::ForOp::getCanonicalizationPatterns(patterns, ctx);
      (void)applyPatternsGreedily(f, std::move(patterns));
    }
  }

  void init_options() {
    targetMemorySpaces.clear();
    if (clAggressiveMode.empty())
      return;
    for (unsigned i = 0; i < clAggressiveMode.size(); ++i) {
      if (clAggressiveMode[i] == "L1")
        targetMemorySpaces.push_back((unsigned)air::MemorySpace::L1);
      else if (clAggressiveMode[i] == "L2")
        targetMemorySpaces.push_back((unsigned)air::MemorySpace::L2);
      else if (clAggressiveMode[i] == "L3")
        targetMemorySpaces.push_back((unsigned)air::MemorySpace::L3);
      LLVM_DEBUG(llvm::outs() << "clAggressiveMode[" << i
                              << "] = " << clAggressiveMode[i] << "\n");
    }
  }

  SmallVector<unsigned> targetMemorySpaces;

private:
  // Get a vector of channel ops which can be fused using a new for loop.
  template <typename T>
  bool areConsistentMemoryAccessPattern(std::vector<T> a_vec,
                                        std::vector<T> b_vec) {
    Value memref = a_vec[0].getMemref();
    SmallVector<Value> offsets = a_vec[0].getOffsets();
    SmallVector<Value> sizes = a_vec[0].getSizes();
    SmallVector<Value> strides = a_vec[0].getStrides();
    for (unsigned i = 1; i < a_vec.size(); i++)
      if ((!memrefsAreAffinitiveToSameChannel(memref, a_vec[i].getMemref())) ||
          (!areTheSameSSAValueLists(offsets, a_vec[i].getOffsets())) ||
          (!areTheSameSSAValueLists(sizes, a_vec[i].getSizes())) ||
          (!areTheSameSSAValueLists(strides, a_vec[i].getStrides())))
        return false; // Inconsistent memory use for all puts
    for (unsigned i = 0; i < b_vec.size(); i++)
      if ((!memrefsAreAffinitiveToSameChannel(memref, b_vec[i].getMemref())) ||
          (!areTheSameSSAValueLists(offsets, b_vec[i].getOffsets())) ||
          (!areTheSameSSAValueLists(sizes, b_vec[i].getSizes())) ||
          (!areTheSameSSAValueLists(strides, b_vec[i].getStrides())))
        return false; // Inconsistent memory use between a puts and b puts
    return true;
  }
  std::vector<air::ChannelPutOp>
  getChannelPutsFusableByFor(air::ChannelOp chanA, air::ChannelOp chanB) {
    std::vector<air::ChannelPutOp> a_puts = getChannelPutOpThroughSymbol(chanA);
    std::vector<air::ChannelPutOp> b_puts = getChannelPutOpThroughSymbol(chanB);

    if (areConsistentMemoryAccessPattern<air::ChannelPutOp>(a_puts, b_puts))
      return a_puts;
    else
      return std::vector<air::ChannelPutOp>{};
  }
  std::vector<air::ChannelGetOp>
  getChannelGetsFusableByFor(air::ChannelOp chanA, air::ChannelOp chanB) {
    std::vector<air::ChannelGetOp> a_gets = getChannelGetOpThroughSymbol(chanA);
    std::vector<air::ChannelGetOp> b_gets = getChannelGetOpThroughSymbol(chanB);

    if (areConsistentMemoryAccessPattern<air::ChannelGetOp>(a_gets, b_gets))
      return a_gets;
    else
      return std::vector<air::ChannelGetOp>{};
  }

  /// Analyzes two channel symbols (`chanA` and `chanB`) and determines which
  /// channel operations can be fused when wrapping them with a `scf.for` loop.
  ///
  /// This function checks both `ChannelPutOp`s and `ChannelGetOp`s associated
  /// with the given channels. If the memory access patterns between the
  /// operations from `chanA` and `chanB` are consistent, it classifies:
  ///   - The ops from `chanA` as **fusableOps** (to be preserved and fused).
  ///   - The ops from `chanB` as **erasedOps** (to be removed after fusion).
  ///
  /// Additionally, any channel ops located under an `air::HerdOp` are excluded
  /// from fusion because they reside in a special execution context.
  ///
  /// \returns A pair `(fusableOps, erasedOps)`:
  ///   - `fusableOps`: channel ops to keep and fuse.
  ///   - `erasedOps`: channel ops to be eliminated after fusion.
  std::pair<std::vector<air::ChannelInterface>,
            std::vector<air::ChannelInterface>>
  getChannelIfOpsFusableByFor(air::ChannelOp chanA, air::ChannelOp chanB) {
    // Collect all channel put/get ops associated with chanA and chanB via
    // symbol references.
    std::vector<air::ChannelPutOp> a_puts = getChannelPutOpThroughSymbol(chanA);
    std::vector<air::ChannelPutOp> b_puts = getChannelPutOpThroughSymbol(chanB);
    std::vector<air::ChannelGetOp> a_gets = getChannelGetOpThroughSymbol(chanA);
    std::vector<air::ChannelGetOp> b_gets = getChannelGetOpThroughSymbol(chanB);
    std::vector<air::ChannelInterface> fusableOps, erasedOps;

    // Step 1: Check if the memory access patterns of all puts (A vs. B) are
    // consistent.
    //   - If consistent, mark puts from A as fusable and puts from B as
    //   erasable.
    //   - Skip any put ops under a HerdOp (cannot fuse them).
    if (areConsistentMemoryAccessPattern<air::ChannelPutOp>(a_puts, b_puts)) {
      for (auto put : a_puts) {
        if (put->getParentOfType<air::HerdOp>())
          continue;
        fusableOps.push_back(put);
      }
      for (auto put : b_puts) {
        if (put->getParentOfType<air::HerdOp>())
          continue;
        erasedOps.push_back(put);
      }
    }
    // Step 2: Perform the same analysis as Step 1 with all gets.
    if (areConsistentMemoryAccessPattern<air::ChannelGetOp>(a_gets, b_gets)) {
      for (auto get : a_gets) {
        if (get->getParentOfType<air::HerdOp>())
          continue;
        fusableOps.push_back(get);
      }
      for (auto get : b_gets) {
        if (get->getParentOfType<air::HerdOp>())
          continue;
        erasedOps.push_back(get);
      }
    }
    // Return the pair of (to-fuse ops, to-erase ops) to guide further
    // transformation.
    return std::make_pair(fusableOps, erasedOps);
  }
  // Create single-iteration for loops around a vector of operations of type T.
  template <typename T>
  void createDummyForOpsAroundOps(std::vector<T> ops) {
    for (auto t_o : ops) {
      Operation *op = t_o.getOperation();
      OpBuilder builder(op);
      IRMapping remap;
      auto loc = op->getLoc();
      auto ctx = op->getContext();
      auto zeroIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
      auto oneIdx = builder.create<arith::ConstantIndexOp>(loc, 1);
      auto newForOp = scf::ForOp();

      if (air::getAsyncTokenFromOp(op)) {
        newForOp = builder.create<scf::ForOp>(
            loc, zeroIdx, oneIdx, oneIdx,
            builder
                .create<air::WaitAllOp>(loc, air::AsyncTokenType::get(ctx),
                                        air::getAsyncDependenciesFromOp(op))
                .getAsyncToken());
        for (auto dep : air::getAsyncDependenciesFromOp(op))
          remap.map(dep, newForOp.getRegionIterArgs()[0]);
      } else
        newForOp = builder.create<scf::ForOp>(loc, zeroIdx, oneIdx, oneIdx);
      builder.setInsertionPointToStart(newForOp.getBody());
      auto newOp = dyn_cast<T>(builder.clone(*op, remap));

      if (auto oldAsyncToken = air::getAsyncTokenFromOp(op)) {
        builder.create<scf::YieldOp>(loc, newOp.getAsyncToken());
        oldAsyncToken.replaceAllUsesWith(newForOp->getResult(0));
      } else
        builder.create<scf::YieldOp>(loc);
    }
    for (auto e : ops)
      e->erase();
    return;
  }

  void sortChannelsByLoopNests(air::ChannelOp &chan_a, air::ChannelOp &chan_b) {
    std::vector<air::ChannelPutOp> a_puts =
        getChannelPutOpThroughSymbol(chan_a);
    std::vector<air::ChannelPutOp> b_puts =
        getChannelPutOpThroughSymbol(chan_b);
    std::vector<air::ChannelGetOp> a_gets =
        getChannelGetOpThroughSymbol(chan_a);
    std::vector<air::ChannelGetOp> b_gets =
        getChannelGetOpThroughSymbol(chan_b);
    if (a_puts.size() != 1 || a_gets.size() != 1) {
      chan_a->emitOpError("has more than one puts or gets.");
      return;
    }
    if (b_puts.size() != 1 || b_gets.size() != 1) {
      chan_b->emitOpError("has more than one puts or gets.");
      return;
    }
    int a_put_loop_nest_size =
        getParentLoopNest(a_puts[0].getOperation()).size();
    int b_put_loop_nest_size =
        getParentLoopNest(b_puts[0].getOperation()).size();
    int a_get_loop_nest_size =
        getParentLoopNest(a_gets[0].getOperation()).size();
    int b_get_loop_nest_size =
        getParentLoopNest(b_gets[0].getOperation()).size();
    if ((a_put_loop_nest_size - b_put_loop_nest_size == 1) &&
        (a_get_loop_nest_size - b_get_loop_nest_size == 1))
      return;
    else if ((b_put_loop_nest_size - a_put_loop_nest_size == 1) &&
             (b_get_loop_nest_size - a_get_loop_nest_size == 1)) {
      air::ChannelOp temp = chan_a;
      chan_a = chan_b;
      chan_b = temp;
      return;
    }
  }

  // Helper function to check if two channels have the same channel_type
  // attribute
  bool haveSameChannelType(air::ChannelOp chan_a, air::ChannelOp chan_b) {
    return chan_a.getChannelType() == chan_b.getChannelType();
  }

  // Check whether puts and gets hit the aggressive mode target memory spaces
  bool hitsMemorySpaceForAggMode(std::vector<air::ChannelPutOp> &puts,
                                 std::vector<air::ChannelGetOp> &gets) {
    for (auto put : puts) {
      BaseMemRefType ty = llvm::cast<BaseMemRefType>(put.getMemref().getType());
      if (llvm::any_of(targetMemorySpaces, [&](unsigned memSpace) {
            return memSpace == ty.getMemorySpaceAsInt();
          })) {
        return true;
      }
    }
    for (auto get : gets) {
      BaseMemRefType ty = llvm::cast<BaseMemRefType>(get.getMemref().getType());
      if (llvm::any_of(targetMemorySpaces, [&](unsigned memSpace) {
            return memSpace == ty.getMemorySpaceAsInt();
          })) {
        return true;
      }
    }
    return false;
  }

  bool checkIfMergeable(air::ChannelOp chan_a, air::ChannelOp chan_b) {
    // Check if channels have the same channel_type
    if (!haveSameChannelType(chan_a, chan_b))
      return false;

    // Check which memory space to time-multiplex channels onto.
    if (targetMemorySpaces.empty())
      return false;
    std::vector<air::ChannelPutOp> a_puts =
        getChannelPutOpThroughSymbol(chan_a);
    std::vector<air::ChannelPutOp> b_puts =
        getChannelPutOpThroughSymbol(chan_b);
    std::vector<air::ChannelGetOp> a_gets =
        getChannelGetOpThroughSymbol(chan_a);
    std::vector<air::ChannelGetOp> b_gets =
        getChannelGetOpThroughSymbol(chan_b);
    if (a_puts.size() != b_puts.size())
      return false;
    if (a_gets.size() != b_gets.size())
      return false;
    // Filter out put/get memory space based on aggressive mode option
    if (!hitsMemorySpaceForAggMode(a_puts, a_gets))
      return false;
    if (!hitsMemorySpaceForAggMode(b_puts, b_gets))
      return false;
    for (unsigned i = 0; i < a_puts.size(); i++) {
      auto a_put_loop_nest = getParentLoopNest(a_puts[i].getOperation());
      auto b_put_loop_nest = getParentLoopNest(b_puts[i].getOperation());
      if (a_put_loop_nest.size() != b_put_loop_nest.size())
        return false;
      for (unsigned i = 0; i < a_put_loop_nest.size(); i++)
        if (!areEquivalentControlLoops(a_put_loop_nest[i], b_put_loop_nest[i]))
          return false;
    }
    for (unsigned i = 0; i < a_gets.size(); i++) {
      auto a_get_loop_nest = getParentLoopNest(a_gets[i].getOperation());
      auto b_get_loop_nest = getParentLoopNest(b_gets[i].getOperation());
      if (a_get_loop_nest.size() != b_get_loop_nest.size())
        return false;
      for (unsigned i = 0; i < a_get_loop_nest.size(); i++)
        if (!areEquivalentControlLoops(a_get_loop_nest[i], b_get_loop_nest[i]))
          return false;
    }
    return true;
  }

  // Find the first control loop that is mismatching. Assume longer_loop_nest is
  // one-loop more than shorter_loop_nest.
  scf::ForOp findMismatchControlLoop(std::vector<Block *> longer_loop_nest,
                                     std::vector<Block *> shorter_loop_nest) {
    if (longer_loop_nest.size() - shorter_loop_nest.size() != 1)
      return scf::ForOp();
    scf::ForOp mismatchScfFor = scf::ForOp();
    unsigned index = 0;
    for (unsigned i = 0; i < shorter_loop_nest.size(); i++) {
      auto aLoop = shorter_loop_nest[i];
      auto bLoop = longer_loop_nest[index++];
      if (!areEquivalentControlLoops(aLoop, bLoop)) {
        mismatchScfFor = dyn_cast<scf::ForOp>(bLoop->getParentOp());
        bLoop = longer_loop_nest[index++];
        if (!areEquivalentControlLoops(aLoop, bLoop))
          return scf::ForOp();
      }
    }
    return mismatchScfFor;
  }

  std::tuple<bool, std::string>
  checkIfTemporalMergeableByScfForImpl(std::vector<Block *> a_loop_nest,
                                       std::vector<Block *> b_loop_nest) {
    std::tuple<bool, std::string> notMergeable = {false, ""};
    std::tuple<bool, std::string> mergeableToLB = {true, "LB"};
    std::tuple<bool, std::string> mergeableToUB = {true, "UB"};
    if (std::abs((int)a_loop_nest.size() - (int)b_loop_nest.size()) != 1)
      return notMergeable;
    if (a_loop_nest.size() < b_loop_nest.size()) {
      auto mismatchScfFor = findMismatchControlLoop(b_loop_nest, a_loop_nest);
      if (!mismatchScfFor)
        return notMergeable;
      if (auto constLB = getConstantIntValue(mismatchScfFor.getLowerBound()))
        if (*constLB < 1)
          return notMergeable;
      // Merge by unpeeling into LB.
      return mergeableToLB;
    } else {
      auto mismatchScfFor = findMismatchControlLoop(a_loop_nest, b_loop_nest);
      if (!mismatchScfFor)
        return notMergeable;
      // Merge by unpeeling into UB.
      return mergeableToUB;
    }
  }
  std::tuple<bool, std::string>
  checkIfTemporalMergeableImpl(std::vector<Block *> a_loop_nest,
                               std::vector<Block *> b_loop_nest) {
    std::tuple<bool, std::string> notMergeable = {false, ""};
    std::tuple<bool, std::string> mergeableToLB = {true, "LB"};
    std::tuple<bool, std::string> mergeableToUB = {true, "UB"};
    std::tuple<bool, std::string> mergeableNoForLoop = {true, "NFL"};
    if (std::abs((int)a_loop_nest.size() - (int)b_loop_nest.size()) == 1)
      return checkIfTemporalMergeableByScfForImpl(a_loop_nest, b_loop_nest);
    else if (a_loop_nest.size() != b_loop_nest.size())
      return notMergeable;

    for (unsigned i = 0; i < a_loop_nest.size(); i++) {
      if (!areEquivalentControlLoops(a_loop_nest[i], b_loop_nest[i]))
        return notMergeable;
    }
    return mergeableNoForLoop;
  }
  Value getHierOperandFromHierBlockArgument(BlockArgument arg) {
    if (!arg)
      return nullptr;
    auto hierArgOwner = air::getHierarchyArgOwner(arg);
    if (!hierArgOwner)
      return nullptr;
    if (auto tiedOper = hierArgOwner.getTiedKernelOperand(arg))
      return tiedOper;
    return nullptr;
  }
  // Returns true if the two memref values are affinitive to the same DMA
  // channel. This is used to determine whether DMA operations between them can
  // share the same channel.
  bool memrefsAreAffinitiveToSameChannel(Value a, Value b) {
    // Extract the memory space (e.g., L1, L2, L3) from the memref types.
    int aMemorySpace =
        llvm::cast<BaseMemRefType>(a.getType()).getMemorySpaceAsInt();
    int bMemorySpace =
        llvm::cast<BaseMemRefType>(b.getType()).getMemorySpaceAsInt();
    // If both memrefs are in L3 (DDR), they are considered affinitive to all
    // shim DMA channels.
    if (aMemorySpace == (int)air::MemorySpace::L3 &&
        bMemorySpace == (int)air::MemorySpace::L3)
      return true;
    // For non-L3 cases, memrefs are only guaranteed to share a channel if they
    // are the same. First, check if both values are proper MemRefType values.
    if (!isa<MemRefType>(a.getType()))
      return false;
    if (!isa<MemRefType>(b.getType()))
      return false;
    // If both values are exactly the same SSA value, they clearly share
    // affinity.
    if (a == b)
      return true;
    // Try to resolve their defining operations through hierarchy, if they're
    // block arguments.
    auto aHierOper =
        getHierOperandFromHierBlockArgument(llvm::dyn_cast<BlockArgument>(a));
    auto bHierOper =
        getHierOperandFromHierBlockArgument(llvm::dyn_cast<BlockArgument>(b));
    // If either couldn't be resolved, conservatively assume no affinity.
    if (!(aHierOper && bHierOper))
      return false;
    // If both resolved to the same defining operation, they share affinity.
    if (aHierOper == bHierOper)
      return true;
    // Otherwise, they are assumed to be affinitive to different channels.
    return false;
  }
  // Check if two ssa value lists are identical.
  bool areTheSameSSAValueLists(SmallVector<Value> a, SmallVector<Value> b) {
    if (a.size() != b.size())
      return false;
    for (unsigned i = 0; i < a.size(); i++) {
      auto constAElem = getConstantIntValue(a[i]);
      auto constBElem = getConstantIntValue(b[i]);
      if (constAElem && constBElem)
        // Unequal constant values
        if (*constAElem != *constBElem)
          return false;
    }
    return true;
  }
  // Check if two channel ops are under identical affine.if condition blocks.
  bool areUnderTheSameAffineIfCond(Operation *a, Operation *b) {
    auto a_loop_nest = getParentLoopNest(a);
    auto b_loop_nest = getParentLoopNest(b);
    affine::AffineIfOp a_aif = nullptr;
    affine::AffineIfOp b_aif = nullptr;
    for (unsigned i = 0; i < a_loop_nest.size(); i++) {
      for (unsigned j = 0; j < b_loop_nest.size(); j++) {
        auto a_parent = a_loop_nest[i]->getParentOp();
        if (!a_parent)
          continue;
        auto b_parent = b_loop_nest[j]->getParentOp();
        if (!b_parent)
          continue;
        a_aif = dyn_cast<affine::AffineIfOp>(a_parent);
        if (!a_aif)
          continue;
        b_aif = dyn_cast<affine::AffineIfOp>(b_parent);
        if (!b_aif)
          continue;
        // Reached innermost affine.if op for both a and b loop nests.
        if (a_aif.getIntegerSet() == b_aif.getIntegerSet())
          return true;
        else
          return false;
      }
    }
    // One of a or b is under affine.if.
    if (a_aif || b_aif)
      return false;
    // Default case when neither a nor b is under affine.if.
    return true;
  }
  // Check of two air.channels are mergeable in time, by fusing into a shared
  // scf.for loop. Returns a tuple of bool of whether mergeable, and string of
  // fusing into for loop lower bound (LB) or upper bound (UB), or fuse with no
  // for loop (NFL).
  std::tuple<bool, std::string>
  checkIfTemporalMergeable(air::ChannelOp chan_a, air::ChannelOp chan_b) {
    std::tuple<bool, std::string> notMergeable = {false, ""};

    // Check if channels have the same channel_type
    if (!haveSameChannelType(chan_a, chan_b))
      return notMergeable;

    std::vector<air::ChannelPutOp> a_puts =
        getChannelPutOpThroughSymbol(chan_a);
    std::vector<air::ChannelPutOp> b_puts =
        getChannelPutOpThroughSymbol(chan_b);
    std::vector<air::ChannelGetOp> a_gets =
        getChannelGetOpThroughSymbol(chan_a);
    std::vector<air::ChannelGetOp> b_gets =
        getChannelGetOpThroughSymbol(chan_b);
    std::tuple<bool, std::string> mergeableToLB = {true, "LB"};
    std::tuple<bool, std::string> mergeableToUB = {true, "UB"};
    std::tuple<bool, std::string> mergeableNoForLoop = {true, "NFL"};
    if (a_puts.size() != b_puts.size())
      return notMergeable;
    if (a_puts.size() != 1)
      return notMergeable;
    if (a_gets.size() != b_gets.size())
      return notMergeable;
    if (a_gets.size() != 1)
      return notMergeable;
    // Check for identical src and dst memrefs, offset, size and stride lists
    Value aMemref = a_puts[0].getMemref();
    SmallVector<Value> aOffsets = a_puts[0].getOffsets();
    SmallVector<Value> aSizes = a_puts[0].getSizes();
    SmallVector<Value> aStrides = a_puts[0].getStrides();
    for (unsigned i = 1; i < a_puts.size(); i++)
      if ((!memrefsAreAffinitiveToSameChannel(aMemref,
                                              a_puts[i].getMemref())) ||
          (!areTheSameSSAValueLists(aOffsets, a_puts[i].getOffsets())) ||
          (!areTheSameSSAValueLists(aSizes, a_puts[i].getSizes())) ||
          (!areTheSameSSAValueLists(aStrides, a_puts[i].getStrides())))
        return notMergeable; // Inconsistent memory use for all puts
    Value bMemref = b_puts[0].getMemref();
    SmallVector<Value> bOffsets = b_puts[0].getOffsets();
    SmallVector<Value> bSizes = b_puts[0].getSizes();
    SmallVector<Value> bStrides = b_puts[0].getStrides();
    for (unsigned i = 1; i < b_puts.size(); i++)
      if ((!memrefsAreAffinitiveToSameChannel(bMemref,
                                              b_puts[i].getMemref())) ||
          (!areTheSameSSAValueLists(bOffsets, b_puts[i].getOffsets())) ||
          (!areTheSameSSAValueLists(bSizes, b_puts[i].getSizes())) ||
          (!areTheSameSSAValueLists(bStrides, b_puts[i].getStrides())))
        return notMergeable; // Inconsistent memory use for all puts
    if ((!memrefsAreAffinitiveToSameChannel(aMemref, bMemref)))
      return notMergeable;
    aMemref = a_gets[0].getMemref();
    aOffsets = a_gets[0].getOffsets();
    aSizes = a_gets[0].getSizes();
    aStrides = a_gets[0].getStrides();
    for (unsigned i = 1; i < a_gets.size(); i++)
      if ((!memrefsAreAffinitiveToSameChannel(aMemref,
                                              a_gets[i].getMemref())) ||
          (!areTheSameSSAValueLists(aOffsets, a_gets[i].getOffsets())) ||
          (!areTheSameSSAValueLists(aSizes, a_gets[i].getSizes())) ||
          (!areTheSameSSAValueLists(aStrides, a_gets[i].getStrides())))
        return notMergeable; // Inconsistent memory use for all gets
    bMemref = b_gets[0].getMemref();
    bOffsets = b_gets[0].getOffsets();
    bSizes = b_gets[0].getSizes();
    bStrides = b_gets[0].getStrides();
    for (unsigned i = 1; i < b_gets.size(); i++)
      if ((!memrefsAreAffinitiveToSameChannel(bMemref,
                                              b_gets[i].getMemref())) ||
          (!areTheSameSSAValueLists(bOffsets, b_gets[i].getOffsets())) ||
          (!areTheSameSSAValueLists(bSizes, b_gets[i].getSizes())) ||
          (!areTheSameSSAValueLists(bStrides, b_gets[i].getStrides())))
        return notMergeable; // Inconsistent memory use for all gets
    if ((!memrefsAreAffinitiveToSameChannel(aMemref, bMemref)) ||
        (!areTheSameSSAValueLists(aOffsets, bOffsets)) ||
        (!areTheSameSSAValueLists(aSizes, bSizes)) ||
        (!areTheSameSSAValueLists(aStrides, bStrides)))
      return notMergeable;
    // If destinations of the two channel ops fall into different affine.if
    // conditions, which imply spatial scaling, then they are not fusable in
    // time.
    for (unsigned i = 0; i < a_gets.size(); i++)
      if ((!areUnderTheSameAffineIfCond(a_gets[i], b_gets[i])))
        return notMergeable;
    std::vector<std::tuple<bool, std::string>> putResults;
    for (unsigned i = 0; i < a_puts.size(); i++) {
      auto a_put_loop_nest = getParentLoopNest(a_puts[i].getOperation());
      auto b_put_loop_nest = getParentLoopNest(b_puts[i].getOperation());
      putResults.push_back(
          checkIfTemporalMergeableImpl(a_put_loop_nest, b_put_loop_nest));
    }
    std::vector<std::tuple<bool, std::string>> getResults;
    for (unsigned i = 0; i < a_gets.size(); i++) {
      auto a_get_loop_nest = getParentLoopNest(a_gets[i].getOperation());
      auto b_get_loop_nest = getParentLoopNest(b_gets[i].getOperation());
      getResults.push_back(
          checkIfTemporalMergeableImpl(a_get_loop_nest, b_get_loop_nest));
    }
    bool overallUBMergeable = true;
    bool overallLBMergeable = true;
    bool overallNFLMergeable = true;
    for (auto putRes : putResults) {
      if (!std::get<0>(putRes))
        return notMergeable;
      overallUBMergeable &= (std::get<1>(putRes) == "UB");
      overallLBMergeable &= (std::get<1>(putRes) == "LB");
      overallNFLMergeable &= (std::get<1>(putRes) == "NFL");
    }
    for (auto getRes : getResults) {
      if (!std::get<0>(getRes))
        return notMergeable;
      overallUBMergeable &= (std::get<1>(getRes) == "UB");
      overallLBMergeable &= (std::get<1>(getRes) == "LB");
      overallNFLMergeable &= (std::get<1>(getRes) == "NFL");
    }
    if (overallNFLMergeable)
      return mergeableNoForLoop;
    else if (overallLBMergeable)
      return mergeableToLB;
    else if (overallUBMergeable)
      return mergeableToUB;
    return notMergeable;
  }
  std::vector<Block *> getParentLoopNest(Operation *op) {
    std::vector<Block *> parent_loop_nest;
    Operation *parent = op;
    while (parent) {
      if (auto forOp = dyn_cast<scf::ForOp>(parent))
        parent_loop_nest.push_back(forOp.getBody());
      else if (auto parOp = dyn_cast<scf::ParallelOp>(parent))
        parent_loop_nest.push_back(parOp.getBody());
      else if (auto hierOp = dyn_cast<air::HierarchyInterface>(parent))
        parent_loop_nest.push_back(&hierOp->getRegion(0).front());
      else if (auto aifOp = dyn_cast<affine::AffineIfOp>(parent)) {
        if (aifOp.getThenBlock()->findAncestorOpInBlock(*op))
          parent_loop_nest.push_back(aifOp.getThenBlock());
        else if (aifOp.hasElse() &&
                 aifOp.getElseBlock()->findAncestorOpInBlock(*op))
          parent_loop_nest.push_back(aifOp.getElseBlock());
      }
      parent = parent->getParentOp();
    }
    return parent_loop_nest;
  }
  bool areEquivalentControlLoops(Block *aBlock, Block *bBlock) {
    Operation *a = aBlock->getParentOp();
    Operation *b = bBlock->getParentOp();
    if (!a)
      return false;
    if (!b)
      return false;
    if (isa<scf::ForOp>(a) && isa<scf::ForOp>(b)) {
      auto a_for = dyn_cast<scf::ForOp>(a);
      auto b_for = dyn_cast<scf::ForOp>(b);
      if (a_for == b_for)
        return true;
      std::optional<int64_t> aLbCstOp =
          mlir::getConstantIntValue(a_for.getLowerBound());
      std::optional<int64_t> aUbCstOp =
          mlir::getConstantIntValue(a_for.getUpperBound());
      std::optional<int64_t> aStepCstOp =
          mlir::getConstantIntValue(a_for.getStep());
      std::optional<int64_t> bLbCstOp =
          mlir::getConstantIntValue(b_for.getLowerBound());
      std::optional<int64_t> bUbCstOp =
          mlir::getConstantIntValue(b_for.getUpperBound());
      std::optional<int64_t> bStepCstOp =
          mlir::getConstantIntValue(b_for.getStep());
      if (aLbCstOp && aUbCstOp && aStepCstOp && bLbCstOp && bUbCstOp &&
          bStepCstOp)
        if (*aLbCstOp == *bLbCstOp && *aUbCstOp == *bUbCstOp &&
            *aStepCstOp == *bStepCstOp)
          return true;
    } else if (isa<scf::ParallelOp>(a) && isa<scf::ParallelOp>(b)) {
      auto a_par = dyn_cast<scf::ParallelOp>(a);
      auto b_par = dyn_cast<scf::ParallelOp>(b);
      if (a_par == b_par)
        return true;
      if (a_par.getStep().size() != b_par.getStep().size())
        return false;
      for (unsigned i = 0; i < a_par.getLowerBound().size(); i++) {
        std::optional<int64_t> aLbCstOp =
            mlir::getConstantIntValue(a_par.getLowerBound()[i]);
        std::optional<int64_t> aUbCstOp =
            mlir::getConstantIntValue(a_par.getUpperBound()[i]);
        std::optional<int64_t> aStepCstOp =
            mlir::getConstantIntValue(a_par.getStep()[i]);
        std::optional<int64_t> bLbCstOp =
            mlir::getConstantIntValue(b_par.getLowerBound()[i]);
        std::optional<int64_t> bUbCstOp =
            mlir::getConstantIntValue(b_par.getUpperBound()[i]);
        std::optional<int64_t> bStepCstOp =
            mlir::getConstantIntValue(b_par.getStep()[i]);
        if (aLbCstOp && aUbCstOp && aStepCstOp && bLbCstOp && bUbCstOp &&
            bStepCstOp) {
          if (*aLbCstOp != *bLbCstOp || *aUbCstOp != *bUbCstOp ||
              *aStepCstOp != *bStepCstOp)
            return false;
        } else
          return false;
      }
      return true;
    } else if (isa<air::HierarchyInterface>(a) &&
               isa<air::HierarchyInterface>(b)) {
      if (a == b)
        return true;
      auto aHierSym =
          a->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
      auto bHierSym =
          b->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
      if (aHierSym && bHierSym && aHierSym.str() == bHierSym.str())
        return true;
    } else if (isa<affine::AffineIfOp>(a) && isa<affine::AffineIfOp>(b)) {
      if (a == b)
        return false; // Sharing the same affine.if means spatially parallel
                      // ops. Cannot merge by for loop (i.e. in time).
      auto aIf = dyn_cast<affine::AffineIfOp>(a);
      auto bIf = dyn_cast<affine::AffineIfOp>(b);
      if (aBlock == aIf.getThenBlock() && bBlock == bIf.getThenBlock())
        return true;
      if (aIf.hasElse() && bIf.hasElse() && aBlock == aIf.getElseBlock() &&
          bBlock == bIf.getElseBlock())
        return true;
    }
    return false;
  }
  void mergeChannelOps(RewriterBase &rewriter, air::ChannelInterface a,
                       air::ChannelInterface b) {
    // fuse a and b under the same loop nest, if a and b are under different
    // loop nests
    if (a->getParentRegion() == b->getParentRegion())
      return;
    IRMapping remap;
    remapAllParentLoopArgs(remap, a, b);
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointAfter(a);
    auto new_b = air::cloneOpAndOperands(rewriter, remap, b);
    if (air::isAsyncOp(a) && air::isAsyncOp(new_b)) {
      auto newWaitAll = rewriter.create<air::WaitAllOp>(
          a->getLoc(), air::AsyncTokenType::get(a->getContext()),
          SmallVector<Value>{air::getAsyncTokenFromOp(a),
                             air::getAsyncTokenFromOp(new_b)});
      air::getAsyncTokenFromOp(a).replaceAllUsesExcept(
          newWaitAll.getAsyncToken(), newWaitAll);
    }
    // Erase b
    if (air::isAsyncOp(b)) {
      IRMapping waitAllRemap;
      rewriter.setInsertionPoint(b);
      auto waitAll = air::replaceAsyncOpWithWaitAll(rewriter, waitAllRemap, b);
      air::getAsyncTokenFromOp(b).replaceAllUsesWith(waitAll.getAsyncToken());
    }
    b->erase();
  }
  // Fuse parent region nests to both a and b, interleaving pairs of
  // air::ChannelInterface ops, originating from a and b loop nests
  // respectively, into the fused loop nest.
  void fuseParentRegionNestByIneterleaving(RewriterBase &rewriter, Operation *a,
                                           Operation *b) {
    if (!a->getParentOfType<LoopLikeOpInterface>())
      return;
    if (!b->getParentOfType<LoopLikeOpInterface>())
      return;
    Region *aRegion = a->getParentRegion();
    Region *bRegion = b->getParentRegion();
    while (!aRegion->getParentOp()->getParentRegion()->isAncestor(
        b->getParentRegion()))
      aRegion = aRegion->getParentOp()->getParentRegion();
    while (!bRegion->getParentOp()->getParentRegion()->isAncestor(
        a->getParentRegion()))
      bRegion = bRegion->getParentOp()->getParentRegion();
    SmallVector<air::ChannelInterface> aChanOps, bChanOps;
    aRegion->walk([&aChanOps](air::ChannelInterface chanOp) {
      aChanOps.push_back(chanOp);
    });
    bRegion->walk([&bChanOps](air::ChannelInterface chanOp) {
      bChanOps.push_back(chanOp);
    });
    if (aChanOps.size() != bChanOps.size())
      return;
    for (auto [aOtherOp, bOtherOp] : llvm::zip_equal(aChanOps, bChanOps))
      mergeChannelOps(rewriter, aOtherOp, bOtherOp);
    return;
  }
  void mergeChannelOpsTemporally(air::ChannelInterface a,
                                 air::ChannelInterface b,
                                 std::string mergeByLBOrUB) {
    scf::ForOp mismatchScfFor =
        findMismatchControlLoop(getParentLoopNest(a), getParentLoopNest(b));
    if (!mismatchScfFor)
      mismatchScfFor =
          findMismatchControlLoop(getParentLoopNest(b), getParentLoopNest(a));
    if (!mismatchScfFor)
      return;

    OpBuilder builder(mismatchScfFor);
    if (mergeByLBOrUB == "LB") {
      int originalLB = *getConstantIntValue(mismatchScfFor.getLowerBound());
      if (originalLB <= 0) {
        mismatchScfFor->emitOpError("non-positive loop lower bound, NYI.");
        return;
      }
      int currLB = originalLB;
      if (a->hasAttr("setLB"))
        currLB = a->getAttrOfType<IntegerAttr>("setLB").getInt();
      a->setAttr("setLB", builder.getI32IntegerAttr(currLB - 1));
    } else if (mergeByLBOrUB == "UB") {
      int originalUB = *getConstantIntValue(mismatchScfFor.getUpperBound());
      int currUB = originalUB;
      if (a->hasAttr("setUB"))
        currUB = a->getAttrOfType<IntegerAttr>("setUB").getInt();
      a->setAttr("setUB", builder.getI32IntegerAttr(currUB + 1));
    } else {
      mismatchScfFor->emitOpError("invalid mergeByLBOrUB flag.");
      return;
    }
    // Erase b.
    if (air::isAsyncOp(b)) {
      IRMapping remap;
      builder.setInsertionPoint(b);
      auto waitAll = air::replaceAsyncOpWithWaitAll(builder, remap, b);
      air::getAsyncTokenFromOp(b).replaceAllUsesWith(waitAll.getAsyncToken());
    }
    b->erase();
  }
  void mergeChannels(RewriterBase &rewriter, air::ChannelOp chan_a,
                     air::ChannelOp chan_b) {
    std::vector<air::ChannelPutOp> a_puts =
        getChannelPutOpThroughSymbol(chan_a);
    std::vector<air::ChannelPutOp> b_puts =
        getChannelPutOpThroughSymbol(chan_b);
    std::vector<air::ChannelGetOp> a_gets =
        getChannelGetOpThroughSymbol(chan_a);
    std::vector<air::ChannelGetOp> b_gets =
        getChannelGetOpThroughSymbol(chan_b);
    // Interleave puts and gets
    for (unsigned i = 0; i < a_puts.size(); i++)
      fuseParentRegionNestByIneterleaving(rewriter, a_puts[i], b_puts[i]);
    for (unsigned i = 0; i < a_gets.size(); i++)
      fuseParentRegionNestByIneterleaving(rewriter, a_gets[i], b_gets[i]);
  }
  void mergeChannelOpsTemporally(air::ChannelOp chan_a, air::ChannelOp chan_b,
                                 std::string mergeByLBOrUB) {
    std::vector<air::ChannelPutOp> a_puts =
        getChannelPutOpThroughSymbol(chan_a);
    std::vector<air::ChannelPutOp> b_puts =
        getChannelPutOpThroughSymbol(chan_b);
    std::vector<air::ChannelGetOp> a_gets =
        getChannelGetOpThroughSymbol(chan_a);
    std::vector<air::ChannelGetOp> b_gets =
        getChannelGetOpThroughSymbol(chan_b);
    if (!b_puts[0]->getParentOfType<air::HerdOp>()) {
      mergeChannelOpsTemporally(a_puts[0], b_puts[0], mergeByLBOrUB);
    }
    if (!b_gets[0]->getParentOfType<air::HerdOp>()) {
      mergeChannelOpsTemporally(a_gets[0], b_gets[0], mergeByLBOrUB);
    }
  }
  void remapAllParentLoopArgs(IRMapping &remap, Operation *a, Operation *b) {
    auto a_loop_nest = getParentLoopNest(a);
    auto b_loop_nest = getParentLoopNest(b);
    if (a_loop_nest.size() != b_loop_nest.size())
      return;
    for (unsigned i = 0; i < a_loop_nest.size(); i++) {
      if (auto a_for = dyn_cast<scf::ForOp>(a_loop_nest[i]->getParentOp())) {
        if (auto b_for = dyn_cast<scf::ForOp>(b_loop_nest[i]->getParentOp())) {
          for (unsigned j = 0; j < a_for.getBody()->getNumArguments(); j++) {
            remap.map(b_for.getBody()->getArgument(j),
                      a_for.getBody()->getArgument(j));
          }
        }
      }
      if (auto a_par =
              dyn_cast<scf::ParallelOp>(a_loop_nest[i]->getParentOp())) {
        if (auto b_par =
                dyn_cast<scf::ParallelOp>(b_loop_nest[i]->getParentOp())) {
          for (unsigned j = 0; j < a_par.getBody()->getNumArguments(); j++)
            remap.map(b_par.getBody()->getArgument(j),
                      a_par.getBody()->getArgument(j));
          for (unsigned j = 0; j < a_par.getInitVals().size(); j++)
            remap.map(b_par.getInitVals()[j], a_par.getInitVals()[j]);
        }
      }
    }
  }

  /// Finds a minimal set of regions in the IR that collectively contain all the
  /// target channel operations (`opSet`). The resulting regions must:
  ///   - Contain **all** the specified channel operations from `opSet`.
  ///   - Not contain any unrelated (non-target) channel operations.
  ///   - Be minimal, meaning if a parent region already qualifies, its
  ///     sub-regions are not included.
  SmallVector<Region *> findMinimalChannelIfOpContainingRegions(
      const llvm::SetVector<Operation *> &opSet) {

    // Helper to check whether a region:
    //   (1) Contains at least one of the target channel operations.
    //   (2) Does not contain any unrelated channel operations.
    auto isValidRegion = [&](Region *region) -> bool {
      bool foundTarget = false;
      for (auto &op : region->getOps()) {
        if (opSet.contains(&op)) {
          foundTarget = true;
        } else if (isa<air::ChannelInterface>(&op)) {
          // Region contains a channel op not in the target set -> invalid.
          return false;
        }
      }
      return foundTarget;
    };

    // Step 1: For each target op, walk up its parent regions to find the
    //         innermost region that is "valid" according to the above criteria.
    //         Collect all such candidate regions.
    SmallVector<Region *> candidateRegions;
    for (auto op : opSet) {
      Region *region = op->getParentRegion();
      while (region) {
        if (isValidRegion(region)) {
          candidateRegions.push_back(region);
          break; // Stop at the first valid region found for this op.
        }
        // Move up to the next enclosing region.
        Operation *parentOp = region->getParentOp();
        if (!parentOp)
          break;
        region = parentOp->getParentRegion();
      }
    }

    // Step 2: Deduplicate candidate regions and remove any region that is
    //         strictly nested inside another already-selected region.
    //         This ensures minimality: only outermost valid regions are kept.
    llvm::SmallPtrSet<Region *, 4> seen;
    SmallVector<Region *> result;

    // Sort by region number (a stable property of regions within an op),
    // so that outer regions tend to be considered before their nested ones.
    llvm::sort(candidateRegions, [](Region *a, Region *b) {
      return a->getRegionNumber() < b->getRegionNumber();
    });

    for (Region *r : candidateRegions) {
      bool skip = false;
      for (Region *other : result) {
        if (other->isAncestor(r)) {
          // If an already-chosen region is an ancestor of this one,
          // we skip the nested region to enforce minimality.
          skip = true;
          break;
        }
      }
      if (!skip)
        result.push_back(r);
    }

    return result;
  }

  /// Wraps the given regions with `scf.for` loops.
  ///
  /// For each region in `regions`, this function:
  ///   1. Inserts a new `scf.for` loop with lower bound = 0, upper bound = 2,
  ///   and step = 1
  ///      **before** the operation that owns the region.
  ///   2. Moves (clones) the parent operation of the region into the body of
  ///   the new loop.
  ///   3. Properly handles asynchronous dependencies if the parent operation
  ///   produces or
  ///      consumes `air::AsyncTokenType`.
  ///   4. Replaces the original parent operation with the new loop and erases
  ///   the old op.
  void wrapRegionsWithForLoops(OpBuilder &builder,
                               const SmallVector<Region *> &regions) {
    for (Region *region : regions) {
      if (!region || region->empty())
        continue;

      // Get the operation that owns this region.
      Operation *parentOp = region->getParentOp();

      // Insert the new `scf.for` loop *before* the parent operation.
      builder.setInsertionPoint(parentOp);

      // Create constant bounds: lb = 0, ub = 2, step = 1.
      auto loc = parentOp->getLoc();
      auto lb = builder.create<arith::ConstantIndexOp>(loc, 0);
      auto ub = builder.create<arith::ConstantIndexOp>(loc, 2);
      auto step = builder.create<arith::ConstantIndexOp>(loc, 1);

      // Prepare to create the new loop. Also set up a remapping for SSA values
      // (used when cloning the original op to preserve async dependencies).
      auto newForOp = scf::ForOp();
      IRMapping remap;

      // Handle the case where the parent operation has an async token:
      //   - Create a `WaitAllOp` to aggregate its dependencies.
      //   - Pass the resulting token as the loop's iter_arg.
      //   - Map original async dependencies to the loop's iter_arg.
      if (air::getAsyncTokenFromOp(parentOp)) {
        newForOp = builder.create<scf::ForOp>(
            loc, lb, ub, step,
            builder
                .create<air::WaitAllOp>(
                    loc, air::AsyncTokenType::get(builder.getContext()),
                    air::getAsyncDependenciesFromOp(parentOp))
                .getAsyncToken());
        for (auto dep : air::getAsyncDependenciesFromOp(parentOp))
          remap.map(dep, newForOp.getRegionIterArgs()[0]);
      } else // Create a standard for loop with no async iter_args.
        newForOp = builder.create<scf::ForOp>(loc, lb, ub, step);
      // Set insertion point to the start of the loop body so that we can clone
      // the op.
      builder.setInsertionPointToStart(newForOp.getBody());
      // Clone the original parent operation into the loop body, applying the
      // SSA remap.
      auto newOp = builder.clone(*parentOp, remap);

      // Emit the yield from the loop body:
      //   - If the cloned op produces an async token, yield it.
      //   - Replace all uses of the old async token with the loop's result.
      if (auto oldAsyncToken = air::getAsyncTokenFromOp(parentOp)) {
        builder.create<scf::YieldOp>(loc, air::getAsyncTokenFromOp(newOp));
        oldAsyncToken.replaceAllUsesWith(newForOp->getResult(0));
      } else
        builder.create<scf::YieldOp>(loc);

      // Finally, erase the original parent operation since it has been replaced
      // by the loop.
      parentOp->erase();
    }
  }
};

struct IsolateAsyncDmaLoopNestInSCFForPattern
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp for_op,
                                PatternRewriter &rewriter) const override {

    auto f = for_op->getParentOfType<func::FuncOp>();
    if (!f)
      return failure();

    // Identify target child ops for hoisting.
    SmallVector<llvm::SetVector<Operation *>> target_ops_sets;

    identifyTargetOpsInSCFFor(f, for_op, target_ops_sets);
    if (target_ops_sets.size() < 2)
      return failure();

    // If necessary, hoist allocs out of the loops, too.
    RewritePatternSet patterns(f.getContext());
    patterns.insert<HoistMemallocInForPattern>(f.getContext(), false);
    (void)applyPatternsGreedily(f, std::move(patterns));

    // Hoist ops out of each scf.for.
    llvm::SetVector<Operation *> erasedOps;
    SmallVector<scf::ForOp> hoistedScfFors;
    for (auto set : target_ops_sets) {
      SmallVector<Operation *> setAsSmallVec = set.takeVector();
      auto hoistedScfFor =
          hoistTargetOpsToNewSCFFor(rewriter, for_op, setAsSmallVec);
      hoistedScfFors.push_back(hoistedScfFor);
      erasedOps.insert(setAsSmallVec.begin(), setAsSmallVec.end());
    }

    // Replace for op uses
    if (air::isAsyncOp(for_op)) {
      if (hoistedScfFors.size() == 1) {
        rewriter.replaceOp(for_op, hoistedScfFors.front());
      } else {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(for_op);
        SmallVector<Value> deps;
        for (auto scfFor : hoistedScfFors) {
          if (air::isAsyncOp(scfFor))
            deps.push_back(air::getAsyncTokenFromOp(scfFor));
        }
        rewriter.replaceOpWithNewOp<air::WaitAllOp>(
            for_op, air::AsyncTokenType::get(f.getContext()), deps);
      }
    } else {
      rewriter.eraseOp(for_op);
    }

    return success();
  }

private:
  // Build a set of child ops from the body of one scf.for op, each of which is
  // to be hoisted into a new loop.
  void identifyTargetOpsInSCFFor(
      func::FuncOp f, scf::ForOp for_op,
      SmallVector<llvm::SetVector<Operation *>> &target_ops_sets) const {
    int for_op_token_count = 0;
    for (auto v : for_op->getResults())
      if (isa<air::AsyncTokenType>(v.getType()))
        for_op_token_count++;
    if (for_op_token_count > 1)
      return; // This for op has more than one loop-carried dep token,
              // suggesting pipelining pattern. Will be handelled by
              // -air-unroll-loop-for-pipelining-pattern instead.
    SmallVector<Operation *> candidate_ops;
    for (auto &o : for_op.getBody()->getOperations()) {
      // Get for_op's immediate child op
      if (!isAsyncOp(&o))
        continue; // This pass requires an async IR.

      if (o.getParentOfType<air::HerdOp>())
        continue; // Skip over herd op's body for now. TODO: generalize this.
      if (o.getParentOfType<affine::AffineIfOp>())
        continue; // Skip over if-else bodies for now. TODO: generalize this.
      if (air::isPure(&o))
        continue; // Pure ops do not touch memory, and therefore do not require
                  // explicit hoisting.
      if (isa<air::WaitAllOp>(o))
        continue;
      if (isa<memref::AllocOp, memref::DeallocOp>(o))
        continue; // Skip over allocs and deallocs; they are hoisted separately
                  // beforehand.
      if (auto exec = dyn_cast<air::ExecuteOp>(o)) {
        if (llvm::any_of(exec.getChildOps(), [](Operation &child) {
              return isa<memref::AllocOp, memref::DeallocOp>(child);
            })) {
          // Same as above.
          continue;
        }
      }
      candidate_ops.push_back(&o);
    }

    // Perform DFS and mark all nodes in the current connected component.
    std::function<void(
        Operation *, llvm::MapVector<Operation *, SmallVector<Operation *>> &,
        llvm::SetVector<Operation *> &, llvm::SetVector<Operation *> &)>
        dfs;
    dfs = [&dfs](Operation *node,
                 llvm::MapVector<Operation *, SmallVector<Operation *>> &graph,
                 llvm::SetVector<Operation *> &visited,
                 llvm::SetVector<Operation *> &component) {
      visited.insert(node);
      component.insert(node);
      for (Operation *neighbour : graph[node])
        if (llvm::find(visited, neighbour) == visited.end())
          dfs(neighbour, graph, visited, component);
      return;
    };
    // Partition the graph into connected subgraphs.
    std::function<SmallVector<llvm::SetVector<Operation *>>(
        llvm::MapVector<Operation *, SmallVector<Operation *>> &)>
        partitionGraph;
    partitionGraph =
        [&dfs](llvm::MapVector<Operation *, SmallVector<Operation *>> &graph) {
          llvm::SetVector<Operation *> visited;
          SmallVector<llvm::SetVector<Operation *>> connectedComponents;
          for (const auto &[node, neighbours] : graph) {
            if (llvm::find(visited, node) == visited.end()) {
              llvm::SetVector<Operation *> component;
              dfs(node, graph, visited, component);
              connectedComponents.push_back(component);
            }
          }
          return connectedComponents;
        };
    llvm::MapVector<Operation *, SmallVector<Operation *>> depGraph;
    for (auto sinkOp : candidate_ops) {
      depGraph[sinkOp] = SmallVector<Operation *>{};
      for (auto sourceOp : candidate_ops)
        if (areAsyncDependent(sourceOp, sinkOp) && sourceOp != sinkOp)
          depGraph[sinkOp].push_back(sourceOp);
    }
    // Partition the graph.
    target_ops_sets = partitionGraph(depGraph);
    // Sort ops in each partition based on their order in block.
    auto sortSetVectorByOpOrder = [](llvm::SetVector<Operation *> &setVec) {
      SmallVector<Operation *> sortedVec = setVec.takeVector();
      llvm::sort(
          sortedVec.begin(), sortedVec.end(),
          [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
      setVec = llvm::SetVector<Operation *>(sortedVec.begin(), sortedVec.end());
      return;
    };
    for (auto &setVec : target_ops_sets)
      sortSetVectorByOpOrder(setVec);

    // Check if any memref.alloc needs to be hoisted.
    for (auto o : candidate_ops) {
      SmallVector<Value, 2> operand_memrefs;
      for (auto operand : o->getOperands())
        if (isa_and_present<MemRefType>(operand.getType()))
          operand_memrefs.push_back(operand);
      for (auto memref : operand_memrefs) {
        auto exec = dyn_cast_if_present<air::ExecuteOp>(memref.getDefiningOp());
        if (!exec)
          continue;
        if (for_op->isProperAncestor(exec))
          exec->setAttr("hoist_alloc",
                        mlir::BoolAttr::get(exec->getContext(), true));
      }
    }
  }
};

LogicalResult AIRIsolateAsyncDmaLoopNestsImpl(Region *region) {
  SmallVector<Operation *> forOps;
  region->walk([&](scf::ForOp op) { forOps.push_back(op); });
  auto ctx = region->getContext();

  RewritePatternSet patterns(ctx);
  patterns.insert<IsolateAsyncDmaLoopNestInSCFForPattern>(ctx);
  (void)applyOpPatternsGreedily(forOps, std::move(patterns));

  // Greedily hoisting air.herd ops out of for loops and merging, and then
  // re-applying loop splitting.
  RewritePatternSet patterns_1(ctx);
  patterns_1
      .insert<MergeAIRHerdsPattern, IsolateAsyncDmaLoopNestInSCFForPattern,
              HoistAIRHerdsToSharedRegionPattern>(ctx);
  air::LaunchOp::getCanonicalizationPatterns(patterns_1, ctx);
  air::SegmentOp::getCanonicalizationPatterns(patterns_1, ctx);
  air::HerdOp::getCanonicalizationPatterns(patterns_1, ctx);
  air::WaitAllOp::getCanonicalizationPatterns(patterns_1, ctx);
  scf::ForOp::getCanonicalizationPatterns(patterns_1, ctx);
  (void)applyPatternsGreedily(*region, std::move(patterns_1));
  return success();
}

// A pass which hoists dma ops out of shared for loops, into perfectly nested
// loops.
class AIRIsolateAsyncDmaLoopNests
    : public xilinx::air::impl::AIRIsolateAsyncDmaLoopNestsBase<
          AIRIsolateAsyncDmaLoopNests> {

public:
  AIRIsolateAsyncDmaLoopNests() = default;
  AIRIsolateAsyncDmaLoopNests(const AIRIsolateAsyncDmaLoopNests &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    SmallVector<Region *> regions;
    if (clScope == "segment")
      module.walk(
          [&](air::SegmentOp op) { regions.push_back(&op.getRegion()); });
    else if (clScope == "launch")
      module.walk(
          [&](air::LaunchOp op) { regions.push_back(&op.getRegion()); });
    else if (clScope == "func")
      module.walk([&](func::FuncOp op) { regions.push_back(&op.getRegion()); });
    else {
      emitError(module.getLoc(),
                "Unknown scope for -air-isolate-async-dma-loop-nests. Must be "
                "one of [segment, launch, func].");
      signalPassFailure();
    }

    for (auto region : regions) {
      if (AIRIsolateAsyncDmaLoopNestsImpl(region).failed())
        signalPassFailure();
    }
  }

private:
};

// A pattern which attempts to fix memref.subview output type, after memref
// shrinkage changes the memref shapes being allocated.
struct UpdateSubViewOutputTypeAfterMemrefShrinkage
    : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    MemRefType newResultType = memref::SubViewOp::inferRankReducedResultType(
        op.getType().getShape(), op.getSourceType(), op.getMixedOffsets(),
        op.getMixedSizes(), op.getMixedStrides());
    if (newResultType != op.getType()) {
      op.getResult().setType(newResultType);
      return success();
    } else
      return failure();
  }
};

// A pattern which attempts to shrink the memref sizes, based on the access
// patterns of all its uses.
struct ShrinkMemrefSizesByAccessPattern
    : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp alloc,
                                PatternRewriter &rewriter) const override {

    // Get memref.
    Value memref = alloc.getMemref();
    if (auto exec = dyn_cast<air::ExecuteOp>(alloc->getParentOp()))
      memref = exec->getResult(1);

    if (alloc->hasAttr("shrinkage"))
      return failure();

    // Get dealloc.
    memref::DeallocOp dealloc;
    // Get channel op users.
    SmallVector<air::ChannelGetOp> gets;
    SmallVector<air::ChannelPutOp> puts;
    SmallVector<Operation *> users;
    if (getAllChanUsers(memref, users, dealloc, rewriter).failed())
      return failure();

    // Analyze data access pattern.
    SmallVector<int64_t> overall_access_bounds =
        air::getDataAccessShapeFromMemcpyOp(memref, users);
    auto memref_shape = getTensorShape(memref.getType());

    bool shrinkMemref = false;
    bool boundsAreAllOnes = true;
    for (unsigned i = 0; i < memref_shape.size(); i++) {
      if (overall_access_bounds[i] != 1)
        boundsAreAllOnes = false;
      if (overall_access_bounds[i] < memref_shape[i])
        shrinkMemref = true;
    }
    if (boundsAreAllOnes)
      return failure(); // Memref access pattern analysis failed
    if (shrinkMemref) {
      // Shrink access patterns to memref.
      for (auto user : users) {
        auto chanOp = dyn_cast<air::ChannelInterface>(user);
        if (!chanOp)
          continue;
        if (updateAccessPatternAfterShrinkage(chanOp, memref_shape,
                                              overall_access_bounds, rewriter)
                .failed()) {
          alloc->setAttr("shrinkage", rewriter.getBoolAttr(false));
          return failure();
        }
      }
      for (auto user : users) {
        // Update access patterns to shrunk memref from memref.subview.
        auto subViewOp = dyn_cast<memref::SubViewOp>(user);
        if (!subViewOp)
          continue;
        if (updateAccessPatternAfterShrinkage(subViewOp, users,
                                              overall_access_bounds, rewriter)
                .failed()) {
          alloc->setAttr("shrinkage", rewriter.getBoolAttr(false));
          return failure();
        }
      }
      for (auto user : users) {
        // Update access patterns to shrunk memref from
        // vector.transfer_read/write.
        auto transReadOp = dyn_cast<vector::TransferReadOp>(user);
        auto transWriteOp = dyn_cast<vector::TransferWriteOp>(user);
        if (transReadOp) {
          if (updateAccessPatternAfterShrinkage(transReadOp, rewriter)
                  .failed()) {
            alloc->setAttr("shrinkage", rewriter.getBoolAttr(false));
            return failure();
          }
        } else if (transWriteOp) {
          if (updateAccessPatternAfterShrinkage(transWriteOp, rewriter)
                  .failed()) {
            alloc->setAttr("shrinkage", rewriter.getBoolAttr(false));
            return failure();
          }
        }
      }

      // Replace memref alloc op;
      Type elemType =
          llvm::cast<BaseMemRefType>(memref.getType()).getElementType();
      Attribute memorySpace =
          llvm::cast<BaseMemRefType>(memref.getType()).getMemorySpace();
      auto newMemrefType = MemRefType::get(overall_access_bounds, elemType,
                                           nullptr, memorySpace);
      if (auto execOp = dyn_cast<air::ExecuteOp>(alloc->getParentOp())) {
        rewriter.setInsertionPoint(execOp);
        auto newExecOp = rewriter.create<air::ExecuteOp>(
            execOp->getLoc(), air::AsyncTokenType::get(rewriter.getContext()),
            newMemrefType, execOp.getAsyncDependencies());
        Block *async_exec_bb = rewriter.createBlock(&newExecOp.getRegion());
        rewriter.setInsertionPointToStart(async_exec_bb);
        auto newAlloc =
            rewriter.create<memref::AllocOp>(alloc->getLoc(), newMemrefType);
        newAlloc->setAttr("shrinkage", rewriter.getBoolAttr(true));
        rewriter.create<air::ExecuteTerminatorOp>(rewriter.getUnknownLoc(),
                                                  newAlloc.getResult());
        for (unsigned i = 0; i < execOp->getNumResults(); i++)
          execOp->getResult(i).replaceAllUsesWith(newExecOp->getResult(i));
        // For air hierarchy ops, also update the argument type
        for (auto res : newExecOp->getResults())
          for (auto user : res.getUsers()) {
            if (!isa<air::HerdOp>(user))
              continue;
            auto herdOp = dyn_cast<air::HerdOp>(user);
            updateHerdArgumentTypes(herdOp);
          }
        rewriter.eraseOp(execOp);

      } else {
        rewriter.setInsertionPoint(alloc);
        auto newAlloc =
            rewriter.create<memref::AllocOp>(alloc->getLoc(), newMemrefType);
        newAlloc->setAttr("shrinkage", rewriter.getBoolAttr(true));
        alloc.getResult().replaceAllUsesWith(newAlloc.getResult());
        rewriter.eraseOp(alloc);
      }
      return success();
    }

    return failure();
  }

private:
  template <typename T>
  void push_back_if_unique(SmallVector<T> &vec, T entry) const {
    if (std::find(vec.begin(), vec.end(), entry) == vec.end()) {
      vec.push_back(entry);
    }
  }

  LogicalResult getAllChanUsers(Value memref, SmallVector<Operation *> &users,
                                memref::DeallocOp &dealloc,
                                OpBuilder &builder) const {
    for (auto user : memref.getUsers()) {
      if (auto da = dyn_cast<memref::DeallocOp>(user))
        dealloc = da;
      else if (isa<air::ChannelInterface>(user))
        users.push_back(user);
      else if (isa<memref::SubViewOp>(user))
        users.push_back(user);
      else if (isa<mlir::vector::TransferReadOp>(user))
        users.push_back(user);
      else if (isa<mlir::vector::TransferWriteOp>(user))
        users.push_back(user);
      else if (auto herdOp = dyn_cast<air::HerdOp>(user)) {
        auto memrefInHerd = herdOp.getTiedKernelArgument(memref);
        if (memrefInHerd &&
            getAllChanUsers(memrefInHerd, users, dealloc, builder).failed())
          return failure();
      } else
        return failure(); // NYI.
    }
    if (users.empty())
      return failure();
    return success();
  }

  // Update herd argument types based on herd operand types
  void updateHerdArgumentTypes(air::HerdOp herdOp) const {
    for (unsigned i = 0; i < herdOp.getNumKernelOperands(); i++) {
      if (herdOp.getKernelArgument(i).getType() ==
          herdOp.getKernelOperand(i).getType())
        continue;
      auto oldArg = herdOp.getKernelArgument(i);
      auto newArg = herdOp.getBody().front().insertArgument(
          herdOp.getNumDims() * 2 + i, herdOp.getKernelOperand(i).getType(),
          herdOp->getLoc());
      for (auto &oldArgUse : oldArg.getUses())
        oldArgUse.assign(newArg);
      oldArg.replaceAllUsesWith(newArg);
      herdOp.getBody().front().eraseArgument(herdOp.getNumDims() * 2 + i + 1);
    }
  }

  // Update access patterns to shrunk memref from air.channel puts and gets.
  LogicalResult
  updateAccessPatternAfterShrinkage(air::ChannelInterface chanOp,
                                    SmallVector<int> memref_shape,
                                    SmallVector<int64_t> overall_access_bounds,
                                    PatternRewriter &rewriter) const {
    rewriter.setInsertionPoint(chanOp);
    // Update offsets.
    auto new_offsets = getUpdatedOffsetsAfterShrinkage(
        memref_shape, overall_access_bounds, chanOp.getOffsets());
    int offsetListIdxOffset =
        dyn_cast<air::AsyncOpInterface>(chanOp.getOperation())
            .getAsyncDependencies()
            .size() +
        chanOp.getIndices().size() + 1;
    for (unsigned i = offsetListIdxOffset;
         i < offsetListIdxOffset + chanOp.getOffsets().size(); i++) {
      if (new_offsets[i - offsetListIdxOffset] < 0)
        continue;
      chanOp->getOpOperand(i).assign(rewriter.create<arith::ConstantIndexOp>(
          chanOp->getLoc(), new_offsets[i - offsetListIdxOffset]));
    }
    // Update strides.
    auto new_strides = getUpdatedStridesAfterShrinkage(
        memref_shape, overall_access_bounds, chanOp.getStrides());
    int strideListIdxOffset = offsetListIdxOffset + chanOp.getOffsets().size() +
                              chanOp.getSizes().size();
    for (unsigned i = strideListIdxOffset;
         i < strideListIdxOffset + chanOp.getStrides().size(); i++) {
      chanOp->getOpOperand(i).assign(rewriter.create<arith::ConstantIndexOp>(
          chanOp->getLoc(), new_strides[i - strideListIdxOffset]));
    }
    return success();
  }

  // Update access patterns to shrunk memref from memref.subview.
  LogicalResult
  updateAccessPatternAfterShrinkage(memref::SubViewOp subViewOp,
                                    SmallVector<Operation *> &users,
                                    SmallVector<int64_t> overall_access_bounds,
                                    PatternRewriter &rewriter) const {
    rewriter.setInsertionPoint(subViewOp);
    auto subview_sizes = subViewOp.getSizes().begin();
    auto subview_strides = subViewOp.getStrides().begin();
    auto subview_offsets = subViewOp.getOffsets().begin();
    auto static_sizes = subViewOp.getStaticSizes();
    auto static_strides = subViewOp.getStaticStrides();
    auto static_offsets = subViewOp.getStaticOffsets();
    // Get MemRefType after shrinkage.
    Type elemType = llvm::cast<BaseMemRefType>(subViewOp.getSource().getType())
                        .getElementType();
    Attribute memorySpace =
        llvm::cast<BaseMemRefType>(subViewOp.getSource().getType())
            .getMemorySpace();
    auto shrunkMemrefType =
        MemRefType::get(overall_access_bounds, elemType, nullptr, memorySpace);
    MemRefType inferredSubViewOutputTy =
        llvm::cast<MemRefType>(memref::SubViewOp::inferRankReducedResultType(
            subViewOp.getType().getShape(), shrunkMemrefType,
            subViewOp.getStaticOffsets(), subViewOp.getStaticSizes(),
            subViewOp.getStaticStrides()));
    // Case 1: static size mismatches the shrunk shape.
    for (unsigned i = 0; i < static_sizes.size(); i++) {
      if (static_sizes[i] < 0) {
        if (*getConstantIntValue(*subview_sizes++) !=
            overall_access_bounds[i]) {
          subViewOp.getResult().setType(inferredSubViewOutputTy);
          if (static_offsets[i] >= 0)
            continue;
          if (auto updatedOffset =
                  getUpdatedOffsetAfterShrinkage(*subview_offsets, rewriter))
            subViewOp->replaceUsesOfWith(*subview_offsets++, updatedOffset);
        }
      } else {
        if (static_sizes[i] != overall_access_bounds[i]) {
          subViewOp.getResult().setType(inferredSubViewOutputTy);
          if (static_offsets[i] >= 0)
            continue;
          if (auto updatedOffset =
                  getUpdatedOffsetAfterShrinkage(*subview_offsets, rewriter))
            subViewOp->replaceUsesOfWith(*subview_offsets++, updatedOffset);
        }
      }
    }
    // Case 2: static strides aren't ones.
    subview_offsets = subViewOp.getOffsets().begin();
    static_offsets = subViewOp.getStaticOffsets();
    for (unsigned i = 0; i < static_strides.size(); i++) {
      if (static_strides[i] < 0) {
        if (*getConstantIntValue(*subview_strides++) != 1) {
          subViewOp.getResult().setType(inferredSubViewOutputTy);
          if (static_offsets[i] >= 0)
            continue;
          if (auto updatedOffset =
                  getUpdatedOffsetAfterShrinkage(*subview_offsets, rewriter))
            subViewOp->replaceUsesOfWith(*subview_offsets++, updatedOffset);
        }
      } else {
        if (static_strides[i] != 1) {
          subViewOp.getResult().setType(inferredSubViewOutputTy);
          if (static_offsets[i] >= 0)
            continue;
          if (auto updatedOffset =
                  getUpdatedOffsetAfterShrinkage(*subview_offsets, rewriter))
            subViewOp->replaceUsesOfWith(*subview_offsets++, updatedOffset);
        }
      }
    }
    // Case 3: offset at dimension where output stride accesses beyond memref
    // volume.
    subview_offsets = subViewOp.getOffsets().begin();
    static_offsets = subViewOp.getStaticOffsets();
    auto outputStrides =
        llvm::cast<StridedLayoutAttr>(inferredSubViewOutputTy.getLayout())
            .getStrides();
    auto memrefVolume = air::getTensorVolume(inferredSubViewOutputTy);
    for (unsigned i = 0; i < outputStrides.size(); i++) {
      if (memrefVolume == 1)
        continue;
      if (outputStrides[i] < (int)memrefVolume)
        continue;
      subViewOp.getResult().setType(inferredSubViewOutputTy);
      if (static_offsets[i] >= 0)
        continue;
      if (auto updatedOffset =
              getUpdatedOffsetAfterShrinkage(*subview_offsets, rewriter))
        subViewOp->replaceUsesOfWith(*subview_offsets++, updatedOffset);
    }

    // Case 4: offset are directly or indirectly herd induction variables.
    subview_offsets = subViewOp.getOffsets().begin();
    for (unsigned i = 0; i < static_offsets.size(); i++) {
      if (static_offsets[i] >= 0)
        continue;
      if (getConstantIntValue(*subview_offsets))
        continue;
      // SSA offset.
      bool offsetIsHerdIndVar = false;
      if (getHerdArgOwner(*subview_offsets))
        offsetIsHerdIndVar = true;
      auto defOp = (*subview_offsets).getDefiningOp();
      if (!defOp)
        continue;
      SetVector<Value> opers = getOperandsToOpOrExecute(defOp);
      if (llvm::any_of(opers, [](Value oper) { return getHerdArgOwner(oper); }))
        offsetIsHerdIndVar = true;
      if (offsetIsHerdIndVar)
        if (auto updatedOffset =
                getUpdatedOffsetAfterShrinkage(*subview_offsets, rewriter))
          subViewOp->replaceUsesOfWith(*subview_offsets, updatedOffset);
      subview_offsets++;
    }

    // Drop memref.subview, if it isn't doing anything useful.
    auto allOffsetsAreZeros = [&]() {
      subview_offsets = subViewOp.getOffsets().begin();
      static_offsets = subViewOp.getStaticOffsets();
      for (unsigned i = 0; i < static_offsets.size(); i++) {
        if (static_offsets[i] > 0)
          return false;
        else if (static_offsets[i] == 0)
          continue;
        if (!getConstantIntValue(*subview_offsets)) {
          auto defOp = (*subview_offsets).getDefiningOp();
          if (!defOp)
            return false;
          SetVector<Value> opers = getOperandsToOpOrExecute(defOp);
          if (llvm::any_of(
                  opers, [](Value oper) { return !getConstantIntValue(oper); }))
            return false;
          if (llvm::any_of(opers, [](Value oper) {
                return *getConstantIntValue(oper) != 0;
              }))
            return false;
        } else if (*getConstantIntValue(*subview_offsets) != 0)
          return false;
        subview_offsets++;
      }
      return true;
    };
    auto allStridesAreOnes = [&]() {
      subview_strides = subViewOp.getStrides().begin();
      for (unsigned i = 0; i < static_strides.size(); i++) {
        if (static_strides[i] > 0 && static_strides[i] != 1)
          return false;
        else if (static_strides[i] < 0) {
          if (!getConstantIntValue(*subview_strides))
            return false;
          if (*getConstantIntValue(*subview_strides++) != 1)
            return false;
        }
      }
      return true;
    };
    auto memrefVolumeUnchanged = [&]() {
      int subviewVolume = 1;
      for (auto s : static_sizes)
        subviewVolume *= s;
      return subviewVolume ==
             (int)air::getTensorVolume(subViewOp.getResult().getType());
    };

    if (allOffsetsAreZeros() && allStridesAreOnes() &&
        memrefVolumeUnchanged()) {
      subViewOp.getResult().replaceAllUsesWith(subViewOp.getSource());
      rewriter.eraseOp(subViewOp);
    }
    return success();
  }

  // Update access patterns to shrunk memref from vector.transfer_read/write.
  LogicalResult
  updateAccessPatternAfterShrinkage(vector::TransferReadOp transReadOp,
                                    PatternRewriter &rewriter) const {
    for (auto index : transReadOp.getIndices())
      if (auto updatedOffset = getUpdatedOffsetAfterShrinkage(index, rewriter))
        transReadOp->replaceUsesOfWith(index, updatedOffset);
    return success();
  }
  LogicalResult
  updateAccessPatternAfterShrinkage(vector::TransferWriteOp transWriteOp,
                                    PatternRewriter &rewriter) const {
    for (auto index : transWriteOp.getIndices())
      if (auto updatedOffset = getUpdatedOffsetAfterShrinkage(index, rewriter))
        transWriteOp->replaceUsesOfWith(index, updatedOffset);
    return success();
  }

  // Update access patterns to shrunk memref implementation.
  Value getUpdatedOffsetAfterShrinkage(Value index,
                                       PatternRewriter &rewriter) const {
    if (!index)
      return nullptr;
    if (getConstantIntValue(index))
      return nullptr;
    auto defOp = index.getDefiningOp();
    if (defOp) {
      SetVector<Value> opers = getOperandsToOpOrExecute(defOp);
      for (auto oper : opers) {
        auto herdOp = air::getHerdArgOwner(oper);
        if (!herdOp)
          continue;
        rewriter.setInsertionPointToStart(&herdOp.getBody().front());
        Value constZero = rewriter.create<arith::ConstantIndexOp>(
            rewriter.getUnknownLoc(), 0);
        defOp->replaceUsesOfWith(oper, constZero);
        for (auto &reg : defOp->getRegions())
          replaceAllUsesInRegionWith(oper, constZero, reg);
      }
    } else if (auto herdOp = air::getHerdArgOwner(index)) {
      rewriter.setInsertionPointToStart(&herdOp.getBody().front());
      return rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(),
                                                     0);
    }
    return nullptr;
  }

  // Get operands to either an operation or an air.execute op. For air.execute
  // op, get all values being used by its region but defined above.
  SetVector<Value> getOperandsToOpOrExecute(Operation *op) const {
    SetVector<Value> opers;
    if (auto execOp = dyn_cast<air::ExecuteOp>(op))
      getUsedValuesDefinedAbove(execOp.getRegion(), opers);
    else
      opers.insert(op->getOperands().begin(), op->getOperands().end());
    return opers;
  }
};

// Get all users to the async op's async token, with type T.
template <typename T>
SmallVector<T> getTokenUsersOfType(air::AsyncOpInterface asyncOp) {
  SmallVector<T> tokenUsers;
  Value token = asyncOp.getAsyncToken();
  for (auto token_user : token.getUsers()) {
    if (auto token_user_of_type = dyn_cast<T>(token_user))
      tokenUsers.push_back(token_user_of_type);
    else if (auto token_user_wait_all = dyn_cast<air::WaitAllOp>(token_user))
      for (auto wa_user : token_user_wait_all.getAsyncToken().getUsers())
        if (auto token_user_of_type = dyn_cast<T>(wa_user))
          tokenUsers.push_back(token_user_of_type);
  }
  return tokenUsers;
}

// Scf.for loop tiling. This simple tiling implementation generates a new
// inner scf.for loop which starts from the original loop's lower bound. It
// may change the meaning of the original scf.for loop, therefore it requires
// a separate check to make sure that it is legal to tile this way.
scf::ForOp simpleScfForLoopTiling(scf::ForOp forOp, int original_step,
                                  int tiled_step) {
  // Check if it is legal to tile the for loop this way, by checking the
  // access pattern of all memrefs within the loop.
  SmallVector<air::ChannelInterface> channel_ops;
  forOp.walk([&](air::ChannelInterface op) { channel_ops.push_back(op); });
  if (channel_ops.size() != 1)
    return scf::ForOp(); // Expected to only have one channel op in loop body.
  auto offsets = channel_ops[0].getOffsets();
  auto sizes = channel_ops[0].getSizes();
  auto strides = channel_ops[0].getStrides();
  int induction_var_dim = -1;
  // Find memref type dimension which the for loop iterates on.
  auto memref_shape = getTensorShape(channel_ops[0].getMemref().getType());
  for (unsigned i = 0; i < offsets.size(); i++) {
    if (scf::getForInductionVarOwner(offsets[i]) == forOp)
      induction_var_dim = i;
  }
  if (induction_var_dim == -1 ||
      (unsigned)induction_var_dim < offsets.size() - memref_shape.size())
    return scf::ForOp(); // NYI.
  if (offsets.size() > memref_shape.size())
    induction_var_dim -= offsets.size() - memref_shape.size();
  unsigned access_volume = 1;
  for (auto v : sizes)
    access_volume *= *getConstantIntValue(v);
  if (offsets.empty() ||
      access_volume == getTensorVolume(channel_ops[0].getMemref().getType()))
    return scf::ForOp(); // May access the whole memref.

  int effective_access_size = getEffectiveMemrefSizeFromAccessPattern(
      memref_shape, sizes, strides)[induction_var_dim];
  effective_access_size *= llvm::divideCeilSigned(original_step, tiled_step);
  if (effective_access_size > original_step)
    return scf::ForOp(); // Loop iteration access out of bound.

  // Tiling.
  auto loc = forOp->getLoc();
  OpBuilder builder(forOp);
  forOp.getStepMutable().assign(
      builder.create<arith::ConstantIndexOp>(loc, original_step));
  builder.setInsertionPointToStart(forOp.getBody());
  auto new_for_op = builder.create<scf::ForOp>(
      loc, builder.create<arith::ConstantIndexOp>(loc, 0),
      builder.create<arith::ConstantIndexOp>(loc, original_step),
      builder.create<arith::ConstantIndexOp>(loc, tiled_step),
      forOp.getRegionIterArgs());
  builder.setInsertionPointToStart(new_for_op.getBody());
  IRMapping remap;
  for (unsigned j = 0; j < forOp.getNumRegionIterArgs(); j++)
    remap.map(forOp.getRegionIterArgs()[j], new_for_op.getRegionIterArgs()[j]);
  remap.map(forOp.getInductionVar(), new_for_op.getInductionVar());
  llvm::SmallSet<Operation *, 1> erased;
  Value yielded_token = nullptr;
  for (auto &o : forOp.getOps()) {
    if (&o != new_for_op && &o != forOp.getBody()->getTerminator()) {
      auto new_o = builder.clone(o, remap);
      if (isAsyncOp(new_o)) {
        yielded_token = new_o->getResult(0);
        erased.insert(&o);
      }
    }
  }
  if (!new_for_op.getBody()->mightHaveTerminator()) {
    if (yielded_token)
      builder.create<scf::YieldOp>(loc, yielded_token);
    else
      builder.create<scf::YieldOp>(loc);
  }
  for (auto o : erased) {
    // Replace all remaining uses of erased op's token with the new for op's.
    for (auto res : o->getResults())
      if (isa<air::AsyncTokenType>(res.getType()) && !res.use_empty())
        res.replaceAllUsesWith(new_for_op->getResult(0));
    o->erase();
  }

  return new_for_op;
}

// Fuse allocs and deallocs into the specified block, if the memref value is
// only used inside the block.
LogicalResult fuseAllocDeallocExecsIntoBlock(
    Block *block, PatternRewriter &rewriter, IRMapping &remap,
    std::vector<std::pair<air::ExecuteOp, air::ExecuteOp>> &allocDeallocExecs) {
  SmallVector<air::ExecuteOp> erase_keys;
  for (auto &[alloc_exec, dealloc_exec] : allocDeallocExecs) {
    Value memref = alloc_exec->getResult(1);
    bool canMove = llvm::all_of(memref.getUsers(), [block](Operation *user) {
      return block->getParentOp()->isAncestor(user) ||
             isa<memref::DeallocOp>(user);
    });
    llvm::SetVector<Value> vals;
    getUsedValuesDefinedAbove(*block->getParent(), vals);
    canMove &= llvm::is_contained(vals, memref);

    if (!canMove) {
      erase_keys.push_back(alloc_exec);
      continue;
    }
  }
  for (auto e : erase_keys)
    for (unsigned i = 0; i < allocDeallocExecs.size(); i++)
      if (e == allocDeallocExecs[i].first)
        allocDeallocExecs.erase(allocDeallocExecs.begin() + i);
  // Move the original allocs/deallocs.
  auto addDepToAllocUsersInBlock = [](air::ExecuteOp alloc, Block *blk) {
    Value memref = alloc->getResult(1);
    SmallVector<Operation *> asyncUsers;
    blk->walk([memref, &asyncUsers](Operation *user) {
      if (!llvm::is_contained(user->getOperands(), memref))
        return;
      if (air::isAsyncOp(user))
        asyncUsers.push_back(user);
      else if (auto exec =
                   dyn_cast_if_present<air::ExecuteOp>(user->getParentOp()))
        asyncUsers.push_back(exec);
    });
    for (auto user : asyncUsers)
      air::addAsyncDependencyIfNew(user, alloc.getAsyncToken());
  };
  auto addDepToDeallocInBlock = [](air::ExecuteOp dealloc, Value memref,
                                   Block *blk) {
    SmallVector<Operation *> asyncUsers;
    blk->walk([dealloc, memref, &asyncUsers](Operation *user) {
      if (user == dealloc)
        return;
      if (!llvm::is_contained(user->getOperands(), memref))
        return;
      if (air::isAsyncOp(user))
        asyncUsers.push_back(user);
      else if (auto exec =
                   dyn_cast_if_present<air::ExecuteOp>(user->getParentOp()))
        asyncUsers.push_back(exec);
    });
    // Find parents to asyncUsers in blk
    llvm::SetVector<Operation *> asyncUsersInBlk;
    for (auto user : asyncUsers) {
      if (!blk->getParent()->isAncestor(user->getParentRegion()))
        continue;
      Operation *userParent = user;
      while (blk->getParent()->isProperAncestor(userParent->getParentRegion()))
        userParent = userParent->getParentOp();
      asyncUsersInBlk.insert(userParent);
    }
    for (auto user : asyncUsersInBlk)
      air::addAsyncDependencyIfNew(dealloc, air::getAsyncTokenFromOp(user));
  };
  auto resolveDepToExecs = [](PatternRewriter &rewriter,
                              SmallVector<air::ExecuteOp> execs, Block *blk) {
    IRMapping remap;
    llvm::DenseMap<air::ExecuteOp, air::WaitAllOp> execToWaitAll;
    for (auto exec : execs) {
      rewriter.setInsertionPoint(exec);
      execToWaitAll[exec] =
          air::replaceAsyncOpWithWaitAll(rewriter, remap, exec);
    }
    for (auto exec : execs) {
      auto waitAll = execToWaitAll[exec];
      rewriter.replaceUsesWithIf(
          exec.getAsyncToken(), waitAll.getAsyncToken(),
          [blk, &execs](OpOperand &u) {
            return !blk->getParent()->isAncestor(
                       u.getOwner()->getParentRegion()) &&
                   !llvm::is_contained(execs, u.getOwner());
          });
    }
    return;
  };
  auto resolveDepFromAllocExec = [](AsyncOpInterface alloc, Block *blk) {
    SmallVector<Value> erasedD;
    for (auto d : alloc.getAsyncDependencies()) {
      if (blk->getParent()->isAncestor(d.getParentRegion()))
        continue;
      erasedD.push_back(d);
      air::addAsyncDependencyIfNew(blk->getParentOp(), d);
    }
    for (auto d : erasedD)
      air::eraseAsyncDependencyFromAsyncOp(alloc, d);
    return;
  };
  auto resolveDepFromDeallocExec = [](AsyncOpInterface delloc, Block *blk) {
    SmallVector<Value> erasedD;
    for (auto d : delloc.getAsyncDependencies()) {
      if (blk->getParent()->isAncestor(d.getParentRegion()))
        continue;
      erasedD.push_back(d);
    }
    for (auto d : erasedD)
      air::eraseAsyncDependencyFromAsyncOp(delloc, d);
    return;
  };
  for (auto &[alloc, dealloc] : allocDeallocExecs) {
    SmallVector<air::ExecuteOp> execs({alloc});
    resolveDepFromAllocExec(alloc, block);
    if (dealloc) {
      resolveDepFromDeallocExec(dealloc, block);
      execs.push_back(dealloc);
    }
    resolveDepToExecs(rewriter, execs, block);
    addDepToAllocUsersInBlock(alloc, block);
    if (dealloc) {
      addDepToDeallocInBlock(dealloc, alloc->getResult(1), block);
    }

    // Move alloc and dealloc
    rewriter.moveOpBefore(alloc, block, block->without_terminator().begin());
    if (dealloc) {
      if (block->mightHaveTerminator())
        rewriter.moveOpBefore(dealloc, block->getTerminator());
      else
        rewriter.moveOpBefore(dealloc, block, block->getOperations().end());
    }
  }
  return success();
}

// Fuse scf.for loops in region. "fuseWithAllocDeallocs" is a bool argument
// configuring whether the pass fuses allocs and deallocs together with the loop
// bands. "maxDepth" is the maximum depth of the fused loop band. "-1" means
// unlimited.
LogicalResult fuseLoopsInRegion(Region *region, PatternRewriter &rewriter,
                                bool fuseWithAllocDeallocs = true,
                                int maxDepth = 1) {
  auto loc = region->getLoc();
  auto ctx = region->getContext();
  // Map from air.execute op containing alloc to air.execute op containing
  // dealloc.
  std::vector<std::pair<air::ExecuteOp, air::ExecuteOp>> alloc_dealloc_execs;
  for (auto execOp : region->getOps<air::ExecuteOp>()) {
    if (llvm::none_of(execOp.getChildOps(), [](Operation &child_op) {
          return isa<memref::AllocOp>(child_op);
        }))
      continue;
    SmallVector<Value> memrefs;
    for (auto res : execOp->getResults())
      if (isa<MemRefType>(res.getType()))
        memrefs.push_back(res);
    // Skip over any memref results used by other than air.channel.put/get ops
    // in loops.
    if (llvm::any_of(memrefs, [](Value v) {
          return llvm::any_of(v.getUsers(), [](Operation *user) {
            return isa<air::ChannelInterface>(user) &&
                   !isa<scf::ForOp>(user->getParentOp());
          });
        }))
      continue;
    alloc_dealloc_execs.push_back(std::make_pair(execOp, nullptr));
  }
  for (auto execOp : region->getOps<air::ExecuteOp>()) {
    if (llvm::none_of(execOp.getChildOps(), [](Operation &child_op) {
          return isa<memref::DeallocOp>(child_op);
        }))
      continue;
    auto dealloc = dyn_cast<memref::DeallocOp>(execOp.getChildOps().front());
    for (auto &pair : alloc_dealloc_execs) {
      if (dealloc.getMemref() == pair.first.getResult(1)) {
        pair.second = execOp;
      }
    }
  }
  // Get roots to perfectly nested scf.for loops.
  auto getNumChannelPutsGetsInBlock = [](Block *block) {
    return (int)llvm::range_size(block->getOps<air::ChannelInterface>());
  };
  auto getNumUniqueChannelsInBlock = [](Block *block) {
    llvm::SmallSet<std::string, 1> chanNamesInBlock;
    for (auto chanOp : block->getOps<air::ChannelInterface>())
      chanNamesInBlock.insert(chanOp.getChanName().str());
    return (int)chanNamesInBlock.size();
  };
  std::function<void(Block *, SmallVector<scf::ForOp> &, int, int)>
      getPerfectlyNestedForInForBody;
  getPerfectlyNestedForInForBody = [&](Block *body,
                                       SmallVector<scf::ForOp> &forOpNest,
                                       int max_depth, int curr_depth) {
    if (max_depth >= 0 && curr_depth >= max_depth)
      return;
    for (auto forOp : body->getOps<scf::ForOp>()) {
      if (getNumUniqueChannelsInBlock(forOp.getBody()) <= 1 &&
          hasNImpureOps(
              forOp.getBody(),
              std::max(getNumChannelPutsGetsInBlock(forOp.getBody()), 1)) &&
          air::getStaticScfForTripCountAsInt(forOp)) {
        forOpNest.push_back(forOp);
        getPerfectlyNestedForInForBody(forOp.getBody(), forOpNest, max_depth,
                                       ++curr_depth);
      }
    }
    return;
  };
  SmallVector<SmallVector<scf::ForOp>> perfectlyNestedForBands;
  for (auto forOp : region->getOps<scf::ForOp>()) {
    // Conditions for candicate scf.for op for fusion: (1) has at most 1
    // unique channels operating in the block, (2) is either perfectly nested,
    // or contains only channel ops, (3) is static for loop.
    if (getNumUniqueChannelsInBlock(forOp.getBody()) <= 1 &&
        hasNImpureOps(
            forOp.getBody(),
            std::max(getNumChannelPutsGetsInBlock(forOp.getBody()), 1)) &&
        air::getStaticScfForTripCountAsInt(forOp)) {
      SmallVector<scf::ForOp> newForOpNest = {forOp};
      getPerfectlyNestedForInForBody(forOp.getBody(), newForOpNest, maxDepth,
                                     1);
      perfectlyNestedForBands.push_back(newForOpNest);
    }
  }
  if (perfectlyNestedForBands.empty())
    return failure();
  if (fuseWithAllocDeallocs && alloc_dealloc_execs.empty())
    return failure();

  // From the loop bands, get fusable scf.for for loop bands.
  SmallVector<SmallVector<scf::ForOp>> equalIterationForOps;
  SmallVector<scf::ForOp> equalIterFirstBand = perfectlyNestedForBands.front();
  equalIterationForOps.push_back(equalIterFirstBand);
  for (unsigned i = 1; i < perfectlyNestedForBands.size(); i++) {
    if (perfectlyNestedForBands[i].size() != equalIterFirstBand.size())
      continue;
    if (llvm::all_of(
            llvm::zip_equal(equalIterFirstBand, perfectlyNestedForBands[i]),
            [&](std::tuple<scf::ForOp, scf::ForOp> pair) {
              int lb = *getConstantIntValue(std::get<0>(pair).getLowerBound());
              int ub = *getConstantIntValue(std::get<0>(pair).getUpperBound());
              int step = *getConstantIntValue(std::get<0>(pair).getStep());
              int band_step = *getConstantIntValue(std::get<1>(pair).getStep());
              int band_lb =
                  *getConstantIntValue(std::get<1>(pair).getLowerBound());
              int band_ub =
                  *getConstantIntValue(std::get<1>(pair).getUpperBound());
              if (band_lb == lb && band_ub == ub && band_step == step) {
                return true;
              } else if (band_lb == lb && band_ub == ub &&
                         llvm::mod(std::max(band_step, step),
                                   std::min(band_step, step)) == 0) {
                // If scf.for loops are not identical, but tilable to having
                // identical roots.
                if (simpleScfForLoopTiling(std::get<1>(pair), step, band_step))
                  return true;
              }
              return false;
            }))
      equalIterationForOps.push_back(perfectlyNestedForBands[i]);
  }
  if (equalIterationForOps.empty())
    return failure();

  // Folding memref.alloc / dealloc ops into fused loop.
  SmallVector<scf::ForOp> fusableForOps;
  for (auto forOpNest : equalIterationForOps) {
    auto nestBack = forOpNest.back();
    if (!fuseWithAllocDeallocs) {
      fusableForOps.push_back(nestBack);
      continue;
    }
    // If fuseWithAllocDeallocs, check whether the loop is using any memrefs
    // with alloc/dealloc hoisted.
    SetVector<Value> vals;
    getUsedValuesDefinedAbove(nestBack.getRegion(), vals);
    if (llvm::any_of(alloc_dealloc_execs,
                     [&](std::pair<air::ExecuteOp, air::ExecuteOp> exec_pair) {
                       return llvm::is_contained(vals,
                                                 exec_pair.first->getResult(1));
                     }))
      fusableForOps.push_back(nestBack);
  }
  if (fusableForOps.size() <= 1)
    return failure();

  rewriter.setInsertionPointAfter(equalIterationForOps.back().front());
  auto new_loop_op_init_arg =
      rewriter
          .create<air::WaitAllOp>(loc, air::AsyncTokenType::get(ctx),
                                  SmallVector<Value>{})
          .getAsyncToken();
  scf::ForOp new_loop_op = nullptr; // Inner-most newly created scf.for op.
  for (unsigned i = 0; i < equalIterFirstBand.size(); i++) {
    new_loop_op = rewriter.create<scf::ForOp>(
        rewriter.getUnknownLoc(), equalIterFirstBand[i].getLowerBound(),
        equalIterFirstBand[i].getUpperBound(), equalIterFirstBand[i].getStep(),
        SmallVector<Value>{new_loop_op_init_arg});
    if (i > 0) {
      // Create scf.yield for perfectly nested scf.for.
      rewriter.create<scf::YieldOp>(loc, air::getAsyncTokenFromOp(new_loop_op));
    }
    // Dive in.
    rewriter.setInsertionPointToStart(new_loop_op.getBody());
    new_loop_op_init_arg =
        air::getLoopCarriedTokenFromScfOp(new_loop_op, "argument");
  }
  IRMapping remap;

  // Loop fusion.
  auto getParentScfForNest = [](Operation *op) {
    SmallVector<scf::ForOp> forNest;
    if (auto forOp = dyn_cast_if_present<scf::ForOp>(op))
      forNest.push_back(forOp);
    scf::ForOp parent = dyn_cast_if_present<scf::ForOp>(op->getParentOp());
    while (parent) {
      forNest.push_back(parent);
      parent = dyn_cast_if_present<scf::ForOp>(parent->getParentOp());
    }
    return forNest;
  };
  for (auto forOp : fusableForOps) {
    for (std::tuple<scf::ForOp, scf::ForOp> pair : llvm::zip_equal(
             getParentScfForNest(forOp), getParentScfForNest(new_loop_op))) {
      remap.map(std::get<0>(pair).getInductionVar(),
                std::get<1>(pair).getInductionVar());
      for (unsigned i = 0; i < std::get<0>(pair).getRegionIterArgs().size();
           i++)
        remap.map(std::get<0>(pair).getRegionIterArgs()[i],
                  std::get<1>(pair).getRegionIterArgs()[i]);
    }
    // Preserve the original outermost scf.for's iter_arg.
    for (unsigned i = 0; i < forOp.getRegionIterArgs().size(); i++)
      if (forOp.getInitArgs()[i].getParentRegion()->isProperAncestor(
              forOp->getParentRegion()))
        remap.map(forOp.getRegionIterArgs()[i], forOp.getInitArgs()[i]);
    rewriter.setInsertionPointToEnd(new_loop_op.getBody());
    // Clone ops
    for (auto &child_op : forOp.getBody()->without_terminator())
      rewriter.clone(child_op, remap);
  }

  // Erase original scf.for ops.
  for (auto forOp : fusableForOps) {
    auto fusableBandHead = getParentScfForNest(forOp).back();
    for (unsigned i = 0; i < fusableBandHead.getNumResults(); i++) {
      fusableBandHead.getResult(i).replaceAllUsesWith(new_loop_op.getResult(i));
    }
    rewriter.eraseOp(fusableBandHead);
  }

  // Fuse allocs and deallocs into the created scf.for loop.
  if (fuseWithAllocDeallocs)
    (void)fuseAllocDeallocExecsIntoBlock(new_loop_op.getBody(), rewriter, remap,
                                         alloc_dealloc_execs);

  // Scf.yield op.
  rewriter.setInsertionPointToEnd(new_loop_op.getBody());
  SmallVector<Value> yield_dep_list;
  for (auto &child_op : new_loop_op.getBody()->getOperations()) {
    if (!child_op.getResults().empty()) {
      if (isa<air::AsyncTokenType>(child_op.getResult(0).getType()) &&
          child_op.getResult(0).getUsers().empty()) {
        yield_dep_list.push_back(child_op.getResult(0));
      }
    }
  }
  auto wa_op = rewriter.create<air::WaitAllOp>(
      loc, air::AsyncTokenType::get(ctx), yield_dep_list);
  rewriter.create<scf::YieldOp>(loc, wa_op.getAsyncToken());

  // Map from channel.put (or any parent scf.for op) to dependent channel.get.
  std::vector<Operation *> put_parents;
  std::map<Operation *, Operation *> put_get_mapping;
  new_loop_op.walk([&](air::ChannelPutOp putOp) {
    air::ChannelGetOp getOp = nullptr;
    for (auto user : putOp.getMemref().getUsers())
      if (auto get_user = dyn_cast<air::ChannelGetOp>(user))
        getOp = get_user;
    Operation *put_parent = putOp;
    while (put_parent->getParentOp() != new_loop_op) {
      put_parent = put_parent->getParentOp();
    }
    Operation *get_parent = getOp;
    if (!get_parent) {
      if (fuseWithAllocDeallocs)
        putOp->emitOpError(
            "is producing data for memref in the fused scf.for loop, but no "
            "consumer is found for this data within the fused loop. This "
            "likely indicates a failure in the compiler pass.");
      return;
    }
    while (get_parent->getParentOp() != new_loop_op) {
      get_parent = get_parent->getParentOp();
    }
    put_get_mapping[put_parent] = get_parent;
    put_parents.push_back(put_parent);
  });
  for (auto put_parent : put_parents) {
    auto get_parent = put_get_mapping[put_parent];
    if (put_parent == get_parent)
      continue;
    if (put_parent->isBeforeInBlock(get_parent))
      put_parent->moveAfter(get_parent);
    Value get_parent_token = air::getAsyncTokenFromOp(get_parent);
    for (unsigned i = 0; i < put_parent->getNumOperands(); i++)
      if (get_parent_token &&
          isa<AsyncTokenType>(put_parent->getOperand(i).getType())) {
        put_parent->getOpOperand(i).assign(get_parent_token);
      }
  }

  return success();
}

struct AIRFuncLoopFusionPattern : public OpRewritePattern<func::FuncOp> {
  using OpRewritePattern<func::FuncOp>::OpRewritePattern;

  AIRFuncLoopFusionPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(func::FuncOp op,
                                PatternRewriter &rewriter) const override {
    return fuseLoopsInRegion(&op.getRegion(), rewriter,
                             /*fuseWithAllocDeallocs*/ false, /*maxDepth*/ -1);
  }

private:
};

struct AIRLaunchLoopFusionPattern : public OpRewritePattern<air::LaunchOp> {
  using OpRewritePattern<air::LaunchOp>::OpRewritePattern;

  AIRLaunchLoopFusionPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(air::LaunchOp op,
                                PatternRewriter &rewriter) const override {
    return fuseLoopsInRegion(&op.getRegion(), rewriter,
                             /*fuseWithAllocDeallocs*/ false, /*maxDepth*/ -1);
  }

private:
};

struct AIRSegmentLoopFusionPattern : public OpRewritePattern<air::SegmentOp> {
  using OpRewritePattern<air::SegmentOp>::OpRewritePattern;

  AIRSegmentLoopFusionPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(air::SegmentOp op,
                                PatternRewriter &rewriter) const override {
    return fuseLoopsInRegion(&op.getRegion(), rewriter);
  }

private:
};

// A pass which performs loop fusion within air.segment op's region.
class AIRLoopFusion
    : public xilinx::air::impl::AIRLoopFusionBase<AIRLoopFusion> {

public:
  AIRLoopFusion() = default;
  AIRLoopFusion(const AIRLoopFusion &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect>();
  }

  void runPreProcPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns
        .insert<CanonicalizeAffineApplyOnLoopInductionVar, UnrollScfParallel>(
            ctx);
    air::WaitAllOp::getCanonicalizationPatterns(patterns, ctx);
    air::ExecuteOp::getCanonicalizationPatterns(patterns, ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }

  void runPostProcPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ShrinkMemrefSizesByAccessPattern,
                    UpdateSubViewOutputTypeAfterMemrefShrinkage>(ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
    // Update func.call declaration post memref shrinkage
    SmallVector<memref::AllocOp> shrunkMemallocs;
    funcOp.walk([&](memref::AllocOp op) {
      if (op->hasAttr("shrinkage"))
        shrunkMemallocs.push_back(op);
    });

    // Find indirect funcCall users of memref.
    auto getFuncCallIndirUser = [](Operation *u,
                                   SmallVector<func::CallOp> &funcCalls) {
      if (auto funcCall = dyn_cast<func::CallOp>(u))
        funcCalls.push_back(funcCall);
      else if (auto subview = dyn_cast<memref::SubViewOp>(u)) {
        for (auto subViewUser : subview.getResult().getUsers())
          if (auto funcCall = dyn_cast<func::CallOp>(subViewUser))
            funcCalls.push_back(funcCall);
      }
    };

    SmallVector<func::CallOp> funcCalls;
    for (auto alloc : shrunkMemallocs) {
      Value memref = alloc.getMemref();
      if (auto exec = dyn_cast<air::ExecuteOp>(alloc->getParentOp()))
        memref = exec.getResult(1);
      for (auto user : memref.getUsers()) {
        getFuncCallIndirUser(user, funcCalls);
        if (auto herdOp = dyn_cast<air::HerdOp>(user)) {
          auto herdArg = herdOp.getTiedKernelArgument(memref);
          if (!herdArg)
            continue;
          for (auto userInHerd : herdArg.getUsers())
            getFuncCallIndirUser(userInHerd, funcCalls);
        }
      }
    }
    for (auto funcCall : funcCalls)
      updateFuncOpInputTypesFromCall(funcCall);
  }

  void runOnOperation() override {
    auto func = getOperation();
    runPreProcPatterns(func);
    RewritePatternSet patterns(func.getContext());
    if (clFusionScope == "segment")
      patterns.insert<AIRSegmentLoopFusionPattern>(func.getContext(), false);
    else if (clFusionScope == "launch")
      patterns.insert<AIRLaunchLoopFusionPattern>(func.getContext(), false);
    else if (clFusionScope == "all")
      patterns.insert<AIRSegmentLoopFusionPattern, AIRLaunchLoopFusionPattern>(
          func.getContext(), false);
    else {
      emitError(func.getLoc(), "Unknown fusion-scope for -air-loop-fusion. "
                               "Must be one of [segment, launch, all].");
      signalPassFailure();
    }
    (void)applyPatternsGreedily(func, std::move(patterns));
    runPostProcPatterns(func);
    func.walk([&](memref::AllocOp op) { op->removeAttr("shrinkage"); });
  }

private:
  void updateFuncOpInputTypesFromCall(func::CallOp callOp) const {
    // Fetch name.
    StringRef fnName = callOp.getCallee();
    auto fnDecl = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(
        callOp->getParentOfType<ModuleOp>(), fnName));
    if (!fnDecl) {
      callOp->emitOpError("expected function declaration");
      return;
    }

    // Update function's argument types.
    auto functionType = fnDecl.getFunctionType();
    auto newArgTypes = llvm::to_vector<6>(callOp.getOperandTypes());
    auto newFunctionType = FunctionType::get(fnDecl.getContext(), newArgTypes,
                                             functionType.getResults());
    fnDecl.setType(newFunctionType);
  }
};

// Air launch is converted to scf for loop nest here, so as to enable
// compile-time shim dma bd optimizations.
struct AIRLaunchToScfForPattern : public OpRewritePattern<air::LaunchOp> {
  using OpRewritePattern<air::LaunchOp>::OpRewritePattern;

  AIRLaunchToScfForPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(air::LaunchOp launch,
                                PatternRewriter &rewriter) const override {
    if (launch->hasAttr("dummyLaunch"))
      return failure(); // Ignore dummy launch having no iteration space
    auto loc = launch->getLoc();
    auto context = rewriter.getContext();

    // Create a dummy single-iteration air launch op in place of the original
    // launch, to preserve the original launch's region which represents the
    // lifetime of all hardware inside it.
    auto dummyLaunch = rewriter.create<air::LaunchOp>(
        loc, /*async_dependencies*/ SmallVector<Value>(),
        /*sizes*/ SmallVector<Value>(), launch.getKernelOperands(),
        /*is_async*/ true);
    dummyLaunch->setAttrs(launch->getDiscardableAttrDictionary());
    dummyLaunch->setAttr("dummyLaunch", BoolAttr::get(context, true));
    rewriter.setInsertionPointToStart(&dummyLaunch.getBody().front());

    SmallVector<Value> lbs, ubs, steps;
    auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // make scf.for loop nest to replace air.launch
    for (auto d : launch.getSizeOperands()) {
      lbs.push_back(c0);
      auto const_d = getConstantIntValue(d);
      ubs.push_back(rewriter.create<arith::ConstantIndexOp>(loc, *const_d));
      steps.push_back(c1);
    }
    if (lbs.empty()) {
      lbs.push_back(c0);
      ubs.push_back(c1);
      steps.push_back(c1);
    }

    SmallVector<Value> ivs;
    Block *body = nullptr;

    // Serialize launch into scf.for.
    SmallVector<Value> iterArgs;
    if (air::isAsyncOp(launch)) {
      auto waitAll = rewriter.create<air::WaitAllOp>(
          loc, air::AsyncTokenType::get(context), SmallVector<Value>{});
      iterArgs.push_back(waitAll.getAsyncToken());
    }
    for (unsigned i = 0; i < lbs.size(); i++) {
      auto scfFor =
          rewriter.create<scf::ForOp>(loc, lbs[i], ubs[i], steps[i], iterArgs);
      if (i != 0 && scfFor->getNumResults())
        rewriter.create<scf::YieldOp>(loc, scfFor->getResults());
      iterArgs.clear();
      for (auto v : scfFor.getRegionIterArgs())
        iterArgs.push_back(v);
      body = scfFor.getBody();
      ivs.push_back(scfFor.getInductionVar());
      rewriter.setInsertionPointToStart(scfFor.getBody());
    }

    IRMapping remap;

    // map launch iteration space to scf.for loop nest's ivs
    for (auto p : llvm::zip(launch.getIds(), ivs))
      remap.map(std::get<0>(p), std::get<1>(p));

    // map launch size to scf.for loop nest's upper bounds
    for (auto p : llvm::zip(launch.getSizeOperands(), ubs))
      remap.map(std::get<0>(p), std::get<1>(p));

    // remap isolated from above launch arguments
    for (unsigned i = 0; i < launch.getNumKernelOperands(); i++)
      remap.map(launch.getKernelArgument(i), dummyLaunch.getKernelArgument(i));

    // clone the body
    rewriter.setInsertionPointToStart(body);
    auto &launchOps = launch.getBody().front().getOperations();
    for (auto bi = launchOps.begin(), be = --launchOps.end(); bi != be; ++bi)
      rewriter.clone(*bi, remap);

    // Create scf.yield to terminate scf.for body.
    if (air::isAsyncOp(launch)) {
      OpBuilder::InsertionGuard guard(rewriter);
      auto wa = generateWaitAllToTerminateBlock(*body, rewriter,
                                                /*isBlocking*/ false);
      rewriter.create<scf::YieldOp>(rewriter.getUnknownLoc(),
                                    wa.getAsyncToken());
    }

    // replace output events with air.wait_all
    if (air::isAsyncOp(launch)) {
      rewriter.setInsertionPoint(launch);
      rewriter.replaceOpWithNewOp<air::WaitAllOp>(
          launch, air::AsyncTokenType::get(context),
          air::getAsyncDependenciesFromOp(launch));
    } else
      rewriter.eraseOp(launch);

    return success();
  }
};

// A pass which performs a series of scf.for loop splitting, fusion and
// specialization, with the goal of generating efficient shim dma block
// descriptors (BD).
class AIROptimizeShimDMABDs
    : public xilinx::air::impl::AIROptimizeShimDMABDsBase<
          AIROptimizeShimDMABDs> {

public:
  AIROptimizeShimDMABDs() = default;
  AIROptimizeShimDMABDs(const AIROptimizeShimDMABDs &pass) {}
  AIROptimizeShimDMABDs(
      const ::xilinx::air::AIROptimizeShimDMABDsOptions &options)
      : AIROptimizeShimDMABDsBase(options) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect, AIE::AIEDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();

    if (func.isExternal())
      return;

    auto ctx = &getContext();
    auto device = AIE::symbolizeAIEDevice(clDevice);
    if (!device) {
      func.emitOpError("Invalid aie.device option");
      signalPassFailure();
      return;
    }

    // AIE1 architecture doesn't support multi-dimensional dma bds. Tiling shim
    // dma for AIE1 does not optimize the schedule.
    if (isa<AIE::AIE1TargetModel>(AIE::getTargetModel(*device))) {
      // AIE1 has no multi-dimensional DMA at shim. No need to tile. Apply dma
      // folding and finish.
      applyAIRL3DmaFoldingPatterns(func, *device);
      return;
    }

    // Convert air.launch to scf.for.
    RewritePatternSet patterns(ctx);
    patterns.insert<AIRLaunchToScfForPattern>(ctx);
    (void)applyPatternsGreedily(func, std::move(patterns));

    // Tile all shim dma scf.for loops into smaller and repeating tiles.
    SmallVector<scf::ForOp> shimFors;
    func.walk([&shimFors](scf::ForOp forOp) {
      // Get for loop band outside of any segment or herd region, and directly
      // nested in a launch or func op.
      if (isa<air::LaunchOp, func::FuncOp>(forOp->getParentOp())) {
        shimFors.push_back(forOp);
      }
    });
    if (shimFors.empty()) {
      // No for loops at shim to tile. Apply dma folding and finish.
      applyAIRL3DmaFoldingPatterns(func, *device);
      return;
    }
    // Helper function converting a vector of unsigned int to a vector of Value.
    auto convertVecOfIntToVecOfValue = [](OpBuilder &b,
                                          SmallVector<unsigned> clTileSizes) {
      OpBuilder::InsertionGuard guard(b);
      SmallVector<Value> optTileSizes;
      for (unsigned i = 0; i < clTileSizes.size(); ++i) {
        optTileSizes.push_back(b.create<arith::ConstantIndexOp>(
            b.getUnknownLoc(), clTileSizes[i]));
      }
      return optTileSizes;
    };
    IRRewriter rewriter(ctx);
    rewriter.setInsertionPoint(shimFors.front());
    SmallVector<Value> optTileSizes = convertVecOfIntToVecOfValue(
        rewriter,
        SmallVector<unsigned>(clTileSizes.begin(), clTileSizes.end()));
    // Helper function getting the actual tiling sizes based on the specified
    // loop band's depth and trip counts.
    auto getActualTileSizesPerScfRoot = [](OpBuilder &b, scf::ForOp root,
                                           SmallVector<Value> optTileSizes) {
      OpBuilder::InsertionGuard guard(b);
      SmallVector<scf::ForOp> perfectlyNestedLoops;
      getPerfectlyNestedLoops(perfectlyNestedLoops, root);
      SmallVector<Value> actualTileSizes;
      for (size_t i = 0;
           i < std::min(perfectlyNestedLoops.size(), optTileSizes.size());
           i++) {
        auto largestFactor = b.create<arith::ConstantIndexOp>(
            b.getUnknownLoc(),
            air::findLargestFactor(
                *air::getStaticScfForTripCountAsInt(perfectlyNestedLoops[i]),
                *getConstantIntValue(optTileSizes[i])));
        actualTileSizes.push_back(largestFactor);
      }
      return actualTileSizes;
    };
    // Tile each for loop band operated by shim dma bds.
    llvm::SetVector<scf::ForOp> forLoopsToUnroll;
    for (auto forOp : shimFors) {
      if (optTileSizes.empty())
        break;
      SmallVector<Value> actualTileSizes =
          getActualTileSizesPerScfRoot(rewriter, forOp, optTileSizes);
      auto tiledLoops = tilePerfectlyNested(forOp, ArrayRef(actualTileSizes));

      // Fixup loop-carried deps in tiled loops
      if (forOp->getNumResults()) {
        // Forward traversing through tiledLoops, to update iter args.
        for (size_t i = 0; i < tiledLoops.size(); i++) {
          scf::ForOp parentFor = tiledLoops[i]->getParentOfType<scf::ForOp>();
          auto replaceRes = tiledLoops[i].replaceWithAdditionalIterOperands(
              rewriter, parentFor.getRegionIterArgs(), true);
          if (failed(replaceRes)) {
            tiledLoops[i]->emitOpError("adding iter operands failed.");
            signalPassFailure();
          }
          tiledLoops[i] = dyn_cast<scf::ForOp>(replaceRes->getOperation());
        }
        // Backward traversing through tiledLoops, to update yields.
        for (auto tiledLoop : llvm::reverse(tiledLoops)) {
          auto loopTerm = tiledLoop.getBody()->getTerminator();
          rewriter.setInsertionPoint(loopTerm);
          rewriter.replaceOpWithNewOp<scf::YieldOp>(
              loopTerm, loopTerm->getPrevNode()->getResult(0));
        }
        // Fixup the yield connecting the innermost outer-loops to the outermost
        // inner-loops; create disconnected async edge between loop iterations
        // to make it blocking.
        rewriter.setInsertionPointAfter(tiledLoops.front());
        auto blockingWaitAll = rewriter.replaceOpWithNewOp<air::WaitAllOp>(
            tiledLoops.front()->getNextNode(), /*result_type*/ std::nullopt,
            tiledLoops.front()->getResult(0));
        rewriter.setInsertionPointAfter(blockingWaitAll);
        auto disconnectedWaitAll = rewriter.create<air::WaitAllOp>(
            tiledLoops.front()->getLoc(),
            air::AsyncTokenType::get(rewriter.getContext()),
            SmallVector<Value>{});
        rewriter.create<scf::YieldOp>(rewriter.getUnknownLoc(),
                                      disconnectedWaitAll.getAsyncToken());
      }

      // Unroll for loop nest from root down until shim-dma-unroll-depth.
      auto forLoopNest = tiledLoops;
      scf::ForOp newForLoopBand = tiledLoops.front();
      while (auto parent = dyn_cast_if_present<scf::ForOp>(
                 newForLoopBand->getParentOp())) {
        forLoopNest.insert(forLoopNest.begin(), parent);
        newForLoopBand = parent;
      }
      // Perform unrolling on the remainder loop nests after tiling, inner loop
      // first.
      for (int i = actualTileSizes.size() - 1; i >= 0; i--) {
        forLoopsToUnroll.insert(forLoopNest[i]);
      }
    }

    auto applyCanonicalizationPatterns = [](MLIRContext *ctx, Region &region) {
      RewritePatternSet affineArithCanoPatterns(ctx);
      mlir::affine::AffineApplyOp::getCanonicalizationPatterns(
          affineArithCanoPatterns, ctx);
      mlir::affine::AffineMinOp::getCanonicalizationPatterns(
          affineArithCanoPatterns, ctx);
      mlir::affine::AffineMaxOp::getCanonicalizationPatterns(
          affineArithCanoPatterns, ctx);
      mlir::arith::AddIOp::getCanonicalizationPatterns(affineArithCanoPatterns,
                                                       ctx);
      mlir::arith::MulIOp::getCanonicalizationPatterns(affineArithCanoPatterns,
                                                       ctx);
      (void)applyPatternsGreedily(region, std::move(affineArithCanoPatterns));
      return;
    };
    // Canonicalize IR to make loop bounds explicitly static.
    applyCanonicalizationPatterns(ctx, func.getBody());

    // Unroll outer scf.for loop nest.
    for (auto scfFor : forLoopsToUnroll) {
      if (failed(air::loopUnrollFullWithAsyncTokenPreserved(scfFor)))
        signalPassFailure();
    }
    // Canonicalize IR to make loop bounds explicitly static.
    applyCanonicalizationPatterns(ctx, func.getBody());

    // Apply DMA folding.
    applyAIRL3DmaFoldingPatterns(func, *device);

    if (forLoopsToUnroll.empty()) {
      // If no loop unrolling was performed, gather all air.channel_put/get
      // tokens from block, and generate a blocking wait all.
      SmallVector<Block *> funcAndLaunchBlocks(1, &func.getBody().front());
      func.walk([&funcAndLaunchBlocks](air::LaunchOp launch) {
        if (air::isAsyncOp(launch))
          funcAndLaunchBlocks.push_back(&launch.getRegion().front());
      });
      for (auto blk : funcAndLaunchBlocks) {
        {
          OpBuilder::InsertionGuard guard(rewriter);
          SmallVector<Value> chanTokens;
          for (auto chan : blk->getOps<air::ChannelInterface>())
            if (air::isAsyncOp(chan))
              chanTokens.push_back(air::getAsyncTokenFromOp(chan));

          if (blk->mightHaveTerminator())
            rewriter.setInsertionPoint(blk->getTerminator());
          else
            rewriter.setInsertionPointToEnd(blk);
          rewriter.create<air::WaitAllOp>(rewriter.getUnknownLoc(),
                                          /*result_type*/ std::nullopt,
                                          chanTokens);
        }
      }
    }
  }

private:
  void applyAIRL3DmaFoldingPatterns(func::FuncOp func, AIE::AIEDevice device) {
    // Preprocess the IR's L3 dma bds by applying loop splitting, fusion and
    // specialization patterns.
    if (isa<AIE::AIE2TargetModel>(AIE::getTargetModel(device)))
      air::applyAIRIsolateAsyncDmaLoopNestsPattern(&func.getBody());

    int maxNumDims =
        isa<AIE::AIE1TargetModel>(AIE::getTargetModel(device)) ? 1 : 4;
    int maxSize =
        isa<AIE::AIE1TargetModel>(AIE::getTargetModel(device)) ? -1 : 1023;
    bool enableForLoopUnrolling =
        isa<AIE::AIE1TargetModel>(AIE::getTargetModel(device)) ? false : true;
    air::applyAIRSpecializeChannelWrapAndStridePattern(
        &func.getRegion(),
        /*maxNumDims=*/maxNumDims,
        /*maxSize=*/maxSize,
        /*enableForLoopUnrolling=*/enableForLoopUnrolling,
        /*enableRepeatAtHighestDim=*/true);
  }
};

// A pass which performs a series of scf.for loop splitting, fusion and
// specialization, with the goal of generating efficient memtile dma block
// descriptors (BD).
class AIROptimizeMemtileDMABDs
    : public xilinx::air::impl::AIROptimizeMemtileDMABDsBase<
          AIROptimizeMemtileDMABDs> {

public:
  AIROptimizeMemtileDMABDs() = default;
  AIROptimizeMemtileDMABDs(const AIROptimizeMemtileDMABDs &pass) {}
  AIROptimizeMemtileDMABDs(
      const ::xilinx::air::AIROptimizeMemtileDMABDsOptions &options)
      : AIROptimizeMemtileDMABDsBase(options) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect, AIE::AIEDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    auto device = AIE::symbolizeAIEDevice(clDevice);
    if (!device) {
      func.emitOpError("Invalid aie.device option");
      signalPassFailure();
      return;
    }
    if (isa<AIE::AIE1TargetModel>(AIE::getTargetModel(*device))) {
      func.emitOpError("AIE1 architecture does not come with memtiles.");
      signalPassFailure();
      return;
    }
    int maxNumDims =
        isa<AIE::AIE2TargetModel>(AIE::getTargetModel(*device)) ? 4 : 4;
    int maxSize =
        isa<AIE::AIE2TargetModel>(AIE::getTargetModel(*device)) ? 1023 : 1023;
    // Preprocess the IR's L2 dma bds by applying loop splitting, fusion and
    // specialization patterns.
    llvm::SetVector<air::SegmentOp> segs;
    func.walk([&](air::SegmentOp seg) { segs.insert(seg); });
    for (auto seg : segs) {
      air::applyAIRIsolateAsyncDmaLoopNestsPattern(&seg.getBody());
      air::applyAIRSpecializeChannelWrapAndStridePattern(
          &seg.getBody(),
          /*maxNumDims*/ maxNumDims, /*maxSize*/ maxSize,
          /*enableForLoopUnrolling*/ true, /*enableRepeatAtHighestDim*/ false);

      // Create wait_all to synchronize body.
      IRRewriter rewriter(func.getContext());
      if (air::isAsyncOp(seg)) {
        OpBuilder::InsertionGuard guard(rewriter);
        if (!generateWaitAllToTerminateBlock(seg.getBody().front(), rewriter,
                                             /*isBlocking*/ true))
          signalPassFailure();
      }
    }
  }

private:
};

// Fuse pairs of alloc and dealloc into the inner-most loop-like op's body,
// which contains all uses of the memref.
template <typename OpTy>
struct AIRFuseAllocDeallocToLoopLike : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    // Skip if the loop-like op is not the inner-most loop of a perfect loop
    // nest.
    Region *opRegion = op.getLoopRegions().front();
    Block *opBlock = &opRegion->front();
    if (hasNImpureOps(opBlock, 1))
      if (llvm::any_of(opBlock->without_terminator(), [](Operation &o) {
            return isa_and_present<LoopLikeOpInterface>(o);
          }))
        return failure();

    llvm::SetVector<Value> usedValsDefedAbove, usedMemrefsDefedAbove;
    getUsedValuesDefinedAbove(*opRegion, usedValsDefedAbove);
    for (auto v : usedValsDefedAbove)
      if (isa<MemRefType>(v.getType()))
        usedMemrefsDefedAbove.insert(v);

    auto memrefIsOnlyUsedInRegionOrByDealloc = [](Region *region,
                                                  Value memref) {
      return llvm::all_of(memref.getUsers(), [region](Operation *user) {
        return isa<memref::DeallocOp>(user) ||
               region->isAncestor(user->getParentRegion());
      });
    };

    // Get candidate allocs and deallocs to be fused.
    std::vector<std::pair<memref::AllocOp, memref::DeallocOp>> allocsDeallocs;
    std::vector<std::pair<air::ExecuteOp, air::ExecuteOp>> allocDeallocExecs;
    for (auto memref : usedMemrefsDefedAbove) {
      if (!memrefIsOnlyUsedInRegionOrByDealloc(opRegion, memref))
        continue;
      air::ExecuteOp allocExec = nullptr;
      air::ExecuteOp deallocExec = nullptr;
      memref::AllocOp alloc = nullptr;
      memref::DeallocOp dealloc = nullptr;
      auto defOp = memref.getDefiningOp();
      if (auto exec = dyn_cast_if_present<air::ExecuteOp>(defOp))
        allocExec = exec;
      else if (auto a = dyn_cast_if_present<memref::AllocOp>(defOp))
        alloc = a;
      else
        continue;
      auto deallocOpt = memref::findDealloc(memref);
      if (deallocOpt && *deallocOpt) {
        if (auto exec = dyn_cast_if_present<air::ExecuteOp>(
                (*deallocOpt)->getParentOp()))
          deallocExec = exec;
        else if (auto d = dyn_cast_if_present<memref::DeallocOp>(*deallocOpt))
          dealloc = d;
      }
      if (allocExec)
        allocDeallocExecs.push_back(std::make_pair(allocExec, deallocExec));
      else if (alloc)
        allocsDeallocs.push_back(std::make_pair(alloc, dealloc));
    }

    // Fuse async allocs and deallocs, i.e. alloc and dealloc enclosed in an
    // air.execute.
    IRMapping remap;
    if (failed(fuseAllocDeallocExecsIntoBlock(opBlock, rewriter, remap,
                                              allocDeallocExecs)))
      return failure();

    // Fuse non-async allocs and deallocs.
    for (auto [alloc, dealloc] : allocsDeallocs) {
      rewriter.moveOpBefore(alloc, opBlock,
                            opBlock->without_terminator().begin());
      if (!dealloc)
        continue;
      if (opBlock->mightHaveTerminator())
        rewriter.moveOpBefore(dealloc, opBlock->getTerminator());
      else
        rewriter.moveOpBefore(dealloc, opBlock, opBlock->getOperations().end());
    }
    return success();
  }

private:
};

// Fuse pairs of alloc and dealloc into the inner-most air.hierarchy op's body,
// which contains all uses of the memref.
template <typename OpTy>
struct AIRFuseAllocDeallocToAIRHierarchy : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    llvm::SetVector<Value> usedMemrefsDefedAbove;
    for (auto v : op->getOperands())
      if (isa<MemRefType>(v.getType()))
        usedMemrefsDefedAbove.insert(v);

    auto memrefIsOnlyUsedByOpOrByDealloc = [](Operation *o, Value memref) {
      return llvm::all_of(memref.getUsers(), [o](Operation *user) {
        return isa<memref::DeallocOp>(user) || o == user;
      });
    };
    // Get candidate allocs and deallocs to be fused.
    std::vector<std::pair<Operation *, Operation *>> allocsDeallocs;
    std::vector<std::pair<Operation *, Operation *>> allocDeallocExecs;
    for (auto memref : usedMemrefsDefedAbove) {
      if (!memrefIsOnlyUsedByOpOrByDealloc(op, memref))
        continue;
      air::ExecuteOp allocExec = nullptr;
      air::ExecuteOp deallocExec = nullptr;
      memref::AllocOp alloc = nullptr;
      memref::DeallocOp dealloc = nullptr;
      auto defOp = memref.getDefiningOp();
      if (auto exec = dyn_cast_if_present<air::ExecuteOp>(defOp))
        allocExec = exec;
      else if (auto a = dyn_cast_if_present<memref::AllocOp>(defOp))
        alloc = a;
      else
        continue;
      auto deallocOpt = memref::findDealloc(memref);
      if (deallocOpt && *deallocOpt) {
        if (auto exec = dyn_cast_if_present<air::ExecuteOp>(
                (*deallocOpt)->getParentOp()))
          deallocExec = exec;
        else if (auto d = dyn_cast_if_present<memref::DeallocOp>(*deallocOpt))
          dealloc = d;
      }
      if (allocExec)
        allocDeallocExecs.push_back(std::make_pair(allocExec, deallocExec));
      else if (alloc)
        allocsDeallocs.push_back(std::make_pair(alloc, dealloc));
    }
    if (allocDeallocExecs.empty() && allocsDeallocs.empty())
      return failure();
    // Fuse async allocs and deallocs, i.e. alloc and dealloc enclosed in an
    // air.execute.
    SmallVector<Value> deps = op.getAsyncDependencies();
    SmallVector<Value> kernelOpers = op.getKernelOperands();
    for (auto [alloc, dealloc] :
         llvm::concat<std::pair<Operation *, Operation *>>(allocDeallocExecs,
                                                           allocsDeallocs)) {
      Value memref = nullptr;
      if (auto exec = dyn_cast<air::ExecuteOp>(alloc))
        memref = alloc->getResult(1);
      else
        memref = dyn_cast<memref::AllocOp>(alloc).getMemref();
      rewriter.replaceAllUsesWith(op.getTiedKernelArgument(memref), memref);
      // Remove alloc and dealloc results from the new herd's arg list.
      llvm::erase(kernelOpers, memref);
      if (auto exec = dyn_cast<air::ExecuteOp>(alloc))
        llvm::erase(deps, exec.getAsyncToken());
      if (auto exec = dyn_cast_if_present<air::ExecuteOp>(dealloc))
        llvm::erase(deps, exec.getAsyncToken());
      auto addDepToAllocUsersInRegion = [](air::ExecuteOp alloc, Region *dest) {
        Value memref = alloc->getResult(1);
        SmallVector<Operation *> asyncUsers;
        dest->walk([memref, &asyncUsers](Operation *user) {
          if (!llvm::is_contained(user->getOperands(), memref))
            return;
          if (air::isAsyncOp(user))
            asyncUsers.push_back(user);
          else if (auto exec =
                       dyn_cast_if_present<air::ExecuteOp>(user->getParentOp()))
            asyncUsers.push_back(exec);
        });
        for (auto user : asyncUsers)
          air::addAsyncDependencyIfNew(user, alloc.getAsyncToken());
      };
      auto addDepToDeallocInRegion = [](air::ExecuteOp dealloc, Value memref,
                                        Region *dest) {
        SmallVector<Operation *> asyncUsers;
        dest->walk([dealloc, memref, &asyncUsers](Operation *user) {
          if (user == dealloc)
            return;
          if (!llvm::is_contained(user->getOperands(), memref))
            return;
          if (air::isAsyncOp(user))
            asyncUsers.push_back(user);
          else if (auto exec =
                       dyn_cast_if_present<air::ExecuteOp>(user->getParentOp()))
            asyncUsers.push_back(exec);
        });
        // Find parents to asyncUsers in blk
        llvm::SetVector<Operation *> asyncUsersInDest;
        for (auto user : asyncUsers) {
          if (!dest->isAncestor(user->getParentRegion()))
            continue;
          Operation *userParent = user;
          while (dest->isProperAncestor(userParent->getParentRegion()))
            userParent = userParent->getParentOp();
          asyncUsersInDest.insert(userParent);
        }
        for (auto user : asyncUsersInDest)
          air::addAsyncDependencyIfNew(dealloc, air::getAsyncTokenFromOp(user));
      };
      auto resolveDepFromAllocExec = [](AsyncOpInterface alloc,
                                        air::HierarchyInterface dest) {
        Region &destRegion = dest.getBody();
        SmallVector<Value> erasedD;
        for (auto d : alloc.getAsyncDependencies()) {
          if (destRegion.isAncestor(d.getParentRegion()))
            continue;
          erasedD.push_back(d);
          air::addAsyncDependencyIfNew(dest.getOperation(), d);
        }
        for (auto d : erasedD)
          air::eraseAsyncDependencyFromAsyncOp(alloc, d);
        return;
      };
      auto resolveDepToExecs = [](PatternRewriter &rewriter,
                                  SmallVector<air::ExecuteOp> execs,
                                  air::HierarchyInterface dest) {
        IRMapping remap;
        llvm::DenseMap<air::ExecuteOp, air::WaitAllOp> execToWaitAll;
        for (auto exec : execs) {
          rewriter.setInsertionPoint(exec);
          execToWaitAll[exec] =
              air::replaceAsyncOpWithWaitAll(rewriter, remap, exec);
        }
        for (auto exec : execs) {
          auto waitAll = execToWaitAll[exec];
          Region &destRegion = dest.getBody();
          rewriter.replaceUsesWithIf(
              exec.getAsyncToken(), waitAll.getAsyncToken(),
              [&destRegion, &execs](OpOperand &u) {
                return !destRegion.isAncestor(
                           u.getOwner()->getParentRegion()) &&
                       !llvm::is_contained(execs, u.getOwner());
              });
        }
        return;
      };
      auto resolveDepFromDeallocExec = [](AsyncOpInterface delloc,
                                          air::HierarchyInterface dest) {
        Region &destRegion = dest.getBody();
        SmallVector<Value> erasedD;
        for (auto d : delloc.getAsyncDependencies()) {
          if (destRegion.isAncestor(d.getParentRegion()))
            continue;
          erasedD.push_back(d);
        }
        for (auto d : erasedD)
          air::eraseAsyncDependencyFromAsyncOp(delloc, d);
        return;
      };
      SmallVector<air::ExecuteOp> execs;
      if (auto exec = dyn_cast_if_present<air::ExecuteOp>(alloc)) {
        resolveDepFromAllocExec(exec, op);
        execs.push_back(exec);
      }
      if (auto exec = dyn_cast_if_present<air::ExecuteOp>(dealloc)) {
        resolveDepFromDeallocExec(exec, op);
        execs.push_back(exec);
      }
      resolveDepToExecs(rewriter, execs, op);
      if (auto exec = dyn_cast_if_present<air::ExecuteOp>(dealloc))
        addDepToDeallocInRegion(exec, alloc->getResult(1), &op.getBody());
      if (auto exec = dyn_cast_if_present<air::ExecuteOp>(alloc))
        addDepToAllocUsersInRegion(exec, &op.getBody());

      // Move alloc and dealloc
      rewriter.moveOpBefore(alloc, &op.getBody().front(),
                            op.getBody().front().without_terminator().begin());
      if (dealloc)
        rewriter.moveOpBefore(dealloc, op.getBody().front().getTerminator());
    }

    // Update air.hierarchy args.
    rewriter.setInsertionPoint(op);
    auto newHerd = rewriter.create<OpTy>(
        op->getLoc(), deps, op.getSizeOperands(), kernelOpers,
        (bool)op.getAsyncToken(), op->getAttrs());
    for (unsigned i = 0; i < op.getNumDims(); i++) {
      rewriter.replaceAllUsesWith(op.getIds()[i], newHerd.getIds()[i]);
      rewriter.replaceAllUsesWith(op.getSize()[i], newHerd.getSize()[i]);
    }
    for (auto oper : kernelOpers)
      rewriter.replaceAllUsesWith(op.getTiedKernelArgument(oper),
                                  newHerd.getTiedKernelArgument(oper));
    auto &bb = newHerd.getBody().front().getOperations();
    auto &body = op.getBody().front().getOperations();
    bb.splice(bb.begin(), body, body.begin(), --body.end());
    rewriter.replaceOp(op, newHerd);
    return success();
  }

private:
};

// Fuse pairs of alloc and dealloc into the inner-most region, which contains
// all uses of the memref.
class AIRFuseAllocDealloc
    : public xilinx::air::impl::AIRFuseAllocDeallocBase<AIRFuseAllocDealloc> {

public:
  AIRFuseAllocDealloc() = default;
  AIRFuseAllocDealloc(const AIRFuseAllocDealloc &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<AIRFuseAllocDeallocToLoopLike<scf::ForOp>,
                    AIRFuseAllocDeallocToLoopLike<scf::ParallelOp>,
                    AIRFuseAllocDeallocToAIRHierarchy<air::HerdOp>,
                    AIRFuseAllocDeallocToAIRHierarchy<air::SegmentOp>,
                    AIRFuseAllocDeallocToAIRHierarchy<air::LaunchOp>>(ctx);
    (void)applyPatternsGreedily(func, std::move(patterns));
  }

private:
};

// Shrink the size of each memref based on the actual access pattern. This
// avoids allocating buffers which are too large.
class AIRShrinkMemrefSizesByAccess
    : public xilinx::air::impl::AIRShrinkMemrefSizesByAccessBase<
          AIRShrinkMemrefSizesByAccess> {
public:
  AIRShrinkMemrefSizesByAccess() = default;
  AIRShrinkMemrefSizesByAccess(const AIRShrinkMemrefSizesByAccess &pass) {}
  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect>();
  }
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    auto funcOp = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ShrinkMemrefSizesByAccessPattern,
                    UpdateSubViewOutputTypeAfterMemrefShrinkage>(ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
    // Update func.call declaration after memref shrinkage
    SmallVector<memref::AllocOp> shrunkMemallocs;
    funcOp.walk([&](memref::AllocOp op) {
      if (op->hasAttr("shrinkage"))
        shrunkMemallocs.push_back(op);
    });
    // Find indirect funcCall users of memref.
    auto getFuncCallIndirUser = [](Operation *u,
                                   SmallVector<func::CallOp> &funcCalls) {
      if (auto funcCall = dyn_cast<func::CallOp>(u))
        funcCalls.push_back(funcCall);
      else if (auto subview = dyn_cast<memref::SubViewOp>(u)) {
        for (auto subViewUser : subview.getResult().getUsers())
          if (auto funcCall = dyn_cast<func::CallOp>(subViewUser))
            funcCalls.push_back(funcCall);
      }
    };
    SmallVector<func::CallOp> funcCalls;
    for (auto alloc : shrunkMemallocs) {
      Value memref = alloc.getMemref();
      if (auto exec = dyn_cast<air::ExecuteOp>(alloc->getParentOp()))
        memref = exec.getResult(1);
      for (auto user : memref.getUsers()) {
        getFuncCallIndirUser(user, funcCalls);
        if (auto herdOp = dyn_cast<air::HerdOp>(user)) {
          auto herdArg = herdOp.getTiedKernelArgument(memref);
          if (!herdArg)
            continue;
          for (auto userInHerd : herdArg.getUsers())
            getFuncCallIndirUser(userInHerd, funcCalls);
        }
      }
    }
    auto updateFuncOpInputTypesFromCall = [](func::CallOp callOp) {
      // Fetch name.
      StringRef fnName = callOp.getCallee();
      auto fnDecl = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(
          callOp->getParentOfType<ModuleOp>(), fnName));
      if (!fnDecl) {
        callOp->emitOpError("expected function declaration");
        return;
      }
      // Update function's argument types.
      auto functionType = fnDecl.getFunctionType();
      auto newArgTypes = llvm::to_vector<6>(callOp.getOperandTypes());
      auto newFunctionType = FunctionType::get(fnDecl.getContext(), newArgTypes,
                                               functionType.getResults());
      fnDecl.setType(newFunctionType);
    };
    for (auto funcCall : funcCalls)
      updateFuncOpInputTypesFromCall(funcCall);
  }

private:
};

} // namespace air
} // namespace xilinx

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRHoistDmaInAccumPattern() {
  return std::make_unique<AIRHoistDmaInAccumPattern>();
}

std::unique_ptr<Pass> createAIRAnnotateFrontAndBackOpsInForPattern() {
  return std::make_unique<AIRAnnotateFrontAndBackOpsInForPattern>();
}

std::unique_ptr<Pass> createAIRHoistMemallocInForPattern() {
  return std::make_unique<AIRHoistMemallocInForPattern>();
}

std::unique_ptr<Pass> createAIRUnrollLoopForPipeliningPattern() {
  return std::make_unique<AIRUnrollLoopForPipeliningPattern>();
}

std::unique_ptr<Pass> createAIRConstructPingPongDependencyPattern() {
  return std::make_unique<AIRConstructPingPongDependencyPattern>();
}

std::unique_ptr<Pass> createAIRHoistOpsNotUsingPingPongPattern() {
  return std::make_unique<AIRHoistOpsNotUsingPingPongPattern>();
}

std::unique_ptr<Pass> createAIRBroadcastDetection() {
  return std::make_unique<AIRBroadcastDetection>();
}

std::unique_ptr<Pass> createAIRPruneLinalgGenericInputDma() {
  return std::make_unique<AIRPruneLinalgGenericInputDma>();
}

std::unique_ptr<Pass> createAIRPingPongTransformationPattern() {
  return std::make_unique<AIRPingPongTransformationPattern>();
}
std::unique_ptr<OperationPass<ModuleOp>> createAIRPingPongTransformationPattern(
    const AIRPingPongTransformationPatternOptions &options) {
  return std::make_unique<AIRPingPongTransformationPattern>(options);
}

std::unique_ptr<Pass> createAIRLabelScfForLoopForPingPongPattern() {
  return std::make_unique<AIRLabelScfForLoopForPingPongPattern>();
}

std::unique_ptr<Pass> createAIRLabelScfForLoopInAIRSegmentPattern() {
  return std::make_unique<AIRLabelScfForLoopInAIRSegmentPattern>();
}

std::unique_ptr<Pass> createAIRSpecializeChannelWrapAndStridePattern() {
  return std::make_unique<AIRSpecializeChannelWrapAndStridePattern>();
}

// std::unique_ptr<mlir::Pass> createAIRDependencyScheduleOptPass() {
//   return std::make_unique<AIRDependencyScheduleOpt>();
// }

std::unique_ptr<Pass> createAIRUnrollChannelByFactorPattern() {
  return std::make_unique<AIRUnrollChannelByFactorPattern>();
}

std::unique_ptr<Pass> createAIRFuseChannels() {
  return std::make_unique<AIRFuseChannels>();
}
std::unique_ptr<OperationPass<ModuleOp>>
createAIRFuseChannels(const AIRFuseChannelsOptions &options) {
  return std::make_unique<AIRFuseChannels>(options);
}

std::unique_ptr<Pass> createAIRIsolateAsyncDmaLoopNests() {
  return std::make_unique<AIRIsolateAsyncDmaLoopNests>();
}

std::unique_ptr<Pass> createAIRLoopFusion() {
  return std::make_unique<AIRLoopFusion>();
}

std::unique_ptr<Pass> createAIROptimizeShimDMABDs() {
  return std::make_unique<AIROptimizeShimDMABDs>();
}
std::unique_ptr<Pass>
createAIROptimizeShimDMABDs(AIROptimizeShimDMABDsOptions options) {
  return std::make_unique<AIROptimizeShimDMABDs>(options);
}

std::unique_ptr<Pass> createAIROptimizeMemtileDMABDs() {
  return std::make_unique<AIROptimizeMemtileDMABDs>();
}
std::unique_ptr<Pass>
createAIROptimizeMemtileDMABDs(AIROptimizeMemtileDMABDsOptions options) {
  return std::make_unique<AIROptimizeMemtileDMABDs>(options);
}

std::unique_ptr<Pass> createAIRFuseAllocDealloc() {
  return std::make_unique<AIRFuseAllocDealloc>();
}

std::unique_ptr<Pass> createAIRShrinkMemrefSizesByAccess() {
  return std::make_unique<AIRShrinkMemrefSizesByAccess>();
}

void populateAIRLoopIndexCanonicalizationPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.insert<CanonicalizeAffineApplyOnLoopInductionVar>(ctx);
}

void populateAIRCanonicalizeChannelWrapAndStridePatterns(
    RewritePatternSet &patterns, int &maxSize, int &maxNumDims,
    bool &enableRepeatAtHighestDim) {
  MLIRContext *ctx = patterns.getContext();
  patterns.insert<
      AIRCanonicalizeChannelPutGetOpWrapAndStrideList<air::ChannelPutOp>,
      AIRCanonicalizeChannelPutGetOpWrapAndStrideList<air::ChannelGetOp>>(
      ctx, maxSize, maxNumDims, enableRepeatAtHighestDim);
}

void applyAIRSpecializeChannelWrapAndStridePattern(
    Region *region, int maxNumDims = -1, int maxSize = -1,
    bool enableForLoopUnrolling = true, bool enableRepeatAtHighestDim = false) {
  (void)AIRSpecializeChannelWrapAndStrideImpl(region, maxNumDims, maxSize,
                                              enableForLoopUnrolling,
                                              enableRepeatAtHighestDim);
}

void populateAIRLoopFusionPattern(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.insert<CanonicalizeAffineApplyOnLoopInductionVar, UnrollScfParallel,
                  AIRSegmentLoopFusionPattern, AIRLaunchLoopFusionPattern,
                  AIRFuncLoopFusionPattern>(ctx);
}

void applyAIRIsolateAsyncDmaLoopNestsPattern(Region *region) {
  (void)AIRIsolateAsyncDmaLoopNestsImpl(region);
}

void populateAIRFuseAllocDeallocToAIRHierPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.insert<AIRFuseAllocDeallocToAIRHierarchy<air::LaunchOp>,
                  AIRFuseAllocDeallocToAIRHierarchy<air::SegmentOp>,
                  AIRFuseAllocDeallocToAIRHierarchy<air::HerdOp>>(ctx);
}

} // namespace air
} // namespace xilinx
