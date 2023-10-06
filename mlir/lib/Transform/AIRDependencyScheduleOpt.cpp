//===- AIRDependencyScheduleOpt.cpp -----------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/AIRDependencyScheduleOpt.h"
#include "air/Util/Dependency.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

#include "mlir/Support/MathExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <map>
#include <numeric>
#include <string>
#include <vector>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

#define DEBUG_TYPE "air-dependency-schedule-opt"

namespace {

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
          air::ExecuteOp alloc_region_op = getRegionOfAllocOpForDmaOp(op_1);
          air::ExecuteOp dealloc_region_op = getRegionOfDeallocOpForDmaOp(op_2);
          if (alloc_region_op.getAsyncDependencies().size() > 1)
            alloc_region_op->emitOpError(
                "alloc event should have only one dependent");

          // Reconnect incoming alloc event
          if (alloc_region_op.getAsyncDependencies().size()) {
            alloc_region_op.eraseAsyncDependency(0);
          }
          // Reconnect incoming dma event
          reconnectIncomingDma(op_1, for_op);
          // Move ops to before the for loop
          alloc_region_op->moveBefore(for_op);
          op_1->moveBefore(for_op);

          // Reconnect outgoing dealloc event
          // Reconnect outgoing dma event
          scf::YieldOp yield_op =
              dyn_cast<scf::YieldOp>(for_op.getBody()->getTerminator());
          air::WaitAllOp wait_all_after_for =
              dyn_cast<air::WaitAllOp>(yield_op->getOperand(0).getDefiningOp());
          reconnectOutgoingEvents(op_2, dealloc_region_op, for_op,
                                  wait_all_after_for);
          // If wait_all depends on outgoing dma, then erase this dependency
          eraseAsyncDependencyFromAsyncOp(
              dyn_cast<air::AsyncOpInterface>(
                  wait_all_after_for.getOperation()),
              dyn_cast<air::AsyncOpInterface>(op_2.getOperation())
                  .getAsyncToken());
          // Move ops to after the for loop
          dealloc_region_op->moveAfter(for_op);
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
        } else if (auto region_op =
                       dyn_cast<air::ExecuteOp>(dep_op.getDefiningOp())) {
          // Found air.ExecuteOp in upstream dependency
          auto child_op =
              &region_op.getRegion().front().getOperations().front();
          if (auto alloc_op = dyn_cast<memref::AllocOp>(child_op)) {
            // Found memref.allocOp inside air.ExecuteOp
            foundMemrefAllocDep = true;
          }
        }
      }
    }
    return foundScfForDep & foundMemrefAllocDep;
  }

  // Return the air.execute op in the dma's dep list which contains memref.alloc
  // op
  air::ExecuteOp getRegionOfAllocOpForDmaOp(air::DmaMemcpyNdOp dma_op) const {
    Operation *current_op = dma_op.getOperation();
    air::AsyncOpInterface current_async_op =
        dyn_cast<air::AsyncOpInterface>(current_op);
    auto dependency_list = current_async_op.getAsyncDependencies();
    if (dependency_list.size()) {
      for (auto dep_op : dependency_list) {
        if (dep_op.getDefiningOp() &&
            dyn_cast<air::ExecuteOp>(dep_op.getDefiningOp())) {
          // Found air.ExecuteOp in upstream dependency
          auto region_op = dyn_cast<air::ExecuteOp>(dep_op.getDefiningOp());
          auto child_op =
              &region_op.getRegion().front().getOperations().front();
          if (auto alloc_op = dyn_cast<memref::AllocOp>(child_op)) {
            // Found memref.allocOp inside air.ExecuteOp
            return region_op;
          }
        }
      }
    }
    return nullptr;
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
      if (auto region_op = dyn_cast<air::ExecuteOp>(user)) {
        // Found air.ExecuteOp in downstream dependency
        auto child_op = &region_op.getRegion().front().getOperations().front();
        if (auto dealloc_op = dyn_cast<memref::DeallocOp>(child_op)) {
          // Found memref.deallocOp inside air.ExecuteOp
          foundDepToMemrefDealloc = true;
        }
      }
      if (dyn_cast<air::WaitAllOp>(user)) {
        foundDepToWaitall = true;
      }
    }
    return foundDepToWaitall & foundDepToMemrefDealloc;
  }

  // Return the air.execute op in the dma's downstream which contains
  // memref.dealloc op
  air::ExecuteOp getRegionOfDeallocOpForDmaOp(air::DmaMemcpyNdOp dma_op) const {
    Operation *current_op = dma_op.getOperation();
    air::AsyncOpInterface current_async_op =
        dyn_cast<air::AsyncOpInterface>(current_op);
    auto dependency_token = current_async_op.getAsyncToken();
    for (auto user : dependency_token.getUsers()) {
      if (auto region_op = dyn_cast<air::ExecuteOp>(user)) {
        // Found air.ExecuteOp in downstream dependency
        auto child_op = &region_op.getRegion().front().getOperations().front();
        if (auto dealloc_op = dyn_cast<memref::DeallocOp>(child_op)) {
          // Found memref.deallocOp inside air.ExecuteOp
          return region_op;
        }
      }
    }
    return nullptr;
  }

  // Reconnect incoming DMA event in the dependency graph
  void reconnectIncomingDma(air::DmaMemcpyNdOp dma_op,
                            scf::ForOp for_op) const {
    Operation *current_op = dma_op.getOperation();
    air::AsyncOpInterface dma_async_op =
        dyn_cast<air::AsyncOpInterface>(current_op);
    auto dependency_list = dma_async_op.getAsyncDependencies();
    if (dependency_list.size()) {
      // Erase dependence to upstream scf.forOp
      eraseAsyncDependencyFromAsyncOp(
          dyn_cast<air::AsyncOpInterface>(dma_async_op.getOperation()),
          for_op.getRegionIterArgs()[0]);
      auto for_op_iter_operand = for_op.getIterOperands()[0];
      dma_op->getResult(0).replaceAllUsesWith(for_op.getRegionIterArgs()[0]);

      replaceAllUsesInRegionWith(for_op_iter_operand, dma_op->getResult(0),
                                 *for_op->getParentRegion());
      dma_async_op.addAsyncDependency(for_op_iter_operand);
    }
  }

  // Reconnect outgoing DMA and dealloc events in the dependency graph
  void reconnectOutgoingEvents(air::DmaMemcpyNdOp dma_op,
                               air::ExecuteOp dealloc_op, scf::ForOp for_op,
                               air::WaitAllOp wait_all_after_for) const {
    Operation *current_op = dma_op.getOperation();
    air::AsyncOpInterface dma_async_op =
        dyn_cast<air::AsyncOpInterface>(current_op);
    auto dependency_list = dma_async_op.getAsyncDependencies();
    if (dependency_list.size()) {
      for (unsigned i = 0; i < dependency_list.size(); i++) {
        wait_all_after_for.addAsyncDependency(dependency_list[i]);
      }
      clearAsyncDependenciesOfAsyncOp(dma_async_op);
    }
    eraseAsyncDependencyFromAsyncOp(
        dyn_cast<air::AsyncOpInterface>(wait_all_after_for.getOperation()),
        dealloc_op.getAsyncToken());
    for_op.getResult(0).replaceAllUsesWith(dealloc_op.getResult(0));
    dma_async_op.addAsyncDependency(for_op.getResult(0));
  }

  // Check if an operation is invariant with respect to for loop iteration
  bool isInvariantWRTForLoop(Operation *op, scf::ForOp for_op) const {
    for (auto op_operand : op->getOperands()) {
      if (op_operand == for_op.getInductionVar()) {
        return false;
      }
      if (op_operand.getDefiningOp() &&
          isa<memref::SubViewOp>(op_operand.getDefiningOp())) {
        auto subview_op =
            dyn_cast<memref::SubViewOp>(op_operand.getDefiningOp());
        for (auto subview_operand : subview_op->getOperands()) {
          if (subview_operand == for_op.getInductionVar()) {
            return false;
          }
        }
      }
    }
    return true;
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
      if (iter_arg.getType().isa<air::AsyncTokenType>()) {
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
        auto child_op = &exec_op.getRegion().front().getOperations().front();
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
        op.setAttr("async_front", rewriter.getBoolAttr(true));
      for (auto token : iterTokens) {
        for (auto dep : dep_list) {
          if (token == dep) {
            setBoolAttrForAsyncOp(rewriter, &op, "async_front");
          }
        }
      }
    }

    // Get ops which are at the back of the loop body's dependency graph
    auto yield = for_op.getBody()->getTerminator();
    SmallVector<Value> yielded_tokens;
    for (auto operand : yield->getOperands()) {
      if (operand.getType().isa<air::AsyncTokenType>()) {
        yielded_tokens.push_back(operand);
      }
    }
    SmallVector<Operation *> back_candidates;
    for (auto token : yielded_tokens) {
      auto back_candidate = token.getDefiningOp();
      if (auto exec_op = dyn_cast<air::ExecuteOp>(back_candidate)) {
        auto child_op = &exec_op.getRegion().front().getOperations().front();
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
      dep_list = async_for_op.getIterOperands();
    } else if (auto async_parallel_op = dyn_cast<scf::ParallelOp>(op)) {
      dep_list = async_parallel_op.getInitVals();
    } else if (auto affine_if_op = dyn_cast<affine::AffineIfOp>(op)) {
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
    Operation *dealloc_op = nullptr;
    auto alloc_exec_memref = alloc_exec->getResults()[1];
    if (!alloc_exec_memref.getType().isa<MemRefType>())
      alloc_op->emitOpError("the ssa value yielded from execute is not memref");
    for (auto user : alloc_exec_memref.getUsers()) {
      if (isa<memref::DeallocOp>(user)) {
        dealloc_op = user;
      }
    }
    if (!dealloc_op)
      return failure();
    auto dealloc_exec = dealloc_op->getParentOfType<air::ExecuteOp>();
    if (!dealloc_exec)
      return failure();

    // Check if alloc is the target
    if (!alloc_op->hasAttr("hoist_alloc"))
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

    for (auto ia : for_op.getIterOperands()) {
      alloc_exec.addAsyncDependency(ia);
      for_op->replaceUsesOfWith(ia, alloc_exec.getAsyncToken());
    }

    // Reconnect dealloc dependency
    skipOverOpInDependencyGraph(rewriter, dealloc_exec.getOperation(),
                                for_op.getRegion());

    if (!keepMemrefDealloc) {
      for (auto for_op_token : for_op->getResults()) {
        dealloc_exec.addAsyncDependency(for_op_token);
      }
    }

    // Hoist alloc and dealloc out of for loop
    alloc_exec->moveBefore(for_op);
    if (!keepMemrefDealloc)
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

    auto deps = async_op.getAsyncDependencies();

    for (int i = deps.size() - 1; i >= 0; i--) {
      for (auto user : async_op.getAsyncToken().getUsers()) {
        if (auto async_user = dyn_cast<air::AsyncOpInterface>(user)) {
          eraseAsyncDependencyFromAsyncOp(async_user, async_op.getAsyncToken());
          addAsyncDependencyIfNew(async_user, deps[i]);
        }
        // Else if user is not an air op, and alloc depends on multiple tokens
        else if (deps.size() > 1) {
          builder.setInsertionPoint(async_op);
          air::WaitAllOp wa = builder.create<xilinx::air::WaitAllOp>(
              async_op->getLoc(),
              air::AsyncTokenType::get(async_op->getContext()),
              async_op.getAsyncDependencies());
          replaceAllUsesInRegionWith(async_op.getAsyncToken(),
                                     wa.getAsyncToken(), region);
        } else {
          replaceAllUsesInRegionWith(async_op.getAsyncToken(), deps[0], region);
        }
      }
      async_op.eraseAsyncDependency(i);
    }
  }
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
    for (auto ia : for_op.getIterOperands()) {
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
    SmallVector<Value> upstream_tokens = alloc_pong_exec.getAsyncDependencies();
    clearAsyncDependenciesOfAsyncOp(alloc_ping_exec);
    for (auto t : upstream_tokens) {
      alloc_ping_exec.addAsyncDependency(t);
    }
    alloc_ping_exec->moveBefore(alloc_pong_exec);

    SmallVector<Value, 1> iter_operands = {alloc_ping_token, alloc_pong_token,
                                           alloc_pong_token, alloc_pong_token};
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
      clearAsyncDependenciesOfAsyncOp(sink);
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
      if (v.getType().isa<air::AsyncTokenType>()) {
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
          exec->getResult(1).getType().isa<MemRefType>()) {
        alloc_execs.push_back(exec.getOperation());
        for (auto dep : exec.getAsyncDependencies()) {
          pushToAllocExecsIfHoistedFromLoop(dep, alloc_execs);
        }
      }
    }
  }
};

struct EnforceLoopCarriedMemrefDeallocPattern
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp for_op,
                                PatternRewriter &rewriter) const override {

    // Check if the loop has already been processed
    scf::YieldOp yield_op =
        dyn_cast<scf::YieldOp>(for_op.getBody()->getTerminator());
    std::vector<air::ExecuteOp> disconnected_deallocs;
    for (auto exec : for_op.getOps<air::ExecuteOp>()) {
      for (auto dealloc : exec.getOps<memref::DeallocOp>()) {
        SmallVector<Operation *, 1> vec;
        if (!hasPath(exec.getOperation(), yield_op.getOperation(), vec)) {
          disconnected_deallocs.push_back(exec);
        }
      }
    }

    // Reconnect deallocs to loop-carried dependency path
    if (!disconnected_deallocs.empty()) {
      rewriter.setInsertionPoint(yield_op);
      auto wa = rewriter.create<air::WaitAllOp>(
          yield_op.getLoc(), air::AsyncTokenType::get(rewriter.getContext()),
          yield_op->getOperands());
      for (auto dealloc_exec : disconnected_deallocs) {
        wa.addAsyncDependency(dealloc_exec.getAsyncToken());
      }
      auto new_yield_op = rewriter.create<scf::YieldOp>(
          yield_op.getLoc(), SmallVector<Value>{wa.getAsyncToken()});
      rewriter.eraseOp(yield_op.getOperation());
    }

    return success();
  }

private:
  std::vector<Operation *> adjacent_events(Operation *event) const {
    SmallVector<Value, 1> returned_tokens = {};
    for (Value result : event->getResults()) {
      if (result.getType().isa<air::AsyncTokenType>()) {
        returned_tokens.push_back(result);
      }
    }
    std::vector<Operation *> adj_set = {};
    for (Value token : returned_tokens) {
      for (Operation *user : token.getUsers()) {
        adj_set.push_back(user);
      }
    }
    return adj_set;
  }

  bool hasPath(Operation *start_event, Operation *end_event,
               SmallVector<Operation *, 1> &vec) const {
    vec.push_back(start_event);
    if (start_event == end_event)
      return true;
    int pathCount = 0;
    std::vector<Operation *> adj_set = adjacent_events(start_event);
    for (auto adj_event : adj_set) {
      SmallVector<Operation *, 1> tmp_vec;
      if (hasPath(adj_event, end_event, tmp_vec)) {
        pathCount++;
        // Concatenate
        vec.insert(vec.end(), tmp_vec.begin(), tmp_vec.end());
      }
    }
    if (pathCount)
      return true;
    vec.pop_back();
    return false;
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
    if (for_op.getIterOperands().size() != 1)
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
    rewriter.setInsertionPoint(for_op);
    IRMapping remap;
    auto new_for_op = rewriter.create<scf::ForOp>(
        for_op.getLoc(), for_op.getLowerBound(), for_op.getUpperBound(),
        for_op.getStep(), for_op.getIterOperands());
    remap.map(for_op.getInductionVar(), new_for_op.getInductionVar());
    remap.map(getLoopCarriedTokenFromScfOp(for_op, "argument"),
              getLoopCarriedTokenFromScfOp(new_for_op, "argument"));
    rewriter.setInsertionPointToStart(new_for_op.getBody());
    SmallVector<Value> yield_operands;
    for (auto op : target_ops) {
      auto new_op = rewriter.clone(*op, remap);
      yield_operands.push_back(new_op->getResult(0));
    }
    rewriter.create<scf::YieldOp>(
        new_for_op.getLoc(),
        SmallVector<Value>{
            rewriter
                .create<air::WaitAllOp>(
                    new_for_op.getLoc(),
                    air::AsyncTokenType::get(rewriter.getContext()),
                    yield_operands)
                ->getResult(0)});

    // Update dependency to hoisted ops
    for (auto herd : new_for_op.getOps<air::HerdOp>()) {
      clearAsyncDependenciesOfAsyncOp(herd);
      herd.addAsyncDependency(
          getLoopCarriedTokenFromScfOp(new_for_op, "argument"));
    }
    for (auto erase_op : target_ops) {
      for (auto user : erase_op->getResult(0).getUsers()) {
        if (auto async_user = dyn_cast<air::AsyncOpInterface>(user)) {
          eraseAsyncDependencyFromAsyncOp(async_user, erase_op->getResult(0));
          for (auto dep : getAsyncDependenciesFromOp(erase_op)) {
            if (dep != getLoopCarriedTokenFromScfOp(for_op, "argument")) {
              addAsyncDependencyIfNew(user, dep);
            }
          }
        }
      }
      erase_op->erase();
    }
    for (auto user : for_op.getResults().front().getUsers()) {
      addAsyncDependencyIfNew(user, new_for_op.getResults().front());
    }

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
    factor = factor;

    // Update channel declaration
    sizes[chanDim] *= factor;
    builder.create<air::ChannelOp>(op->getLoc(), chan_op.getSymName().str(),
                                   builder.getI64ArrayAttr(sizes));

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
      auto memTy = op.getMemref().getType().template cast<MemRefType>();
      for (auto d : getTensorShape(memTy)) {
        new_sizes.push_back(
            builder.create<arith::ConstantIndexOp>(par->getLoc(), d));
      }
    }
    auto size_op = new_sizes[dim].getDefiningOp();
    if (size_op && isa<arith::ConstantIndexOp>(size_op)) {
      auto val = dyn_cast<arith::ConstantIndexOp>(size_op).value();
      val = mlir::ceilDiv(val, factor);
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

  static SmallVector<int> getTensorShape(const ShapedType ty) {

    if (!ty.hasRank())
      return SmallVector<int>(1);

    SmallVector<int> shape = {};
    for (auto &d : ty.getShape())
      shape.push_back(d);
    return shape;
  }

  static SmallVector<int> getTensorShape(const Type ty) {
    if (auto t = ty.dyn_cast<ShapedType>()) {
      return getTensorShape(t);
    } else {
      return SmallVector<int>(1);
    }
  }
};

struct BroadcastDetection {

public:
  // Trace dma ops' dependency to loop induction variables
  void getDmaOpLoopDependency(func::FuncOp f) {
    f.walk([&](Operation *op) {
      if (auto dma_op = mlir::dyn_cast<xilinx::air::DmaMemcpyNdOp>(op)) {
        if (dma_op->getParentOfType<xilinx::air::HerdOp>()) {
          // Start recursively tracing for loop induction variables
          dma_op_history.push_back(dma_op);
          SmallVector<Value, 1> loop_dep_history;
          std::vector<Operation *> op_history;
          traceDependentInductionVar(dma_op, loop_dep_history, op_history);
          dma_op_loop_dep_history.push_back(loop_dep_history);
        }
      }
    });
  }

  // Detect boradcast opportunity based on dependency to loops
  void broadcastDetection() {
    for (unsigned i = 0; i < dma_op_history.size(); i++) {
      auto dma_op = dma_op_history[i];
      SmallVector<Value, 1> loop_dep_history = dma_op_loop_dep_history[i];
      air::HerdOp hl_op = nullptr;
      bool hasDepInHerdRows = false;
      bool hasDepInHerdCols = false;
      // Create an affine set to represent the broadcast pattern
      auto ctx = dma_op->getContext();
      for (auto v : loop_dep_history) {
        if (getHerdArgOwner(v)) {
          hl_op = getHerdArgOwner(v);
          if (v == hl_op.getIds()[0]) {
            hasDepInHerdRows = true;
          }
          if (v == hl_op.getIds()[1]) {
            hasDepInHerdCols = true;
          }
        }
      }

      if (hl_op && hasDepInHerdRows && !hasDepInHerdCols) {
        auto numColsOp = dyn_cast<arith::ConstantIndexOp>(
            hl_op.getSizeOperands()[1].getDefiningOp());
        auto numCols = numColsOp.value();
        if (numCols > 1) {
          SmallVector<AffineExpr, 5> constraints{
              getAffineDimExpr(0, ctx) - getAffineSymbolExpr(0, ctx),
              getAffineDimExpr(1, ctx), numCols - 1 - getAffineDimExpr(1, ctx),
              getAffineSymbolExpr(0, ctx),
              numCols - 1 - getAffineSymbolExpr(0, ctx)};
          SmallVector<bool, 5> eqflags{true, false, false, false, false};
          auto int_set = IntegerSet::get(2, 1, constraints, eqflags);
          dma_op->setAttr("broadcast_pattern",
                          mlir::IntegerSetAttr::get(int_set));
        }
      } else if (hl_op && !hasDepInHerdRows && hasDepInHerdCols) {
        auto numRowsOp = dyn_cast<arith::ConstantIndexOp>(
            hl_op.getSizeOperands()[0].getDefiningOp());
        auto numRows = numRowsOp.value();
        if (numRows > 1) {
          SmallVector<AffineExpr, 5> constraints{
              getAffineDimExpr(0, ctx), numRows - 1 - getAffineDimExpr(0, ctx),
              getAffineDimExpr(1, ctx) - getAffineSymbolExpr(0, ctx),
              getAffineSymbolExpr(0, ctx),
              numRows - 1 - getAffineSymbolExpr(0, ctx)};
          SmallVector<bool, 5> eqflags{false, false, true, false, false};
          auto int_set = IntegerSet::get(2, 1, constraints, eqflags);
          dma_op->setAttr("broadcast_pattern",
                          mlir::IntegerSetAttr::get(int_set));
        }
      } else if (hl_op && !hasDepInHerdRows && !hasDepInHerdCols) {
        auto numRowsOp = dyn_cast<arith::ConstantIndexOp>(
            hl_op.getSizeOperands()[0].getDefiningOp());
        auto numRows = numRowsOp.value();
        auto numColsOp = dyn_cast<arith::ConstantIndexOp>(
            hl_op.getSizeOperands()[1].getDefiningOp());
        auto numCols = numColsOp.value();
        if (numCols > 1 && numRows > 1) {
          SmallVector<AffineExpr, 5> constraints{
              getAffineDimExpr(0, ctx), numRows - 1 - getAffineDimExpr(0, ctx),
              getAffineDimExpr(1, ctx), numCols - 1 - getAffineDimExpr(1, ctx),
              getAffineSymbolExpr(0, ctx)};
          SmallVector<bool, 5> eqflags{false, false, false, false, true};
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
  std::vector<air::DmaMemcpyNdOp> dma_op_history;
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
    if (air::ExecuteOp region_op = op->getParentOfType<air::ExecuteOp>()) {
      async_op = dyn_cast<air::AsyncOpInterface>(region_op.getOperation());
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
      if (upstream_op && dyn_cast<air::DmaMemcpyNdOp>(upstream_op)) {
        auto upstream_dma = dyn_cast<air::DmaMemcpyNdOp>(upstream_op);
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
    : public xilinx::air::AIRHoistDmaInAccumPatternBase<
          AIRHoistDmaInAccumPattern> {

public:
  AIRHoistDmaInAccumPattern() = default;
  AIRHoistDmaInAccumPattern(const AIRHoistDmaInAccumPattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<HoistDmaInAccumPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
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
    : public xilinx::air::AIRBroadcastDetectionBase<AIRBroadcastDetection> {

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
    : public xilinx::air::AIRPruneLinalgGenericInputDmaBase<
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
    : public xilinx::air::AIRAnnotateFrontAndBackOpsInForPatternBase<
          AIRAnnotateFrontAndBackOpsInForPattern> {

public:
  AIRAnnotateFrontAndBackOpsInForPattern() = default;
  AIRAnnotateFrontAndBackOpsInForPattern(
      const AIRAnnotateFrontAndBackOpsInForPattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<AnnotateFrontAndBackOpsInForPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
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
    : public xilinx::air::AIRHoistMemallocInForPatternBase<
          AIRHoistMemallocInForPattern> {

public:
  AIRHoistMemallocInForPattern() = default;
  AIRHoistMemallocInForPattern(const AIRHoistMemallocInForPattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<HoistMemallocInForPattern>(ctx, clKeepMemrefDealloc);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
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
    : public xilinx::air::AIRConstructPingPongDependencyPatternBase<
          AIRConstructPingPongDependencyPattern> {

public:
  AIRConstructPingPongDependencyPattern() = default;
  AIRConstructPingPongDependencyPattern(
      const AIRConstructPingPongDependencyPattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConstructPingPongDependencyPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
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
    : public xilinx::air::AIRUnrollLoopForPipeliningPatternBase<
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
        (void)loopUnrollByFactor(for_op, unroll_factor, annotateFn);
      }
    });
  }

private:
};

class AIRHoistOpsNotUsingPingPongPattern
    : public xilinx::air::AIRHoistOpsNotUsingPingPongPatternBase<
          AIRHoistOpsNotUsingPingPongPattern> {

public:
  AIRHoistOpsNotUsingPingPongPattern() = default;
  AIRHoistOpsNotUsingPingPongPattern(
      const AIRHoistOpsNotUsingPingPongPattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<HoistOpsNotUsingPingPongPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
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
    : public xilinx::air::AIRPingPongTransformationPatternBase<
          AIRPingPongTransformationPattern> {

public:
  AIRPingPongTransformationPattern() = default;
  AIRPingPongTransformationPattern(
      const AIRPingPongTransformationPattern &pass){};

  void runIsolateScfForOpForPingPong(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<HoistOpsNotUsingPingPongPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  void runOpAnnotationPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<AnnotateFrontAndBackOpsInForPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  void runHoistMemallocPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<HoistMemallocInForPattern>(ctx, clKeepMemrefDealloc);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  void runConstructPingPongDependencyPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConstructPingPongDependencyPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
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
    : public xilinx::air::AIRLabelScfForLoopForPingPongPatternBase<
          AIRLabelScfForLoopForPingPongPattern> {

public:
  AIRLabelScfForLoopForPingPongPattern() = default;
  AIRLabelScfForLoopForPingPongPattern(
      const AIRLabelScfForLoopForPingPongPattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<LabelScfForLoopForPingPongPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
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

class AIRUnrollChannelByFactorPattern
    : public xilinx::air::AIRUnrollChannelByFactorPatternBase<
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

class AIRDependencyScheduleOpt
    : public AIRDependencyScheduleOptBase<AIRDependencyScheduleOpt> {

public:
  AIRDependencyScheduleOpt() = default;
  AIRDependencyScheduleOpt(const AIRDependencyScheduleOpt &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect>();
  }

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<HoistDmaInAccumPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  void runOnFunction(func::FuncOp f) {
    // HoistDmaInAccumPattern
    runOptPatterns(f);
    // BroadcastDetection
    BroadcastDetection proc;
    proc.runBroadcastPattern(f);
    // Remove redundant DMA copying into linalg.generic
    PruneLinalgGenericInputDma proc_0;
    proc_0.runLinalgGenericPattern(f);
  }

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps) {
      runOnFunction(f);
      // Renumber the air dma op ids
      xilinx::air::renumberDmaOps(f, "global");
    }
  }

private:
};

class AIREnforceLoopCarriedMemrefDeallocPattern
    : public AIREnforceLoopCarriedMemrefDeallocPatternBase<
          AIREnforceLoopCarriedMemrefDeallocPattern> {

public:
  AIREnforceLoopCarriedMemrefDeallocPattern() = default;
  AIREnforceLoopCarriedMemrefDeallocPattern(
      const AIREnforceLoopCarriedMemrefDeallocPattern &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect>();
  }

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<EnforceLoopCarriedMemrefDeallocPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  void runOnFunction(func::FuncOp f) { runOptPatterns(f); }

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps) {
      runOnFunction(f);
    }
  }

private:
};

} // namespace

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

std::unique_ptr<Pass> createAIRLabelScfForLoopForPingPongPattern() {
  return std::make_unique<AIRLabelScfForLoopForPingPongPattern>();
}

std::unique_ptr<mlir::Pass> createAIRDependencyScheduleOptPass() {
  return std::make_unique<AIRDependencyScheduleOpt>();
}

std::unique_ptr<Pass> createAIRUnrollChannelByFactorPattern() {
  return std::make_unique<AIRUnrollChannelByFactorPattern>();
}

std::unique_ptr<Pass> createAIREnforceLoopCarriedMemrefDeallocPattern() {
  return std::make_unique<AIREnforceLoopCarriedMemrefDeallocPattern>();
}

} // namespace air
} // namespace xilinx