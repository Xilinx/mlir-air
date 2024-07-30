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

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
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
using namespace xilinx;
using namespace xilinx::air;

#define DEBUG_TYPE "air-dependency-schedule-opt"

namespace {

//===----------------------------------------------------------------------===//
// Utility methods for op hoisting
//===----------------------------------------------------------------------===//

// Return the air.execute op in the op's dep list which contains memref.alloc
// op
air::ExecuteOp getRegionOfAllocOpForOp(Operation *op) {
  air::AsyncOpInterface current_async_op = dyn_cast<air::AsyncOpInterface>(op);
  auto dependency_list = current_async_op.getAsyncDependencies();
  if (dependency_list.size()) {
    for (auto dep_op : dependency_list) {
      if (dep_op.getDefiningOp() &&
          dyn_cast<air::ExecuteOp>(dep_op.getDefiningOp())) {
        // Found air.ExecuteOp in upstream dependency
        auto exec_op = dyn_cast<air::ExecuteOp>(dep_op.getDefiningOp());
        auto child_op = exec_op.getChildOp();
        if (auto alloc_op = dyn_cast<memref::AllocOp>(child_op)) {
          // Found memref.allocOp inside air.ExecuteOp
          return exec_op;
        }
      }
    }
  }
  return nullptr;
}

// Return the air.execute op in the dma's downstream which contains
// memref.dealloc op
air::ExecuteOp getRegionOfDeallocOpForOp(Operation *op) {
  air::AsyncOpInterface current_async_op = dyn_cast<air::AsyncOpInterface>(op);
  auto dependency_token = current_async_op.getAsyncToken();
  for (auto user : dependency_token.getUsers()) {
    if (auto exec_op = dyn_cast<air::ExecuteOp>(user)) {
      // Found air.ExecuteOp in downstream dependency
      auto child_op = exec_op.getChildOp();
      if (auto dealloc_op = dyn_cast<memref::DeallocOp>(child_op)) {
        // Found memref.deallocOp inside air.ExecuteOp
        return exec_op;
      }
    }
  }
  return nullptr;
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

// Check if an operation is invariant with respect to for loop iteration
bool isInvariantWRTForLoop(Operation *op, scf::ForOp for_op) {
  for (auto op_operand : op->getOperands()) {
    if (op_operand == for_op.getInductionVar()) {
      return false;
    }
    if (op_operand.getDefiningOp() &&
        isa<memref::SubViewOp>(op_operand.getDefiningOp())) {
      auto subview_op = dyn_cast<memref::SubViewOp>(op_operand.getDefiningOp());
      for (auto subview_operand : subview_op->getOperands()) {
        if (subview_operand == for_op.getInductionVar()) {
          return false;
        }
      }
    }
  }
  return true;
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
          air::ExecuteOp alloc_exec_op = getRegionOfAllocOpForOp(op_1);
          air::ExecuteOp dealloc_exec_op = getRegionOfDeallocOpForOp(op_2);
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
          auto child_op = exec_op.getChildOp();
          if (auto alloc_op = dyn_cast<memref::AllocOp>(child_op)) {
            // Found memref.allocOp inside air.ExecuteOp
            foundMemrefAllocDep = true;
          }
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
      if (auto exec_op = dyn_cast<air::ExecuteOp>(user)) {
        // Found air.ExecuteOp in downstream dependency
        auto child_op = exec_op.getChildOp();
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

struct HoistAIRChannelInAccumPattern : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp for_op,
                                PatternRewriter &rewriter) const override {

    // Currently only able to analyze scf.for in air.herd.
    if (!isa<air::HerdOp>(for_op->getParentOp()))
      return failure();

    SmallVector<Operation *> dataProducers;
    SmallVector<Operation *> dataConsumers;
    // Note: currently not able to analyze air.channel with broadcast (those are
    // skipped over by ignoring affine.if).
    for (auto get : for_op.getOps<air::ChannelGetOp>())
      dataProducers.push_back(get);
    for (auto exec : for_op.getOps<air::ExecuteOp>()) {
      auto child_op = exec.getChildOp();
      if (isa<linalg::FillOp>(child_op))
        dataProducers.push_back(exec);
    }
    for (auto put : for_op.getOps<air::ChannelPutOp>())
      dataConsumers.push_back(put);

    bool foundPairToHoist = false;
    for (auto op_2 : dataConsumers) {
      bool foundPairForThisOp2 = false;
      for (auto op_1 : dataProducers) {
        // Check if the pair of dmas form symmetry in their src and dst
        bool areSymmetric = areSymmetricDataMovements(op_1, op_2);
        // Check if the pair of dmas are invariant with respect to for loop
        // iterations
        bool areInvariantWRTForLoop = true;
        areInvariantWRTForLoop &= isInvariantWRTForLoop(op_1, for_op);
        areInvariantWRTForLoop &= isInvariantWRTForLoop(op_2, for_op);
        // Check if other air.channel ops sharing the same symbolic channel with
        // op_1 and op_2 can also be hoisted out of a for loop.
        bool allOtherChannelOpsCanHoistTogether = true;
        SmallVector<Operation *> other_chan_ops;
        SmallVector<scf::ForOp> other_for_ops;
        if (auto chan_op_1 = dyn_cast<air::ChannelGetOp>(op_1))
          for (auto o : getTheOtherChannelOpThroughSymbol(chan_op_1))
            other_chan_ops.push_back(o.getOperation());
        if (auto chan_op_2 = dyn_cast<air::ChannelPutOp>(op_2))
          for (auto o : getTheOtherChannelOpThroughSymbol(chan_op_2))
            other_chan_ops.push_back(o.getOperation());
        for (auto other_chan_op : other_chan_ops) {
          auto parent_for_op = other_chan_op->getParentOfType<scf::ForOp>();
          if (!parent_for_op)
            continue;
          if (getStaticScfForTripCountAsInt(parent_for_op) &&
              getStaticScfForTripCountAsInt(for_op)) {
            bool equal_trip_count =
                *getStaticScfForTripCountAsInt(parent_for_op) ==
                *getStaticScfForTripCountAsInt(for_op);
            int data_movement_op_count = 0;
            int linalg_op_count = 0;
            parent_for_op->walk([&](Operation *child_op) {
              if (isa<air::MemcpyInterface>(child_op))
                data_movement_op_count++;
              if (isa<linalg::LinalgOp>(child_op))
                linalg_op_count++;
            });
            bool hasOneAsyncEventInForOp =
                (data_movement_op_count + linalg_op_count) == 1;
            allOtherChannelOpsCanHoistTogether &= equal_trip_count;
            allOtherChannelOpsCanHoistTogether &= hasOneAsyncEventInForOp;
            other_for_ops.push_back(parent_for_op);
          }
        }
        if (areSymmetric & areInvariantWRTForLoop &
            allOtherChannelOpsCanHoistTogether) {
          foundPairToHoist = true;
          foundPairForThisOp2 = true;
          // Found a pair of dmas which cancel out each other
          air::ExecuteOp alloc_exec_op = getRegionOfAllocOpForOp(op_1);
          air::ExecuteOp dealloc_exec_op = getRegionOfDeallocOpForOp(op_2);
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
              dyn_cast<air::AsyncOpInterface>(op_2).getAsyncToken());
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
          // Hoist all other channel ops sharing the same channels out of their
          // parent for loops.
          for (auto for_op : other_for_ops) {
            if (*getConstantIntValue(for_op.getLowerBound()) == 0)
              for_op.getUpperBoundMutable().assign(for_op.getStep());
            else
              for_op.getUpperBoundMutable().assign(
                  rewriter.create<arith::ConstantIndexOp>(
                      for_op->getLoc(),
                      *getConstantIntValue(for_op.getLowerBound()) +
                          *getConstantIntValue(for_op.getStep())));
          }
        }
      }
      if (foundPairForThisOp2)
        continue; // Ensure unique pairing
    }
    if (foundPairToHoist)
      return success();
    return failure();
  }

private:
  // Check if two dma ops are symmetric
  bool areSymmetricChannelOps(air::ChannelGetOp op_1,
                              air::ChannelPutOp op_2) const {
    bool areSymmetric = op_1.getMemref() == op_2.getMemref();
    // Check offsets, sizes and strides
    unsigned op_1_dst_num_entries = op_1.getOffsets().size();
    unsigned op_1_src_num_entries = op_1.getOffsets().size();
    unsigned op_2_dst_num_entries = op_2.getOffsets().size();
    unsigned op_2_src_num_entries = op_2.getOffsets().size();
    if (areSymmetric && (op_1_dst_num_entries == op_2_src_num_entries) &&
        (op_1_src_num_entries == op_2_dst_num_entries)) {
      for (unsigned i = 0; i < op_1_dst_num_entries; i++) {
        areSymmetric &=
            areEqualIndices(op_1.getOffsets()[i], op_2.getOffsets()[i]);
        areSymmetric &= areEqualIndices(op_1.getSizes()[i], op_2.getSizes()[i]);
        areSymmetric &=
            areEqualIndices(op_1.getStrides()[i], op_2.getStrides()[i]);
      }
    } else {
      areSymmetric = false;
    }

    return areSymmetric;
  }

  // Check if a pair of data producer and consumer ops are symmetric
  bool areSymmetricDataMovements(Operation *op_1, Operation *op_2) const {
    if (auto chan_1 = dyn_cast<air::ChannelGetOp>(op_1)) {
      if (auto chan_2 = dyn_cast<air::ChannelPutOp>(op_2)) {
        return areSymmetricChannelOps(chan_1, chan_2);
      }
    }
    Operation *actual_op_1 = op_1;
    if (auto exec = dyn_cast<air::ExecuteOp>(op_1)) {
      actual_op_1 = exec.getChildOp();
    }
    Value op_1_memref = nullptr;
    Value op_2_memref = nullptr;
    if (auto linalg_op = dyn_cast<linalg::LinalgOp>(actual_op_1)) {
      if (linalg_op.getDpsInits().size() == 1) {
        op_1_memref = linalg_op.getDpsInits()[0];
      }
    } else if (auto get = dyn_cast<air::ChannelGetOp>(op_1)) {
      op_1_memref = get.getMemref();
    } else
      return false; // Unsupported data producer op.
    if (auto put = dyn_cast<air::ChannelPutOp>(op_2)) {
      op_2_memref = put.getMemref();
    } else
      return false; // Unsupported data consumer op.
    return op_1_memref == op_2_memref;
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
        auto child_op = exec_op.getChildOp();
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
      if (llvm::isa<air::AsyncTokenType>(operand.getType())) {
        yielded_tokens.push_back(operand);
      }
    }
    SmallVector<Operation *> back_candidates;
    for (auto token : yielded_tokens) {
      auto back_candidate = token.getDefiningOp();
      if (auto exec_op = dyn_cast<air::ExecuteOp>(back_candidate)) {
        auto child_op = exec_op.getChildOp();
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
    Operation *dealloc_op = nullptr;
    auto alloc_exec_memref = alloc_exec->getResults()[1];
    if (!llvm::isa<MemRefType>(alloc_exec_memref.getType()))
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

    for (auto ia : for_op.getInitArgs()) {
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

struct HoistAIRHerdInForPattern : public OpRewritePattern<air::HerdOp> {
  using OpRewritePattern<air::HerdOp>::OpRewritePattern;

  HoistAIRHerdInForPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(air::HerdOp herdOp,
                                PatternRewriter &rewriter) const override {
    auto loc = herdOp->getLoc();
    auto ctx = herdOp->getContext();
    auto hasNElements = [](Block *block, unsigned N) {
      auto op_ptr = block->begin();
      for (unsigned i = 0; i < N; i++)
        op_ptr = std::next(op_ptr);
      return op_ptr != block->end() && &*op_ptr == &block->back();
    };
    SmallVector<scf::ForOp> for_loop_nest;
    Operation *parent = herdOp->getParentOp();
    while (parent && !isa<air::SegmentOp>(parent)) {
      if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
        if (hasNElements(forOp.getBody(), 2))
          for_loop_nest.push_back(forOp);
        else
          return failure(); // Herd is not in perfectly nested loop.
      }
      parent = parent->getParentOp();
    }

    if (for_loop_nest.empty())
      return failure();

    llvm::reverse(for_loop_nest);

    scf::ForOp outerMostLoop = for_loop_nest.front();

    // Create new herd op as parent to the scf.for loop nest.
    rewriter.setInsertionPoint(outerMostLoop);
    auto herdOperands = herdOp.getOperands().drop_front(
        herdOp.getAsyncDependencies().size() + herdOp.getNumDims());
    auto forRegionIterOperands = outerMostLoop.getOperands().drop_front(
        outerMostLoop.getNumControlOperands());
    auto newHerdOp = rewriter.create<air::HerdOp>(
        loc, forRegionIterOperands, herdOp.getSizeOperands(), herdOperands,
        true, herdOp->getAttrs());
    outerMostLoop->moveBefore(newHerdOp.getBody().front().getTerminator());
    OpBuilder builder(outerMostLoop);

    // Replace uses of tokens and consts in for loop nest.
    for (auto val : forRegionIterOperands) {
      auto newAsyncToken =
          builder
              .create<air::WaitAllOp>(loc, air::AsyncTokenType::get(ctx),
                                      SmallVector<Value>{})
              .getAsyncToken();
      replaceAllUsesInRegionWith(val, newAsyncToken, newHerdOp.getBody());
    }
    for (auto loop : for_loop_nest) {
      if (auto definingOp = loop.getUpperBound().getDefiningOp())
        replaceAllUsesInRegionWith(loop.getUpperBound(),
                                   builder.clone(*definingOp)->getResult(0),
                                   newHerdOp.getBody());
      if (auto definingOp = loop.getLowerBound().getDefiningOp())
        replaceAllUsesInRegionWith(loop.getLowerBound(),
                                   builder.clone(*definingOp)->getResult(0),
                                   newHerdOp.getBody());
      if (auto definingOp = loop.getStep().getDefiningOp())
        replaceAllUsesInRegionWith(loop.getStep(),
                                   builder.clone(*definingOp)->getResult(0),
                                   newHerdOp.getBody());
    }
    for (auto res : outerMostLoop->getResults())
      res.replaceAllUsesWith(newHerdOp.getAsyncToken());

    // Splice herd block into inner-most for loop.
    scf::ForOp innerMostLoop = for_loop_nest.back();
    auto &bb = innerMostLoop.getBody()->getOperations();
    auto &body = herdOp.getBody().front().getOperations();
    bb.splice(bb.begin(), body, body.begin(), --body.end());

    rewriter.setInsertionPoint(herdOp);
    for (auto res : herdOp->getResults())
      res.replaceAllUsesWith(
          rewriter
              .create<air::WaitAllOp>(loc, air::AsyncTokenType::get(ctx),
                                      SmallVector<Value>{})
              .getAsyncToken());
    for (unsigned i = 0; i < herdOp.getNumKernelOperands(); i++)
      herdOp.getKernelArgument(i).replaceAllUsesWith(
          newHerdOp.getKernelArgument(i));
    for (unsigned i = 0; i < herdOp.getNumDims(); i++) {
      replaceAllUsesInRegionWith(herdOp.getIds()[i], newHerdOp.getIds()[i],
                                 newHerdOp.getBody());
      replaceAllUsesInRegionWith(herdOp.getSize()[i], newHerdOp.getSize()[i],
                                 newHerdOp.getBody());
    }

    rewriter.eraseOp(herdOp);
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
        (void)dealloc;
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
      rewriter.create<scf::YieldOp>(yield_op.getLoc(),
                                    SmallVector<Value>{wa.getAsyncToken()});
      rewriter.eraseOp(yield_op.getOperation());
    }

    return success();
  }

private:
  std::vector<Operation *> adjacent_events(Operation *event) const {
    SmallVector<Value, 1> returned_tokens = {};
    for (Value result : event->getResults()) {
      if (llvm::isa<air::AsyncTokenType>(result.getType())) {
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

    for (auto lb : par.getLowerBound()) {
      auto constLB = getConstantIntValue(lb);
      assert(constLB && "non-static scf.parallel lb, NYI");
      assert(*constLB == 0 && "non-zero scf.parallel lb, NYI");
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
      // Create arith.const ops per position
      SmallVector<Value> positionVals;
      for (unsigned i = 0; i < position.size(); i++) {
        positionVals.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, position[i]));
        remap.map(par.getInductionVars()[i], positionVals[i]);
      }

      // Splice
      for (auto &op : par.getBody()->getOperations()) {
        if (op.mightHaveTrait<OpTrait::IsTerminator>()) {
          if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
            SmallVector<Value> tokens;
            for (auto yieldOper : yieldOp->getOperands())
              if (isa<air::AsyncTokenType>(yieldOper.getType()))
                tokens.push_back(yieldOper);
            auto newWaitAll = rewriter.create<air::WaitAllOp>(
                loc, air::AsyncTokenType::get(rewriter.getContext()), tokens);
            yieldedTokens.push_back(newWaitAll.getAsyncToken());
          }
          continue;
        }
        rewriter.clone(op, remap);
      }
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

struct CanonicalizeAIRExecute : public OpRewritePattern<air::ExecuteOp> {
  using OpRewritePattern<air::ExecuteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(air::ExecuteOp exec,
                                PatternRewriter &rewriter) const override {

    auto childOp = exec.getChildOp();
    assert(childOp && "air.execute op has no child op");
    // Canonicalize air.execute with empty region.
    if (!childOp->mightHaveTrait<OpTrait::IsTerminator>())
      return failure();
    exec.getAsyncToken().replaceAllUsesWith(
        rewriter
            .create<air::WaitAllOp>(
                exec->getLoc(), air::AsyncTokenType::get(rewriter.getContext()),
                exec.getAsyncDependencies())
            .getAsyncToken());

    if (childOp->getNumOperands() != 1)
      return failure();
    assert(childOp->getNumOperands() == 1 &&
           "air.execute_terminator doesn't have exactly one operand, NYI");
    exec.getResult(1).replaceAllUsesWith(childOp->getOperand(0));

    rewriter.eraseOp(exec);
    return success();
  }

private:
};

affine::AffineForOp updateAffineForBounds(OpBuilder builder, IRMapping &remap,
                                          affine::AffineForOp loop_op, int lb,
                                          int ub, int step) {
  affine::AffineForOp new_loop_op = builder.create<affine::AffineForOp>(
      builder.getUnknownLoc(), lb, ub, step);
  remap.map(loop_op.getInductionVar(), new_loop_op.getInductionVar());
  // remap.map(old_apply.getResult(), new_loop_op.getInductionVar());
  auto insertionCheckpoint = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(new_loop_op.getBody());
  for (Operation &child_op : loop_op.getBody()->getOperations()) {
    if (&child_op == loop_op.getBody()->getTerminator()) { /*Skip*/
    } else
      builder.clone(child_op, remap);
  }
  builder.restoreInsertionPoint(insertionCheckpoint);
  return new_loop_op;
}

scf::ForOp updateScfForBounds(OpBuilder builder, IRMapping &remap,
                              scf::ForOp loop_op, int lb, int ub, int step) {
  auto loc = loop_op->getLoc();
  SmallVector<Value, 1> deps =
      loop_op.getOperands().drop_front(loop_op.getNumControlOperands());
  scf::ForOp new_loop_op = builder.create<scf::ForOp>(
      builder.getUnknownLoc(), builder.create<arith::ConstantIndexOp>(loc, lb),
      builder.create<arith::ConstantIndexOp>(loc, ub),
      builder.create<arith::ConstantIndexOp>(loc, step), deps);
  remap.map(loop_op.getInductionVar(), new_loop_op.getInductionVar());
  for (unsigned i = 0; i < loop_op.getRegionIterArgs().size(); i++)
    remap.map(loop_op.getRegionIterArgs()[i],
              new_loop_op.getRegionIterArgs()[i]);
  auto insertionCheckpoint = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(new_loop_op.getBody());
  for (Operation &child_op : loop_op.getBody()->getOperations()) {
    if (&child_op == loop_op.getBody()->getTerminator()) {
      if (!new_loop_op.getBody()->mightHaveTerminator())
        builder.clone(child_op, remap);
    } else
      builder.clone(child_op, remap);
  }
  for (unsigned i = 0; i < loop_op->getNumResults(); i++)
    loop_op->getResult(i).replaceAllUsesWith(new_loop_op->getResult(i));
  builder.restoreInsertionPoint(insertionCheckpoint);
  return new_loop_op;
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
      assert(new_ub && new_lb);
      int newStepInInt = llvm::divideCeilSigned(*new_ub - *new_lb, tripCount);
      IRMapping remap;
      if (auto exec = dyn_cast<air::ExecuteOp>(apply->getParentOp())) {
        rewriter.setInsertionPoint(exec);
        exec.getResult(1).replaceAllUsesWith(sfo.getInductionVar());
        if (sfo.getNumRegionIterArgs())
          exec.getAsyncToken().replaceAllUsesWith(sfo.getRegionIterArgs()[0]);
        else if (exec.getAsyncDependencies().size() == 1)
          exec.getAsyncToken().replaceAllUsesWith(
              exec.getAsyncDependencies()[0]);
        else {
          exec->emitOpError("failed to reconstruct dependency after its erase");
          return failure();
        }
        rewriter.eraseOp(exec);
      } else {
        rewriter.setInsertionPoint(apply);
        apply.getResult().replaceAllUsesWith(sfo.getInductionVar());
        rewriter.eraseOp(apply);
      }
      rewriter.setInsertionPoint(sfo);
      updateScfForBounds(rewriter, remap, sfo, *new_lb, *new_ub, newStepInInt);
      rewriter.eraseOp(sfo);
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
      assert(new_ub && new_lb);
      int newStepInInt = llvm::divideCeilSigned(*new_ub - *new_lb, tripCount);
      IRMapping remap;
      rewriter.setInsertionPoint(afo);
      apply.getResult().replaceAllUsesWith(afo.getInductionVar());
      rewriter.eraseOp(apply);
      updateAffineForBounds(rewriter, remap, afo, *new_lb, *new_ub,
                            newStepInInt);
      rewriter.eraseOp(afo);
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
      IRMapping remap;
      if (auto exec = dyn_cast<air::ExecuteOp>(op->getParentOp())) {
        rewriter.setInsertionPoint(exec);
        exec.getResult(1).replaceAllUsesWith(sfo.getInductionVar());
        if (sfo.getNumRegionIterArgs())
          exec.getAsyncToken().replaceAllUsesWith(sfo.getRegionIterArgs()[0]);
        else if (exec.getAsyncDependencies().size() == 1)
          exec.getAsyncToken().replaceAllUsesWith(
              exec.getAsyncDependencies()[0]);
        else {
          exec->emitOpError("failed to reconstruct dependency after its erase");
          return failure();
        }
        rewriter.eraseOp(exec);
      } else {
        rewriter.setInsertionPoint(op);
        op.getResult().replaceAllUsesWith(sfo.getInductionVar());
        rewriter.eraseOp(op);
      }
      rewriter.setInsertionPoint(sfo);
      updateScfForBounds(rewriter, remap, sfo, new_lb, new_ub, newStepInInt);
      rewriter.eraseOp(sfo);
    } else if (auto afo = dyn_cast<affine::AffineForOp>(containingOp)) {
      if (!afo.hasConstantBounds())
        return failure();
      int tripCount = *getStaticAffineForTripCountAsInt(afo);
      int new_ub = afo.getConstantUpperBound() * muli_factor;
      int new_lb = afo.getConstantLowerBound() * muli_factor;
      int newStepInInt = llvm::divideCeilSigned(new_ub - new_lb, tripCount);
      IRMapping remap;
      rewriter.setInsertionPoint(afo);
      op.getResult().replaceAllUsesWith(afo.getInductionVar());
      rewriter.eraseOp(op);
      updateAffineForBounds(rewriter, remap, afo, new_lb, new_ub, newStepInInt);
      rewriter.eraseOp(afo);
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
      IRMapping remap;
      if (auto exec = dyn_cast<air::ExecuteOp>(op->getParentOp())) {
        rewriter.setInsertionPoint(exec);
        exec.getResult(1).replaceAllUsesWith(sfo.getInductionVar());
        if (sfo.getNumRegionIterArgs())
          exec.getAsyncToken().replaceAllUsesWith(sfo.getRegionIterArgs()[0]);
        else if (exec.getAsyncDependencies().size() == 1)
          exec.getAsyncToken().replaceAllUsesWith(
              exec.getAsyncDependencies()[0]);
        else {
          exec->emitOpError("failed to reconstruct dependency after its erase");
          return failure();
        }
        rewriter.eraseOp(exec);
      } else {
        rewriter.setInsertionPoint(op);
        op.getResult().replaceAllUsesWith(sfo.getInductionVar());
        rewriter.eraseOp(op);
      }
      rewriter.setInsertionPoint(sfo);
      updateScfForBounds(rewriter, remap, sfo, new_lb, new_ub, newStepInInt);
      rewriter.eraseOp(sfo);
    } else if (auto afo = dyn_cast<affine::AffineForOp>(containingOp)) {
      if (!afo.hasConstantBounds())
        return failure();
      int tripCount = *getStaticAffineForTripCountAsInt(afo);
      int new_ub = afo.getConstantUpperBound() + addi_operand;
      int new_lb = afo.getConstantLowerBound() + addi_operand;
      int newStepInInt = llvm::divideCeilSigned(new_ub - new_lb, tripCount);
      IRMapping remap;
      rewriter.setInsertionPoint(afo);
      op.getResult().replaceAllUsesWith(afo.getInductionVar());
      rewriter.eraseOp(op);
      updateAffineForBounds(rewriter, remap, afo, new_lb, new_ub, newStepInInt);
      rewriter.eraseOp(afo);
    } else
      return failure();

    return success();
  }

private:
};

struct AIRSpecializeChannelWrapAndStrideInScfFor
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp for_op,
                                PatternRewriter &rewriter) const override {
    auto loc = for_op->getLoc();
    auto ctx = for_op->getContext();

    // Check if the loop is the outermost loop in a perfect loop nest
    auto hasNElements = [](Block *block, unsigned N) {
      unsigned counter = 0;
      for (auto &o : block->getOperations()) {
        if (isa<air::ChannelInterface>(o))
          counter++;
        else if (isa<LoopLikeOpInterface>(o))
          counter++;
        else if (isa<mlir::linalg::LinalgOp>(o))
          counter++;
      }
      return counter == N;
    };
    if (auto parent_for = dyn_cast<scf::ForOp>(for_op->getParentOp()))
      if (hasNElements(parent_for.getBody(), 1))
        return failure();

    if (!hasNElements(for_op.getBody(), 1))
      return failure();

    // Check if the loop nest contains exactly one channel op
    SmallVector<air::ChannelInterface> channel_ops;
    for_op.getBody()->walk(
        [&](air::ChannelInterface putget) { channel_ops.push_back(putget); });
    if (channel_ops.size() != 1)
      return failure();

    // Fold for loops int channel op's wrap and stride fields
    SmallVector<scf::ForOp> for_loops;
    Operation *parent = channel_ops[0].getOperation();
    while (parent != for_op.getOperation()) {
      parent = parent->getParentOp();
      if (auto for_op_in_nest = dyn_cast<scf::ForOp>(parent))
        for_loops.push_back(for_op_in_nest);
    }
    SmallVector<Value> offsets = channel_ops[0].getOffsets();
    SmallVector<Value> wraps = channel_ops[0].getSizes();
    SmallVector<Value> strides = channel_ops[0].getStrides();
    for (auto o : for_loops) {
      // Check for perfect loop nest containing only air.channel ops
      if (!hasNElements(o.getBody(), 1))
        return failure();
      if (!getStaticScfForTripCountAsInt(o))
        return failure();
    }

    (void)canonicalizeWrapAndStrideList(
        rewriter, offsets, wraps, strides,
        air::getTensorVolume(channel_ops[0].getMemref().getType()));

    // If empty offsets/sizes/strides, then populate the lists with default
    // values.
    if (offsets.empty() && wraps.empty() && strides.empty())
      populateDefaultWrapsAndStrides(rewriter, channel_ops[0].getMemref(),
                                     offsets, wraps, strides);

    auto res = foldForLoopNestAsExtendedSizesAndStrides(
        rewriter, for_op.getOperation(), channel_ops[0].getOperation(), offsets,
        wraps, strides, channel_ops[0].getMemref());
    if (res.failed())
      return failure();

    (void)canonicalizeWrapAndStrideList(
        rewriter, offsets, wraps, strides,
        air::getTensorVolume(channel_ops[0].getMemref().getType()));

    Operation *new_chan_op = nullptr;
    SmallVector<Type, 1> tys;
    if (isAsyncOp(channel_ops[0].getOperation())) {
      tys.push_back(air::AsyncTokenType::get(ctx));
    }
    SmallVector<Value, 1> deps =
        for_op.getOperands().drop_front(for_op.getNumControlOperands());
    if (isa<air::ChannelPutOp>(channel_ops[0]))
      new_chan_op = rewriter.create<air::ChannelPutOp>(
          loc, tys, deps, channel_ops[0].getChanName(),
          channel_ops[0].getIndices(), channel_ops[0].getMemref(), offsets,
          wraps, strides);
    else if (isa<air::ChannelGetOp>(channel_ops[0]))
      new_chan_op = rewriter.create<air::ChannelGetOp>(
          loc, tys, deps, channel_ops[0].getChanName(),
          channel_ops[0].getIndices(), channel_ops[0].getMemref(), offsets,
          wraps, strides);

    for (auto res : for_op.getResults()) {
      if (isa<air::AsyncTokenType>(res.getType())) {
        res.replaceAllUsesWith(
            dyn_cast<air::AsyncOpInterface>(new_chan_op).getAsyncToken());
      }
    }
    rewriter.eraseOp(for_op.getOperation());

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
    auto hasNElements = [](Block *block, unsigned N) {
      unsigned counter = 0;
      for (auto &o : block->getOperations()) {
        if (isa<air::ChannelInterface>(o))
          counter++;
        else if (isa<LoopLikeOpInterface>(o))
          counter++;
        else if (isa<mlir::linalg::LinalgOp>(o))
          counter++;
      }
      return counter == N;
    };
    if (auto parent_for = dyn_cast<affine::AffineForOp>(for_op->getParentOp()))
      if (hasNElements(parent_for.getBody(), 1))
        return failure();

    if (!hasNElements(for_op.getBody(), 1))
      return failure();

    // Check if the loop nest contains exactly one channel op
    SmallVector<air::ChannelInterface> channel_ops;
    for_op.getBody()->walk(
        [&](air::ChannelInterface putget) { channel_ops.push_back(putget); });
    if (channel_ops.size() != 1)
      return failure();

    // Fold for loops int channel op's wrap and stride fields
    SmallVector<affine::AffineForOp> for_loops;
    Operation *parent = channel_ops[0].getOperation();
    while (parent != for_op.getOperation()) {
      parent = parent->getParentOp();
      if (auto for_op_in_nest = dyn_cast<affine::AffineForOp>(parent))
        for_loops.push_back(for_op_in_nest);
    }
    SmallVector<Value> offsets = channel_ops[0].getOffsets();
    SmallVector<Value> wraps = channel_ops[0].getSizes();
    SmallVector<Value> strides = channel_ops[0].getStrides();
    for (auto o : for_loops) {
      // Check for perfect loop nest containing only air.channel ops
      if (!hasNElements(o.getBody(), 1))
        return failure();
      if (!getStaticAffineForTripCountAsInt(o))
        return failure();
    }

    (void)canonicalizeWrapAndStrideList(
        rewriter, offsets, wraps, strides,
        air::getTensorVolume(channel_ops[0].getMemref().getType()));

    auto res = foldForLoopNestAsExtendedSizesAndStrides(
        rewriter, for_op.getOperation(), channel_ops[0].getOperation(), offsets,
        wraps, strides, channel_ops[0].getMemref());
    if (res.failed())
      return failure();

    (void)canonicalizeWrapAndStrideList(
        rewriter, offsets, wraps, strides,
        air::getTensorVolume(channel_ops[0].getMemref().getType()));

    Operation *new_chan_op = nullptr;
    SmallVector<Type, 1> tys;
    if (isAsyncOp(channel_ops[0].getOperation())) {
      tys.push_back(air::AsyncTokenType::get(ctx));
    }
    SmallVector<Value, 1> deps =
        for_op.getOperands().drop_front(for_op.getNumControlOperands());
    if (isa<air::ChannelPutOp>(channel_ops[0]))
      new_chan_op = rewriter.create<air::ChannelPutOp>(
          loc, tys, deps, channel_ops[0].getChanName(),
          channel_ops[0].getIndices(), channel_ops[0].getMemref(), offsets,
          wraps, strides);
    else if (isa<air::ChannelGetOp>(channel_ops[0]))
      new_chan_op = rewriter.create<air::ChannelGetOp>(
          loc, tys, deps, channel_ops[0].getChanName(),
          channel_ops[0].getIndices(), channel_ops[0].getMemref(), offsets,
          wraps, strides);

    for (auto res : for_op.getResults()) {
      if (isa<air::AsyncTokenType>(res.getType())) {
        res.replaceAllUsesWith(
            dyn_cast<air::AsyncOpInterface>(new_chan_op).getAsyncToken());
      }
    }
    rewriter.eraseOp(for_op.getOperation());

    return success();
  }

private:
};

struct AIRCanonicalizeChannelPutOpWrapAndStrideList
    : public OpRewritePattern<air::ChannelPutOp> {
  using OpRewritePattern<air::ChannelPutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(air::ChannelPutOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 1> deps;
    SmallVector<Type, 1> tys;
    if (isAsyncOp(op)) {
      tys.push_back(air::AsyncTokenType::get(op->getContext()));
      deps = op.getAsyncDependencies();
    }

    SmallVector<Value> offsets = op.getOffsets();
    SmallVector<Value> sizes = op.getSizes();
    SmallVector<Value> strides = op.getStrides();

    if (failed(canonicalizeWrapAndStrideList(
            rewriter, offsets, sizes, strides,
            air::getTensorVolume(op.getMemref().getType()))))
      return failure();

    auto new_op = rewriter.create<air::ChannelPutOp>(
        op->getLoc(), tys, deps, op.getChanName(), op.getIndices(),
        op.getMemref(), offsets, sizes, strides);
    for (unsigned i = 0; i < op->getResults().size(); i++)
      op->getResults()[i].replaceAllUsesWith(new_op->getResults()[i]);

    rewriter.eraseOp(op);

    return success();
  }

private:
};

struct AIRCanonicalizeChannelGetOpWrapAndStrideList
    : public OpRewritePattern<air::ChannelGetOp> {
  using OpRewritePattern<air::ChannelGetOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(air::ChannelGetOp op,
                                PatternRewriter &rewriter) const override {

    SmallVector<Value, 1> deps;
    SmallVector<Type, 1> tys;
    if (isAsyncOp(op)) {
      tys.push_back(air::AsyncTokenType::get(op->getContext()));
      deps = op.getAsyncDependencies();
    }

    SmallVector<Value> offsets = op.getOffsets();
    SmallVector<Value> sizes = op.getSizes();
    SmallVector<Value> strides = op.getStrides();

    if (failed(canonicalizeWrapAndStrideList(
            rewriter, offsets, sizes, strides,
            air::getTensorVolume(op.getMemref().getType()))))
      return failure();

    auto new_op = rewriter.create<air::ChannelGetOp>(
        op->getLoc(), tys, deps, op.getChanName(), op.getIndices(),
        op.getMemref(), offsets, sizes, strides);
    for (unsigned i = 0; i < op->getResults().size(); i++)
      op->getResults()[i].replaceAllUsesWith(new_op->getResults()[i]);

    rewriter.eraseOp(op);

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
      auto memTy = llvm::cast<MemRefType>(op.getMemref().getType());
      for (auto d : getTensorShape(memTy)) {
        new_sizes.push_back(
            builder.create<arith::ConstantIndexOp>(par->getLoc(), d));
      }
    }
    auto size_op = new_sizes[dim].getDefiningOp();
    if (size_op && isa<arith::ConstantIndexOp>(size_op)) {
      auto val = dyn_cast<arith::ConstantIndexOp>(size_op).value();
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
        src_memspace = llvm::cast<MemRefType>(dma_op.getSrcMemref().getType())
                           .getMemorySpaceAsInt();
      if (dma_op.getDstMemref())
        dst_memspace = llvm::cast<MemRefType>(dma_op.getDstMemref().getType())
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
      // If not variant wrt herd, then check for fixed row-wise or col-wise
      // offset.
      int src_memspace = llvm::cast<MemRefType>(dma_op.getSrcMemref().getType())
                             .getMemorySpaceAsInt();
      auto externalOffsets = src_memspace == (int)air::MemorySpace::L1
                                 ? dma_op.getDstOffsets()
                                 : dma_op.getSrcOffsets();
      if (!hl_op && externalOffsets.size() ==
                        dma_op->getParentOfType<air::HerdOp>().getNumDims()) {
        hl_op = dma_op->getParentOfType<air::HerdOp>();
        if (getConstantIntValue(externalOffsets[0]))
          isVariantWrtHerdRows = true;
        if (getConstantIntValue(externalOffsets[1]))
          isVariantWrtHerdCols = true;
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
    : public xilinx::air::impl::AIRHoistDmaInAccumPatternBase<
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
    : public xilinx::air::impl::AIRHoistMemallocInForPatternBase<
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
        (void)loopUnrollByFactor(for_op, unroll_factor, annotateFn);
      }
    });
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

class AIRSpecializeChannelWrapAndStridePattern
    : public xilinx::air::impl::AIRSpecializeChannelWrapAndStridePatternBase<
          AIRSpecializeChannelWrapAndStridePattern> {

public:
  AIRSpecializeChannelWrapAndStridePattern() = default;
  AIRSpecializeChannelWrapAndStridePattern(
      const AIRSpecializeChannelWrapAndStridePattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet preproc_patterns(&getContext());
    preproc_patterns.insert<UnrollScfParallel, CanonicalizeAIRExecute,
                            CanonicalizeAffineApplyOnLoopInductionVar,
                            CanonicalizeArithMuliOpOnLoopInductionVar,
                            CanonicalizeArithAddiOpOnLoopInductionVar>(ctx);
    // Canonicalize constant operands in affine.apply.
    mlir::affine::AffineApplyOp::getCanonicalizationPatterns(preproc_patterns,
                                                             ctx);
    air::WaitAllOp::getCanonicalizationPatterns(preproc_patterns, ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(preproc_patterns));

    RewritePatternSet patterns(&getContext());
    patterns.insert<AIRSpecializeChannelWrapAndStrideInScfFor,
                    AIRSpecializeChannelWrapAndStrideInAffineFor>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));

    // Canonicalize wrap and stride list to remove redundant dimensions
    RewritePatternSet cano_patterns(&getContext());
    cano_patterns.insert<AIRCanonicalizeChannelGetOpWrapAndStrideList,
                         AIRCanonicalizeChannelPutOpWrapAndStrideList>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(cano_patterns));
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

class AIRDependencyScheduleOpt
    : public xilinx::air::impl::AIRDependencyScheduleOptBase<
          AIRDependencyScheduleOpt> {

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
    : public xilinx::air::impl::AIREnforceLoopCarriedMemrefDeallocPatternBase<
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

// A pass which de-alias a memref with multiple channel accesses over time, into
// multiple memrefs. Note that this implementation is temporary and not generic.
// TODO: Rewrite as a graph partitioning problem.
class AIRDeAliasMemref
    : public xilinx::air::impl::AIRDeAliasMemrefBase<AIRDeAliasMemref> {

public:
  AIRDeAliasMemref() = default;
  AIRDeAliasMemref(const AIRDeAliasMemref &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect>();
  }

  void runOnFunction(func::FuncOp f) {

    std::vector<memref::AllocOp> allocs;
    f.walk([&](memref::AllocOp alloc) { allocs.push_back(alloc); });

    // Count air.channel references
    for (auto alloc : allocs) {
      Value memref = nullptr;
      if (auto exec = alloc->getParentOfType<air::ExecuteOp>()) {
        memref = exec->getResult(1);
      } else
        memref = alloc.getMemref();
      std::vector<air::ChannelInterface> chan_puts_gets;
      for (auto user : memref.getUsers()) {
        if (auto putget = dyn_cast<air::ChannelInterface>(user))
          if (putget.getMemref() == memref)
            chan_puts_gets.push_back(putget);
      }

      // Partition the subgraph
      std::vector<int> partition_cuts;
      if (!chan_puts_gets.empty()) {
        for (unsigned i = 0; i < chan_puts_gets.size() - 1; i++) {
          if (isa<air::ChannelGetOp>(chan_puts_gets[i].getOperation()) &&
              isa<air::ChannelPutOp>(chan_puts_gets[i + 1].getOperation())) {
            partition_cuts.push_back(i + 1);
          }
        }
      }

      // Allocate new memref per cut
      std::vector<Operation *> new_memallocs;
      for (unsigned i = 0; i < partition_cuts.size(); i++) {
        OpBuilder builder(alloc);
        Operation *new_op = nullptr;
        if (auto exec = alloc->getParentOfType<air::ExecuteOp>()) {
          builder.setInsertionPoint(exec);
          new_op = builder.clone(*exec.getOperation());
        } else
          new_op = builder.clone(*alloc.getOperation());
        new_memallocs.push_back(new_op);

        // Create deallocs for the new memref
        Value new_memref = isa<air::ExecuteOp>(new_op) ? new_op->getResult(1)
                                                       : new_op->getResult(0);
        for (auto user : memref.getUsers()) {
          if (isa<memref::DeallocOp>(user)) {
            if (isa<air::ExecuteOp>(new_op)) {
              builder.setInsertionPoint(
                  user->getParentOfType<air::ExecuteOp>());
              // Async. dealloc
              auto async_exec = builder.create<xilinx::air::ExecuteOp>(
                  user->getLoc(), air::AsyncTokenType::get(alloc->getContext()),
                  SmallVector<Value>{});
              Block *async_exec_bb = builder.createBlock(&async_exec.getBody());
              builder.setInsertionPointToStart(async_exec_bb);
              builder.create<memref::DeallocOp>(user->getLoc(), new_memref);
              builder.create<air::ExecuteTerminatorOp>(user->getLoc());
            } else {
              builder.setInsertionPoint(user);
              // Sync. dealloc
              builder.create<memref::DeallocOp>(user->getLoc(), new_memref);
            }
          }
        }
      }

      // Update references
      partition_cuts.insert(partition_cuts.end(), chan_puts_gets.size());
      for (unsigned i = 0; i < partition_cuts.size() - 1; i++) {
        for (int j = partition_cuts[i]; j < partition_cuts[i + 1]; j++) {
          if (auto old_put = dyn_cast<air::ChannelPutOp>(
                  chan_puts_gets[j].getOperation())) {
            Value new_memref = isa<air::ExecuteOp>(new_memallocs[i])
                                   ? new_memallocs[i]->getResult(1)
                                   : new_memallocs[i]->getResult(0);
            OpBuilder builder(old_put);
            replaceChannelPutOp(builder, old_put, new_memref);
          } else if (auto old_get = dyn_cast<air::ChannelGetOp>(
                         chan_puts_gets[j].getOperation())) {
            Value new_memref = isa<air::ExecuteOp>(new_memallocs[i])
                                   ? new_memallocs[i]->getResult(1)
                                   : new_memallocs[i]->getResult(0);
            OpBuilder builder(old_get);
            replaceChannelGetOp(builder, old_get, new_memref);
          }
        }
      }
    }
  }

  void runOnOperation() override {
    auto module = getOperation();

    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps) {
      runOnFunction(f);
    }
  }

private:
  Operation *replaceChannelPutOp(OpBuilder builder, air::ChannelPutOp old,
                                 Value new_memref) {
    builder.setInsertionPoint(old);
    SmallVector<Type, 1> tys;
    if (old.getAsyncToken()) {
      tys.push_back(air::AsyncTokenType::get(old->getContext()));
    }
    SmallVector<Value, 4> deps = old.getAsyncDependencies();
    auto new_op = builder.create<air::ChannelPutOp>(
        old->getLoc(), tys, deps, old.getChanName(), old.getIndices(),
        new_memref, old.getSrcOffsets(), old.getSrcSizes(),
        old.getSrcStrides());
    if (old.getAsyncToken()) {
      old.getAsyncToken().replaceAllUsesWith(new_op.getAsyncToken());
      // Add dependence to the new memref
      new_op.addAsyncDependency(
          dyn_cast<air::ExecuteOp>(new_memref.getDefiningOp()).getAsyncToken());
    }
    if (old.getId() != -1) {
      new_op->setAttr("id", mlir::IntegerAttr::get(
                                mlir::IntegerType::get(old->getContext(), 32),
                                old.getId()));
    }
    old->erase();
    return new_op.getOperation();
  }
  Operation *replaceChannelGetOp(OpBuilder builder, air::ChannelGetOp old,
                                 Value new_memref) {
    builder.setInsertionPoint(old);
    SmallVector<Type, 1> tys;
    if (old.getAsyncToken()) {
      tys.push_back(air::AsyncTokenType::get(old->getContext()));
    }
    SmallVector<Value, 4> deps = old.getAsyncDependencies();
    auto new_op = builder.create<air::ChannelGetOp>(
        old->getLoc(), tys, deps, old.getChanName(), old.getIndices(),
        new_memref, old.getDstOffsets(), old.getDstSizes(),
        old.getDstStrides());
    if (old.getAsyncToken()) {
      old.getAsyncToken().replaceAllUsesWith(new_op.getAsyncToken());
      // Add dependence to the new memref
      new_op.addAsyncDependency(
          dyn_cast<air::ExecuteOp>(new_memref.getDefiningOp()).getAsyncToken());
    }
    if (old.getId() != -1) {
      new_op->setAttr("id", mlir::IntegerAttr::get(
                                mlir::IntegerType::get(old->getContext(), 32),
                                old.getId()));
    }
    old->erase();
    return new_op.getOperation();
  }
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
    for (unsigned i = 0; i < channelOps.size() - 1; i++) {
      for (unsigned j = i + 1; j < channelOps.size(); j++) {
        std::tuple<bool, std::string> checkScfForMergeableRes =
            checkIfTemporalMergeable(channelOps[i], channelOps[j]);
        if (!std::get<0>(checkScfForMergeableRes))
          continue;
        air::ChannelOp chanA = channelOps[i];
        air::ChannelOp chanB = channelOps[j];
        if (std::get<1>(checkScfForMergeableRes) == "LB" ||
            std::get<1>(checkScfForMergeableRes) == "UB") {
          // Fuse air.channels by scf.for loop unpeeling, i.e. recovering any
          // missing zeroth/last iterations in scf.for loops.
          sortChannelsByLoopNests(chanA, chanB);
          mergeChannelOpsTemporally(chanA, chanB,
                                    std::get<1>(checkScfForMergeableRes));
          chan_merge_map[chanB] = chanA;
        } else if (std::get<1>(checkScfForMergeableRes) == "NFL") {
          // Fuse air.channels temporally, if there isn't any for loop to fuse
          // into.
          createDummyForOpsAroundOps<air::ChannelPutOp>(
              getChannelPutsFusableByFor(chanA, chanB));
          createDummyForOpsAroundOps<air::ChannelGetOp>(
              getChannelGetsFusableByFor(chanA, chanB));
          mergeChannelOpsTemporally(chanA, chanB, "UB");
          // Fuse both channel ops into the loop
          chan_merge_map[chanB] = chanA;
        }
      }
    }
    renameSymbols(channelOps, chan_merge_map);
    if (!targetMemorySpaces.empty()) {
      for (unsigned i = 0; i < channelOps.size() - 1; i++) {
        for (unsigned j = i + 1; j < channelOps.size(); j++) {
          if (!checkIfMergeable(channelOps[i], channelOps[j]))
            continue;
          // Aggressively fuse air.channels by time multiplexing.
          mergeChannels(channelOps[i], channelOps[j]);
          chan_merge_map[channelOps[j]] = channelOps[i];
        }
      }
    }
    renameSymbols(channelOps, chan_merge_map);
  }

  void runOnOperation() override {
    auto module = getOperation();

    SmallVector<func::FuncOp, 4> funcOps;
    std::vector<air::ChannelOp> channelOps;
    module.walk([&](air::ChannelOp op) { channelOps.push_back(op); });
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps) {
      runOnFunction(f, channelOps);
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
      if ((!areTheSameMemref(memref, a_vec[i].getMemref())) ||
          (!areTheSameSSAValueLists(offsets, a_vec[i].getOffsets())) ||
          (!areTheSameSSAValueLists(sizes, a_vec[i].getSizes())) ||
          (!areTheSameSSAValueLists(strides, a_vec[i].getStrides())))
        return false; // Inconsistent memory use for all puts
    for (unsigned i = 0; i < b_vec.size(); i++)
      if ((!areTheSameMemref(memref, b_vec[i].getMemref())) ||
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
    assert(a_puts.size() == 1);
    assert(b_puts.size() == 1);
    assert(a_gets.size() == 1);
    assert(b_gets.size() == 1);
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
    } else
      assert(false && "NYI");
  }

  // Check whether puts and gets hit the aggressive mode target memory spaces
  bool hitsMemorySpaceForAggMode(std::vector<air::ChannelPutOp> &puts,
                                 std::vector<air::ChannelGetOp> &gets) {
    for (auto put : puts) {
      MemRefType ty = llvm::cast<MemRefType>(put.getMemref().getType());
      if (llvm::any_of(targetMemorySpaces, [&](unsigned memSpace) {
            return memSpace == ty.getMemorySpaceAsInt();
          })) {
        return true;
      }
    }
    for (auto get : gets) {
      MemRefType ty = llvm::cast<MemRefType>(get.getMemref().getType());
      if (llvm::any_of(targetMemorySpaces, [&](unsigned memSpace) {
            return memSpace == ty.getMemorySpaceAsInt();
          })) {
        return true;
      }
    }
    return false;
  }

  bool checkIfMergeable(air::ChannelOp chan_a, air::ChannelOp chan_b) {
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

  std::tuple<bool, std::string>
  checkIfTemporalMergeableByScfForImpl(std::vector<Block *> a_loop_nest,
                                       std::vector<Block *> b_loop_nest) {
    std::tuple<bool, std::string> notMergeable = {false, ""};
    std::tuple<bool, std::string> mergeableToLB = {true, "LB"};
    std::tuple<bool, std::string> mergeableToUB = {true, "UB"};
    if (std::abs((int)a_loop_nest.size() - (int)b_loop_nest.size()) != 1)
      return notMergeable;
    unsigned max_loop_nest_count =
        std::max(a_loop_nest.size(), b_loop_nest.size());
    // Skip over the unequal scf.for loop, and check equality for the rest of
    // the loops first.
    int outermostScfFor = -1;
    for (unsigned i = 0; i < max_loop_nest_count; i++) {
      unsigned scfForCount = 0;
      if ((i < a_loop_nest.size()) &&
          isa<scf::ForOp>(a_loop_nest[i]->getParentOp()))
        scfForCount++;
      if ((i < b_loop_nest.size()) &&
          isa<scf::ForOp>(b_loop_nest[i]->getParentOp()))
        scfForCount++;
      if (scfForCount == 1)
        outermostScfFor = (int)i;
    }
    if (outermostScfFor < 0)
      return notMergeable;
    SmallVector<unsigned> controlLoopIndices;
    for (int i = 0; i < (int)max_loop_nest_count; i++)
      if (i != outermostScfFor)
        controlLoopIndices.push_back(i);
    // TODO: Assuming a_loop_nest is before b_loop_nest. Always true? TODO:
    // formalize using async dep.
    unsigned index = 0;
    if (a_loop_nest.size() < b_loop_nest.size()) {
      for (auto i : controlLoopIndices) {
        if (!areEquivalentControlLoops(a_loop_nest[index++], b_loop_nest[i]))
          return notMergeable;
      }
      // Check if the skipped scf.for loop has LB >= 1. This is a sign of
      // peeling, indicating opportunity of merge by unpeeling into LB.
      auto outerMostScfFor =
          dyn_cast<scf::ForOp>(b_loop_nest[outermostScfFor]->getParentOp());
      assert(outerMostScfFor);
      if (auto constLB = getConstantIntValue(outerMostScfFor.getLowerBound()))
        if (*constLB < 1)
          return notMergeable;
      return mergeableToLB;
    } else {
      for (auto i : controlLoopIndices) {
        if (!areEquivalentControlLoops(a_loop_nest[i], b_loop_nest[index++]))
          return notMergeable;
      }
      // Merge by unpeeling into UB.
      auto outerMostScfFor =
          dyn_cast<scf::ForOp>(a_loop_nest[outermostScfFor]->getParentOp());
      assert(outerMostScfFor);
      return mergeableToUB;
    }
    return mergeableToLB;
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
    for (unsigned i = 0; i < hierArgOwner.getNumKernelOperands(); i++)
      if (hierArgOwner.getKernelArgument(i) == arg)
        return hierArgOwner.getKernelOperand(i);
    return nullptr;
  }
  // Check if two values represent the same memref. TODO: Need async dependency
  // to ensure that the *data* is also the same.
  bool areTheSameMemref(Value a, Value b) {
    if (!isa<MemRefType>(a.getType()))
      return false;
    if (!isa<MemRefType>(b.getType()))
      return false;
    if (a == b)
      return true;
    auto aHierOper =
        getHierOperandFromHierBlockArgument(llvm::dyn_cast<BlockArgument>(a));
    auto bHierOper =
        getHierOperandFromHierBlockArgument(llvm::dyn_cast<BlockArgument>(b));
    if (!(aHierOper && bHierOper))
      return false;
    if (aHierOper == bHierOper)
      return true;
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
  // Check of two air.channels are mergeable in time, by fusing into a shared
  // scf.for loop. Returns a tuple of bool of whether mergeable, and string of
  // fusing into for loop lower bound (LB) or upper bound (UB), or fuse with no
  // for loop (NFL).
  std::tuple<bool, std::string>
  checkIfTemporalMergeable(air::ChannelOp chan_a, air::ChannelOp chan_b) {
    std::vector<air::ChannelPutOp> a_puts =
        getChannelPutOpThroughSymbol(chan_a);
    std::vector<air::ChannelPutOp> b_puts =
        getChannelPutOpThroughSymbol(chan_b);
    std::vector<air::ChannelGetOp> a_gets =
        getChannelGetOpThroughSymbol(chan_a);
    std::vector<air::ChannelGetOp> b_gets =
        getChannelGetOpThroughSymbol(chan_b);
    std::tuple<bool, std::string> notMergeable = {false, ""};
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
      if ((!areTheSameMemref(aMemref, a_puts[i].getMemref())) ||
          (!areTheSameSSAValueLists(aOffsets, a_puts[i].getOffsets())) ||
          (!areTheSameSSAValueLists(aSizes, a_puts[i].getSizes())) ||
          (!areTheSameSSAValueLists(aStrides, a_puts[i].getStrides())))
        return notMergeable; // Inconsistent memory use for all puts
    Value bMemref = b_puts[0].getMemref();
    SmallVector<Value> bOffsets = b_puts[0].getOffsets();
    SmallVector<Value> bSizes = b_puts[0].getSizes();
    SmallVector<Value> bStrides = b_puts[0].getStrides();
    for (unsigned i = 1; i < b_puts.size(); i++)
      if ((!areTheSameMemref(bMemref, b_puts[i].getMemref())) ||
          (!areTheSameSSAValueLists(bOffsets, b_puts[i].getOffsets())) ||
          (!areTheSameSSAValueLists(bSizes, b_puts[i].getSizes())) ||
          (!areTheSameSSAValueLists(bStrides, b_puts[i].getStrides())))
        return notMergeable; // Inconsistent memory use for all puts
    if ((!areTheSameMemref(aMemref, bMemref)))
      return notMergeable;
    aMemref = a_gets[0].getMemref();
    aOffsets = a_gets[0].getOffsets();
    aSizes = a_gets[0].getSizes();
    aStrides = a_gets[0].getStrides();
    for (unsigned i = 1; i < a_gets.size(); i++)
      if ((!areTheSameMemref(aMemref, a_gets[i].getMemref())) ||
          (!areTheSameSSAValueLists(aOffsets, a_gets[i].getOffsets())) ||
          (!areTheSameSSAValueLists(aSizes, a_gets[i].getSizes())) ||
          (!areTheSameSSAValueLists(aStrides, a_gets[i].getStrides())))
        return notMergeable; // Inconsistent memory use for all gets
    bMemref = b_gets[0].getMemref();
    bOffsets = b_gets[0].getOffsets();
    bSizes = b_gets[0].getSizes();
    bStrides = b_gets[0].getStrides();
    for (unsigned i = 1; i < b_gets.size(); i++)
      if ((!areTheSameMemref(bMemref, b_gets[i].getMemref())) ||
          (!areTheSameSSAValueLists(bOffsets, b_gets[i].getOffsets())) ||
          (!areTheSameSSAValueLists(bSizes, b_gets[i].getSizes())) ||
          (!areTheSameSSAValueLists(bStrides, b_gets[i].getStrides())))
        return notMergeable; // Inconsistent memory use for all gets
    if ((!areTheSameMemref(aMemref, bMemref)) ||
        (!areTheSameSSAValueLists(aOffsets, bOffsets)) ||
        (!areTheSameSSAValueLists(aSizes, bSizes)) ||
        (!areTheSameSSAValueLists(aStrides, bStrides)))
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
    } else if (isa<affine::AffineIfOp>(a) || isa<affine::AffineIfOp>(b)) {
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
  void mergeChannelOps(air::ChannelInterface a, air::ChannelInterface b) {
    // fuse a and b under the same loop nest, if a and b are under different
    // loop nests
    if (a->getParentRegion() == b->getParentRegion())
      return;
    IRMapping remap;
    remapAllParentLoopArgs(remap, a, b);
    OpBuilder builder(a);
    builder.setInsertionPoint(a->getBlock()->getTerminator());
    auto new_b = cloneOpAndOperands(builder, remap, b);
    eraseParentLoopIfEmpty(*b);
    auto async_b = dyn_cast<air::AsyncOpInterface>(new_b);
    if (async_b.getAsyncToken())
      async_b.addAsyncDependency(
          dyn_cast<air::AsyncOpInterface>(a.getOperation()).getAsyncToken());
  }
  void mergeChannelOpsTemporally(air::ChannelInterface a,
                                 air::ChannelInterface b,
                                 std::string mergeByLBOrUB) {
    scf::ForOp parentForOp = a->getParentOfType<scf::ForOp>();
    if (!parentForOp)
      return;
    while (parentForOp && parentForOp->getParentOfType<scf::ForOp>()) {
      parentForOp = parentForOp->getParentOfType<scf::ForOp>();
    }
    OpBuilder builder(parentForOp);
    if (mergeByLBOrUB == "LB") {
      int originalLB = *getConstantIntValue(parentForOp.getLowerBound());
      assert(originalLB > 0);
      parentForOp->getOpOperand(0).assign(
          builder.create<arith::ConstantIndexOp>(parentForOp->getLoc(),
                                                 originalLB - 1));
    } else if (mergeByLBOrUB == "UB") {
      int originalUB = *getConstantIntValue(parentForOp.getUpperBound());
      parentForOp->getOpOperand(1).assign(
          builder.create<arith::ConstantIndexOp>(parentForOp->getLoc(),
                                                 originalUB + 1));
    } else
      assert(false && "invalid mergeByLBOrUB flag");
    eraseParentLoopIfEmpty(*b);
  }
  void mergeChannels(air::ChannelOp chan_a, air::ChannelOp chan_b) {
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
      mergeChannelOps(a_puts[i], b_puts[i]);
    for (unsigned i = 0; i < a_gets.size(); i++)
      mergeChannelOps(a_gets[i], b_gets[i]);
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
  Operation *cloneOpAndOperands(OpBuilder builder, IRMapping remap,
                                Operation *op) {
    SetVector<Operation *> backwardSlice;
    BackwardSliceOptions bsOptions{[&](Operation *o) {
      return !isa<scf::ForOp>(o) && !isa<scf::ParallelOp>(o) &&
             !isa<air::HierarchyInterface>(o);
    }};
    getBackwardSlice(op, &backwardSlice, bsOptions);
    for (auto operand : op->getOperands()) {
      if (operand.getDefiningOp() &&
          isa<arith::ConstantIndexOp>(operand.getDefiningOp()))
        backwardSlice.insert(operand.getDefiningOp());
    }
    for (auto b : backwardSlice) {
      if (isa<arith::ConstantIndexOp>(b))
        builder.clone(*b, remap);
      else if (b->getNumResults() == 1 &&
               isa<IndexType>(b->getResult(0).getType()))
        builder.clone(*b, remap);
      else if (auto exec = dyn_cast<air::ExecuteOp>(b)) {
        auto child_op = exec.getChildOp();
        if (child_op->getNumResults() == 1 &&
            isa<IndexType>(child_op->getResult(0).getType()))
          builder.clone(*b, remap);
      }
    }
    auto new_op = builder.clone(*op, remap);
    return new_op;
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
          for (unsigned j = 0; j < a_par.getBody()->getNumArguments(); j++) {
            remap.map(b_par.getBody()->getArgument(j),
                      a_par.getBody()->getArgument(j));
          }
        }
      }
    }
  }
  void eraseParentLoopIfEmpty(Operation &op) {
    auto hasNEvents = [](Block *block, unsigned N) {
      unsigned eventCounter = 0;
      for (auto &o : block->getOperations()) {
        if (isa<scf::YieldOp>(o))
          continue;
        if (isa<air::WaitAllOp>(o))
          continue;
        if (auto execOp = dyn_cast<air::ExecuteOp>(o))
          if (execOp->getNumResults() == 2 &&
              llvm::isa<IndexType>(execOp->getResult(1).getType()))
            continue;
        eventCounter++;
      }
      return eventCounter == N;
    };
    Operation *targetOp = &op;
    for (Operation *parent = &op; !isa<air::HierarchyInterface>(parent);
         parent = parent->getParentOp()) {
      if (!hasNEvents(parent->getBlock(), 1)) {
        targetOp = parent;
        break;
      };
    }
    // Erase target op.
    OpBuilder builder(targetOp);
    SmallVector<Value> depList;
    for (auto operand : targetOp->getOperands()) {
      if (llvm::isa<air::AsyncTokenType>(operand.getType()))
        depList.push_back(operand);
    }
    for (auto res : targetOp->getResults()) {
      auto wa = builder.create<air::WaitAllOp>(
          builder.getUnknownLoc(),
          air::AsyncTokenType::get(builder.getContext()), depList);
      res.replaceAllUsesWith(wa.getAsyncToken());
    }
    targetOp->erase();
  }
};

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

  void identifyTargetSCFForAndOps(
      func::FuncOp f, std::vector<air::HierarchyInterface> hierOps,
      std::map<scf::ForOp, SmallVector<Operation *>> &target_ops_map) {
    for (auto hier_op : hierOps) {
      // Identify the target for loops and their target child ops
      for (auto for_op : hier_op->getRegion(0).getOps<scf::ForOp>()) {
        for_op.walk([&](air::MemcpyInterface memcpyOp) {
          // Get for_op's immediate child op
          if (!dyn_cast<air::AsyncOpInterface>(memcpyOp.getOperation())
                   .getAsyncToken())
            return; // This pass requires an async IR.
          int for_op_token_count = 0;
          for (auto v : for_op->getResults())
            if (isa<air::AsyncTokenType>(v.getType()))
              for_op_token_count++;
          if (for_op_token_count > 1)
            return; // This for op has more than one loop-carried dep token,
                    // suggesting pipelining pattern. Will be handelled by
                    // -air-unroll-loop-for-pipelining-pattern instead.
          Operation *parent = memcpyOp.getOperation();
          while (parent->getParentOp() != for_op.getOperation()) {
            parent = parent->getParentOp();
          }
          if (isa<air::HierarchyInterface>(parent))
            return;
          if (isa<affine::AffineIfOp>(parent))
            return;
          // Check if for loop is splittable by tracing air dependency.
          for (auto op : target_ops_map[for_op]) {
            if (areAsyncDependent(op, parent)) {
              target_ops_map.erase(for_op);
              return;
            }
          }
          push_back_if_unique<Operation *>(target_ops_map[for_op], parent);
          // Check if any memref.alloc needs to be hoisted.
          if (memcpyOp.getSrcMemref() &&
              !memcpyOp.getSrcMemref().getDefiningOp())
            return;
          if (memcpyOp.getSrcMemref() &&
              for_op->isProperAncestor(
                  memcpyOp.getSrcMemref().getDefiningOp())) {
            Operation *memref_def = memcpyOp.getSrcMemref().getDefiningOp();
            if (auto exec = dyn_cast<air::ExecuteOp>(memref_def))
              memref_def = exec.getBody()
                               .getBlocks()
                               .front()
                               .getTerminator()
                               ->getOperand(0)
                               .getDefiningOp();
            memref_def->setAttr(
                "hoist_alloc",
                mlir::BoolAttr::get(memref_def->getContext(), true));
          }
          if (memcpyOp.getDstMemref() &&
              !memcpyOp.getDstMemref().getDefiningOp())
            return;
          if (memcpyOp.getDstMemref() &&
              for_op->isProperAncestor(
                  memcpyOp.getDstMemref().getDefiningOp())) {
            Operation *memref_def = memcpyOp.getDstMemref().getDefiningOp();
            if (auto exec = dyn_cast<air::ExecuteOp>(memref_def))
              memref_def = exec.getBody()
                               .getBlocks()
                               .front()
                               .getTerminator()
                               ->getOperand(0)
                               .getDefiningOp();
            memref_def->setAttr(
                "hoist_alloc",
                mlir::BoolAttr::get(memref_def->getContext(), true));
          }
        });
      }
    }
  }

  void runOnOperation() override {
    auto module = getOperation();

    SmallVector<func::FuncOp, 4> funcOps;
    std::vector<air::HierarchyInterface> air_hier_ops;
    // Skipping over loop splitting inside herd.
    module.walk([&](air::HierarchyInterface op) {
      if (!isa<air::HerdOp>(op))
        air_hier_ops.push_back(op);
    });
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });

    // Identify scf.for ops and target child ops for hoisting.
    std::map<scf::ForOp, SmallVector<Operation *>> target_ops_map;
    for (auto f : funcOps) {
      identifyTargetSCFForAndOps(f, air_hier_ops, target_ops_map);
      // If necessary, hoist allocs out of the loops, too.
      RewritePatternSet patterns(f.getContext());
      patterns.insert<HoistMemallocInForPattern>(f.getContext(), false);
      (void)applyPatternsAndFoldGreedily(f, std::move(patterns));
    }

    // Hoist ops out of each scf.for.
    for (auto pair : target_ops_map) {
      OpBuilder builder(pair.first);
      for (auto op : pair.second) {
        auto newForOp = hoistTargetOpsToNewSCFFor(builder, pair.first,
                                                  SmallVector<Operation *>{op});
        if (!newForOp)
          continue;
        // Redo async dependency tracing.
        air::dependencyTracer depTracer;
        depTracer.traceDependencyFromScfForOp(newForOp);
      }
    }

    // Post processing, hoisting air.herd ops out of perfectly nested scf.for
    // loop.
    for (auto f : funcOps) {
      RewritePatternSet patterns_1(f.getContext());
      patterns_1
          .insert<HoistAIRHerdInForPattern, HoistAIRChannelInAccumPattern>(
              f.getContext(), false);
      (void)applyPatternsAndFoldGreedily(f, std::move(patterns_1));
    }
  }

private:
  template <typename T>
  void push_back_if_unique(SmallVector<T> &vec, T entry) const {
    if (std::find(vec.begin(), vec.end(), entry) == vec.end()) {
      vec.push_back(entry);
    }
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
      Type elemType = llvm::cast<MemRefType>(memref.getType()).getElementType();
      Attribute memorySpace =
          llvm::cast<MemRefType>(memref.getType()).getMemorySpace();
      auto newMemrefType = MemRefType::get(overall_access_bounds, elemType,
                                           nullptr, memorySpace);
      if (auto execOp = dyn_cast<air::ExecuteOp>(alloc->getParentOp())) {
        rewriter.setInsertionPoint(execOp);
        auto newExecOp = rewriter.create<air::ExecuteOp>(
            execOp->getLoc(), air::AsyncTokenType::get(rewriter.getContext()),
            newMemrefType, execOp.getAsyncDependencies());
        Block *async_exec_bb = rewriter.createBlock(&newExecOp.getBody());
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
        for (unsigned i = 0; i < herdOp.getNumKernelOperands(); i++) {
          if (herdOp.getKernelOperand(i) == memref) {
            auto memrefInHerd = herdOp.getKernelArgument(i);
            if (getAllChanUsers(memrefInHerd, users, dealloc, builder).failed())
              return failure();
          }
        }
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
    Type elemType = llvm::cast<MemRefType>(subViewOp.getSource().getType())
                        .getElementType();
    Attribute memorySpace =
        llvm::cast<MemRefType>(subViewOp.getSource().getType())
            .getMemorySpace();
    auto shrunkMemrefType =
        MemRefType::get(overall_access_bounds, elemType, nullptr, memorySpace);
    MemRefType inferredSubViewOutputTy =
        llvm::cast<MemRefType>(memref::SubViewOp::inferResultType(
            shrunkMemrefType, subViewOp.getStaticOffsets(),
            subViewOp.getStaticSizes(), subViewOp.getStaticStrides()));
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
      if (outputStrides[i] >= (int)memrefVolume) {
        subViewOp.getResult().setType(inferredSubViewOutputTy);
        if (static_offsets[i] >= 0)
          continue;
        if (auto updatedOffset =
                getUpdatedOffsetAfterShrinkage(*subview_offsets, rewriter))
          subViewOp->replaceUsesOfWith(*subview_offsets++, updatedOffset);
      }
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
      if (!(*subview_offsets).getDefiningOp())
        continue;
      if (auto exec_to_herd_iv =
              dyn_cast<air::ExecuteOp>((*subview_offsets).getDefiningOp())) {
        for (auto oper : exec_to_herd_iv.getChildOp()->getOperands())
          if (getHerdArgOwner(oper))
            offsetIsHerdIndVar = true;
      }
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
          if (!(*subview_offsets).getDefiningOp())
            return false;
          if (auto exec = dyn_cast<air::ExecuteOp>(
                  (*subview_offsets).getDefiningOp())) {
            for (auto oper : exec.getChildOp()->getOperands()) {
              if (!getConstantIntValue(oper))
                return false;
              if (*getConstantIntValue(oper) != 0)
                return false;
            }
          } else
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
      for (auto newUser : subViewOp.getSource().getUsers())
        push_back_if_unique<Operation *>(users, newUser);
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
    if (index.getDefiningOp()) {
      if (auto execOp = dyn_cast<air::ExecuteOp>(index.getDefiningOp())) {
        for (auto oper : execOp.getChildOp()->getOperands()) {
          if (auto herdOp = air::getHerdArgOwner(oper)) {
            rewriter.setInsertionPointToStart(&herdOp.getBody().front());
            execOp.getChildOp()->replaceUsesOfWith(
                oper, rewriter.create<arith::ConstantIndexOp>(
                          rewriter.getUnknownLoc(), 0));
          }
        }
      }
    } else if (auto herdOp = air::getHerdArgOwner(index)) {
      rewriter.setInsertionPointToStart(&herdOp.getBody().front());
      return rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(),
                                                     0);
    }
    return nullptr;
  }
};

struct AIRSegmentLoopFusionPattern : public OpRewritePattern<air::SegmentOp> {
  using OpRewritePattern<air::SegmentOp>::OpRewritePattern;

  AIRSegmentLoopFusionPattern(MLIRContext *ctx) : OpRewritePattern(ctx) {}

  LogicalResult matchAndRewrite(air::SegmentOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    // Get memref.alloc ops.
    SmallVector<air::ExecuteOp> memalloc_execs;
    SmallVector<air::ExecuteOp> memdealloc_execs;
    // Map from air.execute op containing alloc to air.execute op containing
    // dealloc.
    std::vector<std::pair<air::ExecuteOp, air::ExecuteOp>> alloc_dealloc_execs;
    for (auto execOp : op.getOps<air::ExecuteOp>()) {
      if (!execOp.getChildOp())
        continue;
      if (!isa<memref::AllocOp>(execOp.getChildOp()))
        continue;
      auto memref = execOp->getResult(1);
      bool allChannelUsersAreInScfFor = true;
      for (auto user : memref.getUsers())
        if (isa<air::ChannelInterface>(user))
          if (!isa<scf::ForOp>(user->getParentOp()))
            allChannelUsersAreInScfFor = false;
      if (allChannelUsersAreInScfFor)
        alloc_dealloc_execs.push_back(std::make_pair(execOp, nullptr));
    }
    for (auto execOp : op.getOps<air::ExecuteOp>()) {
      if (!execOp.getChildOp())
        continue;
      if (!isa<memref::DeallocOp>(execOp.getChildOp()))
        continue;
      auto dealloc = dyn_cast<memref::DeallocOp>(execOp.getChildOp());
      for (auto &pair : alloc_dealloc_execs) {
        if (dealloc.getMemref() == pair.first.getResult(1)) {
          pair.second = execOp;
        }
      }
    }
    // Get roots to perfectly nested scf.for loops.
    SmallVector<scf::ForOp> perfectlyNestedForBands;
    auto hasNElements = [](Block *block, unsigned N) {
      unsigned counter = 0;
      for (auto &o : block->getOperations()) {
        if (isa<air::ChannelInterface>(o))
          counter++;
        else if (isa<LoopLikeOpInterface>(o))
          counter++;
        else if (isa<mlir::linalg::LinalgOp>(o))
          counter++;
      }
      return counter == N;
    };
    for (auto forOp : op.getOps<scf::ForOp>()) {
      // Conditions for candicate scf.for op for fusion: (1) has at most 1
      // unique channels operating in the block, (2) is either perfectly nested,
      // or contains only channel ops, (3) is static for loop.
      if (getNumUniqueChannelsInBlock(forOp.getBody()) <= 1 &&
          hasNElements(
              forOp.getBody(),
              std::max(getNumChannelPutsGetsInBlock(forOp.getBody()), 1)) &&
          air::getStaticScfForTripCountAsInt(forOp))
        perfectlyNestedForBands.push_back(forOp);
    }
    if (perfectlyNestedForBands.empty())
      return failure();
    if (alloc_dealloc_execs.empty())
      return failure();

    // From the loop bands, get fusable scf.for for loop bands.
    SmallVector<scf::ForOp> equalIterationForOps;
    equalIterationForOps.push_back(perfectlyNestedForBands[0]);
    int lb = *getConstantIntValue(perfectlyNestedForBands[0].getLowerBound());
    int ub = *getConstantIntValue(perfectlyNestedForBands[0].getUpperBound());
    int step = *getConstantIntValue(perfectlyNestedForBands[0].getStep());
    for (unsigned i = 1; i < perfectlyNestedForBands.size(); i++) {
      int band_step =
          *getConstantIntValue(perfectlyNestedForBands[i].getStep());
      int band_lb =
          *getConstantIntValue(perfectlyNestedForBands[i].getLowerBound());
      int band_ub =
          *getConstantIntValue(perfectlyNestedForBands[i].getUpperBound());
      if (band_lb == lb && band_ub == ub && band_step == step) {
        equalIterationForOps.push_back(perfectlyNestedForBands[i]);
      } else if (band_lb == lb && band_ub == ub &&
                 llvm::mod(std::max(band_step, step),
                           std::min(band_step, step)) == 0) {
        // If scf.for loops are not identical, but tilable to having identical
        // roots.
        if (simpleScfForLoopTiling(perfectlyNestedForBands[i], step, band_step))
          equalIterationForOps.push_back(perfectlyNestedForBands[i]);
      }
    }
    if (equalIterationForOps.empty())
      return failure();

    // Folding memref.alloc / dealloc ops into fused loop.
    SmallVector<scf::ForOp> fusableForOps;
    for (auto forOp : equalIterationForOps) {
      for (auto ia : forOp.getInitArgs()) {
        auto iaDefOp = ia.getDefiningOp();
        if (!iaDefOp)
          continue;
        if (llvm::any_of(
                alloc_dealloc_execs,
                [&](std::pair<air::ExecuteOp, air::ExecuteOp> exec_pair) {
                  return exec_pair.first == iaDefOp;
                }))
          fusableForOps.push_back(forOp);
      }
    }
    if (fusableForOps.empty())
      return failure();

    rewriter.setInsertionPoint(equalIterationForOps[0]);
    auto new_loop_op_init_arg =
        rewriter
            .create<air::WaitAllOp>(
                loc, air::AsyncTokenType::get(rewriter.getContext()),
                SmallVector<Value>{})
            .getAsyncToken();
    scf::ForOp new_loop_op = rewriter.create<scf::ForOp>(
        rewriter.getUnknownLoc(), perfectlyNestedForBands[0].getLowerBound(),
        perfectlyNestedForBands[0].getUpperBound(),
        perfectlyNestedForBands[0].getStep(),
        SmallVector<Value>{new_loop_op_init_arg});
    SmallVector<air::ExecuteOp> erase_keys;
    IRMapping remap;
    for (auto execOpPair : alloc_dealloc_execs) {
      bool canMove = false;
      air::ExecuteOp alloc_exec = execOpPair.first;
      for (auto token_user : alloc_exec.getAsyncToken().getUsers())
        if (llvm::any_of(equalIterationForOps, [&](scf::ForOp fusableForOp) {
              return fusableForOp == token_user;
            }))
          canMove = true;
      if (canMove) {
        rewriter.setInsertionPointToEnd(new_loop_op.getBody());
        auto new_alloc_exec = rewriter.clone(*alloc_exec, remap);
        clearAsyncDependenciesOfAsyncOp(
            dyn_cast<air::AsyncOpInterface>(new_alloc_exec));
        for (unsigned i = 0; i < new_alloc_exec->getNumResults(); i++)
          remap.map(alloc_exec->getResult(i), new_alloc_exec->getResult(i));
      } else
        erase_keys.push_back(alloc_exec);
    }
    for (auto e : erase_keys)
      for (unsigned i = 0; i < alloc_dealloc_execs.size(); i++)
        if (e == alloc_dealloc_execs[i].first)
          alloc_dealloc_execs.erase(alloc_dealloc_execs.begin() + i);

    // Loop fusion.
    for (auto forOp : fusableForOps) {
      remap.map(forOp.getInductionVar(), new_loop_op.getInductionVar());
      for (unsigned i = 0; i < forOp.getRegionIterArgs().size(); i++)
        remap.map(forOp.getRegionIterArgs()[i],
                  remap.lookupOrDefault(
                      forOp.getOperand(i + forOp.getNumControlOperands())));
      rewriter.setInsertionPointToEnd(new_loop_op.getBody());
      for (auto &child_op : forOp.getBody()->getOperations())
        if (!child_op.mightHaveTrait<OpTrait::IsTerminator>())
          rewriter.clone(child_op, remap);
    }

    // Fuse dealloc ops.
    for (auto execOpPair : alloc_dealloc_execs) {
      air::ExecuteOp dealloc_exec = execOpPair.second;
      clearAsyncDependenciesOfAsyncOp(dealloc_exec);
      rewriter.setInsertionPointToEnd(new_loop_op.getBody());
      rewriter.clone(*dealloc_exec, remap);
    }

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
        loc, air::AsyncTokenType::get(rewriter.getContext()), yield_dep_list);
    rewriter.create<scf::YieldOp>(loc, wa_op.getAsyncToken());

    // Erase original scf.for ops.
    for (auto forOp : fusableForOps) {
      assert(forOp.getNumResults() == new_loop_op.getNumResults() &&
             "Fused loop has different number of results as original");
      for (unsigned i = 0; i < forOp.getNumResults(); i++) {
        forOp.getResult(i).replaceAllUsesWith(new_loop_op.getResult(i));
      }
      forOp->erase();
    }

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
      while (get_parent->getParentOp() != new_loop_op) {
        get_parent = get_parent->getParentOp();
      }
      put_get_mapping[put_parent] = get_parent;
      put_parents.push_back(put_parent);
    });
    for (auto put_parent : put_parents) {
      auto get_parent = put_get_mapping[put_parent];
      if (put_parent->isBeforeInBlock(get_parent))
        put_parent->moveAfter(get_parent);
      Value get_parent_token = nullptr;
      for (auto res : get_parent->getResults())
        if (isa<AsyncTokenType>(res.getType()))
          get_parent_token = res;
      for (unsigned i = 0; i < put_parent->getNumOperands(); i++)
        if (get_parent_token &&
            isa<AsyncTokenType>(put_parent->getOperand(i).getType())) {
          put_parent->getOpOperand(i).assign(get_parent_token);
        }
    }

    // Erase allocs/deallocs
    for (auto execOpPair : alloc_dealloc_execs) {
      auto alloc = execOpPair.first;
      replaceAsyncOpWithWaitAll(rewriter, alloc);
      alloc->erase();
      auto dealloc = execOpPair.second;
      replaceAsyncOpWithWaitAll(rewriter, dealloc);
      dealloc->erase();
    }

    return success();
  }

private:
  // Get the number of air.channel.puts/gets in block.
  int getNumChannelPutsGetsInBlock(Block *block) const {
    int count = 0;
    for (auto &o : block->getOperations())
      if (auto chanOp = dyn_cast<air::ChannelInterface>(o))
        count++;
    return count;
  }

  // Get the total number of unique air.channels that all air.channel.puts/gets
  // in block operate on.
  int getNumUniqueChannelsInBlock(Block *block) const {
    llvm::SmallSet<std::string, 1> chanNamesInBlock;
    for (auto &o : block->getOperations())
      if (auto chanOp = dyn_cast<air::ChannelInterface>(o))
        chanNamesInBlock.insert(chanOp.getChanName().str());
    return chanNamesInBlock.size();
  }

  // Scf.for loop tiling. This simple tiling implementation generates a new
  // inner scf.for loop which starts from the original loop's lower bound. It
  // may change the meaning of the original scf.for loop, therefore it requires
  // a separate check to make sure that it is legal to tile this way.
  scf::ForOp simpleScfForLoopTiling(scf::ForOp forOp, int original_step,
                                    int tiled_step) const {
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
      remap.map(forOp.getRegionIterArgs()[j],
                new_for_op.getRegionIterArgs()[j]);
    remap.map(forOp.getInductionVar(), new_for_op.getInductionVar());
    SmallVector<Operation *> erased;
    Value yielded_token = nullptr;
    for (auto &o : forOp.getOps()) {
      if (&o != new_for_op && &o != forOp.getBody()->getTerminator()) {
        auto new_o = builder.clone(o, remap);
        if (isAsyncOp(new_o)) {
          yielded_token = new_o->getResult(0);
          erased.push_back(&o);
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
      o->getResult(0).replaceAllUsesWith(new_for_op->getResult(0));
      o->erase();
    }

    return new_for_op;
  }

  // Erase async op and replace with air wait all
  void replaceAsyncOpWithWaitAll(OpBuilder builder, Operation *op) const {
    builder.setInsertionPoint(op);
    auto waitAllReplaceAlloc = builder.create<air::WaitAllOp>(
        builder.getUnknownLoc(), air::AsyncTokenType::get(builder.getContext()),
        SmallVector<Value>{});
    for (unsigned i = 0; i < op->getNumResults(); i++)
      op->getResult(i).replaceAllUsesWith(waitAllReplaceAlloc.getAsyncToken());
  }
};

// A pass which performs loop fusion within air.segment op's region.
class AIRSegmentLoopFusion
    : public xilinx::air::impl::AIRSegmentLoopFusionBase<AIRSegmentLoopFusion> {

public:
  AIRSegmentLoopFusion() = default;
  AIRSegmentLoopFusion(const AIRSegmentLoopFusion &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect>();
  }

  void runPreProcPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns
        .insert<CanonicalizeAffineApplyOnLoopInductionVar, UnrollScfParallel>(
            ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  void runPostProcPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<ShrinkMemrefSizesByAccessPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
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
          if (!getHerdArgumentFromHerdOperand(herdOp, memref))
            continue;
          auto herdArg = getHerdArgumentFromHerdOperand(herdOp, memref);
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
    patterns.insert<AIRSegmentLoopFusionPattern>(func.getContext(), false);
    (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
    runPostProcPatterns(func);
    func.walk([&](memref::AllocOp op) { op->removeAttr("shrinkage"); });
  }

private:
  void updateFuncOpInputTypesFromCall(func::CallOp callOp) const {
    // Fetch name.
    StringRef fnName = callOp.getCallee();
    auto fnDecl = dyn_cast_or_null<func::FuncOp>(SymbolTable::lookupSymbolIn(
        callOp->getParentOfType<ModuleOp>(), fnName));
    assert(fnDecl && "expected function declaration");

    // Update function's argument types.
    auto functionType = fnDecl.getFunctionType();
    auto newArgTypes = llvm::to_vector<6>(callOp.getOperandTypes());
    auto newFunctionType = FunctionType::get(fnDecl.getContext(), newArgTypes,
                                             functionType.getResults());
    fnDecl.setType(newFunctionType);
  }

  // Get herd argument to a herd operand
  BlockArgument getHerdArgumentFromHerdOperand(air::HerdOp herdOp,
                                               Value v) const {
    for (unsigned i = 0; i < herdOp.getNumKernelOperands(); i++)
      if (herdOp.getKernelOperand(i) == v)
        return herdOp.getKernelArgument(i);
    return nullptr;
  }
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

std::unique_ptr<mlir::Pass> createAIRDependencyScheduleOptPass() {
  return std::make_unique<AIRDependencyScheduleOpt>();
}

std::unique_ptr<Pass> createAIRUnrollChannelByFactorPattern() {
  return std::make_unique<AIRUnrollChannelByFactorPattern>();
}

std::unique_ptr<Pass> createAIREnforceLoopCarriedMemrefDeallocPattern() {
  return std::make_unique<AIREnforceLoopCarriedMemrefDeallocPattern>();
}

std::unique_ptr<Pass> createAIRDeAliasMemref() {
  return std::make_unique<AIRDeAliasMemref>();
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

std::unique_ptr<Pass> createAIRSegmentLoopFusion() {
  return std::make_unique<AIRSegmentLoopFusion>();
}

void populateAIRLoopIndexCanonicalizationPatterns(RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.insert<CanonicalizeAffineApplyOnLoopInductionVar>(ctx);
}

} // namespace air
} // namespace xilinx
