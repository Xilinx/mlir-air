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

    SmallVector<air::DmaMemcpyInterface, 1> dmamemcpy_incoming_history;
    SmallVector<air::DmaMemcpyInterface, 1> dmamemcpy_outgoing_history;
    for (auto dma_op : for_op.getOps<air::DmaMemcpyInterface>()) {
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
          assert(alloc_region_op.getAsyncDependencies().size() <= 1 &&
                 "Alloc event having more than one dependant");

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
  bool isIncomingDmaOp(air::DmaMemcpyInterface dma_op) const {
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
  air::ExecuteOp
  getRegionOfAllocOpForDmaOp(air::DmaMemcpyInterface dma_op) const {
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
  bool isOutgoingDmaOp(air::DmaMemcpyInterface dma_op) const {
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
  air::ExecuteOp
  getRegionOfDeallocOpForDmaOp(air::DmaMemcpyInterface dma_op) const {
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
  void reconnectIncomingDma(air::DmaMemcpyInterface dma_op,
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
  void reconnectOutgoingEvents(air::DmaMemcpyInterface dma_op,
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
      for (int i = dependency_list.size() - 1; i >= 0; i--) {
        dma_async_op.eraseAsyncDependency(i);
      }
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
  bool areSymmetricDmaOps(air::DmaMemcpyInterface op_1,
                          air::DmaMemcpyInterface op_2) const {
    bool areSymmetric = op_1.getSrcMemref() == op_2.getDstMemref();
    areSymmetric &= op_2.getSrcMemref() == op_1.getDstMemref();
    if (op_1.getNumDims() == 0 && op_2.getNumDims() == 0) {
      // If both dma ops are nd dmas, then proceed to check offsets, sizes and
      // strides
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
    } else if (op_1.getNumDims() == op_2.getNumDims()) {
      // If both dma ops are of same dma type but not nd dmas, then proceed to
      // check memrefdims etc
      for (unsigned i = 0; i < op_1.getNumDims(); i++) {
        areSymmetric &= op_1.getSrcMemrefDim(i) == op_2.getDstMemrefDim(i);
        areSymmetric &= op_2.getSrcMemrefDim(i) == op_1.getDstMemrefDim(i);
      }
      areSymmetric &= op_1.getLength() == op_2.getLength();
    } else {
      // Two dma ops having different # of dimensions
      areSymmetric = false;
    }

    return areSymmetric;
  }
};

struct HoistMemallocInForPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp alloc_op,
                                PatternRewriter &rewriter) const override {

    // Find parent air.execute
    auto alloc_exec = alloc_op->getParentOfType<air::ExecuteOp>();
    if (!alloc_exec)
      return failure();

    // Find dealloc
    Operation *dealloc_op = nullptr;
    auto alloc_exec_memref = alloc_exec->getResults()[1];
    assert(alloc_exec_memref.getType().isa<MemRefType>() &&
           "the ssa value yielded from execute is not memref");
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

    for (auto for_op_token : for_op->getResults()) {
      dealloc_exec.addAsyncDependency(for_op_token);
    }

    // Hoist alloc and dealloc out of for loop
    alloc_exec->moveBefore(for_op);
    dealloc_exec->moveAfter(for_op);

    // Erase alloc hoisting attr
    alloc_op->removeAttr("hoist_alloc");

    return success();
  }

private:
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

struct BroadcastDetection {

public:
  // Trace dma ops' dependency to loop induction variables
  void getDmaOpLoopDependency(func::FuncOp f) {
    f.walk([&](Operation *op) {
      if (auto dma_op = mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op)) {
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
  std::vector<air::DmaMemcpyInterface> dma_op_history;
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
      if (upstream_op && dyn_cast<air::DmaMemcpyInterface>(upstream_op)) {
        auto upstream_dma = dyn_cast<air::DmaMemcpyInterface>(upstream_op);
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

class AIRHoistMemallocInForPattern
    : public xilinx::air::AIRHoistMemallocInForPatternBase<
          AIRHoistMemallocInForPattern> {

public:
  AIRHoistMemallocInForPattern() = default;
  AIRHoistMemallocInForPattern(const AIRHoistMemallocInForPattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<HoistMemallocInForPattern>(ctx);
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
          op->setAttr("unrolled_iteration", b.getUI32IntegerAttr(i));
        };
        (void)loopUnrollByFactor(for_op, unroll_factor, annotateFn);
      }
    });
  }

private:
};

class AIRPipelineLoweringPattern
    : public xilinx::air::AIRPipelineLoweringPatternBase<
          AIRPipelineLoweringPattern> {

public:
  AIRPipelineLoweringPattern() = default;
  AIRPipelineLoweringPattern(const AIRPipelineLoweringPattern &pass){};

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<HoistMemallocInForPattern>(ctx);
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

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRHoistDmaInAccumPattern() {
  return std::make_unique<AIRHoistDmaInAccumPattern>();
}

std::unique_ptr<Pass> createAIRHoistMemallocInForPattern() {
  return std::make_unique<AIRHoistMemallocInForPattern>();
}

std::unique_ptr<Pass> createAIRUnrollLoopForPipeliningPattern() {
  return std::make_unique<AIRUnrollLoopForPipeliningPattern>();
}

std::unique_ptr<Pass> createAIRBroadcastDetection() {
  return std::make_unique<AIRBroadcastDetection>();
}

std::unique_ptr<Pass> createAIRPruneLinalgGenericInputDma() {
  return std::make_unique<AIRPruneLinalgGenericInputDma>();
}

std::unique_ptr<Pass> createAIRPipelineLoweringPattern() {
  return std::make_unique<AIRPipelineLoweringPattern>();
}

std::unique_ptr<mlir::Pass> createAIRDependencyScheduleOptPass() {
  return std::make_unique<AIRDependencyScheduleOpt>();
}

} // namespace air
} // namespace xilinx