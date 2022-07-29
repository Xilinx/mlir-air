// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

#include "PassDetail.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/AIRDependencyScheduleOpt.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/IR/IntegerSet.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"

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

    // if (for_op->getParentOfType<xilinx::air::HerdLaunchOp>()){
      // Only looking for loops inside of herd launch
      SmallVector<air::DmaMemcpyInterface, 1> dmamemcpy_incoming_history;
      SmallVector<air::DmaMemcpyInterface, 1> dmamemcpy_outgoing_history;
      for (auto dma_op : for_op.getOps<air::DmaMemcpyInterface>()){
        if (isIncomingDmaOp(dma_op)){
          dmamemcpy_incoming_history.push_back(dma_op);
        }
        if (isOutgoingDmaOp(dma_op)){
          dmamemcpy_outgoing_history.push_back(dma_op);
        }
      }
      bool foundDmaPairToHoist = false;
      for (auto op_2 : dmamemcpy_outgoing_history){
        bool foundDmaPairForThisOp2 = false;
        for (auto op_1 : dmamemcpy_incoming_history){
          bool areInvariantWRTForLoop = true;
          // Check if the pair of dmas form symmetry in their src and dst
          bool areSymmetric = areSymmetricDmaOps(op_1, op_2);
          // Check if the pair of dmas are invariant with respect to for loop iterations
          areInvariantWRTForLoop &= isInvariantWRTForLoop(op_1.getOperation(), for_op);
          areInvariantWRTForLoop &= isInvariantWRTForLoop(op_2.getOperation(), for_op);
          if (areSymmetric & areInvariantWRTForLoop){
            foundDmaPairToHoist = true;
            foundDmaPairForThisOp2 = true;
            // Found a pair of dmas which cancel out each other 
            air::RegionOp alloc_region_op = getRegionOfAllocOpForDmaOp(op_1);
            air::RegionOp dealloc_region_op = getRegionOfDeallocOpForDmaOp(op_2);
            assert(alloc_region_op.getAsyncDependencies().size() == 1 && "Alloc event having more than one dependant");

            // Reconnect incoming alloc event
            alloc_region_op.eraseAsyncDependency(0);
            // Reconnect incoming dma event
            reconnectIncomingDma(op_1, for_op);
            // Move ops to before the for loop
            alloc_region_op->moveBefore(for_op); 
            op_1->moveBefore(for_op);

            // Reconnect outgoing dealloc event
            // Reconnect outgoing dma event
            scf::YieldOp yield_op = dyn_cast<scf::YieldOp>(for_op.getBody()->getTerminator());
            air::WaitAllOp wait_all_after_for = dyn_cast<air::WaitAllOp>(yield_op->getOperand(0).getDefiningOp());
            reconnectOutgoingEvents(op_2, dealloc_region_op, for_op, wait_all_after_for);
            // Move ops to after the for loop
            dealloc_region_op->moveAfter(for_op);
            op_2->moveAfter(for_op);
            
            // Move const ops which produce op_2 operands
            // Note: moving consts of which op_1 depends on AFTER op_2 to maintain dominance if consts are shared by both
            for (auto op_2_operand : op_2->getOperands()){
              if (op_2_operand.getDefiningOp() && isa<arith::ConstantOp>(op_2_operand.getDefiningOp())){
                rewriter.setInsertionPoint(op_2);
                rewriter.clone(*op_2_operand.getDefiningOp());
              }
            }
            // Move const ops which produce op_1 operands
            for (auto op_1_operand : op_1->getOperands()){
              if (op_1_operand.getDefiningOp() && isa<arith::ConstantOp>(op_1_operand.getDefiningOp())){
                rewriter.setInsertionPoint(op_1);
                rewriter.clone(*op_1_operand.getDefiningOp());
              }
            }
          }
        }
        if (foundDmaPairForThisOp2) continue; // Ensure unique pairing
      }
      if (foundDmaPairToHoist) return success();
    // }
    return failure();
  }

private:

  // Check if the dma performs memory copy inbound to the for loop with help from dependency graph
  bool isIncomingDmaOp(air::DmaMemcpyInterface dma_op) const {
    bool foundScfForDep = false;
    bool foundMemrefAllocDep = false;
    Operation *current_op = dma_op.getOperation();
    air::AsyncOpInterface current_async_op = dyn_cast<air::AsyncOpInterface>(current_op);
    auto dependency_list = current_async_op.getAsyncDependencies();
    if (dependency_list.size()){
      for (auto dep_op : dependency_list){
        if (dep_op == current_op->getParentOfType<scf::ForOp>().getRegionIterArgs()[0]){
          // Found scf.forOp in upstream dependency
          foundScfForDep = true;
        }
        else if (auto region_op = dyn_cast<air::RegionOp>(dep_op.getDefiningOp())){
          // Found air.regionOp in upstream dependency
          auto child_op = &region_op.getRegion().front().getOperations().front();
          if (auto alloc_op = dyn_cast<memref::AllocOp>(child_op)){
            // Found memref.allocOp inside air.regionOp
            foundMemrefAllocDep = true;
          }
        }
      }
    }
    return foundScfForDep & foundMemrefAllocDep;
  }

  // Return the air.region op in the dma's dep list which contains memref.alloc op
  air::RegionOp getRegionOfAllocOpForDmaOp(air::DmaMemcpyInterface dma_op) const {
    Operation *current_op = dma_op.getOperation();
    air::AsyncOpInterface current_async_op = dyn_cast<air::AsyncOpInterface>(current_op);
    auto dependency_list = current_async_op.getAsyncDependencies();
    if (dependency_list.size()){
      for (auto dep_op : dependency_list){
        if (auto region_op = dyn_cast<air::RegionOp>(dep_op.getDefiningOp())){
          // Found air.regionOp in upstream dependency
          auto child_op = &region_op.getRegion().front().getOperations().front();
          if (auto alloc_op = dyn_cast<memref::AllocOp>(child_op)){
            // Found memref.allocOp inside air.regionOp
            return region_op;
          }
        }
      }
    }
    return nullptr;
  }

  // Check if the dma performs memory copy outbound to the for loop with help from dependency graph
  bool isOutgoingDmaOp(air::DmaMemcpyInterface dma_op) const {
    bool foundDepToWaitall = false;
    bool foundDepToMemrefDealloc = false;
    Operation *current_op = dma_op.getOperation();
    air::AsyncOpInterface current_async_op = dyn_cast<air::AsyncOpInterface>(current_op);
    auto dependency_token = current_async_op.getAsyncToken();
    for (auto user : dependency_token.getUsers()){
      if (auto region_op = dyn_cast<air::RegionOp>(user)){
        // Found air.regionOp in downstream dependency
        auto child_op = &region_op.getRegion().front().getOperations().front();
        if (auto dealloc_op = dyn_cast<memref::DeallocOp>(child_op)){
          // Found memref.deallocOp inside air.regionOp
          foundDepToMemrefDealloc = true;
          for (auto descendant_user : region_op.getAsyncToken().getUsers()){
            if (dyn_cast<air::WaitAllOp>(descendant_user)){
              foundDepToWaitall = true;
            }
          }
        }
      }
    }
    return foundDepToWaitall & foundDepToMemrefDealloc;
  }

  // Return the air.region op in the dma's downstream which contains memref.dealloc op
  air::RegionOp getRegionOfDeallocOpForDmaOp(air::DmaMemcpyInterface dma_op) const {
    Operation *current_op = dma_op.getOperation();
    air::AsyncOpInterface current_async_op = dyn_cast<air::AsyncOpInterface>(current_op);
    auto dependency_token = current_async_op.getAsyncToken();
    for (auto user : dependency_token.getUsers()){
      if (auto region_op = dyn_cast<air::RegionOp>(user)){
        // Found air.regionOp in downstream dependency
        auto child_op = &region_op.getRegion().front().getOperations().front();
        if (auto dealloc_op = dyn_cast<memref::DeallocOp>(child_op)){
          // Found memref.deallocOp inside air.regionOp
          return region_op;
        }
      }
    }
    return nullptr;
  }

  // Reconnect incoming DMA event in the dependency graph
  void reconnectIncomingDma(air::DmaMemcpyInterface dma_op, scf::ForOp for_op) const {
    Operation *current_op = dma_op.getOperation();
    air::AsyncOpInterface dma_async_op = dyn_cast<air::AsyncOpInterface>(current_op);
    auto dependency_list = dma_async_op.getAsyncDependencies();
    if (dependency_list.size()){
      for (unsigned i = 0; i < dependency_list.size(); i++){
        if (dependency_list[i] == for_op.getRegionIterArgs()[0]){
          // Found scf.forOp in upstream dependency
          dma_async_op.eraseAsyncDependency(i);
        }
      }
      auto for_op_iter_operand = for_op.getIterOperands()[0];
      dma_op->getResult(0).replaceAllUsesWith(for_op.getRegionIterArgs()[0]);
      // for_op_iter_operand.replaceAllUsesWith(dma_op->getResult(0));
      
      replaceAllUsesInRegionWith(for_op_iter_operand,
                                  dma_op->getResult(0),
                                  *for_op->getParentRegion());
      dma_async_op.addAsyncDependency(for_op_iter_operand);
    }
  }

  // Reconnect outgoing DMA and dealloc events in the dependency graph
  void reconnectOutgoingEvents(air::DmaMemcpyInterface dma_op, air::RegionOp dealloc_op, scf::ForOp for_op, air::WaitAllOp wait_all_after_for) const {
    Operation *current_op = dma_op.getOperation();
    air::AsyncOpInterface dma_async_op = dyn_cast<air::AsyncOpInterface>(current_op);
    auto dependency_list = dma_async_op.getAsyncDependencies();
    if (dependency_list.size()){
      for (unsigned i = 0; i < dependency_list.size(); i++){
        wait_all_after_for.addAsyncDependency(dependency_list[i]);
      }
      for (unsigned i = 0; i < dependency_list.size(); i++){
        dma_async_op.eraseAsyncDependency(i);
      }
    }
    dependency_list = wait_all_after_for.getAsyncDependencies();
    for (unsigned i = 0; i < dependency_list.size(); i++){
      if (dependency_list[i] == dealloc_op.getResult(0)){
        wait_all_after_for.eraseAsyncDependency(i);
      }
    }
    for_op.getResult(0).replaceAllUsesWith(dealloc_op.getResult(0));
    dma_async_op.addAsyncDependency(for_op.getResult(0));
  }

  bool areEqualIndices (mlir::Value index_0, mlir::Value index_1) const {
    if (index_0 == nullptr || index_1 == nullptr) {
      // Note: memref with index is subset to memref without index (i.e. the entire memref)
      return true;
    }
    else {
      if (index_0 == index_1) return true;
      else if (!index_0.getDefiningOp()) return false;
      else if (!index_1.getDefiningOp()) return false;
      else {
        auto index_0_const_op = dyn_cast<arith::ConstantOp>(index_0.getDefiningOp());
        auto index_1_const_op = dyn_cast<arith::ConstantOp>(index_1.getDefiningOp());
        if (index_0_const_op.getValue() == index_1_const_op.getValue()) return true;
        else return false;
      }
    }
  }

  // Check if an operation is invariant with respect to for loop iteration
  bool isInvariantWRTForLoop(Operation * op, scf::ForOp for_op) const {
    for (auto op_operand : op->getOperands()){
      if (op_operand == for_op.getInductionVar()){
        return false;
      }
      if (op_operand.getDefiningOp() && isa<memref::SubViewOp>(op_operand.getDefiningOp())){
        auto subview_op = dyn_cast<memref::SubViewOp>(op_operand.getDefiningOp());
        for (auto subview_operand : subview_op->getOperands()){
          if (subview_operand == for_op.getInductionVar()){
            return false;
          }
        }
      }
    }
    return true;
  }

  // Check if two dma ops are symmetric
  bool areSymmetricDmaOps(air::DmaMemcpyInterface op_1, air::DmaMemcpyInterface op_2) const {
    bool areSymmetric = op_1.getSrcMemref() == op_2.getDstMemref();
    areSymmetric &= op_2.getSrcMemref() == op_1.getDstMemref();
    if (op_1.getNumDims() == 0 && op_2.getNumDims() == 0){
      // If both dma ops are nd dmas, then proceed to check offsets, sizes and strides
      auto op_1_dmaNd = dyn_cast<air::DmaMemcpyNdOp>(op_1.getOperation());
      auto op_2_dmaNd = dyn_cast<air::DmaMemcpyNdOp>(op_2.getOperation());
      unsigned op_1_dst_num_entries = op_1_dmaNd.getDstOffsets().size();
      unsigned op_1_src_num_entries = op_1_dmaNd.getSrcOffsets().size();
      unsigned op_2_dst_num_entries = op_2_dmaNd.getDstOffsets().size();
      unsigned op_2_src_num_entries = op_2_dmaNd.getSrcOffsets().size();
      if (areSymmetric && (op_1_dst_num_entries == op_2_src_num_entries) 
              && (op_1_src_num_entries == op_2_dst_num_entries)){
        for (unsigned i = 0; i < op_1_dst_num_entries; i++){
          areSymmetric &= areEqualIndices(op_1_dmaNd.getDstOffsets()[i], op_2_dmaNd.getSrcOffsets()[i]);
          areSymmetric &= areEqualIndices(op_1_dmaNd.getDstSizes()[i], op_2_dmaNd.getSrcSizes()[i]);
          areSymmetric &= areEqualIndices(op_1_dmaNd.getDstStrides()[i], op_2_dmaNd.getSrcStrides()[i]);
        }
        for (unsigned i = 0; i < op_1_src_num_entries; i++){
          areSymmetric &= areEqualIndices(op_1_dmaNd.getSrcOffsets()[i], op_2_dmaNd.getDstOffsets()[i]);
          areSymmetric &= areEqualIndices(op_1_dmaNd.getSrcSizes()[i], op_2_dmaNd.getDstSizes()[i]);
          areSymmetric &= areEqualIndices(op_1_dmaNd.getSrcStrides()[i], op_2_dmaNd.getDstStrides()[i]);
        }
      }
      else {
        areSymmetric = false;
      }
    }
    else if (op_1.getNumDims() == op_2.getNumDims()){
      // If both dma ops are of same dma type but not nd dmas, then proceed to check memrefdims etc
      for (unsigned i = 0; i < op_1.getNumDims(); i++){
        areSymmetric &= op_1.getSrcMemrefDim(i) == op_2.getDstMemrefDim(i);
        areSymmetric &= op_2.getSrcMemrefDim(i) == op_1.getDstMemrefDim(i);
      }
      areSymmetric &= op_1.getLength() == op_2.getLength();
    }
    else {
      // Two dma ops having different # of dimensions
      areSymmetric = false;
    }

    return areSymmetric;

  }
};

class AIRDependencyScheduleOpt : public AIRDependencyScheduleOptBase<AIRDependencyScheduleOpt> {

public:
  AIRDependencyScheduleOpt() = default;
  AIRDependencyScheduleOpt(const AIRDependencyScheduleOpt &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {  
    registry.insert<scf::SCFDialect, air::airDialect>();
  }

  // Trace dma ops' dependency to loop induction variables
  void getDmaOpLoopDependency(func::FuncOp f) {
    f.walk([&](Operation *op) {
      if (auto dma_op = mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op)){
        if (dma_op->getParentOfType<xilinx::air::HerdLaunchOp>()){
          // Start recursively tracing for loop induction variables
          dma_op_history.push_back(dma_op);
          SmallVector<Value, 1> loop_dep;
          traceDependentInductionVar(dma_op, loop_dep);
          dma_op_loop_dep.push_back(loop_dep);
        }
      }
    });
  }

  // Collect all for, parallel and hl loops in funcOp
  void pushLoopOpsToHistory(func::FuncOp f) {
    f.walk([&](Operation *op) {
      if (auto for_op = dyn_cast<scf::ForOp>(op)){
        for_op_history.push_back(for_op);
      }
      else if (auto parallel_op = dyn_cast<scf::ParallelOp>(op)){
        parallel_op_history.push_back(parallel_op);
      }
      else if (auto hl_op = dyn_cast<air::HerdLaunchOp>(op)){
        hl_op_history.push_back(hl_op);
      }
    });
  }

  // Detect boradcast opportunity based on dependency to loops
  void broadcastDetection() {
    for (unsigned i = 0; i < dma_op_history.size(); i++){
      auto dma_op = dma_op_history[i];
      SmallVector<Value, 1> loop_dep = dma_op_loop_dep[i];
      auto hl_op = dma_op->getParentOfType<air::HerdLaunchOp>();
      bool hasDepInHerdRows = false;
      bool hasDepInHerdCols = false;
      // Create an affine set to represent the broadcast pattern
      auto ctx = dma_op->getContext();
      for (auto v : loop_dep){
        if (v == hl_op.getTileIds().x){
          hasDepInHerdRows = true;
        }
        if (v == hl_op.getTileIds().y){
          hasDepInHerdCols = true;
        }
      }

      if (hasDepInHerdRows && !hasDepInHerdCols){
        auto numRowsOp = dyn_cast<arith::ConstantIndexOp>(hl_op.getHerdSizeOperands().x.getDefiningOp());
        auto numRows = numRowsOp.value();
        if (numRows > 1){
          SmallVector<AffineExpr, 2> constraints{getAffineDimExpr(0, ctx) - getAffineSymbolExpr(0, ctx),
                                                getAffineDimExpr(1, ctx),
                                                getAffineSymbolExpr(0, ctx),
                                                numRows - 1 - getAffineSymbolExpr(0, ctx)};
          SmallVector<bool, 2> eqflags{true, false, false, false};
          auto int_set = IntegerSet::get(2, 1, constraints, eqflags);
          dma_op->setAttr("broadcast_pattern",
                  mlir::IntegerSetAttr::get(int_set));
        }
      }
      else if (!hasDepInHerdRows && hasDepInHerdCols){
        auto numColsOp = dyn_cast<arith::ConstantIndexOp>(hl_op.getHerdSizeOperands().y.getDefiningOp());
        auto numCols = numColsOp.value();
        if (numCols > 1){
          SmallVector<AffineExpr, 2> constraints{getAffineDimExpr(0, ctx),
                                                getAffineDimExpr(1, ctx) - getAffineSymbolExpr(0, ctx),
                                                getAffineSymbolExpr(0, ctx),
                                                numCols - 1 - getAffineSymbolExpr(0, ctx)};
          SmallVector<bool, 2> eqflags{false, true, false, false};
          auto int_set = IntegerSet::get(2, 1, constraints, eqflags);
          dma_op->setAttr("broadcast_pattern",
                  mlir::IntegerSetAttr::get(int_set));
        }
      }
    }
  }

  void runOptPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<HoistDmaInAccumPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  void runBroadcastPattern(func::FuncOp funcOp){
    // Trace dma ops' dependency to loop induction variables
    // This info will be used for broadcast detection
    pushLoopOpsToHistory(funcOp);
    getDmaOpLoopDependency(funcOp);
    broadcastDetection();
  }

  void runOnFunction(func::FuncOp f) {
    runOptPatterns(f);
    runBroadcastPattern(f);
  }

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps)
      runOnFunction(f);
  }

private:

  // DMA dependency to loop induction variables
  std::vector<air::DmaMemcpyInterface> dma_op_history;
  SmallVector<SmallVector<Value, 1>, 1> dma_op_loop_dep;

  // Loop op history
  std::vector<scf::ForOp> for_op_history;
  std::vector<scf::ParallelOp> parallel_op_history;
  std::vector<air::HerdLaunchOp> hl_op_history;

  // Recursively check for dependency to loop induction vars arising from dma src
  void traceDependentInductionVar (air::DmaMemcpyInterface async_op, SmallVector<Value, 1> &loop_dep) {
    // Check for immediate dependency to loop induction vars
    SmallVector<Value, 1> candidate_scalar_operands;
    for (unsigned i = 0; i < async_op.getNumDims(); i++){
      candidate_scalar_operands.push_back(async_op.getSrcMemrefDim(i));
    }
    if (auto dmaNd_op = dyn_cast<air::DmaMemcpyNdOp>(async_op.getOperation())){
      for (unsigned i = 0; i < dmaNd_op.getSrcOffsets().size(); i++){
        candidate_scalar_operands.push_back(dmaNd_op.getSrcOffsets()[i]);
        candidate_scalar_operands.push_back(dmaNd_op.getSrcSizes()[i]);
        candidate_scalar_operands.push_back(dmaNd_op.getSrcStrides()[i]);
      }
    }
    for (auto operand : candidate_scalar_operands){
      // If parent loop op is an scf.for
      for (auto for_op : for_op_history){
        auto induction_var = for_op.getInductionVar();
        if (operand == induction_var) {
          loop_dep.push_back(induction_var);
        }
      }
      // TODO: Assuming that src.parallel won't exist under herd launch
      // If parent loop op is an scf.parallel
      // for (auto parallel_op : parallel_op_history){
      //   for (auto induction_var : parallel_op.getInductionVars()){
      //     if (operand == induction_var) {
      //       loop_dep.push_back(induction_var);
      //       op->setAttr("debug_parallel",
      //               mlir::StringAttr::get(op->getContext(), "found"));
      //     }
      //   }
      // }
      // If parent loop op is an air.launch_herd
      for (auto hl_op : hl_op_history){
        if (operand == hl_op.getTileIds().x) {
          loop_dep.push_back(hl_op.getTileIds().x);
        }
        else if (operand == hl_op.getTileIds().y) {
          loop_dep.push_back(hl_op.getTileIds().y);
        }
      }
    }

    // Recursively trace dependency to loop induction vars
    for (auto operand : candidate_scalar_operands){
      if (operand && operand.getType().isa<IndexType>()){ // Only tracing scalar operands
        if (operand.getDefiningOp() && mlir::dyn_cast<air::AsyncOpInterface>(operand.getDefiningOp())){
          auto ancestor_async_op = dyn_cast<air::AsyncOpInterface>(operand.getDefiningOp());
          traceDependentInductionVar(ancestor_async_op, loop_dep);
        }
        else {
          // Trace dependency through a for loop
          for (auto for_op : for_op_history){
            if (operand == for_op.getRegionIterArgs()[0]){
              auto ancestor_token = for_op.getIterOperands()[0];
              auto ancestor_async_op = dyn_cast<air::AsyncOpInterface>(ancestor_token.getDefiningOp());
              traceDependentInductionVar(ancestor_async_op, loop_dep);
            }
          }
          // Trace dependency through a parallel loop
          // TODO: decide if parallel should exist in herd launch
        }
      }
    }
  }

  // Recursively check for dependency to any loop induction vars
  void traceDependentInductionVar (air::AsyncOpInterface async_op, SmallVector<Value, 1> &loop_dep) {
    // Get child op if async_op is air.region
    Operation * op = nullptr;
    if (auto air_region_op = dyn_cast<air::RegionOp>(async_op.getOperation())){
      assert(air_region_op.body().front().getOperations().size() == 2 
              && "air::RegionOp should have only one child operation beside the terminator");
      for (auto &child_op : air_region_op.body().front().getOperations()){
        if (!dyn_cast<air::RegionTerminatorOp>(child_op)) op = &child_op;
      }
    }
    else {
      op = async_op.getOperation();
    }

    // Check for immediate dependency to loop induction vars
    for (auto operand : op->getOperands()){
        // If parent loop op is an scf.for
        for (auto for_op : for_op_history){
          auto induction_var = for_op.getInductionVar();
          if (operand == induction_var) {
            loop_dep.push_back(induction_var);
          }
        }
        // If parent loop op is an scf.parallel
        for (auto parallel_op : parallel_op_history){
          for (auto induction_var : parallel_op.getInductionVars()){
            if (operand == induction_var) {
              loop_dep.push_back(induction_var);
            }
          }
        }
        // If parent loop op is an air.launch_herd
        for (auto hl_op : hl_op_history){
          if (operand == hl_op.getTileIds().x) {
            loop_dep.push_back(hl_op.getTileIds().x);
          }
          else if (operand == hl_op.getTileIds().y) {
            loop_dep.push_back(hl_op.getTileIds().y);
          }
        }
    }

    // Recursively trace dependency to loop induction vars
    for (auto operand : op->getOperands()){
      if (operand && operand.getType().isa<IndexType>()){ // Only tracing scalar operands
        if (operand.getDefiningOp() && mlir::dyn_cast<air::AsyncOpInterface>(operand.getDefiningOp())){
          auto ancestor_async_op = dyn_cast<air::AsyncOpInterface>(operand.getDefiningOp());
          traceDependentInductionVar(ancestor_async_op, loop_dep);
        }
        else {
          // Trace dependency through a for loop
          for (auto for_op : for_op_history){
            if (operand == for_op.getRegionIterArgs()[0]){
              auto ancestor_token = for_op.getIterOperands()[0];
              auto ancestor_async_op = dyn_cast<air::AsyncOpInterface>(ancestor_token.getDefiningOp());
              traceDependentInductionVar(ancestor_async_op, loop_dep);
            }
          }
          // Trace dependency through a parallel loop
          // TODO: decide if parallel should exist in herd launch
        }
      }
    }
  }

};

}// namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRDependencyScheduleOptPass() {
  return std::make_unique<AIRDependencyScheduleOpt>();
}

} // namespace air
} // namespace xilinx