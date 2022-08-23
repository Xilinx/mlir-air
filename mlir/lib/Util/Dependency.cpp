// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

#include "air/Util/Dependency.h"

#define DEBUG_TYPE "air-dependency-util"

using namespace mlir;

namespace xilinx {
namespace air {

  bool areEqualIndices (mlir::Value index_0, mlir::Value index_1) {
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

  // Recursively check for dependency to loop induction vars arising from dma src
  void traceDependentInductionVar (air::DmaMemcpyInterface async_op, SmallVector<Value, 1> &loop_dep_history, std::vector<Operation *> &op_history) {
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
      if (auto for_op = mlir::scf::getForInductionVarOwner(operand)){
        loop_dep_history.push_back(for_op.getInductionVar());
      }
      // TODO: Assuming that src.parallel won't exist under herd launch
      // If parent loop op is an scf.parallel
      
      // If parent loop op is an air.launch_herd
      if (auto hl_op = getHerdArgOwner(operand)){
        for (auto id : hl_op.getIds()){
          if (operand == id) {
            loop_dep_history.push_back(id);
          }
        }
      }
    }

    // Recursively trace dependency to loop induction vars
    for (auto operand : candidate_scalar_operands){
      if (operand && operand.getType().isa<IndexType>()){ // Only tracing scalar operands
        if (operand.getDefiningOp() && mlir::dyn_cast<air::AsyncOpInterface>(operand.getDefiningOp())){
          auto ancestor_async_op = dyn_cast<air::AsyncOpInterface>(operand.getDefiningOp());
          op_history.push_back(ancestor_async_op.getOperation());
          traceDependentInductionVar(ancestor_async_op, loop_dep_history, op_history);
        }
        else {
          // Trace dependency through a for loop
          if (auto for_op = getForRegionIterArgsOwner(operand)){
            for (auto iter_arg : for_op.getIterOperands()){
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
  }

  // Recursively check for dependency to any loop induction vars
  void traceDependentInductionVar (air::AsyncOpInterface async_op, SmallVector<Value, 1> &loop_dep_history, std::vector<Operation *> &op_history) {
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
      if (auto for_op = mlir::scf::getForInductionVarOwner(operand)){
        loop_dep_history.push_back(for_op.getInductionVar());
      }
      // If parent loop op is an scf.parallel
      if (auto parallel_op = mlir::scf::getParallelForInductionVarOwner(operand)){
        for (auto induction_var : parallel_op.getInductionVars()){
          if (operand == induction_var) {
            loop_dep_history.push_back(induction_var);
          }
        }
      }
      // If parent loop op is an air.launch_herd
      if (auto hl_op = getHerdArgOwner(operand)){
        for (auto id : hl_op.getIds()){
          if (operand == id) {
            loop_dep_history.push_back(id);
          }
        }
      }
    }

    // Recursively trace dependency to loop induction vars
    for (auto operand : op->getOperands()){
      if (operand && operand.getType().isa<IndexType>()){ // Only tracing scalar operands
        if (operand.getDefiningOp() && mlir::dyn_cast<air::AsyncOpInterface>(operand.getDefiningOp())){
          auto ancestor_async_op = dyn_cast<air::AsyncOpInterface>(operand.getDefiningOp());
          op_history.push_back(ancestor_async_op.getOperation());
          traceDependentInductionVar(ancestor_async_op, loop_dep_history, op_history);
        }
        else {
          // Trace dependency through a for loop
          if (auto for_op = getForRegionIterArgsOwner(operand)){
            for (auto iter_arg : for_op.getIterOperands()){
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
  }

  void eraseAsyncDependencyFromAsyncOp(xilinx::air::AsyncOpInterface op, Value token){
    assert(token && "input value is nullptr");
    assert(token.getType().isa<air::AsyncTokenType>() && "ssa value is not an async token");
    auto dependency_list = op.getAsyncDependencies();
    for (int i = dependency_list.size() - 1; i >= 0; i--){
      if (dependency_list[i] == token){
        op.eraseAsyncDependency(i);
      }
    }
  }

} // namespace air
} // namespace xilinx