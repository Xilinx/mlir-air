//===- Dependency.cpp -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

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
        if (auto index_0_const_op = dyn_cast<arith::ConstantOp>(index_0.getDefiningOp())){
          if (auto index_1_const_op = dyn_cast<arith::ConstantOp>(index_1.getDefiningOp())){
            if (index_0_const_op.getValue() == index_1_const_op.getValue()) return true;
          }
        }
        return false;
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
    // Get child op if async_op is air.execute
    Operation * op = nullptr;
    if (auto air_region_op =
            dyn_cast<air::ExecuteOp>(async_op.getOperation())) {
      assert(air_region_op.body().front().getOperations().size() == 2 &&
             "air::ExecuteOp should have only one child operation beside the "
             "terminator");
      for (auto &child_op : air_region_op.body().front().getOperations()){
        if (!dyn_cast<air::ExecuteTerminatorOp>(child_op))
          op = &child_op;
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

  // Returns the scf parent op from scf.yield op
  template <typename T>
  Operation * getScfParentOpFromYieldOp(scf::YieldOp op){
    if (auto scfop = dyn_cast<T>(op->getParentOp())){
      return scfop.getOperation();
    }
    return nullptr;
  }

  //===----------------------------------------------------------------------===//
  // Dependency graph as a Boost graph object
  //===----------------------------------------------------------------------===//

  void dependencyCanonicalizer::parseCommandGraphs(func::FuncOp &toplevel, dependencyGraph &global_graph, dependencyContext &dep_ctx) {

    // Create vertices for graphs
    // Build up host graph
    toplevel.walk([&](Operation *op) {
      if (!op->getParentOfType<air::LaunchOp>()){
        addVertexFromOpImpls(op, global_graph.g, dep_ctx);
        if (auto launch = dyn_cast<air::LaunchOp>(op)){
          // Build up launch graph
          global_graph.subgraphs.push_back(dependencyGraph(launch.getOperation()));
          dependencyGraph * current_launch_graph = &(global_graph.subgraphs.back());

          launch.walk([&](Operation *launch_childop) {
            if (!launch_childop->getParentOfType<air::PartitionOp>()
                && !dyn_cast<air::LaunchOp>(launch_childop)){
              addVertexFromOpImpls(launch_childop, current_launch_graph->g, dep_ctx);
              if (auto partition = dyn_cast<air::PartitionOp>(launch_childop)){
                // Build up partition graph
                current_launch_graph->subgraphs.push_back(dependencyGraph(partition.getOperation()));
                dependencyGraph * current_part_graph = &(current_launch_graph->subgraphs.back());

                partition.walk([&](Operation *part_childop) {  
                  if (!part_childop->getParentOfType<air::HerdOp>()
                      && !dyn_cast<air::PartitionOp>(part_childop)){
                    addVertexFromOpImpls(part_childop, current_part_graph->g, dep_ctx);
                    if (auto herd = dyn_cast<air::HerdOp>(part_childop)){
                      // Build up herd graph
                      current_part_graph->subgraphs.push_back(dependencyGraph(herd.getOperation()));
                      dependencyGraph * current_herd_graph = &(current_part_graph->subgraphs.back());

                      herd.walk([&](Operation *herd_childop) {
                        if (!dyn_cast<air::HerdOp>(herd_childop)){
                          addVertexFromOpImpls(herd_childop, current_herd_graph->g, dep_ctx);
                        }
                      });
                    }
                  }
                });

              }
            }
          });

        } 
      }
    });

    // Adds edges between async ops
    parseDependencyEdgesInGraph(global_graph.g, dep_ctx);
    for (auto &G_l : global_graph.subgraphs){
      parseDependencyEdgesInGraph(G_l.g, dep_ctx);
      for (auto &G_p : G_l.subgraphs){
        parseDependencyEdgesInGraph(G_p.g, dep_ctx);
        for (auto &G_h : G_p.subgraphs){
          parseDependencyEdgesInGraph(G_h.g, dep_ctx);
        }
      }
    }

    // Connect leaf vertices to launch, partition and herd terminators
    for (auto &G_l : global_graph.subgraphs){
      connectTerminatorInGraph(G_l.g);
      for (auto &G_p : G_l.subgraphs){
        connectTerminatorInGraph(G_p.g);
        for (auto &G_h : G_p.subgraphs){
          connectTerminatorInGraph(G_h.g);
        }
      }
    }

    // Connect the start node per graph as graph inception point;
    // update pointer from graph to air.hierarchy op terminators
    connectStartNodeInCommandGraph(global_graph);
    updatePointerFromGraphToHierarchyTerminator(global_graph);
    updatePointerFromHierarchyOpToGraph(global_graph);
    for (auto &launchGraph : global_graph.subgraphs){
      connectStartNodeInCommandGraph(launchGraph);
      updatePointerFromGraphToHierarchyTerminator(launchGraph);
      updatePointerFromHierarchyTerminatorToGraph(global_graph, launchGraph);
      updatePointerFromHierarchyOpToGraph(launchGraph);
      for (auto &partitionGraph : launchGraph.subgraphs){
        connectStartNodeInCommandGraph(partitionGraph);
        updatePointerFromGraphToHierarchyTerminator(partitionGraph);
        updatePointerFromHierarchyTerminatorToGraph(launchGraph, partitionGraph);
        updatePointerFromHierarchyOpToGraph(partitionGraph);
        for (auto &herdGraph : partitionGraph.subgraphs){
          connectStartNodeInCommandGraph(herdGraph);
          updatePointerFromGraphToHierarchyTerminator(herdGraph);
        }
      }
    }

    // Dump dot graphs
    dump_graph("host.dot", global_graph.g);
    int i = 0;
    for (auto G_l : global_graph.subgraphs){
      std::string name = "launch" + std::to_string(++i) + ".dot";
      dump_graph(name, G_l.g);
      int j = 0;
      for (auto G_p : G_l.subgraphs){
        std::string name = "partition" + std::to_string(i) + "_" + std::to_string(++j) + ".dot";
        dump_graph(name, G_p.g);
        int k = 0;
        for (auto G_h : G_p.subgraphs){
          std::string name = "herd" + std::to_string(i) + "_" + std::to_string(j) + "_" + std::to_string(++k) + ".dot";
          dump_graph(name, G_h.g);
        }
      }
    }

  }

  Graph::vertex_descriptor dependencyCanonicalizer::addVertexFromOpImpls(Operation * op, Graph &G, dependencyContext &dep_ctx){
    if (auto dma_op = mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op)){
      return addVertexFromDmaOp(dma_op, G, dep_ctx);
    }
    else if (auto execute_op = dyn_cast<xilinx::air::ExecuteOp>(op)){
      return addVertexFromExecuteOp(execute_op, G, dep_ctx);
    }
    else if (auto wa_op = dyn_cast<xilinx::air::WaitAllOp>(op)){
      return addVertexFromWaitAllOp(wa_op, G, dep_ctx);
    }
    else if (auto forop = dyn_cast<scf::ForOp>(op)){
      return addVertexFromOp(op, dep_ctx.ForOpID, "for_loop", "ScfForOp", "crimson", "box", G, dep_ctx);
    }
    else if (auto parallelop = dyn_cast<scf::ParallelOp>(op)){
      return addVertexFromOp(op, dep_ctx.ParallelOpID, "parallel_loop", "ScfParallelOp", "crimson", "box", G, dep_ctx);
    }
    else if (auto hier_op = mlir::dyn_cast<xilinx::air::HierarchyInterface>(op)){
      return addVertexFromHierarchyOp(hier_op, G, dep_ctx);
    }
    else if (op->mightHaveTrait<OpTrait::IsTerminator>()){
      return addVertexFromTerminatorOp(op, G, dep_ctx);
    }
    else return 0;
  }

  // Create graph vertex from op
  Graph::vertex_descriptor dependencyCanonicalizer::addVertexFromOp(Operation * op, uint64_t &id, std::string event_type, std::string event_name, std::string color, std::string shape, Graph &G, dependencyContext &dep_ctx, Operation * pointer_op){
    op->setAttr("id",
        mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32),
        ++id));
    auto v = add_vertex(G);
    G[v].asyncEventName = event_name;
    G[v].asyncEventType = event_type;
    G[v].color = color;
    G[v].shape = shape;
    G[v].operationId = id;
    if (pointer_op) G[v].op = pointer_op;
    else G[v].op = op;
    // Update op-to-graph mapping
    auto entry = make_pair(event_type, id);
    dep_ctx.op_to_v.insert(make_pair(entry, v));
    dep_ctx.op_to_g.insert(make_pair(entry, &G));
    return v;
  }

  Graph::vertex_descriptor dependencyCanonicalizer::addVertexFromDmaOp(xilinx::air::DmaMemcpyInterface op, Graph &G, dependencyContext &dep_ctx){
    if (dyn_cast<xilinx::air::DmaMemcpyNdOp>(op.getOperation())){
      return addVertexFromOp(op, dep_ctx.DmaOpID, "dma", "DmaMemcpyNdOp", "cyan", "oval", G, dep_ctx);
    }
    else {
      assert(false && "Unknown dma op");
      return 0;
    }
  }

  Graph::vertex_descriptor dependencyCanonicalizer::addVertexFromHierarchyOp(xilinx::air::HierarchyInterface op, Graph &G, dependencyContext &dep_ctx){
    if (dyn_cast<xilinx::air::LaunchOp>(op.getOperation())){
      return addVertexFromOp(op, dep_ctx.HierarchyOpID, "hierarchy", "LaunchOp", "yellow", "box", G, dep_ctx);
    }
    else if (dyn_cast<xilinx::air::PartitionOp>(op.getOperation())){
      return addVertexFromOp(op, dep_ctx.HierarchyOpID, "hierarchy", "PartitionOp", "yellow", "box", G, dep_ctx);
    }
    else if (dyn_cast<xilinx::air::HerdOp>(op.getOperation())){
      return addVertexFromOp(op, dep_ctx.HierarchyOpID, "hierarchy", "HerdOp", "yellow", "box", G, dep_ctx);
    }
    else {
      assert(false && "Unknown hierarchy op");
      return 0;
    }
  }

  Graph::vertex_descriptor dependencyCanonicalizer::addVertexFromTerminatorOp(Operation * op, Graph &G, dependencyContext &dep_ctx){
    if (dyn_cast<xilinx::air::LaunchTerminatorOp>(op)){
      return addVertexFromOp(op, dep_ctx.TerminatorID, "hierarchy_terminator", "LaunchTerminator", "yellow", "box", G, dep_ctx);
    }
    else if (dyn_cast<xilinx::air::PartitionTerminatorOp>(op)){
      return addVertexFromOp(op, dep_ctx.TerminatorID, "hierarchy_terminator", "PartitionTerminator", "yellow", "box", G, dep_ctx);
    }
    else if (dyn_cast<xilinx::air::HerdTerminatorOp>(op)){
      return addVertexFromOp(op, dep_ctx.TerminatorID, "hierarchy_terminator", "HerdTerminator", "yellow", "box", G, dep_ctx);
    }
    else if (auto yieldop = dyn_cast<scf::YieldOp>(op)){
      if (getScfParentOpFromYieldOp<scf::ParallelOp>(yieldop)){
        return addVertexFromOp(op, dep_ctx.TerminatorID, "terminator", "ScfParallelYieldOp", "crimson", "box", G, dep_ctx);
      }
      else if (getScfParentOpFromYieldOp<scf::ForOp>(yieldop)){
        return addVertexFromOp(op, dep_ctx.TerminatorID, "terminator", "ScfForYieldOp", "crimson", "box", G, dep_ctx);
      }
    }
    return 0;
  }

  Graph::vertex_descriptor dependencyCanonicalizer::addVertexFromExecuteOp(xilinx::air::ExecuteOp op, Graph &G, dependencyContext &dep_ctx){
    int iter_count = 0;
    Graph::vertex_descriptor v_prev = 0;
    Graph::vertex_descriptor v = 0;
    Operation * pointer_op = op;
    for (auto &child_op : op->getRegions().front().getOps()){
      if (dyn_cast<linalg::LinalgOp>(child_op)){
        v = addVertexFromOp(&child_op, dep_ctx.ExecuteOpID, "execute", "LinalgOp", "chartreuse", "oval", G, dep_ctx, pointer_op);
      }
      else if (dyn_cast<memref::AllocOp>(child_op)){
        v = addVertexFromOp(&child_op, dep_ctx.ExecuteOpID, "execute", "AllocOp", "chartreuse", "oval", G, dep_ctx, pointer_op);
      }
      else if (dyn_cast<memref::DeallocOp>(child_op)){
        v = addVertexFromOp(&child_op, dep_ctx.ExecuteOpID, "execute", "DeallocOp", "chartreuse", "oval", G, dep_ctx, pointer_op);
      }
      else if (dyn_cast<memref::CopyOp>(child_op)){
        v = addVertexFromOp(&child_op, dep_ctx.ExecuteOpID, "execute", "CopyOp", "chartreuse", "oval", G, dep_ctx, pointer_op);
      }
      else if (dyn_cast<mlir::AffineApplyOp>(child_op)){
        v = addVertexFromOp(&child_op, dep_ctx.ExecuteOpID, "execute", "AffineApplyOp", "chartreuse", "oval", G, dep_ctx, pointer_op);
      }
      else if (dyn_cast<xilinx::air::ExecuteTerminatorOp>(child_op)){
        v = addVertexFromOp(&child_op, dep_ctx.ExecuteOpID, "execute", "ExecuteTerminatorOp", "chartreuse", "oval", G, dep_ctx, pointer_op);
      }
      else if (dyn_cast<arith::MulIOp>(child_op)){
        v = addVertexFromOp(&child_op, dep_ctx.ExecuteOpID, "execute", "MuliOp", "chartreuse", "oval", G, dep_ctx, pointer_op);
      }
      else if (dyn_cast<arith::AddIOp>(child_op)){
        v = addVertexFromOp(&child_op, dep_ctx.ExecuteOpID, "execute", "AddIOp", "chartreuse", "oval", G, dep_ctx, pointer_op);
      }
      else {
        assert(false && "Unknown op in execute");
      }
      // Make connections within execute
      if (iter_count > 0){
        add_edge(v_prev, v, G);
        pointer_op = nullptr;
      }
      v_prev = v;
      iter_count++;
    }
    return v;
  }

  Graph::vertex_descriptor dependencyCanonicalizer::addVertexFromWaitAllOp(xilinx::air::WaitAllOp op, Graph &G, dependencyContext &dep_ctx){
    if (op.getAsyncToken().hasOneUse()){
      for (auto u : op.getAsyncToken().getUsers()){
        if (dyn_cast<scf::YieldOp>(u)){
          return addVertexFromOp(op, dep_ctx.WaitAllOpID, "wait_all", "WaitAllOp", "crimson", "oval", G, dep_ctx);
        }
        else if (dyn_cast<scf::ReduceReturnOp>(u)){
          return addVertexFromOp(op, dep_ctx.WaitAllOpID, "wait_all", "WaitAllOp", "crimson", "oval", G, dep_ctx);
        }
      }
    }
    return addVertexFromOp(op, dep_ctx.WaitAllOpID, "wait_all", "WaitAllOp", "crimson", "oval", G, dep_ctx);
  }

  // Get type-id pair from op, which will be used to look up vertex in op_to_v
  std::pair<std::string, unsigned> dependencyCanonicalizer::getTypeIdPairFromOp(Operation * op){
    std::pair<std::string, unsigned> output;
    std::string type = getOpTypeFromOpImpls(op);
    output.first = type;
    output.second = xilinx::air::getIdAttr(op);
    return output;
  }

  std::string dependencyCanonicalizer::getOpTypeFromOpImpls(Operation * op){
    if (mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op)){
      return "dma";
    }
    else if (dyn_cast<xilinx::air::WaitAllOp>(op)){
      return "wait_all";
    }
    else if (dyn_cast<xilinx::air::HierarchyInterface>(op)){
      return "hierarchy";
    }
    else if (dyn_cast<scf::ForOp>(op)){
      return "for_loop";
    }
    else if (dyn_cast<scf::ParallelOp>(op)){
      return "parallel_loop";
    }
    else if (dyn_cast<xilinx::air::LaunchTerminatorOp>(op)){
      return "hierarchy_terminator";
    }
    else if (dyn_cast<xilinx::air::PartitionTerminatorOp>(op)){
      return "hierarchy_terminator";
    }
    else if (dyn_cast<xilinx::air::HerdTerminatorOp>(op)){
      return "hierarchy_terminator";
    }
    else if (dyn_cast<scf::YieldOp>(op)){
      return "terminator";
    }
    else {
      if (dyn_cast<xilinx::air::ExecuteOp>(op->getParentOp())){
        return "execute";
      }
      else return "";
    }
  }

  // Get vertex descriptor from op
  // "front_or_back": if op is an air.execute, then "front" returns the first op in region,
  // while "back" returns the terminator in region.
  std::pair<Graph::vertex_descriptor, Graph *> dependencyCanonicalizer::getVertexFromOp(Operation * op, dependencyContext dep_ctx, std::string front_or_back){
    std::pair<Graph::vertex_descriptor, Graph *> output;
    if (auto execute_op = dyn_cast<xilinx::air::ExecuteOp>(op)){
      if (front_or_back == "front"){
        auto execute_front_op = &(*op->getRegions().front().op_begin());
        std::pair<std::string, unsigned> entry_pair = getTypeIdPairFromOp(execute_front_op);
        output.first = dep_ctx.op_to_v[entry_pair];
        output.second = dep_ctx.op_to_g[entry_pair];
      }
      else if (front_or_back == "back"){
        auto execute_end_op = op->getRegions().front().getBlocks().front().getTerminator();
        std::pair<std::string, unsigned> entry_pair = getTypeIdPairFromOp(execute_end_op);
        output.first = dep_ctx.op_to_v[entry_pair];
        output.second = dep_ctx.op_to_g[entry_pair];
      }
      else {
        assert(false && "Unknown string operand (only accepts 'front' or 'back')");
      }
    }
    else {
      std::pair<std::string, unsigned> entry_pair = getTypeIdPairFromOp(op);
        output.first = dep_ctx.op_to_v[entry_pair];
        output.second = dep_ctx.op_to_g[entry_pair];
    }
    return output;
  }

  // Trace dependency of every op in a boost graph
  void dependencyCanonicalizer::parseDependencyEdgesInGraph(Graph &g, dependencyContext dep_ctx){
    auto vp = boost::vertices(g);
    for (auto vit = vp.first; vit != vp.second; ++vit){
      auto op = g[*vit].op;
      if (!op) continue;
      connectOpToItsDepListImpls(op, g, dep_ctx);
    }
  }

  void dependencyCanonicalizer::connectOpToItsDepListImpls(Operation * op, Graph &g, dependencyContext dep_ctx){
    SmallVector<Value, 1> dep_list;
    // air.asyncopinterface
    if (auto async_op = mlir::dyn_cast<xilinx::air::AsyncOpInterface>(op)){
      for (auto dep_token : async_op.getAsyncDependencies()){
        dep_list.push_back(dep_token);
      }
      connectOpToItsDepList(op, dep_list, g, dep_ctx);
    }
    // scf.for
    else if (auto forop = dyn_cast<scf::ForOp>(op)){
      for (auto iter_operand : forop.getIterOperands()){
        dep_list.push_back(iter_operand);
      }
      connectOpToItsDepList(op, dep_list, g, dep_ctx);
    }
    // scf.parallel
    else if (auto parallelop = dyn_cast<scf::ParallelOp>(op)){
      for (auto operand : parallelop->getOperands()){
        dep_list.push_back(operand);
      }
      connectOpToItsDepList(op, dep_list, g, dep_ctx);
    }
    // scf.yield
    else if (auto yieldop = dyn_cast<scf::YieldOp>(op)){
      for (auto operand : yieldop->getOperands()){
        dep_list.push_back(operand);
      }
      connectOpToItsDepList(op, dep_list, g, dep_ctx);
    }

  }

  // Connect an async op to ops in its dependency list
  void dependencyCanonicalizer::connectOpToItsDepList(Operation * op, SmallVector<Value, 1> dep_list, Graph &g, dependencyContext dep_ctx){
    auto dst_v = getVertexFromOp(op, dep_ctx, "front").first;
    if (dep_list.size()){
      for (auto dep_token : dep_list){
        auto src_vector = traceOpFromToken(dep_token);
        if (src_vector.size()){
          for (auto src_op : src_vector){
            auto src_v = getVertexFromOp(src_op, dep_ctx, "back").first;
            if (!edge(src_v, dst_v, g).second){
              add_edge(src_v, dst_v, g);
            }
          }
        }
      }
    }
  }

  // Trace op from a token in dependency list
  std::vector<Operation *> dependencyCanonicalizer::traceOpFromToken(Value dep_token){
    std::vector<Operation *> output;
    // If dependency token originates from async op
    if (dep_token.getDefiningOp() && mlir::dyn_cast<xilinx::air::AsyncOpInterface>(dep_token.getDefiningOp())){
      output.push_back(dep_token.getDefiningOp());
      return output;
    }
    // Else if dependency token is yielded from scf.for
    else if (dep_token.getDefiningOp() && dyn_cast<scf::ForOp>(dep_token.getDefiningOp())){
      auto forop = dyn_cast<scf::ForOp>(dep_token.getDefiningOp());
      auto forop_terminator = forop.getBody()->getTerminator();
      output.push_back(forop_terminator);
      return output;
    }
    // Else if dependency token is yielded from scf.parallel
    else if (dep_token.getDefiningOp() && dyn_cast<scf::ParallelOp>(dep_token.getDefiningOp())){
      auto parallelop = dyn_cast<scf::ParallelOp>(dep_token.getDefiningOp());
      for (auto parallelop_reduceop : parallelop.getOps<scf::ReduceOp>()){
        auto parallelop_terminator = parallelop_reduceop.getRegion().front().getTerminator();
        output.push_back(parallelop_terminator);
        return output;
      }
    }
    // Else if dependency token is the iter arg of an scf for loop
    else if (auto forop = getForRegionIterArgsOwner(dep_token)){
      output.push_back(forop);
      return output;
    }
    // Else if dependency token is from affine if (joint token from multiple ops)
    else if (dep_token.getDefiningOp() && dyn_cast<mlir::AffineIfOp>(dep_token.getDefiningOp())){
      auto aifop = dyn_cast<mlir::AffineIfOp>(dep_token.getDefiningOp());
      auto then_terminator = aifop.getThenBlock()->getTerminator();
      for (auto operand : then_terminator->getOperands()){
        if (auto op = operand.getDefiningOp()){
          output.push_back(op);
        }
      }
      auto else_terminator = aifop.getElseBlock()->getTerminator();
      for (auto operand : else_terminator->getOperands()){
        if (auto op = operand.getDefiningOp()){
          output.push_back(op);
        }
      }
      return output;
    }
    return output;
  }

  // Connects launch, partition and herd terminators
  void dependencyCanonicalizer::connectTerminatorInGraph(Graph &g){
    auto vp = boost::vertices(g);
    Graph::vertex_descriptor terminator_v = 0;
    for (auto vit = vp.first; vit != vp.second; ++vit){
      if (g[*vit].asyncEventType == "hierarchy_terminator"){
        terminator_v = *vit;
      }
    }
    if (terminator_v == 0) return;
    for (auto vit = vp.first; vit != vp.second; ++vit){
      if ((terminator_v != *vit) && !out_degree(*vit, g)
          && (g[*vit].asyncEventType != "start")){
        add_edge(*vit, terminator_v, g);
      }
    }
  }

  // Create start node for graph
  void dependencyCanonicalizer::connectStartNodeInCommandGraph (dependencyGraph &G){
    auto v = G.start_vertex;
    auto vp = boost::vertices(G.g);
    for (auto vit = vp.first; vit != vp.second; ++vit){
      if ((v != *vit) && !in_degree(*vit, G.g)){
        add_edge(v, *vit, G.g);
      }
    }
  }

  // Adds pointer from command graph to launch, partition and herd terminators
  void dependencyCanonicalizer::updatePointerFromGraphToHierarchyTerminator(dependencyGraph &G){
    auto vp = boost::vertices(G.g);
    for (auto v = vp.first; v != vp.second; ++v){
      if (G.g[*v].asyncEventType == "hierarchy_terminator"){
        G.terminator_vertex = *v;
        return;
      }
    }
  }

  // Adds pointer from hierarchy terminator to parent command graph
  void dependencyCanonicalizer::updatePointerFromHierarchyTerminatorToGraph(dependencyGraph &G, dependencyGraph &subG){
    auto vp = boost::vertices(subG.g);
    for (auto v = vp.first; v != vp.second; ++v){
      if (subG.g[*v].asyncEventType == "hierarchy_terminator"){
        subG.g[*v].nextDependencyGraph = &G;
        return;
      }
    }
  }

  // Adds pointer from hierarchy op to sub command graph
  void dependencyCanonicalizer::updatePointerFromHierarchyOpToGraph(dependencyGraph &G){
    unsigned idx = 0;
    auto vp = boost::vertices(G.g);
    for (auto v = vp.first; v != vp.second; ++v){
      if (G.g[*v].asyncEventType == "hierarchy"){
        G.g[*v].nextDependencyGraph = &(G.subgraphs[idx]);
        idx++;
      }
    }
    assert(idx == G.subgraphs.size() && "mismatch between # graphs and hierarchy ops");
  }

  // Dump graphviz
  void dependencyCanonicalizer::dump_graph(std::string filename, Graph G)
  {
    std::ofstream ofs (filename, std::ofstream::out); 
    boost::dynamic_properties dp;
    dp.property("label", boost::get(&dependencyNodeEntry::asyncEventName, G));
    dp.property("color", boost::get(&dependencyNodeEntry::color, G));
    dp.property("shape", boost::get(&dependencyNodeEntry::shape, G));
    dp.property("node_id", boost::get(boost::vertex_index, G));
    dp.property("style", boost::make_constant_property<Graph::vertex_descriptor>(+"filled"));
    write_graphviz_dp(ofs, G, dp);
  }

} // namespace air
} // namespace xilinx