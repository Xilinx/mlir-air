//===- Runner.cpp -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Util/Runner.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/CostModel.h"
#include "air/Util/Util.h"
#include "air/Util/Dependency.h"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/JSON.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/RegionUtils.h"

#include <deque>
#include <float.h>
#include <list>
#include <map>
#include <sstream>
#include <vector>
#include <algorithm>

// boost graph
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/copy.hpp>

#include <algorithm>
#include <numeric> 
#include <string>

#define DEBUG_TYPE "air-runner"

#define INDEX_WIDTH 32

using namespace mlir;
using namespace boost;

namespace xilinx {
namespace air {

struct runnerGraph : dependencyGraph
{
  runnerNode * runner_node;
  std::vector<runnerGraph> subgraphs;

  runnerGraph(mlir::Operation *op = nullptr, bool initStartVertex = false) {
    g = Graph();
    hierarchyOp = op;
    runner_node = nullptr;
    if (initStartVertex) {
      auto v = add_vertex(g);
      g[v].asyncEventType = "start";
      g[v].asyncEventName = "start";
      g[v].color = "yellow";
      g[v].shape = "box";
      start_vertex = v;
    }
  }

  ~runnerGraph() {
    g.clear();
    subgraphs.clear();
  }
};

struct runnerNode {
  dependencyGraph * ctrl_g;
  std::string runner_node_type;
  // Each entry is an std::pair. First element is vertex, and second element is thread id
  std::vector<std::pair<Graph::vertex_descriptor, unsigned>> wavefront;
  std::vector<Graph::vertex_descriptor> processed_vertices;
  // Each entry is an std::pair. First element is for op's id, and second element is counter
  std::vector<std::pair<unsigned, unsigned>> loop_trip_count;
  std::vector<runnerNode> sub_runner_nodes;

  // Private wavefront of each runner node, reserved to interface with resource model
  std::vector<dependencyNodeEntry *> wavefrontNodes() {
    std::vector<dependencyNodeEntry *> output;
    for (auto v : wavefront){
      output.push_back(&ctrl_g->g[v.first]);
    }
    return output;
  }

  runnerNode(dependencyGraph * ctrl_g = nullptr, std::string runner_node_type = "")
      : ctrl_g(ctrl_g), runner_node_type(runner_node_type) {}

  ~runnerNode(){
    wavefront.clear();
    processed_vertices.clear();
    loop_trip_count.clear();
    sub_runner_nodes.clear();
  }
};

static uint64_t ExecuteOpID;
static uint64_t DmaOpID;
static uint64_t HierarchyOpID;
static uint64_t WaitAllOpID;
static uint64_t ForOpID;
static uint64_t ParallelOpID;
static uint64_t TerminatorID;

class AIRRunner::AIRRunner_impl {

  void debugArg(const std::string &head, mlir::Value op,
                const llvm::APInt &value, uint64_t time) {
    LLVM_DEBUG(llvm::dbgs() << "  " << head << ":  " << op << " = " << value
                            << " (llvm::APInt<" << value.getBitWidth() << ">) @"
                            << time << "\n");
  }

  void debugArg(const std::string &head, mlir::Value op,
                const llvm::APFloat &value, uint64_t time) {
    LLVM_DEBUG(llvm::dbgs() << "  " << head << ":  " << op << " = ";
               value.print(llvm::dbgs());
               llvm::dbgs() << " ("
                            << "float"
                            << ") @" << time << "\n");
  }

  void debugArg(const std::string &head, mlir::Value op, const llvm::Any &value,
                uint64_t time) {
    if (llvm::any_isa<llvm::APInt>(value)) {
      debugArg(head, op, llvm::any_cast<llvm::APInt>(value), time);
    } else if (llvm::any_isa<llvm::APFloat>(value)) {
      debugArg(head, op, llvm::any_cast<llvm::APFloat>(value), time);
    } else if (llvm::any_isa<unsigned>(value)) {
      // Represents an allocated buffer.
      LLVM_DEBUG(llvm::dbgs() << "  " << head << ":  " << op << " = Buffer "
                              << llvm::any_cast<unsigned>(value) << "\n");
    } else {
      // llvm_unreachable("unknown type");
    }
  }

public:
  AIRRunner_impl(llvm::raw_ostream &trace_stream, llvm::json::Value &json_model,
                 bool verbose = false)
      : traceStream(trace_stream), jsonModel(json_model), time(1) {

    auto model = jsonModel.getAsObject();

    dispatch_slots = 1;
    if (auto ds = model->getNumber("num_dispatch_queues"))
      dispatch_slots = (unsigned)(*ds);

    dispatch_dma_slots = 1;
    if (auto dd = model->getNumber("num_dispatch_dma_queues"))
      dispatch_dma_slots = (unsigned)(*dd);

    core_dma_slots = 1;
    if (auto cd = model->getNumber("num_core_dma_queues"))
      core_dma_slots = (unsigned)(*cd);

    herd_slots = 1;
    if (auto hs = model->getNumber("num_herd_slots"))
      herd_slots = (unsigned)(*hs);

    LLVM_DEBUG(llvm::dbgs() << "dispatch slots: " << dispatch_slots << "\n");
    LLVM_DEBUG(llvm::dbgs()
               << "dispatch dma slots: " << dispatch_dma_slots << "\n");
    LLVM_DEBUG(llvm::dbgs() << "core dma slots: " << core_dma_slots << "\n");
    LLVM_DEBUG(llvm::dbgs() << "herd slots: " << herd_slots << "\n");
  }

  void emitTraceStart(llvm::raw_ostream &s) { s << "[\n"; }

  void emitTraceEnd(llvm::raw_ostream &s) { s << "{}]\n"; }

  void emitTraceEvent(llvm::raw_ostream &s, std::string name, std::string cat,
                      std::string ph, uint64_t ts, int64_t tid, int64_t pid) {
    s << "{\n";
    s << "  \"name\": \"" << name << "\","
      << "\n";
    s << "  \"cat\": \"" << cat << "\","
      << "\n";
    s << "  \"ph\": \"" << ph << "\","
      << "\n";
    s << "  \"ts\": " << ts << ","
      << "\n";
    s << "  \"pid\": " << pid << ","
      << "\n";
    s << "  \"tid\": " << tid << ","
      << "\n";
    s << "  \"args\": "
      << "{}"
      << ""
      << "\n";
    s << "},\n";
  }

  void executeOp(xilinx::air::HierarchyInterface op, uint64_t time, runnerNode * sub_runner_node, runnerNode &c, Graph::vertex_descriptor it) {
    // Initialize sub runner and sub graph prior to execution
    Graph &G = sub_runner_node->ctrl_g->g;
    auto sub_start_v = sub_runner_node->ctrl_g->start_vertex;
    auto sub_terminator_v = sub_runner_node->ctrl_g->terminator_vertex;
    resetGraphBetweenTwoVertices(sub_start_v, sub_terminator_v, G, *sub_runner_node);
    sub_runner_node->loop_trip_count.clear();

    // Start sub-runner node by pushing start node into its wavefront
    sub_runner_node->ctrl_g->g[sub_start_v].start_time = time;
    sub_runner_node->ctrl_g->g[sub_start_v].end_time = time;
    assert(!sub_runner_node->wavefront.size() && "Sub runner node is busy");
    pushToWavefront(sub_runner_node->wavefront, std::make_pair(sub_start_v, 1));

    sub_runner_node->processed_vertices.clear();

    c.processed_vertices.push_back(it);
  }

  void executeOp(scf::YieldOp op, scf::ForOp for_op, runnerNode &c, Graph::vertex_descriptor it){
    Graph &G = c.ctrl_g->g;
    auto node_entry = G[it];

    // For loop trip counter
    bool trip_count_fulfilled = false;
    for (auto &count_entry : c.loop_trip_count){
      if (count_entry.first == (unsigned)getIdAttr(for_op.getOperation())){
        // Decrement loop trip count
        if (count_entry.second){
          count_entry.second--;
        }

        // Only push yield op to processed_vertices when trip count fulfilled
        if (!count_entry.second){
          c.processed_vertices.push_back(it);
          trip_count_fulfilled = true;
        }
      }
    }

    // If trip count unfulfilled, then iterate.
    // Clear start_time and end_time of all ops in loop body.
    // From processed_vertices, remove all ops which are in loop body.
    if (!trip_count_fulfilled) {
      // Get for op vertex
      auto for_v = getVertexFromOp(for_op.getOperation(), "front").first;
      auto adj_set = boost::adjacent_vertices(for_v, G);
      for (auto adj_v = adj_set.first; adj_v != adj_set.second; ++adj_v){
        resetGraphBetweenTwoVertices(*adj_v, it, G, c);
      }
      
    }
  }

  void executeOp(scf::ForOp op, runnerNode &c, Graph::vertex_descriptor it){
    Graph &G = c.ctrl_g->g;
    auto node_entry = G[it];

    // Get for loop trip count
    auto lb = op.getLowerBound().getDefiningOp();
    int64_t lbv = cast<arith::ConstantIndexOp>(lb).value();
    auto ub = op.getUpperBound().getDefiningOp();
    int64_t ubv = cast<arith::ConstantIndexOp>(ub).value();
    auto step = op.getStep().getDefiningOp();
    int64_t stepv = cast<arith::ConstantIndexOp>(step).value();

    // (ubv - lbv) / stepv, fast round up
    int64_t trip_count = (ubv - lbv + stepv - 1) / stepv;

    // Update for loop trip count
    c.loop_trip_count.push_back(std::make_pair(getIdAttr(op.getOperation()), trip_count));

    c.processed_vertices.push_back(it);
  }

  void executeOp(runnerNode &c, Graph::vertex_descriptor it){
    c.processed_vertices.push_back(it);
  }

  void executeOpImpls(runnerNode &c, Graph::vertex_descriptor it, uint64_t time) {
    Graph G = c.ctrl_g->g;
    auto node = G[it];
    if (node.asyncEventType == "start"){
      executeOp(c, it);
    }
    else if (auto Op = dyn_cast<xilinx::air::HierarchyInterface>(node.op)){
      auto sub_dependency_graph = node.nextDependencyGraph;
      // assert(sub_dependency_graph);
      // assert(sub_dependency_graph->terminator_vertex);
      auto sub_runner_node = sub_dependency_graph->runner_node;
      assert(sub_dependency_graph->g[sub_dependency_graph->start_vertex].asyncEventType == "start");
      assert(sub_runner_node->ctrl_g);
      assert(sub_runner_node->ctrl_g->g[sub_runner_node->ctrl_g->start_vertex].asyncEventType == "start");
      executeOp(Op, time, sub_runner_node, c, it);
    }
    else if (auto Op = dyn_cast<scf::ForOp>(node.op)){
      executeOp(Op, c, it);
    }
    else if (dyn_cast<scf::YieldOp>(node.op)
        && getScfParentOpFromYieldOp<scf::ForOp>(dyn_cast<scf::YieldOp>(node.op))){
      auto Op = dyn_cast<scf::YieldOp>(node.op);
      auto parent_for_op = dyn_cast<scf::ForOp>(getScfParentOpFromYieldOp<scf::ForOp>(Op));
      executeOp(Op, parent_for_op, c, it);
    }
    else {
      executeOp(c, it);
    }
  }

  unsigned modelOp(dependencyNodeEntry c){
    auto type = c.asyncEventType;
    auto name = c.asyncEventName;
    if (type == "terminator") return 1;
    else if (type == "for_loop") return 1;
    else if (type == "hierarchy_terminator") return 1;
    else if (type == "hierarchy") return 1;
    else if (type == "execute" && name == "AllocOp") return 2;
    else if (type == "execute" && name == "DeallocOp") return 1;
    else if (type == "execute" && name == "ExecuteTerminatorOp") return 1;
    else if (type == "wait_all") return 1;
    else return 10;
  }

  std::string to_string(Operation *op) { return op->getName().getStringRef().str(); }

  std::string to_string(dependencyNodeEntry &c) { return to_string(c.op); }

  void processGraph(runnerNode &c, uint64_t time) {

    Graph &G = c.ctrl_g->g;
    
    // Update wavefront
    std::vector<Graph::vertex_descriptor> next_vertex_set_candidates;
    std::vector<Graph::vertex_descriptor> next_vertex_set;
    for (auto it = c.wavefront.begin(); it != c.wavefront.end(); ++it){
      if (G[it->first].is_started() && G[it->first].is_done(time)){

        if (G[it->first].asyncEventType != "start"){
          
          auto runner_id = getIdAttr(c.ctrl_g->hierarchyOp);
          auto sub_tid = it->second;
          emitTraceEvent(traceStream, G[it->first].asyncEventName, "layer", "E", time,
                          sub_tid, runner_id);
        }

        // "ExecuteOp"
        executeOpImpls(c, it->first, time);
        
        // Erase from wavefront
        c.wavefront.erase(it);
        it--;
      }
    }

    // Get all adjacent vertices to the procssed vertices
    findAdjacentVertices(c.processed_vertices, next_vertex_set_candidates, &G);
    // Remove candidate vertices already on wavefront
    removeRepeatedVertices(next_vertex_set_candidates, getVectorOfFirstFromVectorOfPairs(c.wavefront));

    for (auto it = next_vertex_set_candidates.begin(); it != next_vertex_set_candidates.end(); ++it){
      // inv_adj_set is the list of dependent tokens for each candidate op to wavefront
      auto inv_adj_set = boost::inv_adjacent_vertices(*it, G);
      bool dep_fulfilled = true;
      // Build it's dependency list
      std::vector<dependencyNodeEntry *> dep_list;
      for (auto inv_adj_v = inv_adj_set.first; inv_adj_v != inv_adj_set.second; ++inv_adj_v){
        // If dependent on a hierarchy op, then push its terminator into dep_list instead
        if (G[*inv_adj_v].asyncEventType == "hierarchy"){
          auto sub_g = G[*inv_adj_v].nextDependencyGraph;
          auto terminator_v = sub_g->terminator_vertex;
          dep_list.push_back(&sub_g->g[terminator_v]);
        }
        else {
          dep_list.push_back(&G[*inv_adj_v]);
        }
      }
      // Check whether adj_v's dependency list is fulfulled
      for (auto dep : dep_list){
        if ((!dep->is_started()) || (!dep->is_done(time))){
          dep_fulfilled = false;
        }
      }
      if (dep_fulfilled){
        next_vertex_set.push_back(*it);
      }
    }

    for (auto next_vertex : next_vertex_set){
      
      pushToWavefront(c.wavefront, next_vertex);

      G[next_vertex].start_time = time;
      G[next_vertex].end_time = time + modelOp(G[next_vertex]);
      // emit trace event begin
      auto runner_id = getIdAttr(c.ctrl_g->hierarchyOp);
      auto sub_tid = c.wavefront.back().second;
      emitTraceEvent(traceStream, G[next_vertex].asyncEventName, "layer", "B", time,
                    sub_tid, runner_id);
    }

    return;
  }

  void scheduleFunction(func::FuncOp &toplevel) {

    // Walk the launch op and create a boost graph using dependencyCanonicalizer intepreter
    hostGraph.hierarchyOp = toplevel.getOperation();
    dependencyCanonicalizer canonicalizer;
    xilinx::air::dependencyContext dep_ctx;
    canonicalizer.parseCommandGraphs(toplevel, hostGraph, dep_ctx);

    uint64_t time = 1;
    for (auto &launchGraph : hostGraph.subgraphs){

      // air launch iteration space
      int64_t iter_count = 1;
      auto launch_op = dyn_cast<air::LaunchOp>(launchGraph.hierarchyOp);
      for (auto s_op : launch_op.getSizeOperands()){
        int64_t s = cast<arith::ConstantIndexOp>(s_op.getDefiningOp()).value();
        iter_count *= s;
      }

      for (unsigned i = 0; i < iter_count; i++){
      
        // Reset controllers
        launch_runner_node = runnerNode(&launchGraph, "launch");
        // Update pointer to launch runner node in launch graph
        launchGraph.runner_node = &launch_runner_node;
        
        // Walk the launch graph and infer herd/partition runner nodes
        initRunnerNodesFromLaunchGraph(launch_runner_node, launchGraph);

        // Schedule launch runner node and its sub-runner nodes
        scheduleLaunch(launch_runner_node, time);

      }

    }

  }

  void scheduleLaunch(runnerNode &launch, uint64_t &time) {

    bool running = true;
    auto start_v = launch.ctrl_g->start_vertex;
    launch.ctrl_g->g[start_v].start_time = 1;
    launch.ctrl_g->g[start_v].end_time = 1;
    pushToWavefront(launch.wavefront, std::make_pair(start_v, 1));
    launch.processed_vertices.clear();
    while (running) {
      LLVM_DEBUG(llvm::dbgs() << "time: " << time << "\n");

      running = false;
      std::vector<uint64_t> next_times;

      processGraph(launch, time);
      if (launch.wavefront.size()) {
        running = true;
        // getTimeStampsFromWavefront(next_times, launch);
      }

      for (auto &partition_runner_node : launch.sub_runner_nodes){
        processGraph(partition_runner_node, time);
        if (partition_runner_node.wavefront.size()) {
          running = true;
          // getTimeStampsFromWavefront(next_times, partition_runner_node);
        }
        for (auto &herd_runner_node : partition_runner_node.sub_runner_nodes){
          processGraph(herd_runner_node, time);
          if (herd_runner_node.wavefront.size()) {
            running = true;
            // getTimeStampsFromWavefront(next_times, herd_runner_node);
          }
        }
      }

      if (running){
        getTimeStampsFromWavefront(next_times, launch);
        for (auto &partition_runner_node : launch.sub_runner_nodes){
          getTimeStampsFromWavefront(next_times, partition_runner_node);
          for (auto &herd_runner_node : partition_runner_node.sub_runner_nodes){
            getTimeStampsFromWavefront(next_times, herd_runner_node);
          }
        }
      }

      uint64_t next_time = 0;
      if (next_times.size())
        next_time = *std::min_element(next_times.begin(), next_times.end());
      time = std::max(time + 1, next_time);
      if (time > 5000000)
        running = false;
    }
  }

private:
  llvm::raw_ostream &traceStream;
  llvm::json::Value &jsonModel;
  uint64_t time;

  unsigned dispatch_slots;
  unsigned dispatch_dma_slots;
  unsigned core_dma_slots;
  unsigned herd_slots;

  // Dependency graph constructed as Boost graph
  dependencyGraph hostGraph;
  operation_to_vertex_map op_to_v; // Map between ops and vertices in graph
  operation_to_graph_map op_to_g; // Map between ops and graph

  // Host and segment runnerNodes
  runnerNode launch_runner_node;

  // Dump graphviz
  void dump_graph(std::string filename, Graph G)
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

  // Create graph vertex from op
  Graph::vertex_descriptor addVertexFromOp(Operation * op, uint64_t &id, std::string event_type, std::string event_name, std::string color, std::string shape, Graph &G, Operation * pointer_op = nullptr){
    op->removeAttr("id");
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
    G[v].start_time = 0;
    G[v].end_time = 0;
    // Update op-to-graph mapping
    auto entry = make_pair(event_type, id);
    op_to_v.insert(make_pair(entry, v));
    op_to_g.insert(make_pair(entry, &G));
    return v;
  }

  Graph::vertex_descriptor addVertexFromOpImpls(Operation * op, Graph &G){
    if (auto dma_op = mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(op)){
      return addVertexFromDmaOp(dma_op, G);
    }
    else if (auto execute_op = dyn_cast<xilinx::air::ExecuteOp>(op)){
      return addVertexFromExecuteOp(execute_op, G);
    }
    else if (auto wa_op = dyn_cast<xilinx::air::WaitAllOp>(op)){
      return addVertexFromWaitAllOp(wa_op, G);
    }
    else if (auto forop = dyn_cast<scf::ForOp>(op)){
      return addVertexFromOp(op, ForOpID, "for_loop", "ScfForOp", "crimson", "box", G);
    }
    else if (auto parallelop = dyn_cast<scf::ParallelOp>(op)){
      return addVertexFromOp(op, ParallelOpID, "parallel_loop", "ScfParallelOp", "crimson", "box", G);
    }
    else if (auto hier_op = mlir::dyn_cast<xilinx::air::HierarchyInterface>(op)){
      return addVertexFromHierarchyOp(hier_op, G);
    }
    else if (op->mightHaveTrait<OpTrait::IsTerminator>()){
      return addVertexFromTerminatorOp(op, G);
    }
    else return 0;
  }

  Graph::vertex_descriptor addVertexFromDmaOp(xilinx::air::DmaMemcpyInterface op, Graph &G){
    if (dyn_cast<xilinx::air::DmaMemcpyNdOp>(op.getOperation())){
      return addVertexFromOp(op, DmaOpID, "dma", "DmaMemcpyNdOp", "cyan", "oval", G);
    }
    else {
      assert(false && "Unknown dma op");
      return 0;
    }
  }

  Graph::vertex_descriptor addVertexFromHierarchyOp(xilinx::air::HierarchyInterface op, Graph &G){
    if (dyn_cast<xilinx::air::LaunchOp>(op.getOperation())){
      return addVertexFromOp(op, HierarchyOpID, "hierarchy", "LaunchOp", "yellow", "box", G);
    }
    else if (dyn_cast<xilinx::air::PartitionOp>(op.getOperation())){
      return addVertexFromOp(op, HierarchyOpID, "hierarchy", "PartitionOp", "yellow", "box", G);
    }
    else if (dyn_cast<xilinx::air::HerdOp>(op.getOperation())){
      return addVertexFromOp(op, HierarchyOpID, "hierarchy", "HerdOp", "yellow", "box", G);
    }
    else {
      assert(false && "Unknown hierarchy op");
      return 0;
    }
  }

  Graph::vertex_descriptor addVertexFromTerminatorOp(Operation * op, Graph &G){
    if (dyn_cast<xilinx::air::LaunchTerminatorOp>(op)){
      return addVertexFromOp(op, TerminatorID, "hierarchy_terminator", "LaunchTerminator", "yellow", "box", G);
    }
    else if (dyn_cast<xilinx::air::PartitionTerminatorOp>(op)){
      return addVertexFromOp(op, TerminatorID, "hierarchy_terminator", "PartitionTerminator", "yellow", "box", G);
    }
    else if (dyn_cast<xilinx::air::HerdTerminatorOp>(op)){
      return addVertexFromOp(op, TerminatorID, "hierarchy_terminator", "HerdTerminator", "yellow", "box", G);
    }
    else if (auto yieldop = dyn_cast<scf::YieldOp>(op)){
      if (getScfParentOpFromYieldOp<scf::ParallelOp>(yieldop)){
        return addVertexFromOp(op, TerminatorID, "terminator", "ScfParallelYieldOp", "crimson", "box", G);
      }
      else if (getScfParentOpFromYieldOp<scf::ForOp>(yieldop)){
        return addVertexFromOp(op, TerminatorID, "terminator", "ScfForYieldOp", "crimson", "box", G);
      }
    }
    return 0;
  }

  Graph::vertex_descriptor addVertexFromExecuteOp(xilinx::air::ExecuteOp op, Graph &G){
    int iter_count = 0;
    Graph::vertex_descriptor v_prev = 0;
    Graph::vertex_descriptor v = 0;
    Operation * pointer_op = op;
    for (auto &child_op : op->getRegions().front().getOps()){
      if (dyn_cast<linalg::LinalgOp>(child_op)){
        v = addVertexFromOp(&child_op, ExecuteOpID, "execute", "LinalgOp", "chartreuse", "oval", G, pointer_op);
      }
      else if (dyn_cast<memref::AllocOp>(child_op)){
        v = addVertexFromOp(&child_op, ExecuteOpID, "execute", "AllocOp", "chartreuse", "oval", G, pointer_op);
      }
      else if (dyn_cast<memref::DeallocOp>(child_op)){
        v = addVertexFromOp(&child_op, ExecuteOpID, "execute", "DeallocOp", "chartreuse", "oval", G, pointer_op);
      }
      else if (dyn_cast<memref::CopyOp>(child_op)){
        v = addVertexFromOp(&child_op, ExecuteOpID, "execute", "CopyOp", "chartreuse", "oval", G, pointer_op);
      }
      else if (dyn_cast<mlir::AffineApplyOp>(child_op)){
        v = addVertexFromOp(&child_op, ExecuteOpID, "execute", "AffineApplyOp", "chartreuse", "oval", G, pointer_op);
      }
      else if (dyn_cast<xilinx::air::ExecuteTerminatorOp>(child_op)){
        v = addVertexFromOp(&child_op, ExecuteOpID, "execute", "ExecuteTerminatorOp", "chartreuse", "oval", G, pointer_op);
      }
      else if (dyn_cast<arith::MulIOp>(child_op)){
        v = addVertexFromOp(&child_op, ExecuteOpID, "execute", "MuliOp", "chartreuse", "oval", G, pointer_op);
      }
      else if (dyn_cast<arith::AddIOp>(child_op)){
        v = addVertexFromOp(&child_op, ExecuteOpID, "execute", "AddIOp", "chartreuse", "oval", G, pointer_op);
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

  Graph::vertex_descriptor addVertexFromWaitAllOp(xilinx::air::WaitAllOp op, Graph &G){
    if (op.getAsyncToken().hasOneUse()){
      for (auto u : op.getAsyncToken().getUsers()){
        if (dyn_cast<scf::YieldOp>(u)){
          return addVertexFromOp(op, WaitAllOpID, "wait_all", "WaitAllOp", "crimson", "oval", G);
        }
        else if (dyn_cast<scf::ReduceReturnOp>(u)){
          return addVertexFromOp(op, WaitAllOpID, "wait_all", "WaitAllOp", "crimson", "oval", G);
        }
      }
    }
    return addVertexFromOp(op, WaitAllOpID, "wait_all", "WaitAllOp", "crimson", "oval", G);
  }

  // Get type-id pair from op, which will be used to look up vertex in op_to_v
  std::pair<std::string, unsigned> getTypeIdPairFromOp(Operation * op){
    std::pair<std::string, unsigned> output;
    std::string type = getOpTypeFromOpImpls(op);
    output.first = type;
    output.second = getIdAttr(op);
    return output;
  }

  std::string getOpTypeFromOpImpls(Operation * op){
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
  std::pair<Graph::vertex_descriptor, Graph *> getVertexFromOp(Operation * op, std::string front_or_back = "front"){
    std::pair<Graph::vertex_descriptor, Graph *> output;
    if (auto execute_op = dyn_cast<xilinx::air::ExecuteOp>(op)){
      if (front_or_back == "front"){
        auto execute_front_op = &(*op->getRegions().front().op_begin());
        std::pair<std::string, unsigned> entry_pair = getTypeIdPairFromOp(execute_front_op);
        output.first = op_to_v[entry_pair];
        output.second = op_to_g[entry_pair];
      }
      else if (front_or_back == "back"){
        auto execute_end_op = op->getRegions().front().getBlocks().front().getTerminator();
        std::pair<std::string, unsigned> entry_pair = getTypeIdPairFromOp(execute_end_op);
        output.first = op_to_v[entry_pair];
        output.second = op_to_g[entry_pair];
      }
      else {
        assert(false && "Unknown string operand (only accepts 'front' or 'back')");
      }
    }
    else {
      std::pair<std::string, unsigned> entry_pair = getTypeIdPairFromOp(op);
        output.first = op_to_v[entry_pair];
        output.second = op_to_g[entry_pair];
    }
    return output;
  }

  // Trace op from a token in dependency list
  std::vector<Operation *> traceOpFromToken(Value dep_token){
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

  // Insert a vertex v between two vertices a and b which were connected by an edge
  void insertVertexBetweenTwoVertices(Graph::vertex_descriptor a, Graph::vertex_descriptor b, Graph::vertex_descriptor v, Graph &G){
    if ((a != b) && (a != v) && (b != v)){
      if (edge(a, b, G).second){ // if an edge exists
        remove_edge(a, b, G);
        if (!edge(a, v, G).second)
          add_edge(a, v, G);
        if (!edge(v, b, G).second)
          add_edge(v, b, G);
      }
    }
  }

  // Create start node for graph
  void connectStartNodeInDependencyGraph (dependencyGraph &G){
    auto v = G.start_vertex;
    auto vp = boost::vertices(G.g);
    for (auto vit = vp.first; vit != vp.second; ++vit){
      if ((v != *vit) && !in_degree(*vit, G.g)){
        add_edge(v, *vit, G.g);
      }
    }
  }

  // Adds pointer from command graph to launch, partition and herd terminators
  void updatePointerFromGraphToHierarchyTerminator(dependencyGraph &G){
    auto vp = boost::vertices(G.g);
    for (auto v = vp.first; v != vp.second; ++v){
      if (G.g[*v].asyncEventType == "hierarchy_terminator"){
        G.terminator_vertex = *v;
        return;
      }
    }
  }

  // Adds pointer from hierarchy terminator to parent command graph
  void updatePointerFromHierarchyTerminatorToGraph(dependencyGraph &G, dependencyGraph &subG){
    auto vp = boost::vertices(subG.g);
    for (auto v = vp.first; v != vp.second; ++v){
      if (subG.g[*v].asyncEventType == "hierarchy_terminator"){
        subG.g[*v].nextDependencyGraph = &G;
        return;
      }
    }
  }

  // Adds pointer from hierarchy op to sub command graph
  void updatePointerFromHierarchyOpToGraph(dependencyGraph &G){
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

  // Returns the scf parent op from scf.yield op
  template <typename T>
  Operation * getScfParentOpFromYieldOp(scf::YieldOp op){
    if (auto scfop = dyn_cast<T>(op->getParentOp())){
      return scfop.getOperation();
    }
    return nullptr;
  }

  // Connects launch, partition and herd terminators
  void connectTerminatorInGraph(Graph &g){
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

  // Connect an async op to ops in its dependency list
  void connectOpToItsDepList(Operation * op, SmallVector<Value, 1> dep_list, Graph &g){
    auto dst_v = getVertexFromOp(op, "front").first;
    if (dep_list.size()){
      for (auto dep_token : dep_list){
        auto src_vector = traceOpFromToken(dep_token);
        if (src_vector.size()){
          for (auto src_op : src_vector){
            auto src_v = getVertexFromOp(src_op, "back").first;
            if (!edge(src_v, dst_v, g).second){
              add_edge(src_v, dst_v, g);
            }
          }
        }
      }
    }
  }

  // Trace dependency of every op in a boost graph
  void traceDependencyInGraph(Graph &g){
    auto vp = boost::vertices(g);
    for (auto vit = vp.first; vit != vp.second; ++vit){
      auto op = g[*vit].op;
      if (!op) continue;
      connectOpToItsDepListImpls(op, g);
    }
  }

  void connectOpToItsDepListImpls(Operation * op, Graph &g){
    SmallVector<Value, 1> dep_list;
    // air.asyncopinterface
    if (auto async_op = mlir::dyn_cast<xilinx::air::AsyncOpInterface>(op)){
      for (auto dep_token : async_op.getAsyncDependencies()){
        dep_list.push_back(dep_token);
      }
      connectOpToItsDepList(op, dep_list, g);
    }
    // scf.for
    else if (auto forop = dyn_cast<scf::ForOp>(op)){
      for (auto iter_operand : forop.getIterOperands()){
        dep_list.push_back(iter_operand);
      }
      connectOpToItsDepList(op, dep_list, g);
    }
    // scf.parallel
    else if (auto parallelop = dyn_cast<scf::ParallelOp>(op)){
      for (auto operand : parallelop->getOperands()){
        dep_list.push_back(operand);
      }
      connectOpToItsDepList(op, dep_list, g);
    }
    // scf.yield
    else if (auto yieldop = dyn_cast<scf::YieldOp>(op)){
      for (auto operand : yieldop->getOperands()){
        dep_list.push_back(operand);
      }
      connectOpToItsDepList(op, dep_list, g);
    }

  }

  // Find all vertices adjacent to given vertices in graph
  void findAdjacentVertices(std::vector<Graph::vertex_descriptor> vertices, std::vector<Graph::vertex_descriptor> &adjacent_vertices, Graph * G){
    for (auto v : vertices){
      auto adj_set = boost::adjacent_vertices(v, *G);
      for (auto v1 = adj_set.first; v1 != adj_set.second; ++v1){
        bool found_duplicate = false;
        for (auto v2 : adjacent_vertices){
          if (*v1 == v2){
            found_duplicate = true;
          }
        }
        bool is_in_vertices = false;
        for (auto v3 : vertices){
          if (*v1 == v3){
            is_in_vertices = true;
          }
        }
        if (!found_duplicate && !is_in_vertices){
          adjacent_vertices.push_back(*v1);
        }
      }
    }
  }

  // Remove vertices in vector a which already exist in vector b
  void removeRepeatedVertices(std::vector<Graph::vertex_descriptor> &a, std::vector<Graph::vertex_descriptor> b){
    for (auto v : b){
      removeVertexFromVertices(a, v);
    }
  }

  // Remove a vertex from a vector of vertices
  void removeVertexFromVertices(std::vector<Graph::vertex_descriptor> &vector, Graph::vertex_descriptor a){
    if (vector.size()){
      for (auto it = vector.begin(); it != vector.end(); ++it){
        if (*it == a){
          vector.erase(it);
          it--;
        }
      }
    }
  }

  // Recursively reset all vertices in for loop body
  void resetGraphBetweenTwoVertices(Graph::vertex_descriptor start_v, Graph::vertex_descriptor end_v, Graph &G, runnerNode &c){
    
    // Remove start_v from processed_vertices
    removeVertexFromVertices(c.processed_vertices, start_v);

    // Reset start_time and end_time
    G[start_v].start_time = 0;
    G[start_v].end_time = 0;
    
    if (start_v == end_v) return;

    // Recurse until end_v
    auto adj_set = boost::adjacent_vertices(start_v, G);
    for (auto adj_v = adj_set.first; adj_v != adj_set.second; ++adj_v){
      resetGraphBetweenTwoVertices(*adj_v, end_v, G, c);
    }
    // If v is a hierarchy op, then recursively clear the entire subgraph
    if (G[start_v].asyncEventType == "hierarchy"){
      auto sub_c = G[start_v].nextDependencyGraph;
      auto start_v = sub_c->start_vertex;
      auto terminator_v = sub_c->terminator_vertex;
      auto sub_g = sub_c->g;
      auto sub_runner = sub_c->runner_node;
      resetGraphBetweenTwoVertices(start_v, terminator_v, sub_g, *sub_runner);
    }
  }

  // Initialize sub runner nodes from launch graph tree
  void initRunnerNodesFromLaunchGraph(runnerNode &launch_runner_node, dependencyGraph &launchGraph){
    launchGraph.runner_node = &launch_runner_node;
    for (auto &partitionGraph : launchGraph.subgraphs){
      // Create partition runner node
      launch_runner_node.sub_runner_nodes.push_back(runnerNode(&partitionGraph, "partition"));
      auto current_partition_node = &(launch_runner_node.sub_runner_nodes.back());
      for (auto &herdGraph : partitionGraph.subgraphs){
        // Create herd runner node
        current_partition_node->sub_runner_nodes.push_back(runnerNode(&herdGraph, "herd"));
      }
    }
    addPointerBetweenSubRunnerNodeAndSubCommandGraph(launch_runner_node);
    for (auto &partition_runner_node : launch_runner_node.sub_runner_nodes){
      addPointerBetweenSubRunnerNodeAndSubCommandGraph(partition_runner_node);
    }
  }

  // Adds pointer between runner node and command graph
  void addPointerBetweenSubRunnerNodeAndSubCommandGraph(runnerNode &R){
    for (auto r_it = std::begin(R.sub_runner_nodes); r_it != std::end(R.sub_runner_nodes); ++r_it) {
      r_it->ctrl_g->runner_node = &(*r_it);
    }
  }

  // Get time stamps from wavefront
  void getTimeStampsFromWavefront(std::vector<uint64_t> &next_times, runnerNode runner_node){
    for (auto it = runner_node.wavefront.begin(); it != runner_node.wavefront.end(); it++){
      auto command_node = runner_node.ctrl_g->g[it->first];
      if (command_node.is_started() && (command_node.end_time)){
        next_times.push_back(command_node.end_time);
      }
    }
  }

  // Get a vector of first elements from a vector of pairs
  std::vector<Graph::vertex_descriptor> getVectorOfFirstFromVectorOfPairs(std::vector<std::pair<Graph::vertex_descriptor, unsigned>> pairs){
    std::vector<Graph::vertex_descriptor> items;
    std::transform(pairs.begin(), 
                pairs.end(), 
                std::back_inserter(items), 
                [](const std::pair<Graph::vertex_descriptor, unsigned>& p) { return p.first; });
    return items;
  }

  // Push an entry into wavefront
  void pushToWavefront(std::vector<std::pair<Graph::vertex_descriptor, unsigned>> &wavefront, std::pair<Graph::vertex_descriptor, unsigned> entry){
    for (auto i : wavefront){
      assert(i.second != entry.second && "queried thread is busy");
    }
    wavefront.push_back(entry);
  }
  void pushToWavefront(std::vector<std::pair<Graph::vertex_descriptor, unsigned>> &wavefront, Graph::vertex_descriptor v){
    // Acquire available thread id for current op
    unsigned tid = 0;
    for (unsigned i = 1; i < wavefront.size() + 2; i++){
      bool tid_i_unavailable = false;
      for (auto j : wavefront){
        if (j.second == i){
          tid_i_unavailable = true;
        }
      }
      if (!tid_i_unavailable){
        tid = i;
        break;
      }
    }
    wavefront.push_back(std::make_pair(v, tid));
  }
  

}; // AIRRunner_impl

AIRRunner::AIRRunner(llvm::raw_ostream &trace_stream,
                     llvm::json::Value &json_model, bool verbose) {
  impl = std::make_unique<AIRRunner_impl>(trace_stream, json_model, verbose);
  if (verbose) {
    llvm::DebugFlag = true;
    llvm::setCurrentDebugType(DEBUG_TYPE);
  }
}

AIRRunner::~AIRRunner() {}

void AIRRunner::emitTraceStart(llvm::raw_ostream &s) {
  impl->emitTraceStart(s);
}

void AIRRunner::emitTraceEnd(llvm::raw_ostream &s) { impl->emitTraceEnd(s); }

void AIRRunner::scheduleFunction(func::FuncOp &toplevel) {
  impl->scheduleFunction(toplevel);
}

} // namespace air
} // namespace xilinx