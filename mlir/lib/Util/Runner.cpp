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
#include "air/Util/Dependency.h"
#include "air/Util/Util.h"

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

#include <algorithm>
#include <deque>
#include <float.h>
#include <list>
#include <map>
#include <sstream>
#include <vector>

// boost graph
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>

#include <algorithm>
#include <numeric>
#include <string>

#define DEBUG_TYPE "air-runner"

#define INDEX_WIDTH 32

using namespace mlir;
using namespace boost;

namespace xilinx {
namespace air {

struct runnerNode {
  dependencyGraph *ctrl_g;
  std::string runner_node_type;
  // Each entry is an std::pair. First element is vertex, and second element is
  // thread id
  std::vector<std::pair<Graph::vertex_descriptor, unsigned>> wavefront;
  std::vector<Graph::vertex_descriptor> processed_vertices;
  // Each entry is an std::pair. First element is for op's id, and second
  // element is counter
  std::vector<std::pair<unsigned, unsigned>> loop_trip_count;
  std::deque<runnerNode> sub_runner_nodes;

  // Private wavefront of each runner node, reserved to interface with resource
  // model
  std::vector<dependencyNodeEntry *> wavefrontNodes() {
    std::vector<dependencyNodeEntry *> output;
    for (auto v : wavefront) {
      output.push_back(&ctrl_g->g[v.first]);
    }
    return output;
  }

  runnerNode(dependencyGraph *ctrl_g = nullptr,
             std::string runner_node_type = "")
      : ctrl_g(ctrl_g), runner_node_type(runner_node_type) {}

  ~runnerNode() {
    wavefront.clear();
    processed_vertices.clear();
    loop_trip_count.clear();
    sub_runner_nodes.clear();
  }
};

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

  void executeOp(xilinx::air::HierarchyInterface op, uint64_t time,
                 runnerNode *sub_runner_node, runnerNode &c,
                 Graph::vertex_descriptor it) {
    // Initialize sub runner and sub graph prior to execution
    Graph &G = sub_runner_node->ctrl_g->g;
    auto sub_start_v = sub_runner_node->ctrl_g->start_vertex;
    auto sub_terminator_v = sub_runner_node->ctrl_g->terminator_vertex;
    resetGraphBetweenTwoVertices(sub_start_v, sub_terminator_v, G,
                                 *sub_runner_node);
    sub_runner_node->loop_trip_count.clear();

    // Start sub-runner node by pushing start node into its wavefront
    sub_runner_node->ctrl_g->g[sub_start_v].start_time = time;
    sub_runner_node->ctrl_g->g[sub_start_v].end_time = time;
    assert(!sub_runner_node->wavefront.size() && "Sub runner node is busy");
    pushToWavefront(sub_runner_node->wavefront, std::make_pair(sub_start_v, 1));

    sub_runner_node->processed_vertices.clear();

    c.processed_vertices.push_back(it);
  }

  void executeOp(scf::YieldOp op, scf::ForOp for_op, runnerNode &c,
                 Graph::vertex_descriptor it) {
    Graph &G = c.ctrl_g->g;

    // For loop trip counter
    bool trip_count_fulfilled = false;
    for (auto &count_entry : c.loop_trip_count) {
      if (count_entry.first == (unsigned)getIdAttr(for_op.getOperation())) {
        // Decrement loop trip count
        if (count_entry.second) {
          count_entry.second--;
        }

        // If trip count is fulfilled
        if (!count_entry.second) {
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
      auto for_v =
          canonicalizer.getVertexFromOp(for_op.getOperation(), dep_ctx, "front")
              .first;
      auto adj_set = boost::adjacent_vertices(for_v, G);
      for (auto adj_v = adj_set.first; adj_v != adj_set.second; ++adj_v) {
        resetGraphBetweenTwoVertices(*adj_v, it, G, c);
      }
    }
  }

  void executeOp(scf::ForOp op, runnerNode &c, Graph::vertex_descriptor it) {
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
    c.loop_trip_count.push_back(
        std::make_pair(getIdAttr(op.getOperation()), trip_count));

    c.processed_vertices.push_back(it);
  }

  void executeOp(air::ChannelPutOp op, runnerNode &c,
                 Graph::vertex_descriptor it) {
    Graph &G = c.ctrl_g->g;
    G[it].sym_token_count++;

    c.processed_vertices.push_back(it);
  }

  void executeOp(air::ChannelGetOp op, runnerNode &c,
                 Graph::vertex_descriptor it) {
    // Get put-side runner from get op
    air::ChannelPutOp put = air::getTheOtherChannelOpThroughSymbol(op);
    auto put_entry =
        canonicalizer.getVertexFromOp(put.getOperation(), dep_ctx, "front");
    auto put_v = put_entry.first;
    auto &put_g = put_entry.second;
    auto &put_node = put_g->g[put_v];

    put_node.sym_token_count--;

    c.processed_vertices.push_back(it);
  }

  void executeOp(runnerNode &c, Graph::vertex_descriptor it) {
    c.processed_vertices.push_back(it);
  }

  void executeOpImpls(runnerNode &c, Graph::vertex_descriptor it,
                      uint64_t time) {
    Graph G = c.ctrl_g->g;
    auto node = G[it];
    if (node.asyncEventType == "start") {
      executeOp(c, it);
    } else if (auto Op = dyn_cast<xilinx::air::HierarchyInterface>(node.op)) {
      auto sub_dependency_graph = node.nextDependencyGraph;
      auto sub_runner_node = sub_dependency_graph->runner_node;
      executeOp(Op, time, sub_runner_node, c, it);
    } else if (auto Op = dyn_cast<scf::ForOp>(node.op)) {
      executeOp(Op, c, it);
    } else if (dyn_cast<scf::YieldOp>(node.op) &&
               getScfParentOpFromYieldOp<scf::ForOp>(
                   dyn_cast<scf::YieldOp>(node.op))) {
      auto Op = dyn_cast<scf::YieldOp>(node.op);
      auto parent_for_op =
          dyn_cast<scf::ForOp>(getScfParentOpFromYieldOp<scf::ForOp>(Op));
      executeOp(Op, parent_for_op, c, it);
    } else if (auto Op = dyn_cast<air::ChannelPutOp>(node.op)) {
      executeOp(Op, c, it);
    } else if (auto Op = dyn_cast<air::ChannelGetOp>(node.op)) {
      executeOp(Op, c, it);
    } else {
      executeOp(c, it);
    }
  }

  // Model each event's latency
  uint64_t modelOp(dependencyNodeEntry &c) {
    auto type = c.asyncEventType;
    auto name = c.asyncEventName;
    uint64_t execution_time = 1;

    if (type == "wait_all") {
      execution_time = 1;
    } else if (type == "dma") {
      auto Op = mlir::dyn_cast<xilinx::air::DmaMemcpyInterface>(c.op);
      assert(Op);
      MemRefType srcTy = Op.getSrcMemref().getType().cast<MemRefType>();
      MemRefType dstTy = Op.getDstMemref().getType().cast<MemRefType>();
      auto srcSpace = srcTy.getMemorySpaceAsInt();
      auto dstSpace = dstTy.getMemorySpaceAsInt();
      // if there is a size mismatch, it's because we're moving a tile of the
      // larger tensor
      if (getTensorVolume(srcTy) <= getTensorVolume(dstTy))
        execution_time = getTransferCost(srcSpace, dstSpace, srcTy);
      else
        execution_time = getTransferCost(srcSpace, dstSpace, dstTy);
    } else if (type == "channel" &&
               (name.find("ChannelGetOp") != std::string::npos)) {
      auto getOp = mlir::dyn_cast<xilinx::air::ChannelGetOp>(c.op);
      assert(getOp);
      MemRefType dstTy = getOp.getDstMemref().getType().cast<MemRefType>();
      air::ChannelPutOp putOp = air::getTheOtherChannelOpThroughSymbol(getOp);
      assert(putOp);
      MemRefType srcTy = putOp.getSrcMemref().getType().cast<MemRefType>();
      auto srcSpace = srcTy.getMemorySpaceAsInt();
      auto dstSpace = dstTy.getMemorySpaceAsInt();
      // if there is a size mismatch, it's because we're moving a tile of the
      // larger tensor
      if (getTensorVolume(srcTy) <= getTensorVolume(dstTy))
        execution_time = getTransferCost(srcSpace, dstSpace, srcTy);
      else
        execution_time = getTransferCost(srcSpace, dstSpace, dstTy);
    } else if (type == "execute" && name != "ExecuteTerminatorOp") {
      assert(dyn_cast<air::ExecuteOp>(c.op));
      auto child_op = &*(c.op->getRegions().front().getOps().begin());
      if (auto Op = mlir::dyn_cast<linalg::LinalgOp>(child_op)) {
        uint64_t compute_xfer_cost = 0;
        uint64_t compute_op_cost = 0;
        auto opCounts = xilinx::air::CostModel().getOpCounts(child_op);
        std::string skip = "footprint";
        std::string memops = "reads;writes;";
        std::string cpuops = "math.rsqrt;";
        cpuops += "arith.mulf;arith.divf;arith.addf;arith.subf;arith.truncf;"
                  "arith.cmpf;arith.maxf;";
        cpuops += "arith.muli;arith.divi;arith.addi;arith.subi;arith.trunci;"
                  "arith.cmpi;arith.maxi";
        cpuops += "std.select";
        uint64_t memory_op_count = 0;
        uint64_t compute_op_count = 0;
        for (auto &p : opCounts.map) {
          auto name = std::get<0>(p);
          auto count = std::get<1>(p);
          if (memops.find(name) != std::string::npos)
            memory_op_count += count;
          else if (cpuops.find(name) != std::string::npos)
            compute_op_count += count;
          else if (skip.find(name) == std::string::npos)
            LLVM_DEBUG(llvm::dbgs() << name << " not counted\n");
        }

        if (compute_op_count) {
          // defaults
          double num_cores = 1;
          double ops_per_core_per_cycle = 8; // vector width for this type
          double cycles_per_second = 1e9;
          double efficiency = 1.0f;

          auto model = jsonModel.getAsObject();
          assert(model);

          // if kernels exists, assume everthing else exists
          if (model && model->getObject("kernels")) {
            // device level override of defaults
            if (auto d = model->getNumber("cores"))
              num_cores = *d;
            if (auto d = model->getNumber("ops_per_core_per_cycle"))
              ops_per_core_per_cycle = *d;
            if (auto d = model->getNumber("clock"))
              cycles_per_second = *d;
            if (auto d = model->getNumber("efficiency"))
              efficiency = *d;

            // kernel level override of defaults
            auto kernels = model->getObject("kernels");
            assert(kernels && "kernels not found in JSON model");

            if (kernels) {
              auto kernel =
                  kernels->getObject(child_op->getName().getStringRef());
              if (kernel) {
                if (auto d = kernel->getNumber("cores"))
                  num_cores = *d;
                if (auto d = kernel->getNumber("ops_per_core_per_cycle"))
                  ops_per_core_per_cycle = *d;
                if (auto d = kernel->getNumber("clock"))
                  cycles_per_second = *d;
                if (auto d = kernel->getNumber("efficiency"))
                  efficiency = *d;
              }
            }
          }

          double ops_per_cycle =
              num_cores * ops_per_core_per_cycle * efficiency;
          assert(ops_per_cycle > 0 &&
                 "ops per cycle in model must be greater than zero");

          double cycles = ceil(compute_op_count / ops_per_cycle);
          compute_op_cost = cycles;
        }
        execution_time = std::max(compute_op_cost, compute_xfer_cost);
      }
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "WARNING: execution time not modeled for op: '");
      LLVM_DEBUG(llvm::dbgs() << to_string(c.op) << "'\n");
      execution_time = 1;
    }
    return execution_time;
  }

  void buildVertexDependencyList(
      Graph::vertex_descriptor v, Graph G,
      std::vector<std::pair<dependencyNodeEntry *, std::string>> &dep_list) {
    auto inv_adj_set = boost::inv_adjacent_vertices(v, G);
    // If current vertex is ChannelGet, then add implicit ChannelPut vertex to
    // dep list
    if (air::ChannelGetOp channel_get = dyn_cast<air::ChannelGetOp>(G[v].op)) {
      air::ChannelPutOp channel_put =
          air::getTheOtherChannelOpThroughSymbol(channel_get);
      // Get ChannelPut node from op
      auto channel_put_entry = canonicalizer.getVertexFromOp(
          channel_put.getOperation(), dep_ctx, "front");
      auto channel_put_v = channel_put_entry.first;
      auto &channel_put_g = channel_put_entry.second;
      auto &channel_put_node = channel_put_g->g[channel_put_v];
      dep_list.push_back(std::make_pair(&channel_put_node, "sym"));
    }
    for (auto inv_adj_v = inv_adj_set.first; inv_adj_v != inv_adj_set.second;
         ++inv_adj_v) {
      // If dependent on a hierarchy op, then push its terminator into dep_list
      // instead
      if (G[*inv_adj_v].asyncEventType == "hierarchy") {
        auto sub_g = G[*inv_adj_v].nextDependencyGraph;
        auto terminator_v = sub_g->terminator_vertex;
        dep_list.push_back(std::make_pair(&sub_g->g[terminator_v], "ssa"));
      } else {
        dep_list.push_back(std::make_pair(&G[*inv_adj_v], "ssa"));
      }
    }
  }

  std::string to_string(Operation *op) {
    return op->getName().getStringRef().str();
  }

  std::string to_string(dependencyNodeEntry &c) { return to_string(c.op); }

  void processGraph(runnerNode &c, uint64_t time) {

    Graph &G = c.ctrl_g->g;

    // Update wavefront
    std::vector<Graph::vertex_descriptor> next_vertex_set_candidates;
    std::vector<Graph::vertex_descriptor> next_vertex_set;
    for (auto it = c.wavefront.begin(); it != c.wavefront.end(); ++it) {
      if (G[it->first].is_started() && G[it->first].is_done(time)) {

        if (G[it->first].asyncEventType != "start") {

          auto runner_id = getIdAttr(c.ctrl_g->hierarchyOp);
          auto sub_tid = it->second;
          emitTraceEvent(traceStream,
                         G[it->first].asyncEventName +
                             G[it->first].detailed_description,
                         "layer", "E", time, sub_tid, runner_id);
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
    removeRepeatedVertices(next_vertex_set_candidates,
                           getVectorOfFirstFromVectorOfPairs(c.wavefront));

    for (auto it = next_vertex_set_candidates.begin();
         it != next_vertex_set_candidates.end(); ++it) {
      bool dep_fulfilled = true;
      // Build it's dependency list. In each entry, the first field is a pointer
      // to the node, and the second field is a string representing the type of
      // this dependency, either "ssa" or "sym".
      std::vector<std::pair<dependencyNodeEntry *, std::string>> dep_list;
      buildVertexDependencyList(*it, G, dep_list);
      // Check whether adj_v's dependency list is fulfulled
      for (auto &dep : dep_list) {
        if (dep.second == "sym") {
          if (!dep.first->sym_token_count) {
            dep_fulfilled = false;
          }
        } else if (dep.second == "ssa") {
          if ((!dep.first->is_started()) || (!dep.first->is_done(time))) {
            dep_fulfilled = false;
          }
        } else {
          assert(false && "Unknown async token type");
        }
      }
      if (dep_fulfilled) {
        next_vertex_set.push_back(*it);
      }
    }

    for (auto next_vertex : next_vertex_set) {

      pushToWavefront(c.wavefront, next_vertex);

      G[next_vertex].start_time = time;
      G[next_vertex].end_time = time + modelOp(G[next_vertex]);
      // emit trace event begin
      auto runner_id = getIdAttr(c.ctrl_g->hierarchyOp);
      auto sub_tid = c.wavefront.back().second;
      emitTraceEvent(traceStream,
                     G[next_vertex].asyncEventName +
                         G[next_vertex].detailed_description,
                     "layer", "B", time, sub_tid, runner_id);
    }

    return;
  }

  void scheduleFunction(func::FuncOp &toplevel) {

    // Walk the launch op and create a boost graph using dependencyCanonicalizer
    // intepreter
    hostGraph = dependencyGraph(toplevel, true);
    canonicalizer.parseCommandGraphs(toplevel, hostGraph, dep_ctx);

    uint64_t time = 1;
    for (auto &launchGraph : hostGraph.subgraphs) {

      // air launch iteration space
      int64_t iter_count = 1;
      auto launch_op = dyn_cast<air::LaunchOp>(launchGraph.hierarchyOp);
      for (auto s_op : launch_op.getSizeOperands()) {
        int64_t s = cast<arith::ConstantIndexOp>(s_op.getDefiningOp()).value();
        iter_count *= s;
      }

      for (unsigned i = 0; i < iter_count; i++) {

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

    auto start_v = launch.ctrl_g->start_vertex;
    // Reset launch graph
    launch.processed_vertices.clear();
    resetGraphBetweenTwoVertices(start_v, launch.ctrl_g->terminator_vertex,
                                 launch.ctrl_g->g, launch);
    // Start running launch
    bool running = true;
    launch.ctrl_g->g[start_v].start_time = 1;
    launch.ctrl_g->g[start_v].end_time = 1;
    pushToWavefront(launch.wavefront, std::make_pair(start_v, 1));
    while (running) {
      LLVM_DEBUG(llvm::dbgs() << "time: " << time << "\n");

      running = false;
      std::vector<uint64_t> next_times;

      processGraph(launch, time);
      if (launch.wavefront.size()) {
        running = true;
        // getTimeStampsFromWavefront(next_times, launch);
      }

      for (auto &partition_runner_node : launch.sub_runner_nodes) {
        processGraph(partition_runner_node, time);
        if (partition_runner_node.wavefront.size()) {
          running = true;
          // getTimeStampsFromWavefront(next_times, partition_runner_node);
        }
        for (auto &herd_runner_node : partition_runner_node.sub_runner_nodes) {
          processGraph(herd_runner_node, time);
          if (herd_runner_node.wavefront.size()) {
            running = true;
            // getTimeStampsFromWavefront(next_times, herd_runner_node);
          }
        }
      }

      if (running) {
        getTimeStampsFromWavefront(next_times, launch);
        for (auto &partition_runner_node : launch.sub_runner_nodes) {
          getTimeStampsFromWavefront(next_times, partition_runner_node);
          for (auto &herd_runner_node :
               partition_runner_node.sub_runner_nodes) {
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
  dependencyCanonicalizer canonicalizer;
  xilinx::air::dependencyContext dep_ctx;

  llvm::raw_ostream &traceStream;
  llvm::json::Value &jsonModel;
  uint64_t time;

  unsigned dispatch_slots;
  unsigned dispatch_dma_slots;
  unsigned core_dma_slots;
  unsigned herd_slots;

  // Dependency graph constructed as Boost graph
  dependencyGraph hostGraph;

  // Host and segment runnerNodes
  runnerNode launch_runner_node;

  // Dump graphviz
  void dump_graph(std::string filename, Graph G) {
    std::ofstream ofs(filename, std::ofstream::out);
    boost::dynamic_properties dp;
    dp.property("label", boost::get(&dependencyNodeEntry::asyncEventName, G));
    dp.property("color", boost::get(&dependencyNodeEntry::color, G));
    dp.property("shape", boost::get(&dependencyNodeEntry::shape, G));
    dp.property("node_id", boost::get(boost::vertex_index, G));
    dp.property(
        "style",
        boost::make_constant_property<Graph::vertex_descriptor>(+"filled"));
    write_graphviz_dp(ofs, G, dp);
  }

  // Trace op from a token in dependency list
  std::vector<Operation *> traceOpFromToken(Value dep_token) {
    std::vector<Operation *> output;
    // If dependency token originates from async op
    if (dep_token.getDefiningOp() &&
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
        auto parallelop_terminator =
            parallelop_reduceop.getRegion().front().getTerminator();
        output.push_back(parallelop_terminator);
        return output;
      }
    }
    // Else if dependency token is the iter arg of an scf for loop
    else if (auto forop = getForRegionIterArgsOwner(dep_token)) {
      output.push_back(forop);
      return output;
    }
    // Else if dependency token is from affine if (joint token from multiple
    // ops)
    else if (dep_token.getDefiningOp() &&
             dyn_cast<mlir::AffineIfOp>(dep_token.getDefiningOp())) {
      auto aifop = dyn_cast<mlir::AffineIfOp>(dep_token.getDefiningOp());
      auto then_terminator = aifop.getThenBlock()->getTerminator();
      for (auto operand : then_terminator->getOperands()) {
        if (auto op = operand.getDefiningOp()) {
          output.push_back(op);
        }
      }
      auto else_terminator = aifop.getElseBlock()->getTerminator();
      for (auto operand : else_terminator->getOperands()) {
        if (auto op = operand.getDefiningOp()) {
          output.push_back(op);
        }
      }
      return output;
    }
    return output;
  }

  // Insert a vertex v between two vertices a and b which were connected by an
  // edge
  void insertVertexBetweenTwoVertices(Graph::vertex_descriptor a,
                                      Graph::vertex_descriptor b,
                                      Graph::vertex_descriptor v, Graph &G) {
    if ((a != b) && (a != v) && (b != v)) {
      if (edge(a, b, G).second) { // if an edge exists
        remove_edge(a, b, G);
        if (!edge(a, v, G).second)
          add_edge(a, v, G);
        if (!edge(v, b, G).second)
          add_edge(v, b, G);
      }
    }
  }

  // Create start node for graph
  void connectStartNodeInDependencyGraph(dependencyGraph &G) {
    auto v = G.start_vertex;
    auto vp = boost::vertices(G.g);
    for (auto vit = vp.first; vit != vp.second; ++vit) {
      if ((v != *vit) && !in_degree(*vit, G.g)) {
        add_edge(v, *vit, G.g);
      }
    }
  }

  // Adds pointer from command graph to launch, partition and herd terminators
  void updatePointerFromGraphToHierarchyTerminator(dependencyGraph &G) {
    auto vp = boost::vertices(G.g);
    for (auto v = vp.first; v != vp.second; ++v) {
      if (G.g[*v].asyncEventType == "hierarchy_terminator") {
        G.terminator_vertex = *v;
        return;
      }
    }
  }

  // Adds pointer from hierarchy terminator to parent command graph
  void updatePointerFromHierarchyTerminatorToGraph(dependencyGraph &G,
                                                   dependencyGraph &subG) {
    auto vp = boost::vertices(subG.g);
    for (auto v = vp.first; v != vp.second; ++v) {
      if (subG.g[*v].asyncEventType == "hierarchy_terminator") {
        subG.g[*v].nextDependencyGraph = &G;
        return;
      }
    }
  }

  // Adds pointer from hierarchy op to sub command graph
  void updatePointerFromHierarchyOpToGraph(dependencyGraph &G) {
    unsigned idx = 0;
    auto vp = boost::vertices(G.g);
    for (auto v = vp.first; v != vp.second; ++v) {
      if (G.g[*v].asyncEventType == "hierarchy") {
        G.g[*v].nextDependencyGraph = &(G.subgraphs[idx]);
        idx++;
      }
    }
    assert(idx == G.subgraphs.size() &&
           "mismatch between # graphs and hierarchy ops");
  }

  // Returns the scf parent op from scf.yield op
  template <typename T> Operation *getScfParentOpFromYieldOp(scf::YieldOp op) {
    if (auto scfop = dyn_cast<T>(op->getParentOp())) {
      return scfop.getOperation();
    }
    return nullptr;
  }

  // Connects launch, partition and herd terminators
  void connectTerminatorInGraph(Graph &g) {
    auto vp = boost::vertices(g);
    Graph::vertex_descriptor terminator_v = 0;
    for (auto vit = vp.first; vit != vp.second; ++vit) {
      if (g[*vit].asyncEventType == "hierarchy_terminator") {
        terminator_v = *vit;
      }
    }
    if (terminator_v == 0)
      return;
    for (auto vit = vp.first; vit != vp.second; ++vit) {
      if ((terminator_v != *vit) && !out_degree(*vit, g) &&
          (g[*vit].asyncEventType != "start")) {
        add_edge(*vit, terminator_v, g);
      }
    }
  }

  // Find all vertices adjacent to given vertices in graph
  void
  findAdjacentVertices(std::vector<Graph::vertex_descriptor> vertices,
                       std::vector<Graph::vertex_descriptor> &adjacent_vertices,
                       Graph *G) {
    for (auto v : vertices) {
      auto adj_set = boost::adjacent_vertices(v, *G);
      for (auto v1 = adj_set.first; v1 != adj_set.second; ++v1) {
        bool found_duplicate = false;
        for (auto v2 : adjacent_vertices) {
          if (*v1 == v2) {
            found_duplicate = true;
          }
        }
        bool is_in_vertices = false;
        for (auto v3 : vertices) {
          if (*v1 == v3) {
            is_in_vertices = true;
          }
        }
        if (!found_duplicate && !is_in_vertices) {
          adjacent_vertices.push_back(*v1);
        }
      }
    }
  }

  // Remove vertices in vector a which already exist in vector b
  void removeRepeatedVertices(std::vector<Graph::vertex_descriptor> &a,
                              std::vector<Graph::vertex_descriptor> b) {
    for (auto v : b) {
      removeVertexFromVertices(a, v);
    }
  }

  // Remove a vertex from a vector of vertices
  void removeVertexFromVertices(std::vector<Graph::vertex_descriptor> &vector,
                                Graph::vertex_descriptor a) {
    if (vector.size()) {
      for (auto it = vector.begin(); it != vector.end(); ++it) {
        if (*it == a) {
          vector.erase(it);
          it--;
        }
      }
    }
  }

  bool hasPath(Graph::vertex_descriptor start_v, Graph::vertex_descriptor end_v,
               Graph &G, SmallVector<Graph::vertex_descriptor, 1> &vec) {

    vec.push_back(start_v);
    if (start_v == end_v)
      return true;
    int pathCount = 0;
    auto adj_set = boost::adjacent_vertices(start_v, G);
    for (auto adj_v = adj_set.first; adj_v != adj_set.second; ++adj_v) {
      SmallVector<Graph::vertex_descriptor, 1> tmp_vec;
      if (hasPath(*adj_v, end_v, G, tmp_vec)) {
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

  // Recursively reset all vertices in for loop body
  void resetGraphBetweenTwoVertices(Graph::vertex_descriptor start_v,
                                    Graph::vertex_descriptor end_v, Graph &G,
                                    runnerNode &c) {

    // Remove start_v from processed_vertices
    removeVertexFromVertices(c.processed_vertices, start_v);

    // Reset start_time and end_time
    G[start_v].start_time = 0;
    G[start_v].end_time = 0;

    if (start_v == end_v)
      return;

    SmallVector<Graph::vertex_descriptor, 1> vertices;
    if (hasPath(start_v, end_v, G, vertices)) {
      for (auto v : vertices) {
        removeVertexFromVertices(c.processed_vertices, v);
        // Reset start_time and end_time
        G[v].start_time = 0;
        G[v].end_time = 0;
        // If v is a hierarchy op, then recursively clear the entire subgraph
        if (G[v].asyncEventType == "hierarchy") {
          auto sub_c = G[v].nextDependencyGraph;
          auto start = sub_c->start_vertex;
          auto terminator_v = sub_c->terminator_vertex;
          auto sub_g = sub_c->g;
          auto sub_runner = sub_c->runner_node;
          resetGraphBetweenTwoVertices(start, terminator_v, sub_g, *sub_runner);
        }
        // Else if v is an scf.for op, then clear the cached trip count from
        // runner node
        else if (G[v].asyncEventType == "for_loop") {
          // Clear for loop trip count from runner node's cache
          for (auto it = c.loop_trip_count.begin();
               it != c.loop_trip_count.end(); it++) {
            if (it->first == (unsigned)getIdAttr(G[v].op)) {
              c.loop_trip_count.erase(it);
              break;
            }
          }
        }
      }
    }
  }

  // Initialize sub runner nodes from launch graph tree
  void initRunnerNodesFromLaunchGraph(runnerNode &launch_runner_node,
                                      dependencyGraph &launchGraph) {
    launchGraph.runner_node = &launch_runner_node;
    for (auto &partitionGraph : launchGraph.subgraphs) {
      // Create partition runner node
      launch_runner_node.sub_runner_nodes.push_back(
          runnerNode(&partitionGraph, "partition"));
      auto current_partition_node =
          &(launch_runner_node.sub_runner_nodes.back());
      for (auto &herdGraph : partitionGraph.subgraphs) {
        // Create herd runner node
        current_partition_node->sub_runner_nodes.push_back(
            runnerNode(&herdGraph, "herd"));
      }
    }
    addPointerBetweenSubRunnerNodeAndSubCommandGraph(launch_runner_node);
    for (auto &partition_runner_node : launch_runner_node.sub_runner_nodes) {
      addPointerBetweenSubRunnerNodeAndSubCommandGraph(partition_runner_node);
    }
  }

  // Adds pointer between runner node and command graph
  void addPointerBetweenSubRunnerNodeAndSubCommandGraph(runnerNode &R) {
    for (auto r_it = std::begin(R.sub_runner_nodes);
         r_it != std::end(R.sub_runner_nodes); ++r_it) {
      r_it->ctrl_g->runner_node = &(*r_it);
    }
  }

  // Get time stamps from wavefront
  void getTimeStampsFromWavefront(std::vector<uint64_t> &next_times,
                                  runnerNode runner_node) {
    for (auto it = runner_node.wavefront.begin();
         it != runner_node.wavefront.end(); it++) {
      auto command_node = runner_node.ctrl_g->g[it->first];
      if (command_node.is_started() && (command_node.end_time)) {
        next_times.push_back(command_node.end_time);
      }
    }
  }

  // Get a vector of first elements from a vector of pairs
  std::vector<Graph::vertex_descriptor> getVectorOfFirstFromVectorOfPairs(
      std::vector<std::pair<Graph::vertex_descriptor, unsigned>> pairs) {
    std::vector<Graph::vertex_descriptor> items;
    std::transform(pairs.begin(), pairs.end(), std::back_inserter(items),
                   [](const std::pair<Graph::vertex_descriptor, unsigned> &p) {
                     return p.first;
                   });
    return items;
  }

  // Push an entry into wavefront
  void pushToWavefront(
      std::vector<std::pair<Graph::vertex_descriptor, unsigned>> &wavefront,
      std::pair<Graph::vertex_descriptor, unsigned> entry) {
    for (auto i : wavefront) {
      assert(i.second != entry.second && "queried thread is busy");
    }
    wavefront.push_back(entry);
  }
  void pushToWavefront(
      std::vector<std::pair<Graph::vertex_descriptor, unsigned>> &wavefront,
      Graph::vertex_descriptor v) {
    // Acquire available thread id for current op
    unsigned tid = 0;
    for (unsigned i = 1; i < wavefront.size() + 2; i++) {
      bool tid_i_unavailable = false;
      for (auto j : wavefront) {
        if (j.second == i) {
          tid_i_unavailable = true;
        }
      }
      if (!tid_i_unavailable) {
        tid = i;
        break;
      }
    }
    wavefront.push_back(std::make_pair(v, tid));
  }

  uint64_t getTensorVolume(const mlir::ShapedType ty) {

    if (!ty.hasRank())
      return 1;

    uint64_t volume = 1;
    for (auto &d : ty.getShape())
      volume *= d;
    return volume;
  }

  uint64_t getTensorVolume(const mlir::Type ty) {
    if (auto t = ty.dyn_cast<mlir::ShapedType>()) {
      return getTensorVolume(t);
    } else {
      return 1;
    }
  }

  uint64_t getTransferCost(unsigned srcSpace, unsigned dstSpace,
                           mlir::Type ty) {
    return getTransferCost(srcSpace, dstSpace, getTensorVolume(ty));
  }

  uint64_t getTransferCost(unsigned srcSpace, unsigned dstSpace,
                           int64_t volume) {
    std::map<std::pair<unsigned, unsigned>, double> interface_bw;

    // defaults
    interface_bw.insert({{0, 1}, 100});
    interface_bw.insert({{1, 0}, 100});
    interface_bw.insert({{1, 2}, DBL_MAX});
    interface_bw.insert({{2, 1}, DBL_MAX});
    double cps = 0.0f;

    // override of defaults
    auto model = jsonModel.getAsObject();
    unsigned datawidth = 0;
    // if interfaces exists, assume everthing else exists
    if (model && model->getArray("interfaces")) {
      auto interfaces = model->getArray("interfaces");
      assert(interfaces);

      for (auto it = interfaces->begin(), ie = interfaces->end(); it != ie;
           ++it) {
        llvm::json::Value jv = *it;
        llvm::json::Object *interface = jv.getAsObject();
        assert(interface);
        auto srcSpace = interface->getNumber("src");
        auto dstSpace = interface->getNumber("dst");
        auto bps = interface->getNumber("bytes_per_second");
        assert(srcSpace && dstSpace && bps);
        unsigned s = *srcSpace;
        unsigned d = *dstSpace;
        double b = *bps;
        if (interface_bw.count({s, d}))
          interface_bw[{s, d}] = b;
        else
          interface_bw.insert({{s, d}, b});
      }
      if (auto d = model->getNumber("clock"))
        cps = *d;
      if (auto dt = model->getObject("datatype"))
        if (auto bytes = dt->getNumber("bytes"))
          datawidth = *bytes;
    }
    assert(cps != 0.0f && datawidth);

    double bytes = volume * datawidth;
    double bps = interface_bw[{srcSpace, dstSpace}];
    double seconds = bytes / bps;
    return (uint64_t)ceil(seconds * cps);
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