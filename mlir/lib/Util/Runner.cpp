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
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/RegionUtils.h"

#include <algorithm>
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

#include "./Runner/Resource.cpp"
#include "./Runner/RunnerNode.cpp"
// #include "./Runner/ResourceHierarchy.cpp"

#define DEBUG_TYPE "air-runner"

#define INDEX_WIDTH 32

using namespace mlir;
using namespace boost;

namespace xilinx {
namespace air {

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
    if (llvm::any_cast<llvm::APInt>(&value) != nullptr) {
      debugArg(head, op, llvm::any_cast<llvm::APInt>(value), time);
    } else if (llvm::any_cast<llvm::APFloat>(&value) != nullptr) {
      debugArg(head, op, llvm::any_cast<llvm::APFloat>(value), time);
    } else if (llvm::any_cast<unsigned>(&value) != nullptr) {
      // Represents an allocated buffer.
      LLVM_DEBUG(llvm::dbgs() << "  " << head << ":  " << op << " = Buffer "
                              << llvm::any_cast<unsigned>(value) << "\n");
    } else {
      // llvm_unreachable("unknown type");
    }
  }

public:
  AIRRunner_impl(llvm::raw_ostream &trace_stream, llvm::json::Value &json_model,
                 std::string sim_granularity = "herd", bool verbose = false)
      : traceStream(trace_stream), jsonModel(json_model),
        sim_granularity(sim_granularity) {

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

  void emitTraceMetadataEvent(llvm::raw_ostream &s, std::string item_name,
                              std::string arg_name, std::string arg_entry,
                              std::string ph, int64_t pid, int64_t tid = -1) {
    s << "{\n";
    s << "  \"name\": \"" << item_name << "\","
      << "\n";
    s << "  \"ph\": \"" << ph << "\","
      << "\n";
    s << "  \"pid\": " << pid << ","
      << "\n";
    if (tid != -1) {
      s << "  \"tid\": " << tid << ","
        << "\n";
    }
    s << "  \"args\": {\n";
    s << "    \"" << arg_name << "\": \"" << arg_entry << "\""
      << "\n";
    s << "  }\n";
    s << "},\n";
  }

  // Model each event's latency
  uint64_t modelOp(device &d, dependencyNodeEntry &c) {
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
        execution_time = getTransferCost(d, srcSpace, dstSpace, srcTy);
      else
        execution_time = getTransferCost(d, srcSpace, dstSpace, dstTy);
    } else if (type == "channel" &&
               (name.find("ChannelGetOp") != std::string::npos)) {
      auto getOp = mlir::dyn_cast<xilinx::air::ChannelGetOp>(c.op);
      assert(getOp);
      MemRefType dstTy = getOp.getDst().getType().cast<MemRefType>();
      std::vector<air::ChannelPutOp> putOps =
          air::getTheOtherChannelOpThroughSymbol(getOp);
      assert(putOps.size());
      MemRefType srcTy = putOps[0].getSrc().getType().cast<MemRefType>();
      auto srcSpace = srcTy.getMemorySpaceAsInt();
      auto dstSpace = dstTy.getMemorySpaceAsInt();
      // if there is a size mismatch, it's because we're moving a tile of the
      // larger tensor
      if (getTensorVolume(srcTy) <= getTensorVolume(dstTy))
        execution_time = getTransferCost(d, srcSpace, dstSpace, srcTy);
      else
        execution_time = getTransferCost(d, srcSpace, dstSpace, dstTy);
    } else if (type == "execute" && name != "ExecuteTerminatorOp") {
      assert(dyn_cast<air::ExecuteOp>(c.op));
      auto child_op = &*(c.op->getRegions().front().getOps().begin());
      if (auto Op = mlir::dyn_cast<linalg::LinalgOp>(child_op)) {
        uint64_t compute_xfer_cost = 0;
        uint64_t compute_op_cost = getComputeCost(d, child_op);
        execution_time = std::max(compute_op_cost, compute_xfer_cost);
      }
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "WARNING: execution time not modeled for op: '");
      LLVM_DEBUG(llvm::dbgs() << air::to_string(c.op) << "'\n");
      execution_time = 1;
    }
    return execution_time;
  }

  void processGraph(runnerNode &c, device &device_resource_node,
                    uint64_t time) {

    Graph &G = c.ctrl_g->g;

    // Update wavefront
    std::vector<Graph::vertex_descriptor> next_vertex_set_candidates;
    std::vector<Graph::vertex_descriptor> next_vertex_set;
    for (auto it = c.wavefront.begin(); it != c.wavefront.end(); ++it) {
      if (G[it->first].is_started() && G[it->first].is_done(time)) {

        if (G[it->first].asyncEventType != "start") {

          auto runner_id = getIdAttr(c.ctrl_g->hierarchyOp);
          auto tid = it->second;
          emitTraceEvent(traceStream,
                         G[it->first].asyncEventName +
                             G[it->first].detailed_description,
                         "layer", "E", time, tid, runner_id);
        }

        // "ExecuteOp"
        c.executeOpImpls(it->first, time);

        // Consume any loop-carried token
        c.consumeLoopYieldedTokens(it->first);

        // Erase from wavefront
        c.wavefront.erase(it);
        it--;
      }
    }

    // Get all adjacent vertices to the procssed vertices
    c.findAdjacentVerticesToProcessed(next_vertex_set_candidates);
    // Remove candidate vertices already on wavefront
    c.removeRepeatedVertices(next_vertex_set_candidates,
                             getVectorOfFirstFromVectorOfPairs(c.wavefront));
    // Remove candidate vertices which are filtered out by an affine.if, if
    // showing cores
    if (sim_granularity == "core") {
      c.removeOpsFilteredOutByAffineIf(next_vertex_set_candidates);
    }

    for (auto it = next_vertex_set_candidates.begin();
         it != next_vertex_set_candidates.end(); ++it) {
      bool dep_fulfilled = true;
      // Build it's dependency list. In each entry, the first field is a pointer
      // to the node, the second field is a string representing the type of
      // this dependency, either "ssa" or "sym", and the third field is the
      // token index, in case if op contains multiple tokens.
      std::vector<std::pair<dependencyNodeEntry, std::string>> dep_list;
      c.buildVertexDependencyList(*it, dep_list);
      // Check whether adj_v's dependency list is fulfilled
      if (isNonBlocking(G[*it].op)) {
        // If op is non-blocking
        dep_fulfilled =
            c.checkAllDependenciesFulfillment(dep_list, G[*it], time, false);
      } else {
        // Else (op is blocking)
        dep_fulfilled =
            c.checkAllDependenciesFulfillment(dep_list, G[*it], time, true);
      }
      if (dep_fulfilled) {
        next_vertex_set.push_back(*it);
      }
    }

    for (auto next_vertex : next_vertex_set) {

      // Push to wavefront; check if showing cores
      c.pushToWavefront(next_vertex,
                        canonicalizer.getIteratorFromPosition(
                            c.ctrl_g->position, c.ctrl_g->hierarchyOp));

      G[next_vertex].start_time = time;
      G[next_vertex].end_time =
          time + modelOp(device_resource_node, G[next_vertex]);
      // emit trace event begin
      auto runner_id = getIdAttr(c.ctrl_g->hierarchyOp);
      auto tid = c.wavefront.back().second;
      emitTraceEvent(traceStream,
                     G[next_vertex].asyncEventName +
                         G[next_vertex].detailed_description,
                     "layer", "B", time, tid, runner_id);
    }

    return;
  }

  void scheduleFunction(func::FuncOp &toplevel) {

    // Walk the launch op and create a boost graph using dependencyCanonicalizer
    // intepreter
    canonicalizer.removeDepListRepetition(toplevel);
    hostGraph = dependencyGraph(toplevel, true);
    canonicalizer.parseCommandGraphs(toplevel, hostGraph, dep_ctx,
                                     sim_granularity);

    // Walk the launch graph and write process name metadata in trace
    writeTraceMetadataProcNames(hostGraph);

    // Walk the json file and create resource model
    auto model = jsonModel.getAsObject();
    assert(model && "Failed to read JSON model");
    auto device_resource_node = device(model);

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
        launch_runner_node =
            runnerNode(&launchGraph, "launch", &dep_ctx, sim_granularity);
        // Update pointer to launch runner node in launch graph
        launchGraph.runner_node = &launch_runner_node;

        // Walk the launch graph and infer herd/partition runner nodes
        launch_runner_node.initRunnerNodesFromLaunchGraph(launchGraph);

        // Schedule launch runner node and its sub-runner nodes
        scheduleLaunch(launch_runner_node, device_resource_node, time);
      }
    }
  }

  void scheduleLaunch(runnerNode &launch, device &device_resource_node,
                      uint64_t &time) {

    auto start_v = launch.ctrl_g->start_vertex;
    // Reset launch graph
    launch.processed_vertices.clear();
    launch.resetGraphBetweenTwoVertices(
        start_v, launch.ctrl_g->terminator_vertex, launch.ctrl_g->g, time);
    // Start running launch
    bool running = true;
    launch.ctrl_g->g[start_v].start_time = 1;
    launch.ctrl_g->g[start_v].end_time = 1;
    launch.pushToWavefront(std::make_pair(start_v, 1));
    while (running) {
      LLVM_DEBUG(llvm::dbgs() << "time: " << time << "\n");

      running = false;
      std::vector<uint64_t> next_times;

      processGraph(launch, device_resource_node, time);
      if (launch.wavefront.size()) {
        running = true;
        // getTimeStampsFromWavefront(next_times, launch);
      }

      for (auto &partition_runner_node : launch.sub_runner_nodes) {
        processGraph(partition_runner_node, device_resource_node, time);
        if (partition_runner_node.wavefront.size()) {
          running = true;
          // getTimeStampsFromWavefront(next_times, partition_runner_node);
        }
        for (auto &herd_runner_node : partition_runner_node.sub_runner_nodes) {
          processGraph(herd_runner_node, device_resource_node, time);
          if (herd_runner_node.wavefront.size()) {
            running = true;
            // getTimeStampsFromWavefront(next_times, herd_runner_node);
          }
        }
      }

      if (running) {
        launch.getTimeStampsFromWavefront(next_times);
        for (auto &partition_runner_node : launch.sub_runner_nodes) {
          partition_runner_node.getTimeStampsFromWavefront(next_times);
          for (auto &herd_runner_node :
               partition_runner_node.sub_runner_nodes) {
            herd_runner_node.getTimeStampsFromWavefront(next_times);
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
  std::string sim_granularity;

  unsigned dispatch_slots;
  unsigned dispatch_dma_slots;
  unsigned core_dma_slots;
  unsigned herd_slots;

  // Dependency graph constructed as Boost graph
  dependencyGraph hostGraph;

  // Host and segment runnerNodes
  runnerNode launch_runner_node;

  //===----------------------------------------------------------------------===//
  // Trace helper functions
  //===----------------------------------------------------------------------===//

  // Write process names in trace metadata
  void writeTraceMetadataProcNames(dependencyGraph &hostGraph) {
    for (auto &launchGraph : hostGraph.subgraphs) {
      // Write launch process name to trace metadata
      emitTraceMetadataEvent(traceStream, "process_name", "name",
                             air::to_string(launchGraph.hierarchyOp), "M",
                             getIdAttr(launchGraph.hierarchyOp));
      emitTraceMetadataEvent(traceStream, "process_sort_index", "sort_index",
                             std::to_string(getIdAttr(launchGraph.hierarchyOp)),
                             "M", getIdAttr(launchGraph.hierarchyOp));
      for (auto &partitionGraph : launchGraph.subgraphs) {
        // Write partition process name to trace metadata
        emitTraceMetadataEvent(traceStream, "process_name", "name",
                               air::to_string(partitionGraph.hierarchyOp), "M",
                               getIdAttr(partitionGraph.hierarchyOp));
        emitTraceMetadataEvent(
            traceStream, "process_sort_index", "sort_index",
            std::to_string(getIdAttr(partitionGraph.hierarchyOp)), "M",
            getIdAttr(partitionGraph.hierarchyOp));
        for (auto &herdGraph : partitionGraph.subgraphs) {
          // Only write herd process name metadata once per herd
          bool print_pid_metadata_for_herd = true;
          // Write core thread name metadata if showing cores
          bool print_tid_metadata_for_core = false;
          if (herdGraph.position.size()) {
            print_tid_metadata_for_core = true;
            for (auto id : herdGraph.position) {
              if (id != 0) {
                print_pid_metadata_for_herd = false;
              }
            }
          }
          if (print_pid_metadata_for_herd) {
            // Write herd process name to trace metadata
            emitTraceMetadataEvent(traceStream, "process_name", "name",
                                   air::to_string(herdGraph.hierarchyOp), "M",
                                   getIdAttr(herdGraph.hierarchyOp));
            emitTraceMetadataEvent(
                traceStream, "process_sort_index", "sort_index",
                std::to_string(getIdAttr(herdGraph.hierarchyOp)), "M",
                getIdAttr(herdGraph.hierarchyOp));
          }
          if (print_tid_metadata_for_core) {
            // Write herd process name to trace metadata
            std::string thread_name =
                "core [" + to_string(herdGraph.position) + "]";
            // Hardcoded maximum number of threads per core
            unsigned max_num_threads_per_core = 10;
            unsigned core_id = canonicalizer.getIteratorFromPosition(
                                   herdGraph.position, herdGraph.hierarchyOp) *
                                   max_num_threads_per_core +
                               1;
            emitTraceMetadataEvent(traceStream, "thread_name", "name",
                                   thread_name, "M",
                                   getIdAttr(herdGraph.hierarchyOp), core_id);
            // Iteratively write thread sort index for every possible thread in
            // a core
            for (unsigned i = 0; i < max_num_threads_per_core; i++) {
              emitTraceMetadataEvent(traceStream, "thread_sort_index",
                                     "sort_index", std::to_string(core_id + i),
                                     "M", getIdAttr(herdGraph.hierarchyOp),
                                     core_id + i);
            }
          }
        }
      }
    }
  }

  //===----------------------------------------------------------------------===//
  // Latency estimation helper functions
  //===----------------------------------------------------------------------===//

  // Util. functions to estimate event latency
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

  std::string getElementTypeAsString(const mlir::Type ty) {
    if (auto st = ty.dyn_cast<mlir::ShapedType>()) {
      return to_string(st.getElementType());
    } else {
      return to_string(ty);
    }
  }

  uint64_t getTransferCost(device &d, unsigned srcSpace, unsigned dstSpace,
                           mlir::Type ty) {
    return getTransferCost(d, srcSpace, dstSpace, getTensorVolume(ty), ty);
  }

  uint64_t getTransferCost(device &d, unsigned srcSpace, unsigned dstSpace,
                           int64_t volume, mlir::Type ty) {
    double cps = 0.0f;
    unsigned datawidth = 0;
    if (d.ports.size()) {
      cps = d.clock;
      if (auto bytes = d.datatypes[getElementTypeAsString(ty)]) {
        datawidth = bytes;
      } else {
        assert(false && "data type not found in JSON model");
      }
    }
    assert(cps != 0.0f && datawidth);

    double bytes = volume * datawidth;
    assert(d.ports[std::make_pair(srcSpace, dstSpace)].size());
    double bps = d.ports[{srcSpace, dstSpace}][0]->data_rate;
    double seconds = bytes / bps;
    return (uint64_t)ceil(seconds * cps);
  }

  uint64_t getComputeCost(device &d, Operation *op) {
    uint64_t compute_op_cost = 0;
    auto opCounts = xilinx::air::CostModel().getOpCounts(op);
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
      double num_cores = 1;              // one because the post-tiling code in
                                         // air.herd's body is for each core
      double ops_per_core_per_cycle = 8; // vector width for this type
      double cycles_per_second = 1e9;
      double efficiency = 1.0f;

      auto model = jsonModel.getAsObject();
      assert(model);
      assert(d.kernels.size() && "kernels not found in JSON model");

      // if kernels exists, assume everthing else exists
      if (d.kernels.count(air::to_string(op))) {
        ops_per_core_per_cycle =
            d.kernels[air::to_string(op)]->ops_per_core_per_cycle;
        cycles_per_second = d.clock;
        cycles_per_second = d.kernels[air::to_string(op)]->efficiency;
        cycles_per_second = d.clock;
      }

      double ops_per_cycle = num_cores * ops_per_core_per_cycle * efficiency;
      assert(ops_per_cycle > 0 &&
             "ops per cycle in model must be greater than zero");

      double cycles = ceil(compute_op_count / ops_per_cycle);
      compute_op_cost = cycles;
    }
    return compute_op_cost;
  }

  //===----------------------------------------------------------------------===//
  // Dependency helper functions
  //===----------------------------------------------------------------------===//

  // Check if op is a non-blocking event
  bool isNonBlocking(Operation *op) {
    if (auto yield = dyn_cast<scf::YieldOp>(op)) {
      if (yield->getOperands().size() > 1) {
        // Multi-token for loop requires scf.yield to be non-blocking per async
        // token
        return true;
      }
    }
    return false;
  }

  //===----------------------------------------------------------------------===//
  // Misc. helper functions
  //===----------------------------------------------------------------------===//

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

  std::string to_string(std::vector<unsigned> vec) {
    std::string output = "";
    for (unsigned i = 0; i < vec.size(); i++) {
      output += std::to_string(vec[i]);
      if (i != vec.size() - 1) {
        output += ",";
      }
    }
    return output;
  }

  std::string to_string(dependencyNodeEntry &c) { return air::to_string(c.op); }
  std::string to_string(mlir::Type t) {
    std::string type_str;
    llvm::raw_string_ostream rso(type_str);
    t.print(rso);
    return type_str;
  }

}; // AIRRunner_impl

AIRRunner::AIRRunner(llvm::raw_ostream &trace_stream,
                     llvm::json::Value &json_model, std::string sim_granularity,
                     bool verbose) {
  impl = std::make_unique<AIRRunner_impl>(trace_stream, json_model,
                                          sim_granularity, verbose);
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