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

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/Any.h"
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
#include "mlir/Transforms/RegionUtils.h"

#include <algorithm>
#include <float.h>
#include <list>
#include <map>
#include <sstream>
#include <vector>

#include <algorithm>
#include <numeric>
#include <string>

#include "./Runner/Resource.cpp"
#include "./Runner/ResourceHierarchy.cpp"
#include "./Runner/RunnerNode.cpp"

#define DEBUG_TYPE "air-runner"

#define INDEX_WIDTH 32

using namespace mlir;

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
                 std::string sim_granularity = "herd",
                 std::string launch_iterations = "all", bool verbose = false)
      : traceStream(trace_stream), jsonModel(json_model),
        sim_granularity(sim_granularity), launch_iterations(launch_iterations) {

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
                      std::string ph, std::string ts, int64_t tid,
                      int64_t pid) {
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
      auto Op = mlir::dyn_cast<xilinx::air::DmaMemcpyNdOp>(c.op);
      if (!Op)
        c.op->emitOpError("has mismatching event type").attachNote()
            << "Has 'dma' as event type, but op isn't of type "
               "air::DmaMemcpyNdOp";
      MemRefType srcTy = llvm::cast<MemRefType>(Op.getSrcMemref().getType());
      MemRefType dstTy = llvm::cast<MemRefType>(Op.getDstMemref().getType());
      auto srcSpace = srcTy.getMemorySpaceAsInt();
      auto dstSpace = dstTy.getMemorySpaceAsInt();
      // if there is a size mismatch, it's because we're moving a tile of the
      // larger tensor
      if (getTensorVolume(srcTy) <= getTensorVolume(dstTy))
        execution_time = getTransferCost(d, c.op, srcSpace, dstSpace, srcTy);
      else
        execution_time = getTransferCost(d, c.op, srcSpace, dstSpace, dstTy);
    } else if (type == "channel" &&
               (name.find("ChannelGetOp") != std::string::npos)) {
      auto getOp = mlir::dyn_cast<xilinx::air::ChannelGetOp>(c.op);
      if (!getOp)
        c.op->emitOpError("has mismatching event type").attachNote()
            << "Has 'channel' as event type, but op isn't of type "
               "air::ChannelGetOp";
      MemRefType dstTy = llvm::cast<MemRefType>(getOp.getDst().getType());
      std::vector<air::ChannelPutOp> putOps =
          air::getTheOtherChannelOpThroughSymbol(getOp);
      if (!putOps.size())
        getOp->emitOpError("found no put op for air::ChannelGetOp");
      MemRefType srcTy = llvm::cast<MemRefType>(putOps[0].getSrc().getType());
      auto srcSpace = srcTy.getMemorySpaceAsInt();
      auto dstSpace = dstTy.getMemorySpaceAsInt();
      auto srcVolumn = getTransferVolumn(putOps[0]);
      auto dstVolumn = getTransferVolumn(getOp);
      // if there is a size mismatch, it's because we're moving a tile of the
      // larger tensor
      if (srcVolumn <= dstVolumn)
        execution_time =
            getTransferCost(d, c.op, srcSpace, dstSpace, srcVolumn, srcTy);
      else
        execution_time =
            getTransferCost(d, c.op, srcSpace, dstSpace, dstVolumn, dstTy);
    } else if (type == "execute" && name != "ExecuteTerminatorOp") {
      if (!isa<air::ExecuteOp>(c.op))
        c.op->emitOpError("has mismatching event type").attachNote()
            << "Has 'execute' as event type, but op isn't of type "
               "air::ExecuteOp";
      auto child_op = &dyn_cast<air::ExecuteOp>(c.op).getChildOps().front();
      if (auto Op = mlir::dyn_cast<linalg::LinalgOp>(child_op)) {
        uint64_t compute_xfer_cost = 0;
        uint64_t compute_op_cost = getComputeCostFromCostModel(d, child_op);
        execution_time = std::max(compute_op_cost, compute_xfer_cost);
        // Add extra cycles as base latency for linalg ops, to model the
        // overhead of external function.
        execution_time += 100;
      } else if (auto custom_op = dyn_cast<air::CustomOp>(child_op)) {
        execution_time = getComputeCostFromJSON(d, custom_op);
      }
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "WARNING: execution time not modeled for op: '");
      LLVM_DEBUG(llvm::dbgs() << air::to_string(c.op) << "'\n");
      execution_time = 1;
    }
    return execution_time;
  }

  bool processGraph(runnerNode &c, device &device_resource_node,
                    uint64_t time) {

    LLVM_DEBUG(llvm::dbgs() << "\nNEW TIME STAMP @" << time - 1 << " runner "
                            << air::to_string(c.ctrl_g->hierarchyOp) << " loc "
                            << air::to_string(c.ctrl_g->position) << "'\n");

    executeOpsFromWavefrontAndFreeResource(c, device_resource_node, time);
    pushOpsToWavefrontAndAllocateResource(c, device_resource_node, time);

    return !c.wavefront.empty();
  }

  void executeOpsFromWavefrontAndFreeResource(runnerNode &c,
                                              device &device_resource_node,
                                              uint64_t time) {

    Graph &G = c.ctrl_g->g;

    // Pre-process the wavefront by moving terminator ops to the back
    // Note: Reason for sorting the wavefront is because executing terminator
    // event may change the execution status of other ops on wavefront
    for (int i = c.wavefront.size() - 1; i >= 0; i--) {
      if (G[std::get<0>(c.wavefront[i])].asyncEventType == "terminator") {
        moveItemToBack<
            std::tuple<Graph::VertexId, std::vector<resource *>, unsigned>>(
            c.wavefront, i);
      }
    }
    // Update wavefront
    for (auto it = c.wavefront.begin(); it != c.wavefront.end(); ++it) {
      if (G[std::get<0>(*it)].is_started() &&
          G[std::get<0>(*it)].is_done(time)) {

        if (G[std::get<0>(*it)].asyncEventType != "start") {

          auto runner_id = getIdAttr(c.ctrl_g->hierarchyOp);
          auto tid = std::get<2>(*it);
          emitTraceEvent(traceStream,
                         G[std::get<0>(*it)].asyncEventName +
                             G[std::get<0>(*it)].detailed_description,
                         "layer", "E",
                         convertToTimeStampInStr(time, device_resource_node),
                         tid, runner_id);
        }

        // "ExecuteOp"
        c.executeOpImpls(std::get<0>(*it), time);

        // Consume any loop-carried token
        c.consumeLoopYieldedTokens(std::get<0>(*it));

        // Erase from wavefront
        c.wavefront.erase(it);
        it--;
      }
    }
  }

  bool pushOpsToWavefrontAndAllocateResource(runnerNode &c,
                                             device &device_resource_node,
                                             uint64_t time) {

    Graph &G = c.ctrl_g->g;

    // Get candidate vertices to be pushed to wavefront
    std::vector<Graph::VertexId> next_vertex_set_candidates =
        c.getCandidateVerticesForWavefront();

    // Check dependency fulfillment of each candidate
    std::vector<Graph::VertexId> next_vertex_set;
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

    // Check resource fulfillment of each candidate
    for (auto next_vertex : next_vertex_set) {

      // Check whether adj_v's resource requirement has been fulfilled.
      bool res_fulfilled = c.checkResourceFulfillmentForOpImpls(G[next_vertex]);

      if (res_fulfilled) {
        // Delete vertex from latent wavefront candidates
        c.removeVertexFromVertices(c.latent_wavefront_candidates, next_vertex);
        // Push to wavefront; check for sim. granularity
        c.pushToWavefront(next_vertex,
                          canonicalizer.getIteratorFromPosition(
                              c.ctrl_g->position, c.ctrl_g->hierarchyOp));

        G[next_vertex].start_time = time;
        G[next_vertex].end_time =
            time + modelOp(device_resource_node, G[next_vertex]);
        // emit trace event begin
        auto runner_id = getIdAttr(c.ctrl_g->hierarchyOp);
        auto tid = std::get<2>(c.wavefront.back());
        emitTraceEvent(
            traceStream,
            G[next_vertex].asyncEventName + G[next_vertex].detailed_description,
            "layer", "B", convertToTimeStampInStr(time, device_resource_node),
            tid, runner_id);
      }
    }

    return c.wavefront.size() > 0;
  }

  void scheduleFunction(func::FuncOp &toplevel) {

    // Walk the launch op and create a graph using dependencyCanonicalizer
    // intepreter
    hostGraph = dependencyGraph(toplevel, true);
    canonicalizer.parseCommandGraphs(toplevel, hostGraph, dep_ctx,
                                     sim_granularity);

    // Walk the launch graph and write process name metadata in trace
    writeTraceMetadataProcNames(hostGraph);

    // Walk the json file and create resource model
    auto model = jsonModel.getAsObject();
    if (!model)
      toplevel->emitOpError("failed to read JSON model");
    auto device_resource_node = device(model);

    uint64_t time = 1;
    int64_t iter_count = 1;
    for (auto &launchGraph : hostGraph.subgraphs) {

      // air launch iteration space
      iter_count = 1;
      auto launch_op = dyn_cast<air::LaunchOp>(launchGraph.hierarchyOp);
      for (auto s_op : launch_op.getSizeOperands()) {
        int64_t s = cast<arith::ConstantIndexOp>(s_op.getDefiningOp()).value();
        iter_count *= s;
      }

      // Determine iteration count based on launch_iterations option
      int64_t actual_iter_count = iter_count;
      if (launch_iterations == "single") {
        actual_iter_count = 1;
      }

      for (unsigned i = 0; i < actual_iter_count; i++) {

        // Reset controllers
        launch_runner_node = runnerNode(nullptr, &launchGraph, "launch",
                                        &dep_ctx, sim_granularity);
        // Update pointer to launch runner node in launch graph
        launchGraph.runner_node = &launch_runner_node;

        // Walk the launch graph and infer herd/segment runner nodes
        launch_runner_node.initRunnerNodesFromLaunchGraph(launchGraph);

        // Schedule launch runner node and its sub-runner nodes
        scheduleLaunch(launch_runner_node, device_resource_node, time);
      }
    }

    // Simulation performance report
    std::string end_ts = convertToTimeStampInStr(time, device_resource_node);
    if (launch_iterations == "single" && iter_count > 1) {
      // In single-iteration mode, multiply by total iteration count
      double latency_us = std::stod(end_ts) * iter_count;
      std::cout << "Latency (single-iteration mode, estimated for "
                << iter_count << " iterations): " << latency_us << "us\n";
    } else {
      // All-iterations mode or single iteration total
      std::cout << "Latency (all-iterations mode): " << end_ts << "us\n";
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
    launch.pushStartToWavefront(start_v);
    // Consume devices upon launch
    // TODO: multi-device modelling
    launch.resource_hiers.push_back(&device_resource_node);

    while (running) {
      LLVM_DEBUG(llvm::dbgs() << "time: " << time << "\n");

      running = false;
      std::vector<uint64_t> next_times;

      running |= processGraph(launch, device_resource_node, time);

      for (auto &segment_runner_node : launch.sub_runner_nodes) {
        running |=
            processGraph(segment_runner_node, device_resource_node, time);
        for (auto &herd_runner_node : segment_runner_node.sub_runner_nodes) {
          running |= processGraph(herd_runner_node, device_resource_node, time);
        }
      }

      // Check event readiness again after updates to resource allocation
      running |= pushOpsToWavefrontAndAllocateResource(
          launch, device_resource_node, time);

      for (auto &segment_runner_node : launch.sub_runner_nodes) {
        running |= pushOpsToWavefrontAndAllocateResource(
            segment_runner_node, device_resource_node, time);
        for (auto &herd_runner_node : segment_runner_node.sub_runner_nodes) {
          running |= pushOpsToWavefrontAndAllocateResource(
              herd_runner_node, device_resource_node, time);
        }
      }

      if (running) {
        launch.getTimeStampsFromWavefront(next_times);
        for (auto &segment_runner_node : launch.sub_runner_nodes) {
          segment_runner_node.getTimeStampsFromWavefront(next_times);
          for (auto &herd_runner_node : segment_runner_node.sub_runner_nodes) {
            herd_runner_node.getTimeStampsFromWavefront(next_times);
          }
        }
      }

      uint64_t next_time = 0;
      if (next_times.size())
        next_time = *std::min_element(next_times.begin(), next_times.end());
      time = std::max(time + 1, next_time);
      if (time > 5000000000)
        running = false;
    }
  }

private:
  dependencyCanonicalizer canonicalizer;
  xilinx::air::dependencyContext dep_ctx;

  llvm::raw_ostream &traceStream;
  llvm::json::Value &jsonModel;
  std::string sim_granularity;
  std::string launch_iterations;

  unsigned dispatch_slots;
  unsigned dispatch_dma_slots;
  unsigned core_dma_slots;
  unsigned herd_slots;

  // Dependency graph constructed
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
      for (auto &segmentGraph : launchGraph.subgraphs) {
        // Write segment process name to trace metadata
        std::string seg_process_info = "";
        auto seg = dyn_cast<air::SegmentOp>(segmentGraph.hierarchyOp);
        seg_process_info += air::to_string(seg);
        seg_process_info += "[" + std::to_string(*seg.getNumCols()) + ", " +
                            std::to_string(*seg.getNumRows()) + "]";
        emitTraceMetadataEvent(traceStream, "process_name", "name",
                               seg_process_info, "M", getIdAttr(seg));
        emitTraceMetadataEvent(traceStream, "process_sort_index", "sort_index",
                               std::to_string(getIdAttr(seg)), "M",
                               getIdAttr(seg));
        for (auto &herdGraph : segmentGraph.subgraphs) {
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
            std::string herd_process_info = "";
            auto herd = dyn_cast<air::HerdOp>(herdGraph.hierarchyOp);
            herd_process_info += air::to_string(herd);
            herd_process_info += "[" + std::to_string(herd.getNumCols()) +
                                 ", " + std::to_string(herd.getNumRows()) + "]";
            emitTraceMetadataEvent(traceStream, "process_name", "name",
                                   herd_process_info, "M", getIdAttr(herd));
            emitTraceMetadataEvent(
                traceStream, "process_sort_index", "sort_index",
                std::to_string(getIdAttr(herd)), "M", getIdAttr(herd));
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

  // Convert time from cycle count to time stamp in ms (with 3 d.p.)
  std::string convertToTimeStampInStr(uint64_t time, device &d) {
    uint64_t time_in_ns =
        (uint64_t)std::round(((double)time) / (1000000000.0 / (double)d.clock));
    uint64_t int_part = (uint64_t)(time_in_ns / 1000);
    uint64_t frac_part = (uint64_t)(time_in_ns % 1000);
    std::string zero_fill = "";
    if (frac_part < 10)
      zero_fill = "00";
    else if (frac_part < 100)
      zero_fill = "0";
    return std::to_string(int_part) + "." + zero_fill +
           std::to_string(frac_part);
  }

  //===----------------------------------------------------------------------===//
  // Latency estimation helper functions
  //===----------------------------------------------------------------------===//

  uint64_t getTransferCost(device &d, Operation *op, unsigned srcSpace,
                           unsigned dstSpace, mlir::Type ty) {
    return getTransferCost(d, op, srcSpace, dstSpace, getTensorVolume(ty), ty);
  }

  uint64_t getTransferCost(device &d, Operation *op, unsigned srcSpace,
                           unsigned dstSpace, int64_t volume, mlir::Type ty) {
    double cps = d.clock;
    if (cps == 0.0f) {
      op->emitError("device clock frequency not found in JSON model");
    }
    unsigned datawidth = 0;
    if (auto bytes = d.datatypes[getElementTypeAsString(ty)]) {
      datawidth = bytes;
      if (!datawidth)
        op->emitOpError("found data type with zero width in JSON model");
    } else
      op->emitOpError("data type not found in JSON model");

    double bytes = volume * datawidth;
    double bps = d.interfaces[{srcSpace, dstSpace}]->data_rate;
    if (bps == 0.0f)
      op->emitOpError("data rate not found in JSON model");
    double seconds = bytes / bps;
    return (uint64_t)ceil(seconds * cps);
  }

  uint64_t getTransferVolumn(air::ChannelInterface op) {
    MemRefType memTy = llvm::cast<MemRefType>(op.getMemref().getType());
    if (op.getSizes().empty())
      return getTensorVolume(memTy);
    else
      return getVolumnFromSizes(op.getSizes());
  }

  uint64_t getVolumnFromSizes(SmallVector<Value> sizes) {
    uint64_t output = 1;
    for (auto s : sizes) {
      auto op = s.getDefiningOp();
      if (auto cIOp = dyn_cast_if_present<arith::ConstantIndexOp>(op)) {
        output *= cIOp.value();
      } else if (op)
        op->emitOpError("non-static shape for data movement");
    }
    return output;
  }

  uint64_t getComputeCostFromCostModel(device &d, Operation *op) {
    uint64_t compute_op_cost = 0;
    auto opCounts = xilinx::air::CostModel().getOpCounts(op);
    std::string skip = "footprint";
    std::string memops = "reads;writes;";
    std::string cpuops = "math.rsqrt;";
    cpuops += "arith.mulf;arith.divf;arith.addf;arith.subf;arith.truncf;"
              "arith.cmpf;arith.maxf;";
    cpuops += "arith.muli;arith.divsi;arith.divsi;arith.addi;arith.subi;"
              "arith.trunci;arith.cmpi;arith.maxi";
    cpuops += "std.select";
    uint64_t compute_op_count = 0;
    for (auto &p : opCounts.map) {
      auto name = std::get<0>(p);
      auto count = std::get<1>(p);
      if (memops.find(name) != std::string::npos) {
      } else if (cpuops.find(name) != std::string::npos)
        compute_op_count += count;
      else if (skip.find(name) == std::string::npos)
        LLVM_DEBUG(llvm::dbgs() << name << " not counted\n");
    }

    if (compute_op_count) {
      // defaults
      double num_cores = 1;              // one because the post-tiling code in
                                         // air.herd's body is for each core
      double ops_per_core_per_cycle = 8; // vector width for this type
      double efficiency = 1.0f;

      // if kernels exists, assume everthing else exists
      // Get operation datatype as the first operand's datatype
      auto op_datatype = getElementTypeAsString(op->getOperandTypes()[0]);
      if (d.kernels.count(air::to_string(op))) {
        if (d.kernels[air::to_string(op)]->datatypes.count(op_datatype)) {
          ops_per_core_per_cycle =
              d.kernels[air::to_string(op)]->datatypes[op_datatype].second;
          efficiency =
              d.kernels[air::to_string(op)]->datatypes[op_datatype].first;
        }
      }

      double ops_per_cycle = num_cores * ops_per_core_per_cycle * efficiency;
      if (ops_per_cycle <= 0)
        op->emitOpError("ops per cycle in model must be greater than zero");

      double cycles = ceil(compute_op_count / ops_per_cycle);
      compute_op_cost = cycles;
    }
    return compute_op_cost;
  }

  uint64_t getComputeCostFromJSON(device &d, air::CustomOp op) {
    // TODO: read custom kernels directly from device model d
    double cycles = 1.0;
    auto model = jsonModel.getAsObject();
    if (!model)
      op->emitOpError("failed to read JSON model");

    // if kernels exists, look up air.custom op and its symbolic name
    auto op_sym_name =
        op->getAttrOfType<StringAttr>(mlir::SymbolTable::getSymbolAttrName())
            .str();
    auto op_datatype = getElementTypeAsString(op->getOperandTypes()[0]);
    auto kernels = model->getObject("custom_kernels");
    if (kernels) {
      auto kernel = kernels->getObject(op_sym_name);
      if (kernel) {
        auto kernel_ty = kernel->getObject("datatypes")->getObject(op_datatype);
        if (kernel_ty && kernel_ty->getNumber("latency")) {
          cycles = *kernel_ty->getNumber("latency");
        } else {
          op->emitOpError("unknown data type ")
              << op_datatype << " for custom kernel " << op_sym_name;
        }
      } else {
        op->emitOpError("found no custom kernel named ") << op_sym_name;
      }
    } else {
      op->emitOpError("found no custom_kernels obj. in JSON");
    }
    return cycles;
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

  // Move an element of the vector to the back
  template <typename T>
  void moveItemToBack(std::vector<T> &v, size_t itemIndex) {
    auto it = v.begin() + itemIndex;
    std::rotate(it, it + 1, v.end());
  }

}; // AIRRunner_impl

AIRRunner::AIRRunner(llvm::raw_ostream &trace_stream,
                     llvm::json::Value &json_model, std::string sim_granularity,
                     std::string launch_iterations, bool verbose) {
  impl = std::make_unique<AIRRunner_impl>(
      trace_stream, json_model, sim_granularity, launch_iterations, verbose);
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

//===----------------------------------------------------------------------===//
// Runner util. functions
//===----------------------------------------------------------------------===//

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

std::string lookUpMemorySpaceFromInt(unsigned memory_space) {
  std::string output = "";
  if (memory_space == 0) {
    output += "L3";
  } else if (memory_space == 1) {
    output += "L2";
  } else if (memory_space == 2) {
    output += "L1";
  }
  return output;
}

unsigned lookUpMemorySpaceIntFromString(std::string memory_space) {
  unsigned output = 0;
  if (memory_space == "L3") {
    output = 0;
  } else if (memory_space == "L2") {
    output = 1;
  } else if (memory_space == "L1") {
    output = 2;
  }
  return output;
}

template <typename T>
void push_back_if_unique(std::vector<T> &vec, T entry) {
  if (std::find(vec.begin(), vec.end(), entry) == vec.end()) {
    vec.push_back(entry);
  }
}

} // namespace air
} // namespace xilinx
