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
  // Each entry is an std::tuple. First element is for op's id, second element
  // is the loop's async token id, and third element is trip counter.
  std::vector<std::tuple<unsigned, unsigned, unsigned>> loop_trip_count;
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
                              std::string ph, int64_t pid) {
    s << "{\n";
    s << "  \"name\": \"" << item_name << "\","
      << "\n";
    s << "  \"ph\": \"" << ph << "\","
      << "\n";
    s << "  \"pid\": " << pid << ","
      << "\n";
    s << "  \"args\": {\n";
    s << "    \"" << arg_name << "\": \"" << arg_entry << "\""
      << "\n";
    s << "  }\n";
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
                                 *sub_runner_node, time);
    sub_runner_node->loop_trip_count.clear();

    // Start sub-runner node by pushing start node into its wavefront
    sub_runner_node->ctrl_g->g[sub_start_v].start_time = time;
    sub_runner_node->ctrl_g->g[sub_start_v].end_time = time;
    assert(!sub_runner_node->wavefront.size() && "Sub runner node is busy");
    pushToWavefront(sub_runner_node->wavefront, std::make_pair(sub_start_v, 1));

    sub_runner_node->processed_vertices.clear();

    c.processed_vertices.push_back(it);
  }

  void executeOp(scf::YieldOp op, uint64_t time, scf::ForOp for_op,
                 runnerNode &c, Graph::vertex_descriptor it) {
    Graph &G = c.ctrl_g->g;

    // Get async tokens ready to iterate at scf.yield
    std::vector<unsigned> token_ids;
    std::vector<bool> token_is_still_iterating;
    getReadyTokensAtScfYield(token_ids, op, time, G);

    // For loop trip counter
    bool trip_count_fulfilled = false;
    for (auto &count_entry : c.loop_trip_count) {
      if (std::get<0>(count_entry) ==
          (unsigned)getIdAttr(for_op.getOperation())) {
        for (int i = token_ids.size() - 1; i >= 0; i--) {
          if (std::get<1>(count_entry) == token_ids[i]) {
            if (std::get<2>(count_entry)) {
              // Decrement token count if this token still needs to iterate
              std::get<2>(count_entry)--;
            }
            if (!std::get<2>(count_entry)) {
              // If this token's iteration cound is fulfilled, then delete this
              // token from ready token list
              token_is_still_iterating.push_back(false);
            } else {
              token_is_still_iterating.push_back(true);
            }
          }
        }
      }
    }

    // If all async tokens' trip counts are fulfilled
    bool allAsyncTokensFulfilled = true;
    for (auto &count_entry : c.loop_trip_count) {
      if (std::get<0>(count_entry) ==
          (unsigned)getIdAttr(for_op.getOperation())) {
        if (std::get<2>(count_entry)) {
          allAsyncTokensFulfilled = false;
        }
      }
    }

    if (allAsyncTokensFulfilled) {
      c.processed_vertices.push_back(it);
      trip_count_fulfilled = true;
    } else {
      // If trip count unfulfilled, then iterate.
      // Clear start_time and end_time of all ops in loop body.
      // From processed_vertices, remove all ops which are in loop body.
      for (unsigned i = 0; i < token_ids.size(); i++) {
        // Get the yielded token in the next loop iteration (at the beginning of
        // the loop)
        auto next_iter_token = for_op.getRegionIterArgs()[token_ids[i]];
        assert(next_iter_token);

        // Search for vertices corresponding to the next-iteration incarnations
        // of this token
        auto for_v =
            canonicalizer
                .getVertexFromOp(for_op.getOperation(), dep_ctx, "front")
                .first;
        auto adj_set = boost::adjacent_vertices(for_v, G);
        for (auto adj_v = adj_set.first; adj_v != adj_set.second; ++adj_v) {
          auto adj_op = G[*adj_v].op;
          assert(adj_op);
          for (auto d : adj_op->getOperands()) {
            if (d == next_iter_token) {
              // To start the next loop iteration:
              // (1) reset graph wrt this token
              resetGraphBetweenTwoVertices(*adj_v, it, G, c, time);
              // (2) release the token locks, if the token is still iterating
              if (token_is_still_iterating[i]) {
                G[*adj_v].token_count +=
                    tokenSpatialFactor(G[*adj_v].op, c.ctrl_g->position);
              }
            }
          }
        }
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

    // Update for loop trip count per async token
    for (unsigned i = 0; i < op.getRegionIterArgs().size(); i++) {
      c.loop_trip_count.push_back(
          std::make_tuple(getIdAttr(op.getOperation()), i, trip_count));
    }

    // Release the locks for all async tokens adjacent to scf.for, to initiate
    // the first iteration.
    Graph &G = c.ctrl_g->g;
    auto adj_set = boost::adjacent_vertices(it, G);
    for (auto adj_v = adj_set.first; adj_v != adj_set.second; ++adj_v) {
      G[*adj_v].token_count +=
          tokenCountThresholdForExecution(G[*adj_v].op) *
          tokenSpatialFactor(
              G[*adj_v].op,
              c.ctrl_g
                  ->position); // Lock number = number of dependent iter_args
    }

    c.processed_vertices.push_back(it);
  }

  void executeOp(air::ChannelPutOp op, runnerNode &c,
                 Graph::vertex_descriptor it) {
    Graph &G = c.ctrl_g->g;
    G[it].token_count +=
        tokenSpatialFactor(op.getOperation(), c.ctrl_g->position);

    c.processed_vertices.push_back(it);
  }

  void executeOp(air::ChannelGetOp op, runnerNode &c,
                 Graph::vertex_descriptor it) {
    // Get put-side runner from get op
    std::vector<air::ChannelPutOp> puts =
        air::getTheOtherChannelOpThroughSymbol(op);
    // Go through put ops and consume token in order
    for (auto put : puts) {
      auto put_entry =
          canonicalizer.getVertexFromOp(put.getOperation(), dep_ctx, "front");
      auto put_v = put_entry.first;
      auto &put_g = put_entry.second;
      auto &put_node = put_g->g[put_v];
      if (put_node.token_count) {
        put_node.token_count -=
            tokenSpatialFactor(op.getOperation(), c.ctrl_g->position);
        break;
      }
    }

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
      for (auto sub_dependency_graph : node.nextDependencyGraphs) {
        auto sub_runner_node = sub_dependency_graph->runner_node;
        executeOp(Op, time, sub_runner_node, c, it);
      }
    } else if (auto Op = dyn_cast<scf::ForOp>(node.op)) {
      executeOp(Op, c, it);
    } else if (dyn_cast<scf::YieldOp>(node.op) &&
               getScfParentOpFromYieldOp<scf::ForOp>(
                   dyn_cast<scf::YieldOp>(node.op))) {
      auto Op = dyn_cast<scf::YieldOp>(node.op);
      auto parent_for_op =
          dyn_cast<scf::ForOp>(getScfParentOpFromYieldOp<scf::ForOp>(Op));
      executeOp(Op, time, parent_for_op, c, it);
    } else if (auto Op = dyn_cast<air::ChannelPutOp>(node.op)) {
      executeOp(Op, c, it);
    } else if (auto Op = dyn_cast<air::ChannelGetOp>(node.op)) {
      executeOp(Op, c, it);
    } else {
      executeOp(c, it);
    }
  }

  // Consume tokens upon op execution
  void consumeLoopYieldedTokens(runnerNode &c, Graph::vertex_descriptor it) {

    Graph &G = c.ctrl_g->g;
    auto inv_adj_set = boost::inv_adjacent_vertices(it, G);
    for (auto inv_adj_v = inv_adj_set.first; inv_adj_v != inv_adj_set.second;
         ++inv_adj_v) {
      if (G[*inv_adj_v].asyncEventType == "for_loop") {
        unsigned th = tokenCountThresholdForExecution(
            G[it].op); // Consume all iter_arg tokens
        assert(G[it].token_count >=
               th * tokenSpatialFactor(G[it].op, c.ctrl_g->position));
        G[it].token_count -=
            th * tokenSpatialFactor(G[it].op, c.ctrl_g->position);
        return;
      }
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
          double num_cores = 1; // one because the post-tiling code in
                                // air.herd's body is for each core
          double ops_per_core_per_cycle = 8; // vector width for this type
          double cycles_per_second = 1e9;
          double efficiency = 1.0f;

          auto model = jsonModel.getAsObject();
          assert(model);

          // if kernels exists, assume everthing else exists
          if (model && model->getObject("kernels")) {
            // device level override of defaults
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
      std::vector<air::ChannelPutOp> channel_puts =
          air::getTheOtherChannelOpThroughSymbol(channel_get);
      // Get any ChannelPut node with token. Otherwise push the last one to dep
      // list
      bool pushed_to_dep_list = false;
      for (auto channel_put : channel_puts) {
        auto channel_put_entry = canonicalizer.getVertexFromOp(
            channel_put.getOperation(), dep_ctx, "front");
        auto channel_put_v = channel_put_entry.first;
        auto &channel_put_g = channel_put_entry.second;
        auto &channel_put_node = channel_put_g->g[channel_put_v];
        if (channel_put_node.token_count > 0) {
          dep_list.push_back(std::make_pair(&channel_put_node, "sym"));
          pushed_to_dep_list = true;
          break;
        }
        if (channel_put == channel_puts.back()) {
          dep_list.push_back(std::make_pair(&channel_put_node, "sym"));
        }
      }
    }
    for (auto inv_adj_v = inv_adj_set.first; inv_adj_v != inv_adj_set.second;
         ++inv_adj_v) {
      // If dependent on a hierarchy op, then push its terminator into dep_list
      // instead
      if (G[*inv_adj_v].asyncEventType == "hierarchy") {
        for (auto sub_g : G[*inv_adj_v].nextDependencyGraphs) {
          auto terminator_v = sub_g->terminator_vertex;
          dep_list.push_back(std::make_pair(&sub_g->g[terminator_v], "ssa"));
        }
      } else if (G[*inv_adj_v].asyncEventType == "for_loop") {
        dep_list.push_back(std::make_pair(&G[*inv_adj_v], "ssa_loop_yield"));
      } else {
        dep_list.push_back(std::make_pair(&G[*inv_adj_v], "ssa"));
      }
    }
  }

  // Check if a dependence has been fulfilled
  bool checkEachDependenceFulfillment(
      std::pair<dependencyNodeEntry *, std::string> dep,
      dependencyNodeEntry *node, uint64_t time) {
    if (dep.second == "ssa") {
      if ((!dep.first->is_started()) || (!dep.first->is_done(time))) {
        // If source and sink of dep are both under the same loop
        if (node && dep.first &&
            shareInnerMostForLoop(node->op, dep.first->op)) {
          // Check node's timestamp log, in case if it has executed in previous
          // loop iterations
          unsigned dep_iter_count = dep.first->start_end_time_log.size();
          unsigned node_iter_count = node->start_end_time_log.size();
          if (node->is_started() && node->is_done(time))
            node_iter_count++;
          if (dep_iter_count <= node_iter_count) {
            return false;
          }
        } else
          return false;
      }
    } else if (dep.second == "ssa_loop_yield") {
      // Threshold token_count for dep fulfillment = how many iter_args does
      // node depend on
      unsigned th = tokenCountThresholdForExecution(node->op);
      if (node->token_count < th * tokenSpatialFactor(node->op, {})) {
        return false;
      }
    } else if (dep.second == "sym") {
      if (dep.first->token_count <= 0) {
        return false;
      }
    } else {
      assert(false && "Unknown async token type");
    }
    return true;
  }

  std::string to_string(Operation *op) {
    return op->getName().getStringRef().str();
  }

  // Check if all dependencies of an async op have been fulfilled
  bool checkAllDependenciesFulfillment(
      std::vector<std::pair<dependencyNodeEntry *, std::string>> dep_list,
      dependencyNodeEntry *node, uint64_t time, bool isBlocking) {
    bool dep_fulfilled = true;
    if (isBlocking) {
      dep_fulfilled = true;
      for (auto &dep : dep_list) {
        dep_fulfilled =
            dep_fulfilled && checkEachDependenceFulfillment(dep, node, time);
      }
    } else {
      dep_fulfilled = false;
      for (auto &dep : dep_list) {
        dep_fulfilled =
            dep_fulfilled || checkEachDependenceFulfillment(dep, node, time);
      }
    }
    return dep_fulfilled;
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
          auto tid = it->second;
          emitTraceEvent(traceStream,
                         G[it->first].asyncEventName +
                             G[it->first].detailed_description,
                         "layer", "E", time, tid, runner_id);
        }

        // "ExecuteOp"
        executeOpImpls(c, it->first, time);

        // Consume any loop-carried token
        consumeLoopYieldedTokens(c, it->first);

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
      // to the node, the second field is a string representing the type of
      // this dependency, either "ssa" or "sym", and the third field is the
      // token index, in case if op contains multiple tokens.
      std::vector<std::pair<dependencyNodeEntry *, std::string>> dep_list;
      buildVertexDependencyList(*it, G, dep_list);
      // Check whether adj_v's dependency list is fulfilled
      if (isNonBlocking(G[*it].op)) {
        // If op is non-blocking
        dep_fulfilled =
            checkAllDependenciesFulfillment(dep_list, &G[*it], time, false);
      } else {
        // Else (op is blocking)
        dep_fulfilled =
            checkAllDependenciesFulfillment(dep_list, &G[*it], time, true);
      }
      if (dep_fulfilled) {
        next_vertex_set.push_back(*it);
      }
    }

    for (auto next_vertex : next_vertex_set) {

      // Push to wavefront; check if showing cores
      pushToWavefront(c.wavefront, next_vertex,
                      canonicalizer.getIteratorFromPosition(
                          c.ctrl_g->position, c.ctrl_g->hierarchyOp));

      G[next_vertex].start_time = time;
      G[next_vertex].end_time = time + modelOp(G[next_vertex]);
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
                                 launch.ctrl_g->g, launch, time);
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
  std::string sim_granularity;

  unsigned dispatch_slots;
  unsigned dispatch_dma_slots;
  unsigned core_dma_slots;
  unsigned herd_slots;

  // Dependency graph constructed as Boost graph
  dependencyGraph hostGraph;

  // Host and segment runnerNodes
  runnerNode launch_runner_node;

  // Check if op is a non-blocking event
  bool isNonBlocking(Operation *op) {
    if (dyn_cast<scf::YieldOp>(op)) {
      return true;
    } else
      return false;
  }

  // Check if two ops are under the same scf for loop
  bool shareInnerMostForLoop(Operation *a, Operation *b) {
    if (a) {
      if (auto a_parent_for = a->getParentOfType<scf::ForOp>()) {
        if (b) {
          if (auto b_parent_for = b->getParentOfType<scf::ForOp>()) {
            if (a_parent_for == b_parent_for) {
              return true;
            }
          }
        }
      }
    }
    return false;
  }

  // Async for loop race condition: calculate the minimum number of tokens
  // required for dep fulfillment
  unsigned tokenCountThresholdForExecution(Operation *op) {
    // Threshold token_count for dep fulfillment = how many iter_args does node
    // depend on
    unsigned th = 0;
    if (auto async_op = dyn_cast<air::AsyncOpInterface>(op)) {
      for (auto token : async_op.getAsyncDependencies()) {
        if (getForRegionIterArgsOwner(token)) {
          th++;
        }
      }
    } else {
      th = 1;
    }
    return th;
  }

  // Walk affine.if then and else blocks and check if current core lies in
  // condition
  bool positionHitsAffineIfCondition(Operation *op, Operation *spatial_loop,
                                     std::vector<Operation *> affine_if_nest,
                                     std::vector<unsigned> position) {
    SmallVector<int, 2> lbs_spatial;
    SmallVector<int, 2> ubs_spatial;
    if (auto scf_par = dyn_cast<scf::ParallelOp>(spatial_loop)) {
      for (unsigned i = 0; i < scf_par.getLowerBound().size(); i++) {
        auto lbCstOp =
            scf_par.getLowerBound()[i].getDefiningOp<arith::ConstantIndexOp>();
        auto ubCstOp =
            scf_par.getUpperBound()[i].getDefiningOp<arith::ConstantIndexOp>();
        auto stepCstOp =
            scf_par.getStep()[i].getDefiningOp<arith::ConstantIndexOp>();
        lbs_spatial.push_back(
            mlir::ceilDiv(lbCstOp.value(), stepCstOp.value()) + 1);
        ubs_spatial.push_back(
            mlir::ceilDiv(ubCstOp.value(), stepCstOp.value()));
      }
    } else if (auto hier = dyn_cast<air::HierarchyInterface>(spatial_loop)) {
      for (unsigned i = 0; i < hier.getSizeOperands().size(); i++) {
        lbs_spatial.push_back(1);
        ubs_spatial.push_back(hier.getSizeOperands()[i]
                                  .getDefiningOp<arith::ConstantIndexOp>()
                                  .value());
      }
    }

    // Walk through affine.if nest (in reverse order through vector)
    for (auto it = affine_if_nest.rbegin(); it != affine_if_nest.rend(); ++it) {
      auto affine_if = dyn_cast<mlir::AffineIfOp>(*it);
      // Get then integerset sizes
      SmallVector<int, 2> lbs_int = {-1, -1};
      SmallVector<int, 2> ubs_int = {-1, -1};
      IntegerSet int_set = affine_if.getIntegerSet();
      getSizesFromIntegerSet(affine_if->getContext(), int_set, lbs_int,
                             ubs_int);
      // If found then block containing op
      if (affine_if.getThenBlock()->findAncestorOpInBlock(*op)) {
        bool hit = true;
        for (unsigned i = 0; i < lbs_int.size(); i++) {
          if ((position[i] + 1 < lbs_int[i]) ||
              (position[i] + 1 > ubs_int[i])) {
            hit = false;
          }
        }
        return hit;
      }
      // Else keep going, while updating the spatial sizes wrt else condition
      else {
        getElseSizesFromAffineIf(lbs_spatial, ubs_spatial, lbs_int, ubs_int);
      }
    }
    // If op isn't in any then blocks in affine.if nest
    bool hit = true;
    for (unsigned i = 0; i < lbs_spatial.size(); i++) {
      if ((position[i] + 1 < lbs_spatial[i]) ||
          (position[i] + 1 > ubs_spatial[i])) {
        hit = false;
      }
    }
    return hit;
  }

  // Walk affine.if then and else blocks and infer block sizes of op's ancestor
  unsigned getSizeThroughAffineIf(Operation *op, Operation *spatial_loop,
                                  std::vector<Operation *> affine_if_nest) {
    unsigned output = 1;
    SmallVector<int, 2> lbs_spatial;
    SmallVector<int, 2> ubs_spatial;
    if (auto scf_par = dyn_cast<scf::ParallelOp>(spatial_loop)) {
      for (unsigned i = 0; i < scf_par.getLowerBound().size(); i++) {
        auto lbCstOp =
            scf_par.getLowerBound()[i].getDefiningOp<arith::ConstantIndexOp>();
        auto ubCstOp =
            scf_par.getUpperBound()[i].getDefiningOp<arith::ConstantIndexOp>();
        auto stepCstOp =
            scf_par.getStep()[i].getDefiningOp<arith::ConstantIndexOp>();
        lbs_spatial.push_back(
            mlir::ceilDiv(lbCstOp.value(), stepCstOp.value()) + 1);
        ubs_spatial.push_back(
            mlir::ceilDiv(ubCstOp.value(), stepCstOp.value()));
      }
    } else if (auto hier = dyn_cast<air::HierarchyInterface>(spatial_loop)) {
      for (unsigned i = 0; i < hier.getSizeOperands().size(); i++) {
        lbs_spatial.push_back(1);
        ubs_spatial.push_back(hier.getSizeOperands()[i]
                                  .getDefiningOp<arith::ConstantIndexOp>()
                                  .value());
      }
    }

    // Walk through affine.if nest (in reverse order through vector)
    for (auto it = affine_if_nest.rbegin(); it != affine_if_nest.rend(); ++it) {
      auto affine_if = dyn_cast<mlir::AffineIfOp>(*it);
      // Get then integerset sizes
      SmallVector<int, 2> lbs_int = {-1, -1};
      SmallVector<int, 2> ubs_int = {-1, -1};
      IntegerSet int_set = affine_if.getIntegerSet();
      getSizesFromIntegerSet(affine_if->getContext(), int_set, lbs_int,
                             ubs_int);
      // If found then block containing op
      if (affine_if.getThenBlock()->findAncestorOpInBlock(*op)) {
        for (unsigned i = 0; i < lbs_int.size(); i++) {
          output *= ubs_int[i] - lbs_int[i] + 1;
        }
        return output;
      }
      // Else keep going, while updating the spatial sizes wrt else condition
      else {
        getElseSizesFromAffineIf(lbs_spatial, ubs_spatial, lbs_int, ubs_int);
      }
    }
    // If op isn't in any then blocks in affine.if nest
    for (unsigned i = 0; i < lbs_spatial.size(); i++) {
      output *= ubs_spatial[i] - lbs_spatial[i] + 1;
    }
    return output;
  }

  // Get else sizes from affine.if. Assumption: rectangular input, then and else
  // sizes only
  void getElseSizesFromAffineIf(SmallVector<int, 2> &lbs_in,
                                SmallVector<int, 2> &ubs_in,
                                SmallVector<int, 2> &lbs_then,
                                SmallVector<int, 2> &ubs_then) {
    for (unsigned i = 0; i < lbs_in.size(); i++) {
      if ((lbs_in[i] != lbs_then[i])) {
        ubs_in[i] = lbs_then[i] - 1;
        lbs_in[i] = lbs_in[i];
        return;
      } else if ((ubs_in[i] != ubs_then[i])) {
        lbs_in[i] = ubs_then[i] + 1;
        ubs_in[i] = ubs_in[i];
        return;
      }
    }
  }

  // Calculate the number of spatially parallel tokens produced/consumed per op
  unsigned tokenSpatialFactor(Operation *op, std::vector<unsigned> position) {
    unsigned output = 1;
    // If op is producer to a channel broadcast, then bump up token count by
    // fanout
    if (isa<air::ChannelPutOp>(op)) {
      auto channel_op = dyn_cast<air::ChannelInterface>(op);
      auto chan = getChannelDeclarationThroughSymbol(channel_op);
      if (chan->hasAttr("broadcast_shape")) {
        auto size = extractFromI64ArrayAttr(chan->getAttr("broadcast_shape"));
        for (auto s : size) {
          output *= s;
        }
        size = extractFromI64ArrayAttr(chan.getSize());
        for (auto s : size) {
          output /= s;
        }
      }
    }
    for (auto parent = op->getParentOp(); !isa<func::FuncOp>(parent);
         parent = parent->getParentOp()) {
      if (auto scf_par = dyn_cast<scf::ParallelOp>(parent)) {
        for (unsigned i = 0; i < scf_par.getNumLoops(); i++) {
          auto lbCstOp = scf_par.getLowerBound()[i]
                             .getDefiningOp<arith::ConstantIndexOp>();
          auto ubCstOp = scf_par.getUpperBound()[i]
                             .getDefiningOp<arith::ConstantIndexOp>();
          auto stepCstOp =
              scf_par.getStep()[i].getDefiningOp<arith::ConstantIndexOp>();
          int64_t tripCount = mlir::ceilDiv(ubCstOp.value() - lbCstOp.value(),
                                            stepCstOp.value());
          output *= tripCount;
        }
      } else if (auto hier = dyn_cast<air::HierarchyInterface>(parent)) {
        if (sim_granularity == "core" && isa<air::HerdOp>(parent)) {
        } else {
          output *= canonicalizer.getTripCountInHierarchyOp(hier);
        }
      } else if (auto affine_if = dyn_cast<mlir::AffineIfOp>(parent)) {
        // Fast forward through affine.if nest
        std::vector<Operation *> affine_if_nest;
        Operation *spatial_loop = nullptr;
        while ((!isa<scf::ParallelOp>(parent)) &&
               (!isa<air::HierarchyInterface>(parent))) {
          if (isa<mlir::AffineIfOp>(parent)) {
            affine_if_nest.push_back(parent);
          }
          parent = parent->getParentOp();
        }
        // Skip over the first parent hierarchy or parallel loop
        spatial_loop = parent;
        parent = parent->getParentOp();

        // If showing cores
        auto herd = dyn_cast<air::HerdOp>(spatial_loop);
        if (herd && sim_granularity == "core") {
          output = (positionHitsAffineIfCondition(op, spatial_loop,
                                                  affine_if_nest, position))
                       ? (output)
                       : (0);
        } else {
          unsigned size =
              getSizeThroughAffineIf(op, spatial_loop, affine_if_nest);
          output *= size;
        }
      }
    }
    return output;
  }

  // Returns the scf parent op from scf.yield op
  template <typename T> Operation *getScfParentOpFromYieldOp(scf::YieldOp op) {
    if (auto scfop = dyn_cast<T>(op->getParentOp())) {
      return scfop.getOperation();
    }
    return nullptr;
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

  // Reset a vertex in dependency graph
  void resetVertex(Graph::vertex_descriptor v, Graph &G, runnerNode &c,
                   uint64_t time) {

    // Remove start_v from processed_vertices
    removeVertexFromVertices(c.processed_vertices, v);

    // Reset node's start_time and end_time, if the async event represented by
    // the vertex is complete
    if (G[v].is_started() && G[v].is_done(time)) {
      G[v].start_end_time_log.push_back(
          std::make_pair(G[v].start_time, G[v].end_time));
      G[v].start_time = 0;
      G[v].end_time = 0;
    }
  }

  // Recursively reset all vertices in for loop body
  void resetGraphBetweenTwoVertices(Graph::vertex_descriptor start_v,
                                    Graph::vertex_descriptor end_v, Graph &G,
                                    runnerNode &c, uint64_t time) {

    resetVertex(start_v, G, c, time);

    if (start_v == end_v)
      return;

    SmallVector<Graph::vertex_descriptor, 1> vertices;
    if (hasPath(start_v, end_v, G, vertices)) {
      for (auto v : vertices) {
        resetVertex(v, G, c, time);
        // If v is a hierarchy op, then recursively clear the entire subgraph
        if (G[v].asyncEventType == "hierarchy") {
          for (auto sub_c : G[v].nextDependencyGraphs) {
            auto start = sub_c->start_vertex;
            auto terminator_v = sub_c->terminator_vertex;
            auto sub_g = sub_c->g;
            auto sub_runner = sub_c->runner_node;
            resetGraphBetweenTwoVertices(start, terminator_v, sub_g,
                                         *sub_runner, time);
          }
        }
        // Else if v is an scf.for op, then clear the cached trip count from
        // runner node
        else if (G[v].asyncEventType == "for_loop") {
          // Clear for loop trip count from runner node's cache
          for (auto it = c.loop_trip_count.begin();
               it != c.loop_trip_count.end(); it++) {
            if (std::get<0>(*it) == (unsigned)getIdAttr(G[v].op)) {
              c.loop_trip_count.erase(it);
              it--;
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

  // Write process names in trace metadata
  void writeTraceMetadataProcNames(dependencyGraph &hostGraph) {
    for (auto &launchGraph : hostGraph.subgraphs) {
      // Write launch process name to trace metadata
      emitTraceMetadataEvent(traceStream, "process_name", "name",
                             to_string(launchGraph.hierarchyOp), "M",
                             getIdAttr(launchGraph.hierarchyOp));
      emitTraceMetadataEvent(traceStream, "process_sort_index", "sort_index",
                             std::to_string(getIdAttr(launchGraph.hierarchyOp)),
                             "M", getIdAttr(launchGraph.hierarchyOp));
      for (auto &partitionGraph : launchGraph.subgraphs) {
        // Write partition process name to trace metadata
        emitTraceMetadataEvent(traceStream, "process_name", "name",
                               to_string(partitionGraph.hierarchyOp), "M",
                               getIdAttr(partitionGraph.hierarchyOp));
        emitTraceMetadataEvent(
            traceStream, "process_sort_index", "sort_index",
            std::to_string(getIdAttr(partitionGraph.hierarchyOp)), "M",
            getIdAttr(partitionGraph.hierarchyOp));
        for (auto &herdGraph : partitionGraph.subgraphs) {
          // Write herd process name to trace metadata
          emitTraceMetadataEvent(traceStream, "process_name", "name",
                                 to_string(herdGraph.hierarchyOp), "M",
                                 getIdAttr(herdGraph.hierarchyOp));
          emitTraceMetadataEvent(
              traceStream, "process_sort_index", "sort_index",
              std::to_string(getIdAttr(herdGraph.hierarchyOp)), "M",
              getIdAttr(herdGraph.hierarchyOp));
        }
      }
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
      Graph::vertex_descriptor v, unsigned core_id = 0) {
    // Acquire available thread id for current op
    unsigned max_num_threads_per_core =
        10; // Hardcoded maximum number of threads per core
    unsigned offset = core_id * max_num_threads_per_core;
    unsigned tid = 0;
    for (unsigned i = offset + 1; i < offset + wavefront.size() + 2; i++) {
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

  // Get a vector of async tokens which are ready to advance to the next loop
  // iteration at scf.yield
  void getReadyTokensAtScfYield(std::vector<unsigned> &token_ids,
                                scf::YieldOp op, uint64_t time, Graph &G) {

    unsigned token_id = 0;
    for (auto operand : op->getOperands()) {
      // With scf.for possibly having multiple tokens, check for the token ids
      // which are ready to advance to the next iteration
      auto dep_op = operand.getDefiningOp();
      auto dep_entry = canonicalizer.getVertexFromOp(dep_op, dep_ctx, "front");
      auto &dep_node = G[dep_entry.first];

      // Check each token's dependence fulfillment at scf.yield
      std::string node_type = "ssa";
      if (checkEachDependenceFulfillment(std::make_pair(&dep_node, node_type),
                                         nullptr, time)) {
        token_ids.push_back(token_id);
      }

      token_id++;
    }
  }

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