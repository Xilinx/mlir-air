//===- Dependency.h ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

//===- Dependency.h - AIR Dependency Tracing and Parsing utilities --------===//
//
// This header file defines utility functions that are commonly used in passes,
// primarily AIR dependency tracing passes.
//===----------------------------------------------------------------------===//

#pragma once

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/RegionUtils.h"

#include <numeric>
#include <string>

// boost graph
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/subgraph.hpp>
#include <boost/graph/transitive_reduction.hpp>

using namespace mlir;

namespace xilinx {
namespace air {

//===----------------------------------------------------------------------===//
// Dependency tracing helper functions
//===----------------------------------------------------------------------===//

bool areEqualIndices(mlir::Value index_0, mlir::Value index_1);
void traceDependentInductionVar(air::DmaMemcpyNdOp async_op,
                                SmallVector<Value, 1> &loop_dep_history,
                                std::vector<Operation *> &op_history);
void traceDependentInductionVar(air::AsyncOpInterface async_op,
                                SmallVector<Value, 1> &loop_dep_history,
                                std::vector<Operation *> &op_history);
void eraseAsyncDependencyFromAsyncOp(xilinx::air::AsyncOpInterface op,
                                     Value token);
void clearAsyncDependenciesOfAsyncOp(Operation *op);
Value getLoopCarriedTokenFromScfOp(scf::ParallelOp op);
Value getLoopCarriedTokenFromScfOp(scf::ForOp op,
                                   std::string operand_or_argument = "operand");
scf::ReduceOp createSCFReduceForAsyncSCFParallel(OpBuilder builder,
                                                 Location loc, Value token,
                                                 MLIRContext *ctx);
SmallVector<Value> getAsyncDependenciesFromOp(Operation *op);
void addAsyncDependencyIfNew(Operation *op, Value token);
bool isAsyncOp(Operation *op);

//===----------------------------------------------------------------------===//
// Dependency graph parsed as a Boost graph object
//===----------------------------------------------------------------------===//

struct dependencyNodeEntry;
struct dependencyGraph;
class runnerNode;

// GraphViz node properties for visualization
struct graphNodeProperties {
  std::string color;
  std::string shape;
  std::string detailed_description;

  graphNodeProperties(std::string color, std::string shape,
                      std::string detailed_description)
      : color(color), shape(shape), detailed_description(detailed_description) {
  }
  graphNodeProperties(std::string nodeType, std::string details = "") {
    detailed_description = details;
    if (nodeType == "hierarchy") {
      color = "yellow";
      shape = "box";
    } else if (nodeType == "control") {
      color = "crimson";
      shape = "box";
    } else if (nodeType == "data") {
      color = "cyan";
      shape = "oval";
    } else if (nodeType == "compute") {
      color = "chartreuse";
      shape = "oval";
    } else {
      color = "";
      shape = "";
    }
  }
};

// Node entry for dependency graph
struct dependencyNodeEntry {
  std::string asyncEventName;
  std::string asyncEventType;
  std::string color;
  std::string shape;
  std::string detailed_description;
  unsigned operationId;
  mlir::Operation *op;
  std::vector<dependencyGraph *> nextDependencyGraphs;
  uint64_t start_time;
  uint64_t end_time;
  std::vector<std::pair<uint64_t, uint64_t>> start_end_time_log;
  // Token count is used to synchronize operations which consumes/produces
  // multiple async tokens.
  int token_count;

  bool is_started() { return (start_time != 0) && (end_time != 0); }
  bool is_done(uint64_t t) { return t >= end_time; }

  dependencyNodeEntry(std::string asyncEventName = "",
                      std::string asyncEventType = "", std::string color = "",
                      std::string shape = "",
                      std::string detailed_description = "",
                      unsigned operationId = 0, mlir::Operation *op = nullptr,
                      uint64_t start_time = 0, uint64_t end_time = 0,
                      int token_count = 0)
      : asyncEventName(asyncEventName), asyncEventType(asyncEventType),
        color(color), shape(shape), detailed_description(detailed_description),
        operationId(operationId), op(op), start_time(start_time),
        end_time(end_time), token_count(token_count) {}
};

// Boost dependency graph
typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                              dependencyNodeEntry>
    Graph;
typedef boost::graph_traits<Graph>::in_edge_iterator in_edge_iterator;
typedef boost::graph_traits<Graph>::out_edge_iterator out_edge_iterator;
typedef boost::graph_traits<Graph>::vertex_iterator vertex_iterator;

// Dependency graph object
struct dependencyGraph {
  Graph g;
  mlir::Operation *hierarchyOp;
  std::vector<dependencyGraph> subgraphs;
  runnerNode *runner_node;
  Graph::vertex_descriptor start_vertex;
  Graph::vertex_descriptor terminator_vertex;
  std::vector<unsigned>
      position; // Position (coordinates) of each core in herd, if showing cores

  dependencyGraph(mlir::Operation *op = nullptr, bool initStartVertex = false) {
    g = Graph();
    hierarchyOp = op;
    if (initStartVertex) {
      auto v = add_vertex(g);
      g[v].asyncEventType = "start";
      g[v].asyncEventName = "start";
      g[v].color = "yellow";
      g[v].shape = "box";
      start_vertex = v;
    }
  }

  ~dependencyGraph() {
    g.clear();
    subgraphs.clear();
    position.clear();
  }
};

// Maps involving Graph and vertex
typedef std::map<std::pair<std::string, unsigned>, Graph::vertex_descriptor>
    operation_to_vertex_map;
typedef std::map<std::pair<std::string, unsigned>, dependencyGraph *>
    operation_to_graph_map;
typedef std::map<Graph::vertex_descriptor, Graph::vertex_descriptor>
    vertex_to_vertex_map;

struct vertex_to_vertex_map_tree {
  vertex_to_vertex_map a_to_b;
  vertex_to_vertex_map b_to_a;
  std::vector<vertex_to_vertex_map_tree> submaps;

  vertex_to_vertex_map_tree() {}
};

struct dependencyContext {
  uint64_t ExecuteOpID;
  uint64_t DmaOpID;
  uint64_t ChannelOpID;
  uint64_t HierarchyOpID;
  uint64_t WaitAllOpID;
  uint64_t ForOpID;
  uint64_t ParallelOpID;
  uint64_t TerminatorID;
  operation_to_vertex_map op_to_v;
  operation_to_graph_map op_to_g;

  dependencyContext()
      : ExecuteOpID(0), DmaOpID(0), ChannelOpID(0), HierarchyOpID(0),
        WaitAllOpID(0), ForOpID(0), ParallelOpID(0), TerminatorID(0) {}
};

// Flat boost graph for visualization
typedef std::map<std::string, std::string> GraphvizAttributes;
typedef boost::subgraph<boost::adjacency_list<
    boost::vecS, boost::vecS, boost::directedS,
    boost::property<boost::vertex_attribute_t, GraphvizAttributes>,
    boost::property<
        boost::edge_index_t, int,
        boost::property<boost::edge_attribute_t, GraphvizAttributes>>,
    boost::property<
        boost::graph_name_t, std::string,
        boost::property<
            boost::graph_graph_attribute_t, GraphvizAttributes,
            boost::property<boost::graph_vertex_attribute_t, GraphvizAttributes,
                            boost::property<boost::graph_edge_attribute_t,
                                            GraphvizAttributes>>>>>>
    FlatGraph;
typedef std::map<Graph::vertex_descriptor, FlatGraph::vertex_descriptor>
    vertex_to_flat_vertex_map;
typedef std::map<std::string,
                 std::pair<std::vector<FlatGraph::vertex_descriptor>,
                           std::vector<FlatGraph::vertex_descriptor>>>
    ChannelMap;

class dependencyCanonicalizer {

  typedef std::tuple<bool, bool, bool, bool> graphGranularityProperties;

public:
  void parseCommandGraphs(func::FuncOp &toplevel, dependencyGraph &global_graph,
                          dependencyContext &dep_ctx,
                          std::string granularity = "herd",
                          bool dump_dot = false, std::string dump_dir = "");
  void canonicalizeGraphs(dependencyGraph &global_graph,
                          dependencyGraph &tr_graph,
                          vertex_to_vertex_map_tree &g_to_tr,
                          bool dump_graph = false, std::string dump_dir = "");
  void updateDepList(func::FuncOp func, dependencyGraph &global_graph);
  void removeDepListRepetition(func::FuncOp func);
  void removeUnusedExecuteOp(func::FuncOp func);
  void removeRedundantWaitAllOps(func::FuncOp func);
  void dumpDotGraphFiles(dependencyGraph global_graph,
                         std::string dump_dir = "");
  void copyDependencyGraphToFlatGraphAndVisualize(func::FuncOp &toplevel,
                                                  dependencyGraph &global_graph,
                                                  dependencyContext &dep_ctx,
                                                  bool dump_dot = false,
                                                  std::string dump_dir = "");
  std::pair<Graph::vertex_descriptor, dependencyGraph *>
  getVertexFromOp(Operation *op, dependencyContext dep_ctx,
                  std::string front_or_back = "front");
  // CDFG show cores in herd
  unsigned getTripCountInHierarchyOp(air::HierarchyInterface hier);
  std::vector<unsigned> getPositionFromIterator(unsigned iter,
                                                air::HerdOp herd);
  std::string toPositionString(std::vector<unsigned> position);
  unsigned getIteratorFromPosition(std::vector<unsigned> position,
                                   Operation *hier_op);
  void redoDepTraceIfDepOnHier(func::FuncOp func);

private:
  void addVerticesInHerd(std::vector<dependencyGraph> &herd_subgraphs,
                         air::HerdOp herd, dependencyContext &dep_ctx,
                         graphGranularityProperties expandHier = {true, true,
                                                                  true, false});
  void addVerticesInSegment(std::vector<dependencyGraph> &part_subgraphs,
                            air::SegmentOp segment, dependencyContext &dep_ctx,
                            graphGranularityProperties expandHier = {
                                true, true, true, false});
  void addVerticesInLaunch(std::vector<dependencyGraph> &launch_subgraphs,
                           air::LaunchOp launch, dependencyContext &dep_ctx,
                           graphGranularityProperties expandHier = {
                               true, true, true, false});
  Graph::vertex_descriptor addVertexFromOpImpls(Operation *op,
                                                dependencyGraph *G,
                                                dependencyContext &dep_ctx);
  Graph::vertex_descriptor
  addVertexFromOp(Operation *op, uint64_t &id, std::string event_type,
                  std::string event_name, graphNodeProperties properties,
                  dependencyGraph *G, dependencyContext &dep_ctx,
                  Operation *pointer_op = nullptr);
  Graph::vertex_descriptor addVertexFromDmaOp(xilinx::air::DmaMemcpyNdOp op,
                                              dependencyGraph *G,
                                              dependencyContext &dep_ctx);
  Graph::vertex_descriptor
  addVertexFromChannelOp(xilinx::air::ChannelInterface op, dependencyGraph *G,
                         dependencyContext &dep_ctx);
  Graph::vertex_descriptor
  addVertexFromHierarchyOp(xilinx::air::HierarchyInterface op,
                           dependencyGraph *G, dependencyContext &dep_ctx);
  Graph::vertex_descriptor
  addVertexFromTerminatorOp(Operation *op, dependencyGraph *G,
                            dependencyContext &dep_ctx);
  Graph::vertex_descriptor addVertexFromReduceOp(Operation *op,
                                                 dependencyGraph *G,
                                                 dependencyContext &dep_ctx);
  Graph::vertex_descriptor addVertexFromExecuteOp(xilinx::air::ExecuteOp op,
                                                  dependencyGraph *G,
                                                  dependencyContext &dep_ctx);
  Graph::vertex_descriptor addVertexFromWaitAllOp(xilinx::air::WaitAllOp op,
                                                  dependencyGraph *G,
                                                  dependencyContext &dep_ctx);
  std::pair<std::string, unsigned> getTypeIdPairFromOp(Operation *op);
  std::string getOpTypeFromOpImpls(Operation *op);
  void parseDependencyEdgesInGraph(Graph &g, dependencyContext dep_ctx);
  void copyFromDependencyGraphToFlatGraph(Graph g_src,
                                          std::vector<unsigned> position,
                                          FlatGraph &g_dst,
                                          vertex_to_flat_vertex_map &map,
                                          bool copyEdges = false);
  void updateSubgraphFromDependencyGraph(Graph subg_src,
                                         std::vector<unsigned> position,
                                         FlatGraph &subg_dst,
                                         vertex_to_flat_vertex_map map,
                                         bool copyEdges = false);
  void connectOpToItsDepListImpls(Operation *op, Graph &g,
                                  dependencyContext dep_ctx);
  void connectOpToItsDepList(Operation *op, SmallVector<Value, 1> dep_list,
                             Graph &g, dependencyContext dep_ctx);
  std::vector<Operation *> traceOpFromToken(Operation *op, Value dep_token);
  void connectTerminatorInGraph(Graph &g);
  void connectStartNodeInCommandGraph(dependencyGraph &G);
  void updatePointerFromGraphToHierarchyTerminator(dependencyGraph &G);
  void updatePointerFromHierarchyTerminatorToGraph(dependencyGraph &G,
                                                   dependencyGraph &subG);
  void updatePointerFromHierarchyOpToGraph(dependencyGraph &G);
  void dump_graph(std::string filename, Graph G);
  void boostTransitiveReductionImpl(Graph &asyncExecuteGraph,
                                    Graph &asyncExecuteGraphTR,
                                    vertex_to_vertex_map &g_to_tr,
                                    vertex_to_vertex_map &tr_to_g);
  void purgeAIRDepList(dependencyGraph &graph);
  void fillAIRDepListUsingGraphTR(dependencyGraph &graph);
  void collectAIRChannelPutAndGetInGraph(Graph g,
                                         std::vector<unsigned> position,
                                         vertex_to_flat_vertex_map map,
                                         ChannelMap &channel_map);
  void updateSubgraphFromDependencyGraphAsGraphVizCluster(
      dependencyGraph &G, FlatGraph &flat_subg, vertex_to_flat_vertex_map map,
      unsigned global_idx, unsigned &subg_idx, std::string hier_name = "");
  std::vector<Graph::vertex_descriptor>
  getVerticesWithAffineIf(Graph g, std::vector<unsigned> position);
};

//===----------------------------------------------------------------------===//
// Dependency tracing
//===----------------------------------------------------------------------===//

struct partialMemref {
  Value memrefValue;
  unsigned numDims;
  SmallVector<Value, 2> memrefIndices;
};

class dependencyTracer {

public:
  // Get partial memref tiles from op
  void
  getPartialMemrefFromOp(Operation *sink_op,
                         SmallVector<partialMemref, 1> &sink_op_memref_reads,
                         SmallVector<partialMemref, 1> &sink_op_memref_writes,
                         SmallVector<Value, 1> &sink_op_scalar_ins,
                         SmallVector<Value, 1> &sink_op_scalar_outs);

  // Trace dependency from op
  template <typename T>
  void traceDependencyFromOp(SmallVector<partialMemref, 1> operands,
                             T sink_air_op, std::string dep_type) {

    char dep_tracing_mode = 'n';
    if (dep_type == "RAW")
      dep_tracing_mode = 'w';
    else if (dep_type == "WAW/WAR")
      dep_tracing_mode = 'n';
    else
      assert(false && "Unknown dependency type");

    // Detect deps
    for (auto operand : operands) {
      // Trace the defining op of sink op, RAW
      traceDefiningOpAsDep<T>(operand.memrefValue, sink_air_op);

      // If sink op and operand's use are under the same scope
      auto async_op =
          dyn_cast<air::AsyncOpInterface>(sink_air_op.getOperation());
      pushDepsAtCurrentScope(operand.memrefValue, async_op, dep_tracing_mode,
                             &operand);
    }
  }

  // Recursively reconnect loop-carried dependency in scf loop nest
  void reconnectLoopCarriedDependencyFromOp(Operation *op);

  // Trace tile index deps
  void traceTileIndices(SmallVector<partialMemref, 1> read_operands,
                        SmallVector<partialMemref, 1> write_operands,
                        SmallVector<Value, 1> in_scalars,
                        SmallVector<Value, 1> out_scalars,
                        air::AsyncOpInterface sink_air_op);

private:
  // Trace the defining op of sink op, RAW
  template <typename T> void traceDefiningOpAsDep(Value operand, T op) {
    // Check memref deps
    if (auto defop = operand.getDefiningOp<air::ExecuteOp>()) {
      // addNewAsyncDepToGraph<T>(defop.getResult(0), op);
      op.addAsyncDependency(defop.getAsyncToken());
    }
  }

  // If sink op and operand's use are under the same scope
  void pushDepsAtCurrentScope(mlir::Value operand, air::AsyncOpInterface op,
                              char rw = 'n', partialMemref *tile = nullptr);

  // Create partial memref
  partialMemref createPartialMemref(mlir::Value memrefValue, unsigned numDims);
  partialMemref createPartialMemref(mlir::Value memrefValue, unsigned numDims,
                                    SmallVector<Value, 2> memrefIndices);

  // Check if two partial memref tiles have identical indices
  bool areEqualIndexPartialMemrefs(partialMemref *tile_0,
                                   partialMemref *tile_1);

  char checkOperandReadOrWrite(mlir::Value operand);

  // Add dependency edge
  void addDependencyBetweenOps(Operation *source, Operation *sink);

  // Add tile index deps to op
  void pushTileIndexAsDep(mlir::Value tile_index, air::AsyncOpInterface op);
};

} // namespace air
} // namespace xilinx
