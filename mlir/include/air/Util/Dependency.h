//===- Dependency.h ---------------------------------------------*- C++ -*-===//
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

#include <numeric>
#include <string>
#include <sys/stat.h>

// boost graph
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/copy.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/transitive_reduction.hpp>

using namespace mlir;

namespace xilinx {
namespace air {

//===----------------------------------------------------------------------===//
// Dependency tracing helper functions
//===----------------------------------------------------------------------===//

bool areEqualIndices(mlir::Value index_0, mlir::Value index_1);
void traceDependentInductionVar(air::DmaMemcpyInterface async_op,
                                SmallVector<Value, 1> &loop_dep_history,
                                std::vector<Operation *> &op_history);
void traceDependentInductionVar(air::AsyncOpInterface async_op,
                                SmallVector<Value, 1> &loop_dep_history,
                                std::vector<Operation *> &op_history);
void eraseAsyncDependencyFromAsyncOp(xilinx::air::AsyncOpInterface op,
                                     Value token);

//===----------------------------------------------------------------------===//
// Dependency graph parsed as a Boost graph object
//===----------------------------------------------------------------------===//

struct dependencyNodeEntry;
struct dependencyGraph;

struct dependencyNodeEntry {
  std::string asyncEventName;
  std::string asyncEventType;
  std::string color;
  std::string shape;
  unsigned operationId;
  mlir::Operation *op;
  dependencyGraph *nextDependencyGraph;

  dependencyNodeEntry(std::string asyncEventName = "",
                      std::string asyncEventType = "", std::string color = "",
                      std::string shape = "", unsigned operationId = 0,
                      mlir::Operation *op = nullptr,
                      dependencyGraph *nextDependencyGraph = nullptr)
      : asyncEventName(asyncEventName), asyncEventType(asyncEventType),
        color(color), shape(shape), operationId(operationId), op(op),
        nextDependencyGraph(nextDependencyGraph) {}
};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS,
                              dependencyNodeEntry>
    Graph;
typedef boost::graph_traits<Graph>::in_edge_iterator in_edge_iterator;
typedef boost::graph_traits<Graph>::out_edge_iterator out_edge_iterator;
typedef boost::graph_traits<Graph>::vertex_iterator vertex_iterator;

typedef std::map<std::pair<std::string, unsigned>, Graph::vertex_descriptor>
    operation_to_vertex_map;
typedef std::map<std::pair<std::string, unsigned>, Graph *>
    operation_to_graph_map;
typedef std::map<Graph::vertex_descriptor, Graph::vertex_descriptor>
    vertex_to_vertex_map;

struct dependencyGraph {
  Graph g;
  mlir::Operation *hierarchyOp;
  std::vector<dependencyGraph> subgraphs;
  Graph::vertex_descriptor start_vertex;
  Graph::vertex_descriptor terminator_vertex;

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
  }
};

struct vertex_to_vertex_map_tree {
  vertex_to_vertex_map a_to_b;
  vertex_to_vertex_map b_to_a;
  std::vector<vertex_to_vertex_map_tree> submaps;

  vertex_to_vertex_map_tree() {}
};

struct dependencyContext {
  uint64_t ExecuteOpID;
  uint64_t DmaOpID;
  uint64_t HierarchyOpID;
  uint64_t WaitAllOpID;
  uint64_t ForOpID;
  uint64_t ParallelOpID;
  uint64_t TerminatorID;
  operation_to_vertex_map op_to_v;
  operation_to_graph_map op_to_g;

  dependencyContext()
      : ExecuteOpID(0), DmaOpID(0), HierarchyOpID(0), WaitAllOpID(0),
        ForOpID(0), ParallelOpID(0), TerminatorID(0) {}
};

class dependencyCanonicalizer {

public:
  void parseCommandGraphs(func::FuncOp &toplevel, dependencyGraph &global_graph,
                          dependencyContext &dep_ctx, bool dump_dot = false, std::string dump_dir = "");
  void canonicalizeGraphs(dependencyGraph &global_graph,
                          dependencyGraph &tr_graph,
                          vertex_to_vertex_map_tree &g_to_tr,
                          bool dump_graph = false,
                          std::string dump_dir = "");
  void updateDepList(func::FuncOp func, dependencyGraph &global_graph);
  void removeDepListRepitition(func::FuncOp func);
  void removeRedundantWaitAllOps(func::FuncOp func);
  void dumpDotGraphFiles(dependencyGraph global_graph, std::string dump_dir = "");

private:
  void addVerticesInHerd(std::vector<dependencyGraph> &herd_subgraphs, air::HerdOp herd, dependencyContext &dep_ctx);
  void addVerticesInPartition(std::vector<dependencyGraph> &part_subgraphs, air::PartitionOp partition, dependencyContext &dep_ctx);
  void addVerticesInLaunch(std::vector<dependencyGraph> &launch_subgraphs, air::LaunchOp launch, dependencyContext &dep_ctx);
  Graph::vertex_descriptor addVertexFromOpImpls(Operation *op, Graph &G,
                                                dependencyContext &dep_ctx);
  Graph::vertex_descriptor addVertexFromOp(Operation *op, uint64_t &id,
                                           std::string event_type,
                                           std::string event_name,
                                           std::string color, std::string shape,
                                           Graph &G, dependencyContext &dep_ctx,
                                           Operation *pointer_op = nullptr);
  Graph::vertex_descriptor
  addVertexFromDmaOp(xilinx::air::DmaMemcpyInterface op, Graph &G,
                     dependencyContext &dep_ctx);
  Graph::vertex_descriptor
  addVertexFromHierarchyOp(xilinx::air::HierarchyInterface op, Graph &G,
                           dependencyContext &dep_ctx);
  Graph::vertex_descriptor
  addVertexFromTerminatorOp(Operation *op, Graph &G,
                            dependencyContext &dep_ctx);
  Graph::vertex_descriptor addVertexFromExecuteOp(xilinx::air::ExecuteOp op,
                                                  Graph &G,
                                                  dependencyContext &dep_ctx);
  Graph::vertex_descriptor addVertexFromWaitAllOp(xilinx::air::WaitAllOp op,
                                                  Graph &G,
                                                  dependencyContext &dep_ctx);
  std::pair<std::string, unsigned> getTypeIdPairFromOp(Operation *op);
  std::string getOpTypeFromOpImpls(Operation *op);
  std::pair<Graph::vertex_descriptor, Graph *>
  getVertexFromOp(Operation *op, dependencyContext dep_ctx,
                  std::string front_or_back = "front");
  void parseDependencyEdgesInGraph(Graph &g, dependencyContext dep_ctx);
  void connectOpToItsDepListImpls(Operation *op, Graph &g,
                                  dependencyContext dep_ctx);
  void connectOpToItsDepList(Operation *op, SmallVector<Value, 1> dep_list,
                             Graph &g, dependencyContext dep_ctx);
  std::vector<Operation *> traceOpFromToken(Value dep_token);
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
};

} // namespace air
} // namespace xilinx
