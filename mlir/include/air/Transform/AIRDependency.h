// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_DEPENDENCY_H
#define AIR_DEPENDENCY_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <vector>

// boost graph
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/transitive_reduction.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>

// air region op
#include "air/Dialect/AIR/AIRDialect.h"

namespace xilinx {
namespace air {

// Construction of a dependency graph as a Boost graph
using namespace boost;

struct regionNode {
    std::string asyncEventName;
    std::string asyncEventType;
    unsigned operationId;
};

typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::bidirectionalS, regionNode> Graph;
typedef graph_traits<Graph>::in_edge_iterator in_edge_iterator;
typedef graph_traits<Graph>::out_edge_iterator out_edge_iterator;
typedef graph_traits<Graph>::vertex_iterator vertex_iterator;

typedef std::map<Graph::vertex_descriptor, Graph::vertex_descriptor> vertex_map;
typedef std::map<unsigned, Graph::vertex_descriptor> region_id_to_vertex_map;

std::unique_ptr<mlir::Pass> createAIRDependencyPass();

} // namespace air
} // namespace xilinx

#endif // AIR_DEPENDENCY_H