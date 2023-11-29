//===- AIRDependencyParseGraph.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/AIRDependencyParseGraph.h"
#include "air/Util/Dependency.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;
using namespace boost;

#define DEBUG_TYPE "air-dependency-parse-graph"

namespace {

class AIRDependencyParseGraph
    : public xilinx::air::impl::AIRDependencyParseGraphBase<
          AIRDependencyParseGraph> {

public:
  AIRDependencyParseGraph() = default;
  AIRDependencyParseGraph(const AIRDependencyParseGraph &pass) {}

  dependencyCanonicalizer canonicalizer;

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    for (auto func : module.getOps<func::FuncOp>()) {
      // Parse dependency graphs
      std::string graphGranularity = (clShowCores) ? ("core") : ("herd");
      hostGraph = dependencyGraph(func, true);
      canonicalizer.parseCommandGraphs(func, hostGraph, dep_ctx,
                                       graphGranularity);
      // Purge id attribute
      func.walk([&](Operation *op) { op->removeAttr("id"); });

      // Visualize graph with flattened hierarchy
      canonicalizer.copyDependencyGraphToFlatGraphAndVisualize(
          func, hostGraph, dep_ctx, true, clDumpDir);
    }
  }

private:
  xilinx::air::dependencyGraph hostGraph;
  xilinx::air::dependencyContext dep_ctx;
  xilinx::air::vertex_to_vertex_map_tree
      g_to_tr; // Map between graph g and graph tr (post-tr graph)
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRDependencyParseGraphPass() {
  return std::make_unique<AIRDependencyParseGraph>();
}

} // namespace air
} // namespace xilinx