//===- AIRDependencyCanonicalize.cpp ----------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/AIRDependencyCanonicalize.h"
#include "air/Util/Dependency.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;
using namespace boost;

#define DEBUG_TYPE "air-dependency-canonicalize"

namespace {

class AIRDependencyCanonicalize
    : public AIRDependencyCanonicalizeBase<AIRDependencyCanonicalize> {

public:
  AIRDependencyCanonicalize() = default;
  AIRDependencyCanonicalize(const AIRDependencyCanonicalize &pass) {}

  dependencyCanonicalizer canonicalizer;

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();

    for (auto func : module.getOps<func::FuncOp>()) {
      // Parse dependency graphs
      hostGraph = dependencyGraph(func, true);
      canonicalizer.parseCommandGraphs(func, hostGraph, dep_ctx);

      // Transitive reduction
      xilinx::air::dependencyGraph trHostGraph;
      canonicalizer.canonicalizeGraphs(hostGraph, trHostGraph, g_to_tr);

      // Update dependency list
      canonicalizer.updateDepList(func, trHostGraph);

      // Clean up
      canonicalizer.removeDepListRepitition(func);
      canonicalizer.removeRedundantWaitAllOps(func);

      if (clDumpGraph){
        // Dump graphs
        canonicalizer.dumpDotGraphFiles(trHostGraph, clDumpDir);
      }
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

std::unique_ptr<mlir::Pass> createAIRDependencyCanonicalizePass() {
  return std::make_unique<AIRDependencyCanonicalize>();
}

} // namespace air
} // namespace xilinx