//===- AIRDependencyCanonicalize.cpp ----------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

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
    : public xilinx::air::impl::AIRDependencyCanonicalizeBase<
          AIRDependencyCanonicalize> {

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
      // Pre processing
      // Re-trace ops which depend on air.hierarchies
      // (Removes obsolete dep edges after -canonicalize)
      canonicalizer.redoDepTraceIfDepOnHier(func);

      // Parse dependency graphs
      hostGraph = dependencyGraph(func, true);
      canonicalizer.parseCommandGraphs(func, hostGraph, dep_ctx);

      // Transitive reduction
      xilinx::air::dependencyGraph trHostGraph;
      canonicalizer.canonicalizeGraphs(hostGraph, trHostGraph);

      // Post processing
      // Update dependency list
      canonicalizer.updateDepList(func, trHostGraph);

      // Clean up
      canonicalizer.removeUnusedExecuteOp(func);
      canonicalizer.removeRedundantWaitAllOps(func);
      canonicalizer.removeDepListRepetition(func);
    }
  }

private:
  xilinx::air::dependencyGraph hostGraph;
  xilinx::air::dependencyContext dep_ctx;
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRDependencyCanonicalizePass() {
  return std::make_unique<AIRDependencyCanonicalize>();
}

} // namespace air
} // namespace xilinx
