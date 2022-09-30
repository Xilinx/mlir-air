//===- AIRDependencyCanonicalize.cpp -----------------------------*- C++ -*-===//
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


class AIRDependencyCanonicalize : public AIRDependencyCanonicalizeBase<AIRDependencyCanonicalize> {

public:
  AIRDependencyCanonicalize() = default;
  AIRDependencyCanonicalize(const AIRDependencyCanonicalize &pass) {}

  dependencyCanonicalizer canonicalizer;

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();

    // Parse dependency graphs
    hostGraph = dependencyGraph(func, true);
    canonicalizer.parseCommandGraphs(func, hostGraph, dep_ctx);

    // Transitive reduction
    xilinx::air::dependencyGraph trHostGraph;
    canonicalizer.canonicalizeGraphs(hostGraph, trHostGraph, g_to_tr, clDumpGraph);

    // Update dependency list
    canonicalizer.updateDepList(func, trHostGraph);

  }

private:
  xilinx::air::dependencyGraph hostGraph;
  xilinx::air::dependencyContext dep_ctx;
  xilinx::air::vertex_to_vertex_map_tree g_to_tr; // Map between graph g and graph tr (post-tr graph)

};
    

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRDependencyCanonicalizePass() {
  return std::make_unique<AIRDependencyCanonicalize>();
}

} // namespace air
} // namespace xilinx