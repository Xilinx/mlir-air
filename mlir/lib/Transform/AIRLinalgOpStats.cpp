//===- AIRLinalgOpStats.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
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

#include "air/Transform/AIRLinalgOpStats.h"
#include "PassDetail.h"
#include "air/Util/CostModel.h"

#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "air-linalg-op-stats"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

class AIRLinalgOpStats : public AIRLinalgOpStatsBase<AIRLinalgOpStats> {
public:
  AIRLinalgOpStats() = default;
  AIRLinalgOpStats(const AIRLinalgOpStats &pass) {}

  Option<std::string> OutputFileName{*this, "outputfile",
                                     llvm::cl::desc("Output filename"),
                                     llvm::cl::init("-")};

  void runOnOperation() override;
};

void AIRLinalgOpStats::runOnOperation() {
  xilinx::air::CostModel model;
  auto json = model.opCountsToJSON(getOperation());
  if (OutputFileName != "-") {
    std::error_code EC;
    llvm::raw_fd_ostream os(OutputFileName, EC);
    os << json;
  } else {
    llvm::outs() << json << "\n";
  }
}

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRLinalgOpStatsPass() {
  return std::make_unique<AIRLinalgOpStats>();
}

} // namespace air
} // namespace xilinx