//===- AIRLinalgOpStats.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
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