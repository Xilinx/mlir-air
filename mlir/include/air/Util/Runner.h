//===- Runner.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_UTIL_RUNNER_H
#define AIR_UTIL_RUNNER_H

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "llvm/Support/JSON.h"

namespace xilinx {
namespace air {

struct AIRRunner {

  AIRRunner(llvm::raw_ostream &trace_stream, llvm::json::Value &json_model,
            std::string sim_granularity = "herd", bool verbose = false);
  ~AIRRunner();

  void emitTraceStart(llvm::raw_ostream &s);
  void emitTraceEnd(llvm::raw_ostream &s);

  void scheduleFunction(mlir::func::FuncOp &toplevel);

private:
  class AIRRunner_impl;
  std::unique_ptr<AIRRunner_impl> impl;
};

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_RUNNER_H