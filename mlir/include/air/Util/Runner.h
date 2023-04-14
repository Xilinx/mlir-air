//===- Runner.h -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_UTIL_RUNNER_H
#define AIR_UTIL_RUNNER_H

#include "air/Util/Dependency.h"

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

//===----------------------------------------------------------------------===//
// Runner util. functions
//===----------------------------------------------------------------------===//

std::string to_string(std::vector<unsigned> vec);
std::string to_string(dependencyNodeEntry &c);
std::string to_string(mlir::Type t);
std::string getElementTypeAsString(const mlir::Type ty);
std::string lookUpMemorySpaceFromInt(unsigned memory_space);

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_RUNNER_H