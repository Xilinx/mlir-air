//===- AIRRunnerModule.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "AIRRunnerModule.h"

#include "air-c/Runner.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

#include <string>

namespace xilinx {
namespace air {

void defineAIRRunnerModule(pybind11::module &m) {
  m.def("run",
        [](MlirModule module, std::string json, std::string outfile,
           std::string function, std::string sim_granularity, bool verbose) {
          airRunnerRun(module, json.c_str(), outfile.c_str(), function.c_str(),
                       sim_granularity.c_str(), verbose);
        });
}

} // namespace air
} // namespace xilinx