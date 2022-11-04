//===- AIRRunnerModule.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_RUNNER_MODULE_H
#define AIR_RUNNER_MODULE_H

#include <pybind11/pybind11.h>

namespace xilinx {
namespace air {

void defineAIRRunnerModule(pybind11::module &m);

}
} // namespace xilinx

#endif // AIR_RUNNER_MODULE_H