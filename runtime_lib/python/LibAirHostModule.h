//===- LibAirHostModule.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_HOST_MODULE_H
#define AIR_HOST_MODULE_H

#include <pybind11/pybind11.h>

namespace xilinx {
namespace air {

void defineAIRHostModule(pybind11::module &m);

}
}

#endif // AIR_HOST_MODULE_H