//===- Registration.cpp -----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air-c/Registration.h"

#include "air/InitAll.h"

#include "mlir/CAPI/IR.h"
#include "mlir/InitAllPasses.h"

void airRegisterAllDialects(MlirDialectRegistry registry) {
  auto r = unwrap(registry);
  xilinx::air::registerAllDialects(*r);
}

void airRegisterAllPasses() { xilinx::air::registerAllPasses(); }
