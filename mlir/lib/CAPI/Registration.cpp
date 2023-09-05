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

void airRegisterAllDialects(MlirContext context) {
  mlir::DialectRegistry registry;
  xilinx::air::registerAllDialects(registry);
  unwrap(context)->appendDialectRegistry(registry);
  // TODO: Don't eagerly load once D88162 is in and clients can do this.
  unwrap(context)->loadAllAvailableDialects();
}

void airRegisterAllPasses() { xilinx::air::registerAllPasses(); }
