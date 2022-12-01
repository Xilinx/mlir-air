//===- InitAll.h ------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_INITALL_H
#define AIR_INITALL_H

#include "mlir/IR/Dialect.h"

namespace xilinx {
namespace air {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllPasses();

} // namespace air
} // namespace xilinx

#endif // AIR_INITALL_H
