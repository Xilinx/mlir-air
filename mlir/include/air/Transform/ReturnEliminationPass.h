//===- ReturnEliminationPass.h ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef RETURN_ELIMINATION_H
#define RETURN_ELIMINATION_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>
namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createReturnEliminationPass();

} // namespace air
} // namespace xilinx

#endif