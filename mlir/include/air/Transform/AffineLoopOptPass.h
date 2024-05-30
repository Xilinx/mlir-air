//===- AffineLoopOptPass.h --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AFFINE_LOOP_OPT_H
#define AFFINE_LOOP_OPT_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"

#include <memory>
#include <vector>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAffineLoopOptPass();
std::unique_ptr<mlir::Pass>
createAffineLoopOptPass(AffineLoopOptPassOptions options);

} // namespace air
} // namespace xilinx

#endif // AFFINE_LOOP_OPT_H