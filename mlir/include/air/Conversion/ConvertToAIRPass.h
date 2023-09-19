//===- ConvertToAIRPass.h ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2019-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef CONVERT_TO_AIR_H
#define CONVERT_TO_AIR_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createParallelToHerdPass();
std::unique_ptr<mlir::Pass> createParallelToLaunchPass();
std::unique_ptr<mlir::Pass> createCopyToDmaPass();
std::unique_ptr<mlir::Pass> createDmaToChannelPass();
std::unique_ptr<mlir::Pass> createInsertEmptyLaunchOverHerdPass();

} // namespace air
} // namespace xilinx

#endif // CONVERT_TO_AIR_H
