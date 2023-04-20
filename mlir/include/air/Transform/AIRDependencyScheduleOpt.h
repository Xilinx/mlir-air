//===- AIRDependencyScheduleOpt.h -------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_DEPENDENCY_SCHEDULE_OPT_H
#define AIR_DEPENDENCY_SCHEDULE_OPT_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRHoistDmaInAccumPattern();

std::unique_ptr<mlir::Pass> createAIRHoistMemallocInForPattern();

std::unique_ptr<mlir::Pass> createAIRUnrollLoopForPipeliningPattern();

std::unique_ptr<mlir::Pass> createAIRBroadcastDetection();

std::unique_ptr<mlir::Pass> createAIRPruneLinalgGenericInputDma();

std::unique_ptr<mlir::Pass> createAIRPipelineLoweringPattern();

std::unique_ptr<mlir::Pass> createAIRDependencyScheduleOptPass();

} // namespace air
} // namespace xilinx

#endif // AIR_DEPENDENCY_SCHEDULE_OPT_H