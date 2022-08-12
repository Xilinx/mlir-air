// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.
#ifndef AIR_DEPENDENCY_SCHEDULE_OPT_H
#define AIR_DEPENDENCY_SCHEDULE_OPT_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRHoistDmaInAccumPattern();

std::unique_ptr<mlir::Pass> createAIRBroadcastDetection();

std::unique_ptr<mlir::Pass> createAIRDependencyScheduleOptPass();

} // namespace air
} // namespace xilinx

#endif // AIR_DEPENDENCY_SCHEDULE_OPT_H