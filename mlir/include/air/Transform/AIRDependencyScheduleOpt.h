// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.
#ifndef AIR_LOOP_DMA_DEDUPE_H
#define AIR_LOOP_DMA_DEDUPE_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRDependencyScheduleOptPass();

} // namespace air
} // namespace xilinx

#endif // AIR_LOOP_DMA_DEDUPE_H