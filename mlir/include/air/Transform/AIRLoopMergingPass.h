// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_LOOP_MERGING_PASS_H
#define AIR_LOOP_MERGING_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRLoopMergingPass();

} // namespace air
} // namespace xilinx

#endif // AIR_LOOP_MERGING_PASS_H