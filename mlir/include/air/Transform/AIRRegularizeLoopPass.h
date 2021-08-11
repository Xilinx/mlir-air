// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_REGULARIZE_LOOP_PASS_H
#define AIR_REGULARIZE_LOOP_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRRegularizeLoopPass();

} // namespace air
} // namespace xilinx

#endif // AIR_REGULARIZE_LOOP_PASS_H