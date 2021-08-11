// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_TO_AIE_PASS_H
#define AIR_TO_AIE_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>
namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRToAIEPass();

} // namespace air
} // namespace xilinx

#endif // AIR_TO_AIE_PASS_H