// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#ifndef AIR_TO_STD_H
#define AIR_TO_STD_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRLoweringPass();

} // namespace air
} // namespace xilinx

#endif // AIR_TO_STD_H