// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#pragma once

#include "AIRPasses.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRAutomaticTilingPass();
void registerAIRAutomaticTilingPass();

} // namespace air
} // namespace xilinx
