// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_AUTOMATIC_TILING_PASS_H
#define AIR_AUTOMATIC_TILING_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <vector>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRAutomaticTilingPass();

} // namespace air
} // namespace xilinx

#endif
