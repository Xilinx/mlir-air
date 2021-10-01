// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_LOWER_LINALG_TENSORS_H
#define AIR_LOWER_LINALG_TENSORS_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <vector>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRLowerLinalgTensorsPass();

} // namespace air
} // namespace xilinx

#endif // AIR_LOWER_LINALG_TENSORS_H