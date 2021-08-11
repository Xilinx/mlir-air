// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#ifndef AFFINE_TO_AIR_H
#define AFFINE_TO_AIR_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAffineToAIRPass();

} // namespace air
} // namespace xilinx

#endif // AFFINE_TO_AIR_H
