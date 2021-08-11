// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AFFINE_TO_AIR_PASS_H
#define AFFINE_TO_AIR_PASS_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <vector>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRLinalgCodegenPass();

} // namespace air
} // namespace xilinx

#endif // AFFINE_TO_AIR_PASS_H