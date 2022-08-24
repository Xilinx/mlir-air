// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#ifndef CONVERT_TO_AIR_H
#define CONVERT_TO_AIR_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createParallelToHerdPass();
std::unique_ptr<mlir::Pass> createParallelToLaunchPass();
std::unique_ptr<mlir::Pass> createCopyToDmaPass();

} // namespace air
} // namespace xilinx

#endif // CONVERT_TO_AIR_H
