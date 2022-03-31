// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_DEPENDENCY_H
#define AIR_DEPENDENCY_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRDependencyPass();

} // namespace air
} // namespace xilinx

#endif // AIR_DEPENDENCY_H