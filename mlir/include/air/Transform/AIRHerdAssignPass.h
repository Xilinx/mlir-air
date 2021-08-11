// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_HERD_ASSIGN_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRHerdAssignPass();

} // namespace air
} // namespace xilinx
#endif