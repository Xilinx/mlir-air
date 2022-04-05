// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.
#ifndef AIR_MISC_PASSES_H
#define AIR_MISC_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRExamplePass();
std::unique_ptr<mlir::Pass> createAIRSpecializeDma();
std::unique_ptr<mlir::Pass> createAIRPromoteUniformL1Dma();
std::unique_ptr<mlir::Pass> createAIRLinalgNamePass();
std::unique_ptr<mlir::Pass> createAIRRemoveLinalgNamePass();

} // namespace air
} // namespace xilinx

#endif // AIR_MISC_PASSES_H