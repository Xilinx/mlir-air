// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_LINALG_CODEGEN_H
#define AIR_LINALG_CODEGEN_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <vector>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRLinalgCodegenPass();

} // namespace air
} // namespace xilinx

#endif // AIR_LINALG_CODEGEN_H