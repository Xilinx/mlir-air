// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#pragma once

namespace mlir {
class Pass;
} // namespace mlir

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRLoweringPass();
void registerAIRLoweringPass();

} // namespace aten
} // namespace xilinx
