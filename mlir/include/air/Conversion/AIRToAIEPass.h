// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_TO_AIE_PASS_H
#define AIR_TO_AIE_PASS_H

#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"

#include <memory>

namespace mlir {
class ModuleOp;
class RewriterBase;
}

namespace xilinx {
namespace air {
class PartitionOp;

mlir::FailureOr<mlir::ModuleOp> convertAIRToAIE(mlir::RewriterBase &rewriter, air::PartitionOp partition);
std::unique_ptr<mlir::Pass> createAIRToAIEPass();

} // namespace air
} // namespace xilinx

#endif // AIR_TO_AIE_PASS_H