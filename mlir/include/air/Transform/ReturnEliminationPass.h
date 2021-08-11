// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#ifndef RETURN_ELIMINATION_H
#define RETURN_ELIMINATION_H

#include "mlir/Pass/Pass.h"
#include <memory>
namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createReturnEliminationPass();

} // namespace air
} // namespace xilinx

#endif