// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_LINALG_OP_STATS_H
#define AIR_LINALG_OP_STATS_H

#include "mlir/Pass/Pass.h"
#include <memory>
#include <vector>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRLinalgOpStatsPass();

} // namespace air
} // namespace xilinx

#endif // AIR_LINALG_OP_STATS_H