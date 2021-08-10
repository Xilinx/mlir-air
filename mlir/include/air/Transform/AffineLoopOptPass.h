// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#pragma once

#include "ATenPasses.h"

namespace mlir {
class Pass;
} // namespace mlir

namespace xilinx {
namespace aten {

std::unique_ptr<mlir::Pass> createAffineLoopOptPass();
void registerAffineLoopOptPass();

extern std::vector<uint64_t> AffineLoopOptCopyDepths;
extern std::vector<uint64_t> AffineLoopOptTileSizes;
extern uint64_t AffineLoopOptFastSpace;
extern uint64_t AffineLoopOptSlowSpace;

} // namespace aten
} // namespace xilinx
