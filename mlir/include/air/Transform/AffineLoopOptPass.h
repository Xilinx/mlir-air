// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#ifndef AFFINE_LOOP_OPT_H
#define AFFINE_LOOP_OPT_H

#include <mlir/Pass/Pass.h>
#include <memory>
#include <vector>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAffineLoopOptPass();

extern std::vector<uint64_t> AffineLoopOptCopyDepths;
extern std::vector<uint64_t> AffineLoopOptTileSizes;
extern uint64_t AffineLoopOptFastSpace;
extern uint64_t AffineLoopOptSlowSpace;

} // namespace air
} // namespace xilinx

#endif // AFFINE_LOOP_OPT_H