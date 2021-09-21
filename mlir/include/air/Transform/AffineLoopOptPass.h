// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#ifndef AFFINE_LOOP_OPT_H
#define AFFINE_LOOP_OPT_H

#include <mlir/Pass/Pass.h>
#include <memory>
#include <vector>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAffineLoopOptPass();

} // namespace air
} // namespace xilinx

#endif // AFFINE_LOOP_OPT_H