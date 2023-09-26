//===- Outliner.h -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_UTIL_OUTLINER_H
#define AIR_UTIL_OUTLINER_H

#include <string>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace xilinx {
namespace air {

struct AIROutliner {

public:
  AIROutliner() {}

  mlir::func::CallOp outline(std::vector<mlir::Operation *> ops,
                             std::string fname = "acap_outline_fn");
  mlir::func::CallOp outline(mlir::AffineForOp forOp,
                             std::string fname = "acap_outline_fn");

private:
  // mlir::ModuleOp &module;
};

} // namespace air
} // namespace xilinx
#endif // AIR_UTIL_OUTLINER_H