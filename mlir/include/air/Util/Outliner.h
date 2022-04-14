// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#ifndef AIR_UTIL_OUTLINER_H
#define AIR_UTIL_OUTLINER_H

#include <string>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace xilinx {
namespace air {

struct AIROutliner {

public:
  AIROutliner()
  {
  }

  mlir::func::CallOp outline(std::vector<mlir::Operation *> ops,
                             std::string fname = "acap_outline_fn");
  mlir::func::CallOp outline(mlir::AffineForOp forOp,
                             std::string fname = "acap_outline_fn");

private:

  //mlir::ModuleOp &module;
};

} // namespace air
} // namespace xilinx
#endif // AIR_UTIL_OUTLINER_H