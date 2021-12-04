// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.
#ifndef AIR_UTIL_OUTLINER_H
#define AIR_UTIL_OUTLINER_H

#include <string>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

namespace xilinx {
namespace air {

struct AIROutliner {

public:
  AIROutliner()
  {
  }

  mlir::CallOp outline(std::vector<mlir::Operation*> ops, std::string fname="acap_outline_fn");
  mlir::CallOp outline(mlir::AffineForOp forOp, std::string fname="acap_outline_fn");

private:

  //mlir::ModuleOp &module;
};

} // namespace air
} // namespace xilinx
#endif // AIR_UTIL_OUTLINER_H