// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_UTIL_COSTMODEL_H
#define AIR_UTIL_COSTMODEL_H

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "llvm/Support/JSON.h"

namespace xilinx {
namespace air {

typedef std::map<std::string, int> OpCountMap;

class CostModel {
public:
  CostModel() {}

  OpCountMap getLinalgOpCounts(mlir::linalg::LinalgOp op);
  std::string opCountsToJSON(mlir::ModuleOp module);
  void linalgOpCountsToJSON(mlir::linalg::LinalgOp op, llvm::json::Object &top);

private:
  int LayerID;
};

} // namespace air
} // namespace xilinx
#endif // AIR_UTIL_COSTMODEL_H