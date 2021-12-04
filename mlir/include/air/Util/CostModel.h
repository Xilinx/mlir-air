// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_UTIL_COSTMODEL_H
#define AIR_UTIL_COSTMODEL_H

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"


namespace xilinx {
namespace air {

typedef std::map<std::string, int> OpCountMap;

class CostModel {
public:
  CostModel() {}

  OpCountMap getLinalgOpCounts(mlir::linalg::LinalgOp op);
  void dumpLinalgOpCounts(mlir::linalg::LinalgOp op);
  void dumpLinalgOpCountsToJSON(mlir::linalg::LinalgOp op);
};
} // namespace air
} // namespace xilinx
#endif // AIR_UTIL_COSTMODEL_H