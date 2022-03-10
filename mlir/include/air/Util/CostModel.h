// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#ifndef AIR_UTIL_COSTMODEL_H
#define AIR_UTIL_COSTMODEL_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "llvm/Support/JSON.h"

namespace xilinx {
namespace air {

class CostModel {
public:
  CostModel() {}

  class OpCountMap {
  public:
    size_t count(std::string& s) {
      return map.count(s);
    }
    uint64_t &operator[](const std::string& s) {
      assert(map.count(s) && "key not found");
      return map[s];
    }
    std::string name;
    std::map<std::string, uint64_t> map;
    std::vector<OpCountMap> ops;
  };

  OpCountMap getOpCounts(mlir::Operation* op);
  std::string opCountsToJSON(mlir::ModuleOp module);
  void opCountToJSON(OpCountMap &opCounts, llvm::json::Object &top);

private:
  void getScfForOpCounts(OpCountMap &map, mlir::scf::ForOp op);
  void getLinalgOpCounts(OpCountMap &map, mlir::linalg::LinalgOp op);

  int LayerID;
};

} // namespace air
} // namespace xilinx
#endif // AIR_UTIL_COSTMODEL_H