// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#include "air/Util/CostModel.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <string>

#define DEBUG_TYPE "air-util-costmodel"

using namespace mlir;

namespace xilinx {
namespace air {

OpCountMap CostModel::getLinalgOpCounts(linalg::LinalgOp op) {
  OpBuilder b(op);
  auto loc = op.getLoc();

  std::map<std::string, int> map;
  // use getStaticLoopRanges instead?
  auto allShapeSizes = op.createFlatListOfOperandDims(b, loc);
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap)
    return OpCountMap();

  auto shapeSizes =
      linalg::applyMapToValues(b, loc, shapeSizesToLoopsMap, allShapeSizes);
  int64_t iters = 1;
  int64_t reads = 0;
  int64_t writes = 0;
  for (auto size : shapeSizes) {
    auto c = dyn_cast<ConstantIndexOp>(size.getDefiningOp());
    if (!c) {
      LLVM_DEBUG(llvm::outs() << "Found non-constant dim!\n");
      return map;
    }
    iters *= c.getValue();
  }
  Region &region = op.getOperation()->getRegion(0);
  region.walk([&](Operation *op) {
    OperationName name = op->getName();
    std::string s = name.getStringRef().str();
    if (map.count(s) == 0)
      map.insert({s, 0});
    map[s] = map[s] + 1;
  });
  for (auto &oper : op.getInputOperands())
    if (op.payloadUsesValueFromOperand(oper))
      reads++;
  for (auto &oper : op.getOutputOperands()) {
    if (op.payloadUsesValueFromOperand(oper))
      reads++;
    writes++;
  }
  map["reads"] = reads;
  map["writes"] = writes;
  map.erase("linalg.yield");

  for (auto &m : map)
    m.second = m.second * iters;
  return map;
}

std::string CostModel::opCountsToJSON(ModuleOp module) {
  llvm::json::Object top;

  module.walk([&](mlir::FuncOp fop) {
    LayerID = 0;
    llvm::json::Object function;
    fop.walk(
        [&](linalg::LinalgOp lop) { linalgOpCountsToJSON(lop, function); });
    top[fop.sym_name()] = llvm::json::Value(std::move(function));
  });

  llvm::json::Value topv(std::move(top));
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << llvm::formatv("{0:2}", topv) << "\n";
  return ss.str();
}

void CostModel::linalgOpCountsToJSON(linalg::LinalgOp op,
                                     llvm::json::Object &parent) {
  OpCountMap opCounts = getLinalgOpCounts(op);
  llvm::json::Object layerStatsJSON;
  for (auto p : opCounts) {
    auto name = p.first;
    auto count = p.second;
    layerStatsJSON[name] = count;
  }
  parent[op->getName().getStringRef().str() + std::to_string(LayerID++)] =
      llvm::json::Value(std::move(layerStatsJSON));
}

} // namespace air
} // namespace xilinx
