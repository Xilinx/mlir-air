// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.
#include "air/Util/CostModel.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/SCF/SCF.h"

#include "llvm/ADT/TypeSwitch.h"
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

void
CostModel::getLinalgOpCounts(OpCountMap &map, linalg::LinalgOp op) {
  OpBuilder b(op);
  auto loc = op.getLoc();

  // use getStaticLoopRanges instead?
  auto allShapeSizes = op.createFlatListOfOperandDims(b, loc);
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap)
    return;

  auto shapeSizes =
      linalg::applyMapToValues(b, loc, shapeSizesToLoopsMap, allShapeSizes);
  int64_t iters = 1;
  int64_t reads = 0;
  int64_t writes = 0;
  for (auto size : shapeSizes) {
    auto c = dyn_cast<ConstantIndexOp>(size.getDefiningOp());
    if (!c) {
      LLVM_DEBUG(llvm::outs() << "Found non-constant dim!\n");
      return;
    }
    iters *= c.getValue();
  }
  Region &region = op.getOperation()->getRegion(0);
  region.walk([&](Operation *op) {
    OperationName name = op->getName();
    std::string s = name.getStringRef().str();
    if (map.count(s) == 0)
      map.map.insert({s, 0});
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
  map.map.erase("linalg.yield");

  for (auto &m : map.map)
    m.second = m.second * iters;
  return;
}

void
CostModel::getScfForOpCounts(CostModel::OpCountMap &map, scf::ForOp op)
{
  // everything must be a constant
  auto step = op.step();
  if (!step.getDefiningOp<ConstantIndexOp>())
    return;

  auto lowerBound = op.lowerBound();
  if (!lowerBound.getDefiningOp<ConstantIndexOp>())
    return;

  auto upperBound = op.upperBound();
  if (!upperBound.getDefiningOp<ConstantIndexOp>())
    return;

  auto stepI64 = cast<ConstantIndexOp>(step.getDefiningOp()).getValue();
  auto lowerBoundI64 = cast<ConstantIndexOp>(lowerBound.getDefiningOp()).getValue();
  auto upperBoundI64 = cast<ConstantIndexOp>(upperBound.getDefiningOp()).getValue();

  auto iters = (upperBoundI64 - lowerBoundI64) / stepI64;

  map.map.insert({"step", stepI64});
  map.map.insert({"lb", lowerBoundI64});
  map.map.insert({"ub", upperBoundI64});
  map.map.insert({"iters", iters});

  auto body = op.getBody();
  for (auto &o : body->getOperations())
    map.ops.push_back(getOpCounts(&o));

  return;
}

CostModel::OpCountMap
CostModel::getOpCounts(Operation* op)
{
  OpCountMap map;
  map.name = op->getName().getStringRef().str();
  llvm::TypeSwitch<Operation*>(op)
      .Case<linalg::LinalgOp>([&](linalg::LinalgOp o){
        getLinalgOpCounts(map, o);
      })
      .Case<scf::ForOp>([&](scf::ForOp o){
        getScfForOpCounts(map, o);
      })
      .Default([&](Operation *op){
        return map;//map.insert({"unknown", 1});
      });
  return map;
}

void
CostModel::opCountToJSON(OpCountMap &opCounts,
                         llvm::json::Object &parent) {
  llvm::json::Object layerStatsJSON;
  for (auto p : opCounts.map) {
    auto name = p.first;
    auto count = p.second;
    layerStatsJSON[name] = count;
  }
  for (auto oc : opCounts.ops) {
    opCountToJSON(oc, layerStatsJSON);
  }
  parent[opCounts.name + std::to_string(LayerID++)] =
      llvm::json::Value(std::move(layerStatsJSON));
}

std::string
CostModel::opCountsToJSON(ModuleOp module) {
  llvm::json::Object top;

  module.walk([&](mlir::FuncOp fop) {
    LayerID = 0;
    llvm::json::Object function;
    fop.walk([&](Operation *op) {
      auto opCounts = getOpCounts(op);
      opCountToJSON(opCounts, function);
    });
    top[fop.sym_name()] = llvm::json::Value(std::move(function));
  });

  llvm::json::Value topv(std::move(top));
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << llvm::formatv("{0:2}", topv) << "\n";
  return ss.str();
}

} // namespace air
} // namespace xilinx
