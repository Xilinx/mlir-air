//===- CostModel.cpp --------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Util/CostModel.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

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


static uint64_t getTensorVolume(const ShapedType ty) {

  if (!ty.hasRank())
    return 1;

  uint64_t volume = 1;
  for (auto &d : ty.getShape())
    volume *= d;
  return volume * (ty.getElementTypeBitWidth()/8);
}

static uint64_t getTensorVolume(const Type ty) {
  if (auto t = llvm::dyn_cast<ShapedType>(ty)) {
    return getTensorVolume(t);
  }
  else {
    return 1;
  }
}


void
CostModel::getLinalgOpCounts(OpCountMap &map, linalg::LinalgOp op) {
  OpBuilder b(op);
  auto loc = op.getLoc();

  auto allShapeSizes = op.createFlatListOfOperandDims(b, loc);
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap)
    return;

  SmallVector<OpFoldResult> shapeSizes =
      affine::makeComposedFoldedMultiResultAffineApply(
          b, loc, shapeSizesToLoopsMap, allShapeSizes);
  int64_t iters = 1;
  int64_t reads = 0;
  int64_t writes = 0;
  uint64_t footprint = 0;
  for (auto size : shapeSizes) {
    if (auto v = llvm::dyn_cast<Value>(size)) {
      auto c = dyn_cast<arith::ConstantIndexOp>(v.getDefiningOp());
      if (!c) {
        LLVM_DEBUG(llvm::outs() << "Found non-constant dim!\n");
        return;
      }
      iters *= c.value();
    } else {
      auto a = llvm::dyn_cast<Attribute>(size);
      auto c = llvm::dyn_cast<IntegerAttr>(a);
      if (!c) {
        LLVM_DEBUG(llvm::outs() << "unhandled addr!\n");
        return;
      }
      iters *= c.getInt();
    }
  }
  Region &region = op.getOperation()->getRegion(0);
  region.walk([&](Operation *op) {
    OperationName name = op->getName();
    std::string s = name.getStringRef().str();
    if (map.count(s) == 0)
      map.map.insert({s, 0});
    map[s] = map[s] + 1;
  });
  for (auto &oper : op.getDpsInputOperands()) {
    if (op.payloadUsesValueFromOperand(oper))
      reads++;
    footprint += getTensorVolume(oper->get().getType());
  }
  for (int i = 0; i < op.getNumDpsInits(); i++) {
    auto oper = op.getDpsInitOperand(i);
    if (op.payloadUsesValueFromOperand(oper))
      reads++;
    writes++;
    footprint += getTensorVolume(oper->get().getType());
  }
  map.map.insert({"reads", reads});
  map.map.insert({"writes", writes});
  map.map.erase("linalg.yield");

  for (auto &m : map.map)
    m.second = m.second * iters;

  map.map.insert({"footprint", footprint});

  return;
}

void
CostModel::getScfForOpCounts(CostModel::OpCountMap &map, scf::ForOp op)
{
  // everything must be a constant
  auto step = op.getStep();
  if (!step.getDefiningOp<arith::ConstantIndexOp>())
    return;

  auto lowerBound = op.getLowerBound();
  if (!lowerBound.getDefiningOp<arith::ConstantIndexOp>())
    return;

  auto upperBound = op.getUpperBound();
  if (!upperBound.getDefiningOp<arith::ConstantIndexOp>())
    return;

  auto stepI64 = cast<arith::ConstantIndexOp>(step.getDefiningOp()).value();
  auto lowerBoundI64 = cast<arith::ConstantIndexOp>(lowerBound.getDefiningOp()).value();
  auto upperBoundI64 = cast<arith::ConstantIndexOp>(upperBound.getDefiningOp()).value();

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

  module.walk([&](func::FuncOp fop) {
    LayerID = 0;
    llvm::json::Object function;
    fop.walk([&](Operation *op) {
      auto opCounts = getOpCounts(op);
      opCountToJSON(opCounts, function);
    });
    top[fop.getSymName()] = llvm::json::Value(std::move(function));
  });

  llvm::json::Value topv(std::move(top));
  std::string ret;
  llvm::raw_string_ostream ss(ret);
  ss << llvm::formatv("{0:2}", topv) << "\n";
  return ss.str();
}

} // namespace air
} // namespace xilinx
