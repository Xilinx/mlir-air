//===- MatmulCodegenConfig.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Util/MatmulCodegenConfig.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

using namespace mlir;

namespace xilinx {
namespace air {

std::optional<DictionaryAttr> findMatmulCodegenConfig(func::FuncOp funcOp) {
  StringRef name = getMatmulCodegenConfigAttrName();
  std::optional<DictionaryAttr> found;
  funcOp.walk([&](Operation *op) {
    if (auto attr = op->getDiscardableAttr(name)) {
      if (auto dict = dyn_cast<DictionaryAttr>(attr)) {
        found = dict;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

SmallVector<int64_t> getI64Array(DictionaryAttr cfg, StringRef key) {
  SmallVector<int64_t> out;
  if (!cfg)
    return out;
  auto entry = cfg.get(key);
  auto arr = dyn_cast_if_present<ArrayAttr>(entry);
  if (!arr)
    return out;
  for (Attribute a : arr) {
    if (auto i = dyn_cast<IntegerAttr>(a))
      out.push_back(i.getInt());
  }
  return out;
}

int64_t getI64(DictionaryAttr cfg, StringRef key, int64_t defaultVal) {
  if (!cfg)
    return defaultVal;
  auto entry = cfg.get(key);
  if (auto i = dyn_cast_if_present<IntegerAttr>(entry))
    return i.getInt();
  return defaultVal;
}

bool getBool(DictionaryAttr cfg, StringRef key, bool defaultVal) {
  if (!cfg)
    return defaultVal;
  auto entry = cfg.get(key);
  if (auto b = dyn_cast_if_present<BoolAttr>(entry))
    return b.getValue();
  return defaultVal;
}

bool writeMatmulCodegenConfig(func::FuncOp funcOp, DictionaryAttr dict,
                              StringRef markerName) {
  StringRef name = getMatmulCodegenConfigAttrName();
  Operation *target = nullptr;
  if (!markerName.empty()) {
    funcOp.walk([&](Operation *op) {
      if (op->hasAttr(markerName)) {
        target = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }
  if (!target) {
    funcOp.walk([&](linalg::MatmulOp op) {
      target = op.getOperation();
      return WalkResult::interrupt();
    });
  }
  if (!target)
    return false;
  target->setDiscardableAttr(name, dict);
  return true;
}

DictionaryAttr buildMatmulCodegenConfig(MLIRContext *ctx,
                                        ArrayRef<NamedAttribute> entries) {
  SmallVector<NamedAttribute> filtered;
  filtered.reserve(entries.size());
  for (const NamedAttribute &e : entries)
    if (e.getValue())
      filtered.push_back(e);
  return DictionaryAttr::get(ctx, filtered);
}

} // namespace air
} // namespace xilinx
