//===- AIRTilingUtils.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRTilingUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/IR/BuiltinOps.h"

#define DEBUG_TYPE "air-tiling-utils"

using namespace mlir;
using namespace mlir::affine;

namespace xilinx {
namespace air {
/// Function from mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp
/// Identify valid and profitable bands of loops to tile. This is currently just
/// a temporary placeholder to test the mechanics of tiled code generation.
/// Returns all maximal outermost perfect loop nests to tile.
void getTileableBands(func::FuncOp f,
                      std::vector<SmallVector<affine::AffineForOp, 6>> &bands,
                      const char *attrName, StringRef label) {
  // Get maximal perfect nest of 'affine.for' insts starting from root
  // (inclusive).
  auto getMaximalPerfectLoopNest = [&](affine::AffineForOp root) {
    SmallVector<affine::AffineForOp, 6> band;
    getPerfectlyNestedLoops(band, root);
    bands.push_back(band);
  };

  for (auto &block : f)
    for (auto &op : block)
      if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
        auto targetForOp = getLabel(forOp, label, attrName);
        if (targetForOp) {
          getMaximalPerfectLoopNest(targetForOp);
        }
      }
}

affine::AffineForOp getLabel(affine::AffineForOp root, StringRef label,
                             const char *attrName) {
  affine::AffineForOp res;

  root.walk([&](affine::AffineForOp forOp) {
    auto stringAttr = forOp->getAttrOfType<StringAttr>(attrName);
    if (!stringAttr)
      return WalkResult::advance();
    auto forOpCodegenName = stringAttr.getValue();
    if (label.empty() || forOpCodegenName == label) {
      res = forOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return res;
}

} // namespace air
} // namespace xilinx
