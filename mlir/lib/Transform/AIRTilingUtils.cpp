//===- AIRTilingUtils.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRTilingUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"

#define DEBUG_TYPE "air-tiling-utils"

using namespace mlir;

namespace xilinx {
namespace air {
/// Function from mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp
/// Identify valid and profitable bands of loops to tile. This is currently just
/// a temporary placeholder to test the mechanics of tiled code generation.
/// Returns all maximal outermost perfect loop nests to tile.
void getTileableBands(func::FuncOp f,
                      std::vector<SmallVector<AffineForOp, 6>> &bands,
                      const char *attrName, StringRef label) {
  // Get maximal perfect nest of 'affine.for' insts starting from root
  // (inclusive).
  auto getMaximalPerfectLoopNest = [&](AffineForOp root) {
    SmallVector<AffineForOp, 6> band;
    getPerfectlyNestedLoops(band, root);
    bands.push_back(band);
  };

  for (auto &block : f)
    for (auto &op : block)
      if (auto forOp = dyn_cast<AffineForOp>(op)) {
        auto targetForOp = getLabel(forOp, label, attrName);
        if (targetForOp) {
          getMaximalPerfectLoopNest(targetForOp);
        }
      }
}

AffineForOp getLabel(AffineForOp root, StringRef label,
                                  const char* attrName) {
  AffineForOp res;

  root.walk([&](AffineForOp forOp) {
    auto stringAttr = forOp->getAttrOfType<StringAttr>(attrName);
    if (!stringAttr)
      return WalkResult::advance();
    auto forOpCodegenName = stringAttr.getValue();
    if (label.empty() || forOpCodegenName.equals(label)) {
      res = forOp;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return res;
}

} // namespace air
} // namespace xilinx


