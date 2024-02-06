//===- AffineLoopOptPass.cpp ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2020-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

#include "air/Transform/AffineLoopOptPass.h"
#include "air/Util/Outliner.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"

#include <optional>
#include <set>

#define DEBUG_TYPE "affine-loop-opt"

using namespace mlir;
using namespace xilinx::air;

namespace {

class AffineLoopOptPass
    : public xilinx::air::impl::AffineLoopOptPassBase<AffineLoopOptPass> {

public:
  AffineLoopOptPass() = default;
  AffineLoopOptPass(const AffineLoopOptPass &pass){};
  AffineLoopOptPass(const ::xilinx::air::AffineLoopOptPassOptions &options)
      : AffineLoopOptPassBase(options) {}

  void init_options() {
    optTileSizes.clear();
    if (clTileSizes.size() > 0) {
      for (unsigned i = 0; i < clTileSizes.size(); ++i) {
        optTileSizes.push_back(clTileSizes[i]);
        LLVM_DEBUG(llvm::outs()
                   << "clTileSizes[" << i << "] = " << clTileSizes[i] << "\n");
      }
    }

    optCopyDepths.clear();
    if (clCopyDepths.size() > 0) {
      for (unsigned i = 0; i < clCopyDepths.size(); ++i) {
        optCopyDepths.push_back(clCopyDepths[i]);
        LLVM_DEBUG(llvm::outs() << "clCopyDepths[" << i
                                << "] = " << clCopyDepths[i] << "\n");
      }
    }

    erasedOps.clear();
    dataCopyNests.clear();
  }

  void runOnOperation() override;
  // void runOnBlock(Block *block);
  // static const llvm::DenseMap<StringRef, unsigned> optConf;

  void tileLoops(std::vector<SmallVector<affine::AffineForOp, 6>> *bands);
  void
  generateDataCopyLoops(std::vector<SmallVector<affine::AffineForOp, 6>> *bands,
                        std::optional<Value> filterMemRef = std::nullopt);
  void outlineDataCopyLoops();

  static void
  getTileableBands(func::FuncOp,
                   std::vector<SmallVector<affine::AffineForOp, 6>> *bands,
                   StringRef label);

  SmallVector<unsigned, 6> optTileSizes;
  SmallVector<unsigned, 6> optCopyDepths;
  std::set<Operation *> erasedOps;
  SmallVector<DenseSet<Operation *>, 3> dataCopyNests;

  static const char *affineOptAttrName;

private:
};

const char *AffineLoopOptPass::affineOptAttrName = "affine_opt_label";

void AffineLoopOptPass::runOnOperation() {

  init_options();
  auto func = getOperation();

  // Bands of loops to tile.
  std::vector<SmallVector<affine::AffineForOp, 6>> bands;
  getTileableBands(func, &bands, clAffineOptLabel);

  // Tile loops
  if (optTileSizes.size())
    tileLoops(&bands);

  // Bands of loops to generate copies
  bands.clear();
  getTileableBands(func, &bands, clAffineOptLabel);

  // Generate copies
  generateDataCopyLoops(&bands);
  outlineDataCopyLoops();

  // remove unused ops
  for (auto o : erasedOps) {
    o->erase();
  }
}

static affine::AffineForOp getLabel(affine::AffineForOp root, StringRef label) {
  affine::AffineForOp res;

  root.walk([&](affine::AffineForOp forOp) {
    auto stringAttr =
        forOp->getAttrOfType<StringAttr>(AffineLoopOptPass::affineOptAttrName);
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

void AffineLoopOptPass::tileLoops(
    std::vector<SmallVector<affine::AffineForOp, 6>> *bands) {
  // Tile loops

  // Tile each band.
  for (auto &band : *bands) {

    SmallVector<affine::AffineForOp, 6> tiledNest;
    SmallVector<unsigned, 6> actualTileSizes = optTileSizes;

    unsigned loop_depth = band.size();
    actualTileSizes.resize(loop_depth, 1);

    if (failed(tilePerfectlyNested(band, actualTileSizes, &tiledNest)))
      return signalPassFailure();

    // Separate full and partial tiles.
    if (clSeparate) {
      auto intraTileLoops =
          MutableArrayRef<affine::AffineForOp>(tiledNest).drop_front(
              band.size());
      (void)separateFullTiles(intraTileLoops);
    }

    auto stringAttr = band[0]->getAttrOfType<StringAttr>(
        AffineLoopOptPass::affineOptAttrName);
    if (stringAttr)
      tiledNest[0]->setAttr(
          AffineLoopOptPass::affineOptAttrName,
          StringAttr::get(clAffineOptPostLabel, stringAttr.getType()));
  }
}

void AffineLoopOptPass::generateDataCopyLoops(
    std::vector<SmallVector<affine::AffineForOp, 6>> *bands,
    std::optional<Value> filterMemRef) {

  if (bands->size() == 0)
    return;

  if (optCopyDepths.size() == 0)
    return;

  auto fors = (*bands)[0];

  for (auto fors : *bands) {

    DenseSet<Operation *> copyNests;

    affine::AffineCopyOptions copyOptions;
    copyOptions.generateDma = true;
    copyOptions.slowMemorySpace = clSlowSpace;
    copyOptions.fastMemorySpace = clFastSpace;
    copyOptions.tagMemorySpace = 0;
    copyOptions.fastMemCapacityBytes = 100e9;

    for (auto depth : optCopyDepths) {
      if (depth >= fors.size())
        continue;
      (void)affineDataCopyGenerate(fors[depth], copyOptions, filterMemRef,
                                   copyNests);
      dataCopyNests.push_back(copyNests);
    }
  }
}

void AffineLoopOptPass::outlineDataCopyLoops() {
  for (auto &nest : dataCopyNests) {
    for (auto o : nest) {
      xilinx::air::AIROutliner olnr;
      /*auto call = */ olnr.outline(cast<affine::AffineForOp>(o),
                                    "air_dma_copy");
      erasedOps.insert(o);
    }
  }
}

// Functions from mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp

// Identify valid and profitable bands of loops to tile. This is currently just
// a temporary placeholder to test the mechanics of tiled code generation.
// Returns all maximal outermost perfect loop nests to tile.
void AffineLoopOptPass::getTileableBands(
    func::FuncOp f, std::vector<SmallVector<affine::AffineForOp, 6>> *bands,
    StringRef label) {
  // Get maximal perfect nest of 'affine.for' insts starting from root
  // (inclusive).
  auto getMaximalPerfectLoopNest = [&](affine::AffineForOp root) {
    SmallVector<affine::AffineForOp, 6> band;
    getPerfectlyNestedLoops(band, root);
    bands->push_back(band);
  };

  for (auto &block : f)
    for (auto &op : block)
      if (auto forOp = dyn_cast<affine::AffineForOp>(op)) {
        auto targetForOp = getLabel(forOp, label);
        if (targetForOp) {
          getMaximalPerfectLoopNest(targetForOp);
        }
      }
}

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAffineLoopOptPass() {
  return std::make_unique<AffineLoopOptPass>();
}

std::unique_ptr<Pass>
createAffineLoopOptPass(AffineLoopOptPassOptions options) {
  return std::make_unique<AffineLoopOptPass>(options);
}

} // namespace air
} // namespace xilinx
