
// (c) Copyright 2020 Xilinx Inc. All Rights Reserved.

#include "air/Transform/AffineLoopOptPass.h"
#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "air/Util/Outliner.h"

#include "PassDetail.h"
#include "mlir/Analysis/AffineAnalysis.h"
#include "mlir/Analysis/LoopAnalysis.h"
#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/LoopUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Utils.h"

#include "llvm/Support/Debug.h"

#include <set>

#define DEBUG_TYPE "affine-loop-opt"

using namespace mlir;

namespace xilinx {
namespace air {

std::vector<uint64_t> AffineLoopOptCopyDepths;
std::vector<uint64_t> AffineLoopOptTileSizes;
uint64_t AffineLoopOptFastSpace = 1;
uint64_t AffineLoopOptSlowSpace = 0;

}
}

namespace {
  
class AffineLoopOptPass : public PassWrapper<AffineLoopOptPass, FunctionPass> {

public:
  AffineLoopOptPass() = default;
  AffineLoopOptPass(const AffineLoopOptPass &pass){};

  void init_options()
  {
    optTileSizes.clear();
    if (clTileSizes.size() > 0) {
      for (unsigned i = 0; i < clTileSizes.size(); ++i) {
        optTileSizes.push_back(clTileSizes[i]);
        LLVM_DEBUG(llvm::outs() << "clTileSizes[" << i << "] = " << clTileSizes[i] << "\n");
      }
    }
    else if (xilinx::air::AffineLoopOptTileSizes.size() > 0) {
      for (unsigned i = 0; i < xilinx::air::AffineLoopOptTileSizes.size(); ++i) {
        optTileSizes.push_back(xilinx::air::AffineLoopOptTileSizes[i]);
        LLVM_DEBUG(llvm::outs() << "AffineLoopOptTileSizes[" << i << "] = "
          << xilinx::air::AffineLoopOptTileSizes[i] << "\n");
      }
    }

    optCopyDepths.clear();
    if (clCopyDepths.size() > 0) {
      for (unsigned i = 0; i < clCopyDepths.size(); ++i) {
        optCopyDepths.push_back(clCopyDepths[i]);
        LLVM_DEBUG(llvm::outs() << "clCopyDepths[" << i << "] = " << clCopyDepths[i] << "\n");
      }
    }
    else if (xilinx::air::AffineLoopOptCopyDepths.size() > 0) {
      for (unsigned i = 0; i < xilinx::air::AffineLoopOptCopyDepths.size(); ++i) {
        optCopyDepths.push_back(xilinx::air::AffineLoopOptCopyDepths[i]);
        LLVM_DEBUG(llvm::outs() << "AffineLoopOptCopyDepths[" << i << "] = "
          << xilinx::air::AffineLoopOptCopyDepths[i] << "\n");
      }
    }

    // if (clFastSpace.getNumOccurrences() == 0)
    //   clFastSpace.setValue(xilinx::air::AffineLoopOptFastSpace);

    // if (clSlowSpace.getNumOccurrences() == 0)
    //   clSlowSpace.setValue(xilinx::air::AffineLoopOptSlowSpace);

    erasedOps.clear();
    dataCopyNests.clear();
  }
 
  // Option<bool> clUnroll{*this, "affine-opt-unroll",
  //                              llvm::cl::desc("Affine loop unrolling"),
  //                              llvm::cl::init(false));

  // Option<bool> clPermute("affine-opt-permute",
  //                         llvm::cl::desc("Affine loop permutation"),
  //                         llvm::cl::init(false));

  ListOption<unsigned> clTileSizes{*this, "affine-opt-tile-sizes",
                                        llvm::cl::desc("Affine loop tiling sizes"),
                                        llvm::cl::ZeroOrMore,
                                        llvm::cl::CommaSeparated};

  ListOption<unsigned> clCopyDepths{*this, "affine-opt-copy-depths",
                                        llvm::cl::desc("Affine loop data copy loop depths"),
                                        llvm::cl::ZeroOrMore,
                                        llvm::cl::CommaSeparated};

  Option<unsigned> clFastSpace{*this, "affine-opt-copy-fast-space",
                              llvm::cl::desc("Fast memory space to use for affine data copy"),
                              llvm::cl::init(1)};

  Option<unsigned> clSlowSpace{*this, "affine-opt-copy-slow-space",
                              llvm::cl::desc("slow memory space to use for affine data copy"),
                              llvm::cl::init(0)};

  Option<unsigned> clCacheSize{*this, "affine-opt-cache-size",
                              llvm::cl::desc("Affine loop tiling cache size in KiB"),
                              llvm::cl::init(32)};

  Option<bool> clSeparate{*this, "affine-opt-tile-separate",
                          llvm::cl::desc("Affine loop tiling separates full and partial tiles"),
                          llvm::cl::init(false)};

  Option<std::string> clAffineOptLabel{*this, "affine-opt-label",
                          llvm::cl::desc("Transform loops with the given label"),
                          llvm::cl::init("")};
                          
  Option<std::string> clAffineOptPostLabel{*this, "affine-opt-post-label",
                          llvm::cl::desc("Label to apply to transformed loop nest"),
                          llvm::cl::init("")};

  void runOnFunction() override;
  // void runOnBlock(Block *block);
  // static const llvm::DenseMap<StringRef, unsigned> optConf;

  void tileLoops(std::vector<SmallVector<AffineForOp, 6>> *bands);
  void generateDataCopyLoops(std::vector<SmallVector<AffineForOp, 6>> *bands, Optional<Value> filterMemRef = None);
  void outlineDataCopyLoops();

  static void checkDependences(ArrayRef<Operation *> loadsAndStores);
  static void getTileableBands(FuncOp f, std::vector<SmallVector<AffineForOp, 6>> *bands, StringRef label);
  static void adjustToDivisorsOfTripCounts(ArrayRef<AffineForOp> band,
                                           SmallVectorImpl<unsigned> *tileSizes);
  void getTileSizes(ArrayRef<AffineForOp> band, SmallVectorImpl<unsigned> *tileSizes);

  // Default tile size if nothing is provided.
  //constexpr static unsigned kDefaultTileSize = 4;

  SmallVector<unsigned, 6> optTileSizes;
  SmallVector<unsigned, 6> optCopyDepths;
  std::set<Operation*>  erasedOps;
  SmallVector<DenseSet<Operation *>, 3> dataCopyNests; 

  // If true, tile sizes are set to avoid max/min in bounds if possible.
  bool avoidMaxMinBounds = true;

  static const char *affineOptAttrName;

private:

};

const char *AffineLoopOptPass::affineOptAttrName = "affine_opt_label";

void AffineLoopOptPass::runOnFunction() {

  init_options();
  auto func = getFunction();

  // Bands of loops to tile.
  std::vector<SmallVector<AffineForOp, 6>> bands;
  getTileableBands(func, &bands, clAffineOptLabel);

  //llvm::outs() << "FOOT: " << getMemoryFootprintBytes(bands[0][0]) <<"\n";

  // Tile loops
  if (optTileSizes.size())
    tileLoops(&bands);

  // Bands of loops to generate copies
  bands.clear();
  getTileableBands(func, &bands, clAffineOptLabel);

  //llvm::outs() << "PRINT: " << getMemoryFootprintBytes(bands[0][3])<<"\n";

  // Generate copies
  generateDataCopyLoops(&bands);
  outlineDataCopyLoops();

  // remove unused ops
  for (auto o : erasedOps) {
    o->erase();
  }

  // sinkSequentialLoops(f); 
  // analyzeDependences(f);
  // permuteLoops(f); 
}


static AffineForOp getLabel(AffineForOp root, StringRef label) {
  AffineForOp res;

  root.walk([&](AffineForOp forOp) {
    auto stringAttr = forOp->getAttrOfType<StringAttr>(
        AffineLoopOptPass::affineOptAttrName);
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

void AffineLoopOptPass::tileLoops(std::vector<SmallVector<AffineForOp, 6>> *bands) {
  // Tile loops

  // Tile each band.
  for (auto &band : *bands) {
    // Set up tile sizes; fill missing tile sizes at the end with default tile
    // size or clTileSize if one was provided.
    // SmallVector<unsigned, 6> tileSizes;
    // getTileSizes(band, &tileSizes);
    // if (llvm::DebugFlag) {
    //   auto diag = band[0].emitRemark("using tile sizes [");
    //   for (auto tSize : tileSizes)
    //     diag << tSize << ' ';
    //   diag << "]\n";
    // }
    SmallVector<AffineForOp, 6> tiledNest;
    SmallVector<unsigned, 6> actualTileSizes = optTileSizes;
    //unsigned arch_levels = actualTileSizes.size();
    unsigned loop_depth  = band.size();
    actualTileSizes.resize(loop_depth, 1);

    // if (failed(tilePerfectlyNested(band, tileSizes, &tiledNest)))
    if (failed(tilePerfectlyNested(band, actualTileSizes, &tiledNest)))
      return signalPassFailure();

    // Separate full and partial tiles.
    if (clSeparate) {
      auto intraTileLoops =
          MutableArrayRef<AffineForOp>(tiledNest).drop_front(band.size());
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


void AffineLoopOptPass::generateDataCopyLoops(std::vector<SmallVector<AffineForOp, 6>> *bands, Optional<Value> filterMemRef) {

  if (bands->size() == 0)
    return;

  if (optCopyDepths.size() == 0)
    return;

  auto fors = (*bands)[0];

  for (auto fors : *bands) {
    DenseSet<Operation *> copyNests;

    AffineCopyOptions copyOptions;
    copyOptions.generateDma = true;
    copyOptions.slowMemorySpace = clSlowSpace;
    copyOptions.fastMemorySpace = clFastSpace;
    copyOptions.tagMemorySpace = 0;
    copyOptions.fastMemCapacityBytes = 100e9;

    for (auto depth : optCopyDepths) {
      affineDataCopyGenerate(fors[depth], copyOptions, filterMemRef, copyNests);
      dataCopyNests.push_back(copyNests);
    }
  }
}

void AffineLoopOptPass::outlineDataCopyLoops() {
  for (auto &nest : dataCopyNests) {
    for (auto o : nest) {
      xilinx::air::AIROutliner olnr;
      /*auto call = */olnr.outline(cast<AffineForOp>(o), "air_dma_copy");
      erasedOps.insert(o);
    }
  }
}


// void AffineLoopOptPass::sinkSequentialLoops(FuncOp f) {
//   // Move sequential loops inside
//   f.walk([&](Operation *op) {
//     // if (auto forOp = dyn_cast<AffineForOp>(op)) {
//     if (isa<AffineForOp>(op)) {
//       AffineForOp newRootForOp = sinkSequentialLoops(cast<AffineForOp>(op));
//       // *op = newRootForOp.getOperation();
//       op->replaceAllUsesWith(newRootForOp.getOperation());
//     }
//   });
// }

// void AffineLoopOptPass::analyzeDependences(FuncOp f) {
//   // Dependence test
//   // Collect the loads and stores within the function.
//   SmallVector<Operation *, 4> loadsAndStores;
//   loadsAndStores.clear();
//   f.walk([&](Operation *op) {
//     if (isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op))
//       loadsAndStores.push_back(op);
//   });
//   checkDependences(loadsAndStores);
// }

// void AffineLoopOptPass::permuteLoops(FuncOp f) {
//   // loop permutation
//   // Get the first maximal perfect nest.
//   SmallVector<AffineForOp, 6> nest;
//   for (auto &op : f.front()) {
//     if (auto forOp = dyn_cast<AffineForOp>(op)) {
//       getPerfectlyNestedLoops(nest, forOp);
//       break;
//     }
//   }

//   // Nothing to do.
//   if (nest.size() < 2)
//     return;

//   SmallVector<unsigned, 4> permMap(permList.begin(), permList.end());
//   permuteLoops(nest, permMap);

// }



// getDirectionVectorStr and checkDependences functions: 
// from mlir/test/lib/Transforms/TestMemRefDependenceCheck.cpp

// Returns a result string which represents the direction vector (if there was
// a dependence), returns the string "false" otherwise.
static std::string
getDirectionVectorStr(bool ret, unsigned numCommonLoops, unsigned loopNestDepth,
                      ArrayRef<DependenceComponent> dependenceComponents) {
  if (!ret)
    return "false";
  if (dependenceComponents.empty() || loopNestDepth > numCommonLoops)
    return "true";
  std::string result;
  for (unsigned i = 0, e = dependenceComponents.size(); i < e; ++i) {
    std::string lbStr = "-inf";
    if (dependenceComponents[i].lb.hasValue() &&
        dependenceComponents[i].lb.getValue() !=
            std::numeric_limits<int64_t>::min())
      lbStr = std::to_string(dependenceComponents[i].lb.getValue());

    std::string ubStr = "+inf";
    if (dependenceComponents[i].ub.hasValue() &&
        dependenceComponents[i].ub.getValue() !=
            std::numeric_limits<int64_t>::max())
      ubStr = std::to_string(dependenceComponents[i].ub.getValue());

    result += "[" + lbStr + ", " + ubStr + "]";
  }
  return result;
}

// For each access in 'loadsAndStores', runs a dependence check between this
// "source" access and all subsequent "destination" accesses in
// 'loadsAndStores'. Emits the result of the dependence check as a note with
// the source access.
void AffineLoopOptPass::checkDependences(ArrayRef<Operation *> loadsAndStores) {
  for (unsigned i = 0, e = loadsAndStores.size(); i < e; ++i) {
    auto *srcOpInst = loadsAndStores[i];
    MemRefAccess srcAccess(srcOpInst);
    for (unsigned j = 0; j < e; ++j) {
      auto *dstOpInst = loadsAndStores[j];
      MemRefAccess dstAccess(dstOpInst);

      unsigned numCommonLoops =
          getNumCommonSurroundingLoops(*srcOpInst, *dstOpInst);
      for (unsigned d = 1; d <= numCommonLoops + 1; ++d) {
        FlatAffineConstraints dependenceConstraints;
        SmallVector<DependenceComponent, 2> dependenceComponents;
        DependenceResult result = checkMemrefAccessDependence(
            srcAccess, dstAccess, d, &dependenceConstraints,
            &dependenceComponents);
        assert(result.value != DependenceResult::Failure);
        bool ret = hasDependence(result);
        // TODO(andydavis) Print dependence type (i.e. RAW, etc) and print
        // distance vectors as: ([2, 3], [0, 10]). Also, shorten distance
        // vectors from ([1, 1], [3, 3]) to (1, 3).
        srcOpInst->emitRemark("dependence from ")
            << i << " to " << j << " at depth " << d << " = "
            << getDirectionVectorStr(ret, numCommonLoops, d,
                                     dependenceComponents);
      }
    }
  }
}

// Functions from mlir/lib/Dialect/Affine/Transforms/LoopTiling.cpp

// Identify valid and profitable bands of loops to tile. This is currently just
// a temporary placeholder to test the mechanics of tiled code generation.
// Returns all maximal outermost perfect loop nests to tile.
void AffineLoopOptPass::getTileableBands(FuncOp f,
                      std::vector<SmallVector<AffineForOp, 6>> *bands,
                      StringRef label) {
  // Get maximal perfect nest of 'affine.for' insts starting from root
  // (inclusive).
  auto getMaximalPerfectLoopNest = [&](AffineForOp root) {
    SmallVector<AffineForOp, 6> band;
    getPerfectlyNestedLoops(band, root);
    bands->push_back(band);
  };

  for (auto &block : f)
    for (auto &op : block)
      if (auto forOp = dyn_cast<AffineForOp>(op)) {
        auto targetForOp = getLabel(forOp, label);
        if (targetForOp) {
          getMaximalPerfectLoopNest(targetForOp);
        }
      }
}

/// Reduces each tile size to the largest divisor of the corresponding trip
/// count (if the trip count is known).
void AffineLoopOptPass::adjustToDivisorsOfTripCounts(ArrayRef<AffineForOp> band,
                                         SmallVectorImpl<unsigned> *tileSizes) {
  assert(band.size() == tileSizes->size() && "invalid tile size count");
  for (unsigned i = 0, e = band.size(); i < e; i++) {
    unsigned &tSizeAdjusted = (*tileSizes)[i];
    auto mayConst = getConstantTripCount(band[i]);
    if (!mayConst)
      continue;
    // Adjust the tile size to largest factor of the trip count less than
    // tSize.
    uint64_t constTripCount = mayConst.getValue();
    if (constTripCount > 1 && tSizeAdjusted > constTripCount / 2)
      tSizeAdjusted = constTripCount / 2;
    while (constTripCount % tSizeAdjusted != 0)
      tSizeAdjusted--;
  }
}

// Returns tile sizes to use. Checks CL options; if none are specified, sets it
// based on a simple model that looks at the memory footprint and determines
// tile sizes assuming identity accesses / 1:1 tile size proportional footprint
// along each of the dimensions being tiled.
// TODO(mlir-team): evolve this model. Tile size determination is a large area
// to play with in general.
void AffineLoopOptPass::getTileSizes(ArrayRef<AffineForOp> band,
                                     SmallVectorImpl<unsigned> *tileSizes) {
  if (band.empty())
    return;

  unsigned clTileSize = 4; // tmp

  // Use command-line clTileSize for all loops if specified.
  if (clTileSize) {
    tileSizes->assign(band.size(), clTileSize);
    return;
  }

  // // Use tileSizes and fill them with default tile size if it's short.
  // if (!this->tileSizes.empty()) {
  //   tileSizes->assign(this->tileSizes.begin(), this->tileSizes.end());
  //   tileSizes->resize(band.size(), kDefaultTileSize);
  //   return;
  // }
  tileSizes->resize(band.size());

  // The first loop in the band.
  auto rootForOp = band[0];
  (void)rootForOp;

  // Obtain memory footprint and set tile sizes so that a tile fits in
  // the cache size. This is an approximation with the assumption that the
  // footprint increases with the tile size linearly in that dimension (i.e.,
  // assumes one-to-one access function).
  auto fp = getMemoryFootprintBytes(band[0], 0);
  if (!fp) {
    // Fill with default tile sizes if footprint is unknown.
    std::fill(tileSizes->begin(), tileSizes->end(), clTileSize);
    if (avoidMaxMinBounds)
      adjustToDivisorsOfTripCounts(band, tileSizes);
    LLVM_DEBUG(
        rootForOp.emitWarning("memory footprint unknown: using default tile "
                              "sizes adjusted to trip count divisors"));
    return;
  }

  // Check how many times larger the cache size is when compared to footprint.
  uint64_t cacheSizeBytes = clCacheSize * 1024;
  uint64_t excessFactor = llvm::divideCeil(fp.getValue(), cacheSizeBytes);
  if (excessFactor <= 1) {
    // No need of any tiling - set tile size to 1.
    std::fill(tileSizes->begin(), tileSizes->end(), 1);
    return;
  }

  // Divide all loops equally in an attempt to reduce footprint.
  // TODO(bondhugula): this is approximate. Ideally, obtain reuse factor /
  // profitability along each dimension and weight tile sizes based on that as
  // one possible approach. Or compute a polynomial in tile sizes and solve for
  // it.

  // For an n-d tileable band, compute the n^th root of the excess.
  unsigned tSize =
      static_cast<unsigned>(floorl(std::pow(excessFactor, 1.0 / band.size())));
  // We'll keep a running product to determine the last tile size better.
  unsigned cumulProductOfTileSizes = 1;
  for (unsigned i = 0, e = band.size(); i < e; i++) {
    if (i < e - 1)
      (*tileSizes)[i] = tSize;
    else
      // Set last tile size to cover the balance.
      (*tileSizes)[i] = std::max(
          1U, static_cast<unsigned>(excessFactor / cumulProductOfTileSizes));
    cumulProductOfTileSizes *= (*tileSizes)[i];
  }
  if (avoidMaxMinBounds)
    adjustToDivisorsOfTripCounts(band, tileSizes);
}


} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAffineLoopOptPass() {
  return std::make_unique<AffineLoopOptPass>();
}

} // namespace air
} // namespace xilinx
