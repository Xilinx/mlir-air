//===- AIRSplitLaunchForPadding.cpp ----------------------------*- C++ -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRSplitLaunchForPadding.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace xilinx::air;

namespace xilinx {
namespace air {

namespace {

// ===----------------------------------------------------------------------===//
// Shared helpers
// ===----------------------------------------------------------------------===//

// Trace a memref Value through block args of air.HierarchyInterface ops to
// its function argument index.
static unsigned traceFuncArgIdx(Value memref) {
  Value traced = memref;
  while (auto ba = dyn_cast<BlockArgument>(traced)) {
    auto *parentOp = ba.getOwner()->getParentOp();
    if (auto hier = dyn_cast<HierarchyInterface>(parentOp)) {
      auto bodyArgs = hier.getKernelArguments();
      auto operands = hier.getKernelOperands();
      bool found = false;
      for (unsigned i = 0; i < bodyArgs.size(); i++) {
        if (bodyArgs[i] == ba) {
          traced = operands[i];
          found = true;
          break;
        }
      }
      if (!found)
        return UINT_MAX;
    } else if (isa<func::FuncOp>(parentOp)) {
      return ba.getArgNumber();
    } else {
      return UINT_MAX;
    }
  }
  return UINT_MAX;
}

// Collect tile-size candidates from arith.muli uses of blockIdx.
static void collectTileSizeCandidates(Value blockIdx,
                                      SmallVectorImpl<int64_t> &candidates) {
  for (auto &use : blockIdx.getUses()) {
    if (auto mulOp = dyn_cast<arith::MulIOp>(use.getOwner())) {
      Value other =
          (mulOp.getLhs() == blockIdx) ? mulOp.getRhs() : mulOp.getLhs();
      if (auto constVal = getConstantIntValue(other))
        if (*constVal > 0)
          candidates.push_back(*constVal);
    }
  }
}

// Infer tile size from arith.muli of a launch block index. Also follows the
// ID through air.segment hierarchy. When multiple multipliers exist, picks
// the smallest (tile offset, not output stride).
static int64_t inferTileSize(LaunchOp launchOp, unsigned dimIdx) {
  auto ids = launchOp.getIds();
  if (dimIdx >= ids.size())
    return 0;
  Value blockIdx = ids[dimIdx];

  SmallVector<int64_t> candidates;
  collectTileSizeCandidates(blockIdx, candidates);

  // Follow through segment hierarchy.
  for (auto &use : blockIdx.getUses()) {
    if (auto hier = dyn_cast<HierarchyInterface>(use.getOwner())) {
      auto operands = hier.getKernelOperands();
      auto bodyArgs = hier.getKernelArguments();
      for (unsigned i = 0; i < operands.size(); ++i) {
        if (operands[i] == blockIdx) {
          collectTileSizeCandidates(bodyArgs[i], candidates);
          break;
        }
      }
    }
  }

  if (candidates.empty())
    return 0;
  return *llvm::min_element(candidates);
}

// Returns true if this input operand index should be skipped for padding.
// argIdx==0 -> A (M-padded); argIdx==1 -> B (N-padded).
static bool skipInput(unsigned argIdx, bool padM, bool padN) {
  if (argIdx == 0)
    return !padM;
  if (argIdx == 1)
    return !padN;
  return true;
}

} // anonymous namespace

// ===----------------------------------------------------------------------===//
// AIRSplitLaunchForPadding
// ===----------------------------------------------------------------------===//

class AIRSplitLaunchForPadding
    : public xilinx::air::impl::AIRSplitLaunchForPaddingBase<
          AIRSplitLaunchForPadding> {
public:
  // Padding location: memtile (L2 MM2S, 3-level hierarchy) or source.
  enum class PadLocation { Memtile, Source };
  // IR representation: channel ops or dma_memcpy_nd ops.
  enum class OpKind { Channel, Dma };

  AIRSplitLaunchForPadding() = default;
  AIRSplitLaunchForPadding(const AIRSplitLaunchForPadding &pass)
      : AIRSplitLaunchForPaddingBase(pass) {}
  AIRSplitLaunchForPadding(const AIRSplitLaunchForPaddingOptions &options)
      : AIRSplitLaunchForPaddingBase(options) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, air::airDialect, arith::ArithDialect>();
  }

  // Resolve effective pad location from options (handles deprecated flag).
  PadLocation getEffectivePadLocation() const {
    if (clUseDmaMemcpy)
      return PadLocation::Source;
    if (clPadLocation == "source")
      return PadLocation::Source;
    return PadLocation::Memtile;
  }

  // Auto-detect whether the launch uses channel ops or DMA ops.
  static OpKind detectOpKind(air::LaunchOp launch) {
    bool hasChannels = false;
    launch.walk([&](Operation *op) {
      if (isa<air::ChannelPutOp, air::ChannelGetOp>(op))
        hasChannels = true;
    });
    return hasChannels ? OpKind::Channel : OpKind::Dma;
  }

  // Per-shim padding info for a channel.put/get op.
  struct ShimPadInfo {
    int64_t shimIdx;
    int64_t chunkSize;
    int64_t actualForShim;
    int64_t padForShim;
    int64_t padDimIdx; // dimension index of the padded dim in sizes
  };

  // Pre-computed info for an L3->L2 channel.get that needs explicit
  // sizes/strides added (Step 5 of addMemtilePaddingForChannels).
  struct S2MMReplaceInfo {
    air::ChannelGetOp getOp;
    int64_t padDimInMemref;
    int64_t shimIdx;
    int64_t actualForShim;
  };

  // Find the dimension index in L3->L2 channel.put sizes that corresponds to
  // the padded dimension (M for A, N for B). Traces which offset depends on
  // the launch block index BEFORE the launch is cloned/specialized.
  //
  // Must be called on the ORIGINAL launch (before cloneAndSpecializeLaunch
  // replaces block indices with constants). Returns {A_padDimIdx, B_padDimIdx}.
  std::pair<int64_t, int64_t>
  inferL3PadDimIndices(air::LaunchOp launchOp,
                       DenseMap<StringRef, unsigned> &chanToArgIdx) {
    int64_t aPadDim = 0;
    int64_t bPadDim = -1; // Will use last dim as fallback

    auto launchIds = launchOp.getIds();
    if (launchIds.size() < 2)
      return {aPadDim, bPadDim};

    // Find launch offset values for M (dim 0) and N (dim 1).
    Value mOffset = nullptr, nOffset = nullptr;
    for (auto &use : launchIds[0].getUses()) {
      if (auto mulOp = dyn_cast<arith::MulIOp>(use.getOwner())) {
        mOffset = mulOp.getResult();
        break;
      }
    }
    for (auto &use : launchIds[1].getUses()) {
      if (auto mulOp = dyn_cast<arith::MulIOp>(use.getOwner())) {
        nOffset = mulOp.getResult();
        break;
      }
    }

    // Scan L3->L2 channel.puts to find which offset dim uses the launch offset.
    launchOp.walk([&](air::ChannelPutOp putOp) {
      if (putOp->getParentOfType<air::SegmentOp>())
        return;
      if (!chanToArgIdx.count(putOp.getChanName()))
        return;
      unsigned argIdx = chanToArgIdx.lookup(putOp.getChanName());
      auto offsets = putOp.getOffsets();
      auto sizes = putOp.getSizes();
      if (offsets.empty() || sizes.empty())
        return;

      for (unsigned i = 0; i < offsets.size() && i < sizes.size(); ++i) {
        if (argIdx == 0 && mOffset && offsets[i] == mOffset) {
          aPadDim = i;
        } else if (argIdx == 1 && nOffset && offsets[i] == nOffset) {
          bPadDim = i;
        }
      }
    });

    return {aPadDim, bPadDim};
  }

  // DMA counterpart to inferL3PadDimIndices. Traces launch offset Values
  // through DmaMemcpyNdOp src_offsets to find which dimension is padded.
  std::pair<int64_t, int64_t>
  inferL3PadDimIndicesForDma(air::LaunchOp launchOp) {
    int64_t aPadDim = 0;
    int64_t bPadDim = -1;

    auto launchIds = launchOp.getIds();
    if (launchIds.size() < 2)
      return {aPadDim, bPadDim};

    // Find launch offset values for M/N (first arith.muli of each ID).
    Value mOffset = nullptr, nOffset = nullptr;
    for (auto &use : launchIds[0].getUses()) {
      if (auto mulOp = dyn_cast<arith::MulIOp>(use.getOwner())) {
        mOffset = mulOp.getResult();
        break;
      }
    }
    for (auto &use : launchIds[1].getUses()) {
      if (auto mulOp = dyn_cast<arith::MulIOp>(use.getOwner())) {
        nOffset = mulOp.getResult();
        break;
      }
    }

    // Scan launch-level DMA ops (not inside segment).
    launchOp.walk([&](air::DmaMemcpyNdOp dmaOp) {
      if (dmaOp->getParentOfType<air::SegmentOp>())
        return;
      unsigned argIdx = traceFuncArgIdx(dmaOp.getSrcMemref());
      if (argIdx > 1)
        return;
      auto offsets = dmaOp.getSrcOffsets();
      auto sizes = dmaOp.getSrcSizes();
      if (offsets.empty() || sizes.empty())
        return;
      for (unsigned i = 0; i < offsets.size() && i < sizes.size(); ++i) {
        if (argIdx == 0 && mOffset && offsets[i] == mOffset)
          aPadDim = i;
        else if (argIdx == 1 && nOffset && offsets[i] == nOffset)
          bPadDim = i;
      }
    });

    return {aPadDim, bPadDim};
  }

  // Compute per-shim padding info for an L3 channel op.
  // padDimIdxA/B are pre-computed from the original launch (before cloning).
  std::optional<ShimPadInfo> computeShimPadInfo(air::ChannelInterface chOp,
                                                bool isA, int64_t mActualLast,
                                                int64_t nActualLast,
                                                int64_t padDimIdxA,
                                                int64_t padDimIdxB) {
    auto sizes = chOp.getSizes();
    if (sizes.empty())
      return std::nullopt;

    int64_t padDimIdx =
        isA ? padDimIdxA
            : (padDimIdxB >= 0 ? padDimIdxB : ((int64_t)sizes.size() - 1));
    if (padDimIdx < 0 || padDimIdx >= (int64_t)sizes.size())
      return std::nullopt;
    auto chunkSizeOpt = getConstantIntValue(sizes[padDimIdx]);
    if (!chunkSizeOpt)
      return std::nullopt;
    int64_t chunkSize = *chunkSizeOpt;

    auto indices = chOp.getIndices();
    int64_t shimIdx = 0;
    if (!indices.empty()) {
      auto shimIdxOpt = getConstantIntValue(indices[0]);
      if (shimIdxOpt)
        shimIdx = *shimIdxOpt;
    }

    int64_t dimActualLast = isA ? mActualLast : nActualLast;
    int64_t shimOffset = shimIdx * chunkSize;
    int64_t actualForShim =
        std::max(int64_t(0), std::min(chunkSize, dimActualLast - shimOffset));
    int64_t padForShim = chunkSize - actualForShim;

    return ShimPadInfo{shimIdx, chunkSize, actualForShim, padForShim,
                       padDimIdx};
  }

  // DMA counterpart to computeShimPadInfo. No channel indices, so
  // shimIdx is always 0 (no multiplexing).
  std::optional<ShimPadInfo>
  computeShimPadInfoForDma(air::DmaMemcpyNdOp dmaOp, bool isA,
                           int64_t mActualLast, int64_t nActualLast,
                           int64_t padDimIdxA, int64_t padDimIdxB) {
    auto sizes = dmaOp.getSrcSizes();
    if (sizes.empty())
      return std::nullopt;

    int64_t padDimIdx =
        isA ? padDimIdxA
            : (padDimIdxB >= 0 ? padDimIdxB : ((int64_t)sizes.size() - 1));
    if (padDimIdx < 0 || padDimIdx >= (int64_t)sizes.size())
      return std::nullopt;
    auto chunkSizeOpt = getConstantIntValue(sizes[padDimIdx]);
    if (!chunkSizeOpt)
      return std::nullopt;
    int64_t chunkSize = *chunkSizeOpt;

    int64_t shimIdx = 0; // DMA: no channel multiplexing
    int64_t dimActualLast = isA ? mActualLast : nActualLast;
    int64_t shimOffset = shimIdx * chunkSize;
    int64_t actualForShim =
        std::max(int64_t(0), std::min(chunkSize, dimActualLast - shimOffset));
    int64_t padForShim = chunkSize - actualForShim;

    return ShimPadInfo{shimIdx, chunkSize, actualForShim, padForShim,
                       padDimIdx};
  }

  // Find the dimension in a memref shape where shimIdx * shape[d] partially
  // overlaps dimActualLast, indicating that dimension needs padding.
  static int64_t findPadDimInMemrefShape(ArrayRef<int64_t> shape,
                                         int64_t shimIdx, int64_t dimActualLast,
                                         bool searchForward) {
    auto check = [&](int64_t d) -> bool {
      if (shape[d] <= 1)
        return false;
      int64_t shimOffset = shimIdx * shape[d];
      int64_t actual =
          std::max(int64_t(0), std::min(shape[d], dimActualLast - shimOffset));
      return actual > 0 && actual < shape[d];
    };
    if (searchForward) {
      for (int64_t d = 0; d < (int64_t)shape.size(); ++d)
        if (check(d))
          return d;
    } else {
      for (int64_t d = (int64_t)shape.size() - 1; d >= 0; --d)
        if (check(d))
          return d;
    }
    return -1;
  }

  // Find the padded dimension in L2->L1 channel sizes/strides.
  static int64_t findPadDimInChannelSizes(OperandRange sizes,
                                          OperandRange strides, int64_t shimIdx,
                                          int64_t dimActualLast,
                                          int64_t defaultDim = 1) {
    auto checkDim = [&](int64_t d) -> bool {
      if (d < 0 || d >= (int64_t)sizes.size())
        return false;
      auto sizeOpt = getConstantIntValue(sizes[d]);
      if (!sizeOpt || *sizeOpt <= 1)
        return false;
      auto strideAtDimOpt = getConstantIntValue(strides[d]);
      auto lastSizeOpt = getConstantIntValue(sizes[sizes.size() - 1]);
      int64_t innerBlock = 1;
      if (strideAtDimOpt && lastSizeOpt && *strideAtDimOpt == *lastSizeOpt)
        innerBlock = *lastSizeOpt;
      int64_t chunkSz = *sizeOpt * innerBlock;
      int64_t shimOff = shimIdx * chunkSz;
      int64_t actual =
          std::max(int64_t(0), std::min(chunkSz, dimActualLast - shimOff));
      return actual > 0 && actual < chunkSz;
    };

    if (checkDim(defaultDim))
      return defaultDim;
    for (int64_t d = 0; d < (int64_t)sizes.size(); ++d) {
      if (d == defaultDim)
        continue;
      if (checkDim(d))
        return d;
    }
    return -1;
  }

  // Modify sizes on a channel op for a boundary partition.
  void reduceChannelSizes(air::ChannelInterface chOp, ShimPadInfo &info) {
    OpBuilder builder(chOp);
    Location loc = chOp.getLoc();
    auto actualVal =
        arith::ConstantIndexOp::create(builder, loc, info.actualForShim);

    auto sizes = chOp.getSizes();
    if (sizes.empty() || info.padDimIdx >= (int64_t)sizes.size())
      return;

    unsigned sizeBegin = sizes.getBeginOperandIndex();
    chOp->setOperand(sizeBegin + info.padDimIdx, actualVal);
  }

  // Add pad_after on an L2->L1 channel.put (segment level, memtile MM2S).
  void addMemtilePadding(air::ChannelPutOp putOp, int64_t padDimIdx,
                         int64_t padAmount) {
    auto sizes = putOp.getSizes();
    int numDims = sizes.size();
    SmallVector<int32_t> padBefore(numDims, 0);
    SmallVector<int32_t> padAfter(numDims, 0);
    padAfter[padDimIdx] = padAmount;
    putOp->setAttr("pad_before",
                   DenseI32ArrayAttr::get(putOp.getContext(), padBefore));
    putOp->setAttr("pad_after",
                   DenseI32ArrayAttr::get(putOp.getContext(), padAfter));
  }

  // Clone an air.launch and specialize it for a boundary partition.
  air::LaunchOp cloneAndSpecializeLaunch(air::LaunchOp origLaunch,
                                         OpBuilder &builder, Location loc,
                                         int64_t fixedDimM, int64_t fixedDimN,
                                         int64_t interiorM, int64_t interiorN,
                                         StringRef suffix) {
    unsigned numDims = origLaunch.getSizeOperands().size();
    SmallVector<int64_t> newSizes(numDims);
    for (unsigned i = 0; i < numDims; i++) {
      if (i == 0)
        newSizes[i] = fixedDimM >= 0 ? 1 : interiorM;
      else if (i == 1)
        newSizes[i] = fixedDimN >= 0 ? 1 : interiorN;
      else {
        auto origSzOpt = getConstantIntValue(origLaunch.getSizeOperands()[i]);
        if (!origSzOpt)
          return origLaunch;
        newSizes[i] = *origSzOpt;
      }
    }

    SmallVector<Value> newSizeVals;
    for (unsigned i = 0; i < numDims; i++)
      newSizeVals.push_back(
          arith::ConstantIndexOp::create(builder, loc, newSizes[i]));

    auto clonedOp = cast<air::LaunchOp>(builder.clone(*origLaunch));

    {
      auto sizeRange = clonedOp.getSizeOperands();
      unsigned sizeBegin = sizeRange.getBeginOperandIndex();
      for (unsigned i = 0; i < numDims; i++)
        clonedOp->setOperand(sizeBegin + i, newSizeVals[i]);
    }

    auto &bodyBlock = clonedOp.getBody().front();
    unsigned numIds = numDims;
    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&bodyBlock);
    if (fixedDimM >= 0 && numIds > 0) {
      Value mConst =
          arith::ConstantIndexOp::create(bodyBuilder, loc, fixedDimM);
      bodyBlock.getArgument(0).replaceAllUsesWith(mConst);
    }
    if (fixedDimN >= 0 && numIds > 1) {
      Value nConst =
          arith::ConstantIndexOp::create(bodyBuilder, loc, fixedDimN);
      bodyBlock.getArgument(1).replaceAllUsesWith(nConst);
    }

    for (unsigned i = 0; i < numDims; i++) {
      Value szConst =
          arith::ConstantIndexOp::create(bodyBuilder, loc, newSizes[i]);
      bodyBlock.getArgument(numIds + i).replaceAllUsesWith(szConst);
    }

    clonedOp.walk([&](air::SegmentOp segOp) {
      if (auto name = segOp.getSymName()) {
        segOp.setSymName((*name + suffix).str());
      }
    });

    // Clone channel declarations and rename references.
    auto module = clonedOp->getParentOfType<ModuleOp>();
    DenseMap<StringRef, std::string> chanRenameMap;

    clonedOp.walk([&](air::ChannelInterface chOp) {
      auto chanName = chOp.getChanName();
      if (chanRenameMap.count(chanName))
        return;
      std::string newName = (chanName + suffix).str();
      chanRenameMap[chanName] = newName;
    });

    OpBuilder moduleBuilder = OpBuilder::atBlockEnd(module.getBody());
    for (auto &[origName, newName] : chanRenameMap) {
      auto origChanOp = module.lookupSymbol<air::ChannelOp>(origName);
      if (!origChanOp)
        continue;
      if (module.lookupSymbol<air::ChannelOp>(newName))
        continue;
      moduleBuilder.setInsertionPointAfter(origChanOp);
      auto clonedChan = cast<air::ChannelOp>(moduleBuilder.clone(*origChanOp));
      clonedChan.setSymName(newName);
    }

    clonedOp.walk([&](air::ChannelInterface chOp) {
      auto it = chanRenameMap.find(chOp.getChanName());
      if (it != chanRenameMap.end()) {
        chOp->setAttr("chan_name",
                      FlatSymbolRefAttr::get(chOp->getContext(), it->second));
      }
    });

    return clonedOp;
  }

  // ===--------------------------------------------------------------------===//
  // Multi-launch mode (existing AIE-path behavior)
  // ===--------------------------------------------------------------------===//

  void runMultiLaunchMode() {
    auto module = getOperation();

    SmallVector<air::LaunchOp> launchOps;
    module.walk([&](air::LaunchOp op) { launchOps.push_back(op); });

    for (auto launchOp : launchOps) {
      auto actualSizesAttr =
          launchOp->getAttrOfType<DenseI64ArrayAttr>("air.actual_sizes");
      if (!actualSizesAttr)
        continue;
      auto actualSizes = actualSizesAttr.asArrayRef();

      auto launchIds = launchOp.getIds();
      if (launchIds.size() < 2 || actualSizes.size() < 2)
        continue;

      int64_t actualM = actualSizes[0];
      int64_t actualN = actualSizes[1];
      if (actualM == 0 && actualN == 0)
        continue;

      int64_t tileM = inferTileSize(launchOp, 0);
      int64_t tileN = inferTileSize(launchOp, 1);

      if ((actualM > 0 && tileM <= 0) || (actualN > 0 && tileN <= 0)) {
        launchOp.emitError("air-split-launch-for-padding: could not infer tile "
                           "sizes from launch body offset computations");
        return signalPassFailure();
      }

      int64_t mRem = actualM > 0 ? (actualM % tileM) : 0;
      int64_t nRem = actualN > 0 ? (actualN % tileN) : 0;
      if (!mRem && !nRem)
        continue;

      auto sizeOperands = launchOp.getSizeOperands();
      auto launchMOpt = getConstantIntValue(sizeOperands[0]);
      auto launchNOpt = getConstantIntValue(sizeOperands[1]);
      if (!launchMOpt || !launchNOpt)
        continue;
      int64_t launchM = *launchMOpt;
      int64_t launchN = *launchNOpt;

      [[maybe_unused]] int64_t mActualLast =
          mRem ? (actualM - (launchM - 1) * tileM) : tileM;
      [[maybe_unused]] int64_t nActualLast =
          nRem ? (actualN - (launchN - 1) * tileN) : tileN;

      OpBuilder builder(launchOp);
      Location loc = launchOp.getLoc();

      auto padLoc = getEffectivePadLocation();
      auto opKind = detectOpKind(launchOp);

      int64_t padDimIdxA = 0, padDimIdxB = -1;
      if (padLoc == PadLocation::Memtile) {
        if (opKind == OpKind::Channel) {
          DenseMap<StringRef, unsigned> chanToArgIdxOrig;
          launchOp.walk([&](air::ChannelPutOp putOp) {
            if (putOp->getParentOfType<air::SegmentOp>())
              return;
            auto memrefType =
                dyn_cast<BaseMemRefType>(putOp.getMemref().getType());
            if (memrefType &&
                getMemorySpace(memrefType) == air::MemorySpace::L3) {
              unsigned argIdx = traceFuncArgIdx(putOp.getMemref());
              chanToArgIdxOrig[putOp.getChanName()] = argIdx;
            }
          });
          std::tie(padDimIdxA, padDimIdxB) =
              inferL3PadDimIndices(launchOp, chanToArgIdxOrig);
        } else {
          std::tie(padDimIdxA, padDimIdxB) =
              inferL3PadDimIndicesForDma(launchOp);
        }
      }

      auto applyPadding = [&](air::LaunchOp boundary, bool padM, bool padN) {
        if (padLoc == PadLocation::Source) {
          if (opKind == OpKind::Channel)
            addSourcePaddingForChannels(boundary, padM, padN, mActualLast,
                                        nActualLast, tileM, tileN);
          else
            addSourcePaddingForDma(boundary, padM, padN, mActualLast,
                                   nActualLast, tileM, tileN);
        } else {
          if (opKind == OpKind::Channel)
            addMemtilePaddingForChannels(boundary, padM, padN, mActualLast,
                                         nActualLast, padDimIdxA, padDimIdxB);
          else
            addMemtilePaddingForDma(boundary, padM, padN, mActualLast,
                                    nActualLast, padDimIdxA, padDimIdxB);
        }
      };

      int64_t interiorM = mRem ? (launchM - 1) : launchM;
      int64_t interiorN = nRem ? (launchN - 1) : launchN;

      if (interiorM > 0 && interiorN > 0) {
        auto interior = cloneAndSpecializeLaunch(
            launchOp, builder, loc, /*fixedDimM=*/-1, /*fixedDimN=*/-1,
            interiorM, interiorN, "_interior");
        (void)interior;
      }
      if (mRem && interiorN > 0) {
        auto mBound = cloneAndSpecializeLaunch(
            launchOp, builder, loc, /*fixedDimM=*/launchM - 1,
            /*fixedDimN=*/-1, 1, interiorN, "_m_boundary");
        applyPadding(mBound, /*padM=*/true, /*padN=*/false);
      }
      if (nRem && interiorM > 0) {
        auto nBound = cloneAndSpecializeLaunch(
            launchOp, builder, loc, /*fixedDimM=*/-1,
            /*fixedDimN=*/launchN - 1, interiorM, 1, "_n_boundary");
        applyPadding(nBound, /*padM=*/false, /*padN=*/true);
      }
      if (mRem && nRem) {
        auto corner = cloneAndSpecializeLaunch(
            launchOp, builder, loc, /*fixedDimM=*/launchM - 1,
            /*fixedDimN=*/launchN - 1, 1, 1, "_corner");
        applyPadding(corner, /*padM=*/true, /*padN=*/true);
      }

      launchOp.erase();
    }
  }

  // ===--------------------------------------------------------------------===//
  // Single-launch mode (GPU-path: scf.if on block indices)
  // ===--------------------------------------------------------------------===//

  void runSingleLaunchMode() {
    auto module = getOperation();

    SmallVector<LaunchOp> launchOps;
    module.walk([&](LaunchOp op) { launchOps.push_back(op); });

    for (auto launchOp : launchOps) {
      auto actualSizesAttr =
          launchOp->getAttrOfType<DenseI64ArrayAttr>("air.actual_sizes");
      if (!actualSizesAttr)
        continue;
      auto actualSizes = SmallVector<int64_t>(actualSizesAttr.asArrayRef());

      // Single-launch mode only supports dma_memcpy_nd ops.
      if (detectOpKind(launchOp) == OpKind::Channel) {
        launchOp.emitError(
            "air-split-launch-for-padding: single-launch mode does not "
            "support air.channel.put/get ops; use split-mode=multi-launch");
        return signalPassFailure();
      }

      auto launchIds = launchOp.getIds();
      if (launchIds.empty() || actualSizes.empty())
        continue;

      int64_t actualM = actualSizes[0];
      int64_t actualN = (actualSizes.size() >= 2) ? actualSizes[1] : 0;
      if (actualM == 0 && actualN == 0)
        continue;

      bool has2D = (launchIds.size() >= 2 && actualSizes.size() >= 2);

      int64_t tileM = inferTileSize(launchOp, 0);
      int64_t tileN = has2D ? inferTileSize(launchOp, 1) : 1;

      if (actualM > 0 && tileM <= 0) {
        launchOp.emitError("air-split-launch-for-padding: could not infer M "
                           "tile size from launch body");
        return signalPassFailure();
      }
      if (actualN > 0 && tileN <= 0) {
        launchOp.emitError("air-split-launch-for-padding: could not infer N "
                           "tile size from launch body");
        return signalPassFailure();
      }

      int64_t mRem = actualM > 0 ? (actualM % tileM) : 0;
      int64_t nRem = (has2D && actualN > 0) ? (actualN % tileN) : 0;
      if (!mRem && !nRem)
        continue;

      auto sizeOperands = launchOp.getSizeOperands();
      auto launchMOpt = getConstantIntValue(sizeOperands[0]);
      if (!launchMOpt)
        continue;
      int64_t launchM = *launchMOpt;
      int64_t launchN = 1;
      if (has2D) {
        auto launchNOpt = getConstantIntValue(sizeOperands[1]);
        if (!launchNOpt)
          continue;
        launchN = *launchNOpt;
      }

      int64_t mActualLast = mRem ? (actualM - (launchM - 1) * tileM) : tileM;
      int64_t nActualLast = nRem ? (actualN - (launchN - 1) * tileN) : tileN;

      int64_t interiorM = mRem ? (launchM - 1) : launchM;
      int64_t interiorN = nRem ? (launchN - 1) : launchN;

      OpBuilder builder(launchOp);
      Location loc = launchOp.getLoc();

      Block &origBody = launchOp.getBody().front();
      auto ids = launchOp.getIds();

      auto *terminator = origBody.getTerminator();

      // Pre-compute pad dimension indices from the original launch using
      // offset-based tracing. This must be done before cloning replaces
      // block indices with constants.
      auto padDims = inferL3PadDimIndicesForDma(launchOp);
      int64_t aPadDim = padDims.first;
      int64_t bPadDim = padDims.second;

      // Partition body ops into branch-invariant (hoistable) and
      // branch-varying (cloneable). An op is branch-varying if any of its
      // operands transitively depend on a block index that differs across
      // partitions. Hoistable ops are emitted once before the scf.if tree;
      // only cloneable ops are duplicated into each branch.
      DenseSet<Value> branchVarying;
      branchVarying.insert(ids[0]); // blockM
      if (has2D)
        branchVarying.insert(ids[1]); // blockN

      SmallVector<Operation *> hoistableOps, cloneableOps;
      for (auto &op : origBody) {
        if (&op == terminator)
          continue;
        bool dependsOnBranch = llvm::any_of(op.getOperands(), [&](Value v) {
          return branchVarying.contains(v);
        });
        if (dependsOnBranch) {
          cloneableOps.push_back(&op);
          for (Value r : op.getResults())
            branchVarying.insert(r);
        } else {
          hoistableOps.push_back(&op);
        }
      }

      builder.setInsertionPoint(terminator);

      Value blockM = ids[0];
      Value blockN = has2D ? Value(ids[1]) : Value();

      Value interiorMConst =
          arith::ConstantIndexOp::create(builder, loc, interiorM);
      Value isMInterior = arith::CmpIOp::create(
          builder, loc, arith::CmpIPredicate::ult, blockM, interiorMConst);

      auto createNInteriorCond = [&](OpBuilder &b) -> Value {
        Value nc = arith::ConstantIndexOp::create(b, loc, interiorN);
        return arith::CmpIOp::create(b, loc, arith::CmpIPredicate::ult, blockN,
                                     nc);
      };

      // Clone only branch-varying ops into a partition, optionally replacing
      // block IDs with constants and adding pad_after to boundary DMAs.
      // Hoistable ops remain before the scf.if tree and are visible to all
      // branches without cloning.
      auto cloneAndPad = [&](OpBuilder &ifBuilder, int64_t fixedM,
                             int64_t fixedN, bool padM, bool padN) {
        IRMapping mapping;
        if (fixedM >= 0) {
          Value cst = arith::ConstantIndexOp::create(ifBuilder, loc, fixedM);
          mapping.map(ids[0], cst);
        }
        if (has2D && fixedN >= 0) {
          Value cst = arith::ConstantIndexOp::create(ifBuilder, loc, fixedN);
          mapping.map(ids[1], cst);
        }

        for (auto *op : cloneableOps) {
          if (auto *blockTerm = ifBuilder.getInsertionBlock()->getTerminator())
            ifBuilder.setInsertionPoint(blockTerm);
          ifBuilder.clone(*op, mapping);
        }

        if (padM || padN) {
          Block *currentBlock = ifBuilder.getInsertionBlock();
          SmallVector<DmaMemcpyNdOp> dmas;
          currentBlock->walk([&](DmaMemcpyNdOp dma) { dmas.push_back(dma); });
          for (auto dmaOp : dmas) {
            unsigned argIdx = traceFuncArgIdx(dmaOp.getSrcMemref());
            if (argIdx > 1)
              continue;
            if (skipInput(argIdx, padM, padN))
              continue;

            bool isA = (argIdx == 0);
            int64_t tileSize = isA ? tileM : tileN;
            int64_t actualLast = isA ? mActualLast : nActualLast;

            auto srcSizes = dmaOp.getSrcSizes();
            if (srcSizes.empty())
              continue;

            // Use pre-computed pad dimension from offset-based tracing.
            int64_t padDimIdx = isA ? aPadDim : bPadDim;
            // Fallback: if offset tracing didn't find a match, use dim 0.
            if (padDimIdx < 0)
              padDimIdx = 0;
            if (padDimIdx >= (int64_t)srcSizes.size())
              continue;

            int64_t padAmount = tileSize - actualLast;
            if (padAmount <= 0)
              continue;

            OpBuilder dmaBuilder(dmaOp);
            Value actualVal =
                arith::ConstantIndexOp::create(dmaBuilder, loc, actualLast);
            dmaOp.getSrcSizesMutable().slice(padDimIdx, 1).assign(actualVal);
            auto dstSizes = dmaOp.getDstSizes();
            if (padDimIdx < (int64_t)dstSizes.size())
              dmaOp.getDstSizesMutable().slice(padDimIdx, 1).assign(actualVal);

            int64_t numDims = srcSizes.size();
            SmallVector<int32_t> padBefore(numDims, 0);
            SmallVector<int32_t> padAfter(numDims, 0);
            padAfter[padDimIdx] = padAmount;
            dmaOp->setAttr("pad_before", DenseI32ArrayAttr::get(
                                             dmaOp.getContext(), padBefore));
            dmaOp->setAttr("pad_after", DenseI32ArrayAttr::get(
                                            dmaOp.getContext(), padAfter));
          }
        }
      };

      // Build the scf.if tree.
      if (mRem && nRem) {
        auto outerIf = scf::IfOp::create(builder, loc, /*types=*/TypeRange{},
                                         isMInterior, /*withElse=*/true);

        {
          OpBuilder thenBuilder =
              OpBuilder::atBlockTerminator(&outerIf.getThenRegion().front());
          Value isNInt1 = createNInteriorCond(thenBuilder);
          auto innerIf = scf::IfOp::create(thenBuilder, loc, TypeRange{},
                                           isNInt1, /*withElse=*/true);
          {
            OpBuilder ib =
                OpBuilder::atBlockTerminator(&innerIf.getThenRegion().front());
            cloneAndPad(ib, -1, -1, false, false);
          }
          {
            OpBuilder ib =
                OpBuilder::atBlockTerminator(&innerIf.getElseRegion().front());
            cloneAndPad(ib, -1, launchN - 1, false, true);
          }
        }

        {
          OpBuilder elseBuilder =
              OpBuilder::atBlockTerminator(&outerIf.getElseRegion().front());
          Value isNInt2 = createNInteriorCond(elseBuilder);
          auto innerIf = scf::IfOp::create(elseBuilder, loc, TypeRange{},
                                           isNInt2, /*withElse=*/true);
          {
            OpBuilder ib =
                OpBuilder::atBlockTerminator(&innerIf.getThenRegion().front());
            cloneAndPad(ib, launchM - 1, -1, true, false);
          }
          {
            OpBuilder ib =
                OpBuilder::atBlockTerminator(&innerIf.getElseRegion().front());
            cloneAndPad(ib, launchM - 1, launchN - 1, true, true);
          }
        }
      } else if (mRem) {
        auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, isMInterior,
                                      /*withElse=*/true);
        {
          OpBuilder ib =
              OpBuilder::atBlockTerminator(&ifOp.getThenRegion().front());
          cloneAndPad(ib, -1, -1, false, false);
        }
        {
          OpBuilder ib =
              OpBuilder::atBlockTerminator(&ifOp.getElseRegion().front());
          cloneAndPad(ib, launchM - 1, -1, true, false);
        }
      } else if (nRem) {
        Value isNInt = createNInteriorCond(builder);
        auto ifOp = scf::IfOp::create(builder, loc, TypeRange{}, isNInt,
                                      /*withElse=*/true);
        {
          OpBuilder ib =
              OpBuilder::atBlockTerminator(&ifOp.getThenRegion().front());
          cloneAndPad(ib, -1, -1, false, false);
        }
        {
          OpBuilder ib =
              OpBuilder::atBlockTerminator(&ifOp.getElseRegion().front());
          cloneAndPad(ib, -1, launchN - 1, false, true);
        }
      }

      // Erase the original branch-varying ops (now duplicated inside scf.if).
      // Hoistable ops remain in the launch body, before the scf.if tree.
      for (auto *op : llvm::reverse(cloneableOps))
        op->erase();
    }
  }

  // ===--------------------------------------------------------------------===//
  // Source-level padding helpers
  // ===--------------------------------------------------------------------===//

  void addSourcePaddingForDma(air::LaunchOp launch, bool padM, bool padN,
                              int64_t mActualLast, int64_t nActualLast,
                              int64_t tileM, int64_t tileN) {
    launch.walk([&](air::DmaMemcpyNdOp dmaOp) {
      unsigned argIdx = traceFuncArgIdx(dmaOp.getSrcMemref());
      if (argIdx > 1)
        return;
      if (skipInput(argIdx, padM, padN))
        return;

      bool isA = (argIdx == 0);
      int64_t tileSize = isA ? tileM : tileN;
      int64_t actualLast = isA ? mActualLast : nActualLast;

      auto srcSizes = dmaOp.getSrcSizes();
      if (srcSizes.empty())
        return;

      int64_t padDimIdx = -1;
      for (unsigned i = 0; i < srcSizes.size(); ++i) {
        auto sizeOpt = getConstantIntValue(srcSizes[i]);
        if (sizeOpt && *sizeOpt == tileSize) {
          padDimIdx = i;
          break;
        }
      }
      if (padDimIdx < 0)
        return;

      int64_t padAmount = tileSize - actualLast;
      if (padAmount <= 0)
        return;

      OpBuilder builder(dmaOp);
      Location loc = dmaOp.getLoc();
      Value actualVal =
          arith::ConstantIndexOp::create(builder, loc, actualLast);
      auto srcSizeMutable = dmaOp.getSrcSizesMutable();
      srcSizeMutable.slice(padDimIdx, 1).assign(actualVal);

      int64_t numDims = srcSizes.size();
      SmallVector<int32_t> padBefore(numDims, 0);
      SmallVector<int32_t> padAfter(numDims, 0);
      padAfter[padDimIdx] = padAmount;
      dmaOp->setAttr("pad_before",
                     DenseI32ArrayAttr::get(dmaOp.getContext(), padBefore));
      dmaOp->setAttr("pad_after",
                     DenseI32ArrayAttr::get(dmaOp.getContext(), padAfter));
    });
  }

  void addSourcePaddingForChannels(air::LaunchOp launch, bool padM, bool padN,
                                   int64_t mActualLast, int64_t nActualLast,
                                   int64_t tileM, int64_t tileN) {
    launch.walk([&](air::ChannelPutOp putOp) {
      unsigned argIdx = traceFuncArgIdx(putOp.getMemref());
      if (argIdx > 1)
        return;
      if (skipInput(argIdx, padM, padN))
        return;

      bool isA = (argIdx == 0);
      int64_t tileSize = isA ? tileM : tileN;
      int64_t actualLast = isA ? mActualLast : nActualLast;

      auto sizes = putOp.getSizes();
      if (sizes.empty())
        return;

      int64_t padDimIdx = -1;
      for (unsigned i = 0; i < sizes.size(); ++i) {
        auto sizeOpt = getConstantIntValue(sizes[i]);
        if (sizeOpt && *sizeOpt == tileSize) {
          padDimIdx = i;
          break;
        }
      }
      if (padDimIdx < 0)
        return;

      int64_t padAmount = tileSize - actualLast;
      if (padAmount <= 0)
        return;

      OpBuilder builder(putOp);
      Location loc = putOp.getLoc();
      Value actualVal =
          arith::ConstantIndexOp::create(builder, loc, actualLast);
      unsigned sizeBegin = sizes.getBeginOperandIndex();
      putOp->setOperand(sizeBegin + padDimIdx, actualVal);

      int64_t numDims = sizes.size();
      SmallVector<int32_t> padBefore(numDims, 0);
      SmallVector<int32_t> padAfter(numDims, 0);
      padAfter[padDimIdx] = padAmount;
      putOp->setAttr("pad_before",
                     DenseI32ArrayAttr::get(putOp.getContext(), padBefore));
      putOp->setAttr("pad_after",
                     DenseI32ArrayAttr::get(putOp.getContext(), padAfter));
    });
  }

  // ===--------------------------------------------------------------------===//
  // Memtile padding helpers
  // ===--------------------------------------------------------------------===//

  void addMemtilePaddingForChannels(air::LaunchOp launch, bool padM, bool padN,
                                    int64_t mActualLast, int64_t nActualLast,
                                    int64_t padDimIdxA, int64_t padDimIdxB) {
    // Step 1: Build mapping from L3 channel names to input type (A=0, B=1).
    DenseMap<StringRef, unsigned> chanToArgIdx;
    launch.walk([&](air::ChannelPutOp putOp) {
      if (putOp->getParentOfType<air::SegmentOp>())
        return;
      auto memrefType = dyn_cast<BaseMemRefType>(putOp.getMemref().getType());
      if (memrefType && getMemorySpace(memrefType) == air::MemorySpace::L3) {
        unsigned argIdx = traceFuncArgIdx(putOp.getMemref());
        chanToArgIdx[putOp.getChanName()] = argIdx;
      }
    });

    // Step 2: Map L2 buffers to their source input via segment-level gets.
    DenseMap<Value, unsigned> l2BufToArgIdx;
    DenseMap<Value, int64_t> l2BufToShimIdx;
    launch.walk([&](air::ChannelGetOp getOp) {
      if (!getOp->getParentOfType<air::SegmentOp>())
        return;
      if (!chanToArgIdx.count(getOp.getChanName()))
        return;
      unsigned argIdx = chanToArgIdx.lookup(getOp.getChanName());
      l2BufToArgIdx[getOp.getMemref()] = argIdx;
      auto indices = getOp.getIndices();
      if (!indices.empty()) {
        auto shimIdxOpt = getConstantIntValue(indices[0]);
        if (shimIdxOpt)
          l2BufToShimIdx[getOp.getMemref()] = *shimIdxOpt;
      }
    });

    // Step 3: Add padding on L2->L1 channel.put ops.
    launch.walk([&](air::ChannelPutOp putOp) {
      if (!putOp->getParentOfType<air::SegmentOp>())
        return;
      Value srcBuf = putOp.getMemref();
      auto it = l2BufToArgIdx.find(srcBuf);
      if (it == l2BufToArgIdx.end())
        return;

      unsigned argIdx = it->second;
      if (skipInput(argIdx, padM, padN))
        return;
      bool isA = (argIdx == 0);

      auto shimIt = l2BufToShimIdx.find(srcBuf);
      if (shimIt == l2BufToShimIdx.end())
        return;
      int64_t shimIdx = shimIt->second;

      auto sizes = putOp.getSizes();
      auto strides = putOp.getStrides();
      if (sizes.empty() || strides.empty())
        return;

      int64_t dimActualLast = isA ? mActualLast : nActualLast;
      int64_t l2l1PadDimIdx = findPadDimInChannelSizes(
          sizes, strides, shimIdx, dimActualLast, /*defaultDim=*/1);
      if (l2l1PadDimIdx < 0 || l2l1PadDimIdx >= (int64_t)sizes.size())
        return;

      auto fullBlocksOpt = getConstantIntValue(sizes[l2l1PadDimIdx]);
      if (!fullBlocksOpt)
        return;
      int64_t fullBlocks = *fullBlocksOpt;

      auto strideAtPadDimOpt = getConstantIntValue(strides[l2l1PadDimIdx]);
      auto lastSizeOpt = getConstantIntValue(sizes[sizes.size() - 1]);
      int64_t innerBlockSize = 1;
      if (strideAtPadDimOpt && lastSizeOpt &&
          *strideAtPadDimOpt == *lastSizeOpt)
        innerBlockSize = *lastSizeOpt;

      int64_t l3ChunkSize = fullBlocks * innerBlockSize;
      int64_t shimOffset = shimIdx * l3ChunkSize;
      int64_t actualForShim = std::max(
          int64_t(0), std::min(l3ChunkSize, dimActualLast - shimOffset));

      if (actualForShim >= l3ChunkSize || actualForShim <= 0)
        return;

      int64_t actualBlocks =
          (actualForShim + innerBlockSize - 1) / innerBlockSize;
      int64_t padBlocks = fullBlocks - actualBlocks;
      if (padBlocks <= 0)
        return;
      if (padBlocks > 31) {
        putOp.emitRemark("memtile padding of ")
            << padBlocks << " blocks exceeds hardware limit of 31; "
            << "host-side zero-padded buffers required for this shim";
        return;
      }

      ShimPadInfo info{shimIdx, fullBlocks, actualBlocks, padBlocks,
                       l2l1PadDimIdx};
      reduceChannelSizes(cast<air::ChannelInterface>(putOp.getOperation()),
                         info);
      addMemtilePadding(putOp, l2l1PadDimIdx, padBlocks);
    });

    // Step 4: Reduce L3->L2 channel.put sizes.
    launch.walk([&](air::ChannelPutOp putOp) {
      if (putOp->getParentOfType<air::SegmentOp>())
        return;
      if (!chanToArgIdx.count(putOp.getChanName()))
        return;
      unsigned argIdx = chanToArgIdx.lookup(putOp.getChanName());
      if (skipInput(argIdx, padM, padN))
        return;
      bool isA = (argIdx == 0);

      auto info = computeShimPadInfo(
          cast<air::ChannelInterface>(putOp.getOperation()), isA, mActualLast,
          nActualLast, padDimIdxA, padDimIdxB);
      if (!info || info->actualForShim >= info->chunkSize ||
          info->actualForShim <= 0)
        return;

      reduceChannelSizes(cast<air::ChannelInterface>(putOp.getOperation()),
                         *info);
    });

    // Step 5: Add explicit sizes/strides to L3->L2 channel.get.
    SmallVector<S2MMReplaceInfo> getsToReplace;
    launch.walk([&](air::ChannelGetOp getOp) {
      if (!getOp->getParentOfType<air::SegmentOp>())
        return;
      if (!chanToArgIdx.count(getOp.getChanName()))
        return;
      if (!getOp.getSizes().empty())
        return;
      unsigned argIdx = chanToArgIdx.lookup(getOp.getChanName());
      if (skipInput(argIdx, padM, padN))
        return;
      bool isA = (argIdx == 0);
      auto memrefType = dyn_cast<MemRefType>(getOp.getMemref().getType());
      if (!memrefType || memrefType.getRank() < 2)
        return;

      auto shape = memrefType.getShape();
      int64_t dimActualLast = isA ? mActualLast : nActualLast;

      auto shimIt = l2BufToShimIdx.find(getOp.getMemref());
      int64_t shimIdx = 0;
      if (shimIt != l2BufToShimIdx.end())
        shimIdx = shimIt->second;

      int64_t padDimInMemref = findPadDimInMemrefShape(
          shape, shimIdx, dimActualLast, /*searchForward=*/isA);
      if (padDimInMemref < 0)
        return;

      int64_t chunkSize = shape[padDimInMemref];
      int64_t shimOffset = shimIdx * chunkSize;
      int64_t actualForShim =
          std::max(int64_t(0), std::min(chunkSize, dimActualLast - shimOffset));
      if (actualForShim >= chunkSize || actualForShim <= 0)
        return;

      getsToReplace.push_back({getOp, padDimInMemref, shimIdx, actualForShim});
    });
    for (auto &replaceInfo : getsToReplace) {
      auto getOp = replaceInfo.getOp;
      auto memrefType = cast<MemRefType>(getOp.getMemref().getType());
      auto shape = memrefType.getShape();
      int64_t rank = memrefType.getRank();

      OpBuilder builder(getOp);
      Location loc = getOp.getLoc();

      SmallVector<Value> offsets, sizes, strides;
      for (int64_t d = 0; d < rank; ++d) {
        offsets.push_back(arith::ConstantIndexOp::create(builder, loc, 0));
        int64_t dimSize = (d == replaceInfo.padDimInMemref)
                              ? replaceInfo.actualForShim
                              : shape[d];
        sizes.push_back(arith::ConstantIndexOp::create(builder, loc, dimSize));
        int64_t stride = 1;
        for (int64_t j = d + 1; j < rank; ++j)
          stride *= shape[j];
        strides.push_back(arith::ConstantIndexOp::create(builder, loc, stride));
      }

      auto newGet = air::ChannelGetOp::create(
          builder, loc,
          getOp.getAsyncToken() ? getOp.getAsyncToken().getType() : Type(),
          getOp.getAsyncDependencies(), getOp.getChanName(), getOp.getIndices(),
          getOp.getMemref(), offsets, sizes, strides,
          /*pad_before=*/DenseI32ArrayAttr(),
          /*pad_after=*/DenseI32ArrayAttr());
      for (auto attr : getOp->getAttrs()) {
        if (attr.getName() != "operandSegmentSizes")
          newGet->setAttr(attr.getName(), attr.getValue());
      }
      if (getOp.getAsyncToken())
        getOp.getAsyncToken().replaceAllUsesWith(newGet.getAsyncToken());
      getOp.erase();
    }
  }

  void addMemtilePaddingForDma(air::LaunchOp launch, bool padM, bool padN,
                               int64_t mActualLast, int64_t nActualLast,
                               int64_t padDimIdxA, int64_t padDimIdxB) {
    // Step 1: Map L3->L2 DMAs to arg indices.
    DenseMap<Value, unsigned> l2BufToArgIdx;
    launch.walk([&](air::DmaMemcpyNdOp dmaOp) {
      if (dmaOp->getParentOfType<air::SegmentOp>())
        return;
      auto srcType = dyn_cast<BaseMemRefType>(dmaOp.getSrcMemref().getType());
      if (!srcType || getMemorySpace(srcType) != air::MemorySpace::L3)
        return;
      unsigned argIdx = traceFuncArgIdx(dmaOp.getSrcMemref());
      l2BufToArgIdx[dmaOp.getDstMemref()] = argIdx;
    });

    // Step 2: Trace L2 buffers through segment hierarchy.
    DenseMap<Value, Value> segArgToLaunchL2;
    launch.walk([&](air::SegmentOp segOp) {
      auto kernelOperands = segOp.getKernelOperands();
      auto kernelBodyArgs = segOp.getKernelArguments();
      for (unsigned i = 0; i < kernelOperands.size(); ++i) {
        if (l2BufToArgIdx.count(kernelOperands[i]))
          segArgToLaunchL2[kernelBodyArgs[i]] = kernelOperands[i];
      }
    });

    // Step 3: Add padding on L2->L1 DMA ops.
    launch.walk([&](air::DmaMemcpyNdOp dmaOp) {
      if (!dmaOp->getParentOfType<air::SegmentOp>())
        return;
      auto srcType = dyn_cast<BaseMemRefType>(dmaOp.getSrcMemref().getType());
      auto dstType = dyn_cast<BaseMemRefType>(dmaOp.getDstMemref().getType());
      if (!srcType || !dstType)
        return;
      if (getMemorySpace(srcType) != air::MemorySpace::L2 ||
          getMemorySpace(dstType) != air::MemorySpace::L1)
        return;

      Value srcBuf = dmaOp.getSrcMemref();
      auto segIt = segArgToLaunchL2.find(srcBuf);
      if (segIt != segArgToLaunchL2.end())
        srcBuf = segIt->second;
      auto argIt = l2BufToArgIdx.find(srcBuf);
      if (argIt == l2BufToArgIdx.end())
        return;

      unsigned argIdx = argIt->second;
      if (skipInput(argIdx, padM, padN))
        return;
      bool isA = (argIdx == 0);

      int64_t dimActualLast = isA ? mActualLast : nActualLast;

      auto sizes = dmaOp.getSrcSizes();
      auto strides = dmaOp.getSrcStrides();
      if (sizes.empty() || strides.empty())
        return;

      int64_t l2l1PadDimIdx =
          isA ? padDimIdxA
              : (padDimIdxB >= 0 ? padDimIdxB : ((int64_t)sizes.size() - 1));
      if (l2l1PadDimIdx < 0 || l2l1PadDimIdx >= (int64_t)sizes.size())
        return;

      auto fullBlocksOpt = getConstantIntValue(sizes[l2l1PadDimIdx]);
      if (!fullBlocksOpt)
        return;
      int64_t fullBlocks = *fullBlocksOpt;

      int64_t actualBlocks = dimActualLast;
      if (actualBlocks >= fullBlocks || actualBlocks <= 0)
        return;

      int64_t padBlocks = fullBlocks - actualBlocks;
      if (padBlocks <= 0)
        return;
      if (padBlocks > 31) {
        dmaOp.emitRemark("memtile padding of ")
            << padBlocks << " blocks exceeds hardware limit of 31; "
            << "host-side zero-padded buffers required for this shim";
        return;
      }

      OpBuilder builder(dmaOp);
      Location loc = dmaOp.getLoc();
      Value actualVal =
          arith::ConstantIndexOp::create(builder, loc, actualBlocks);
      dmaOp.getSrcSizesMutable().slice(l2l1PadDimIdx, 1).assign(actualVal);

      int64_t numDims = sizes.size();
      SmallVector<int32_t> padBefore(numDims, 0);
      SmallVector<int32_t> padAfter(numDims, 0);
      padAfter[l2l1PadDimIdx] = padBlocks;
      dmaOp->setAttr("pad_before",
                     DenseI32ArrayAttr::get(dmaOp.getContext(), padBefore));
      dmaOp->setAttr("pad_after",
                     DenseI32ArrayAttr::get(dmaOp.getContext(), padAfter));
    });

    // Step 4: Reduce L3->L2 DMA src_sizes.
    launch.walk([&](air::DmaMemcpyNdOp dmaOp) {
      if (dmaOp->getParentOfType<air::SegmentOp>())
        return;
      auto srcType = dyn_cast<BaseMemRefType>(dmaOp.getSrcMemref().getType());
      if (!srcType || getMemorySpace(srcType) != air::MemorySpace::L3)
        return;
      unsigned argIdx = traceFuncArgIdx(dmaOp.getSrcMemref());
      if (skipInput(argIdx, padM, padN))
        return;
      bool isA = (argIdx == 0);

      auto info = computeShimPadInfoForDma(dmaOp, isA, mActualLast, nActualLast,
                                           padDimIdxA, padDimIdxB);
      if (!info || info->actualForShim >= info->chunkSize ||
          info->actualForShim <= 0)
        return;

      OpBuilder builder(dmaOp);
      Location loc = dmaOp.getLoc();
      Value actualVal =
          arith::ConstantIndexOp::create(builder, loc, info->actualForShim);
      dmaOp.getSrcSizesMutable().slice(info->padDimIdx, 1).assign(actualVal);
    });

    // Step 5: Add explicit dst sizes/strides to segment-level L3->L2 DMAs.
    struct DmaS2MMInfo {
      air::DmaMemcpyNdOp dmaOp;
      int64_t padDimInMemref;
      int64_t actualForShim;
    };
    SmallVector<DmaS2MMInfo> dmasToFixup;

    launch.walk([&](air::DmaMemcpyNdOp dmaOp) {
      if (!dmaOp->getParentOfType<air::SegmentOp>())
        return;
      auto dstType = dyn_cast<BaseMemRefType>(dmaOp.getDstMemref().getType());
      auto srcType = dyn_cast<BaseMemRefType>(dmaOp.getSrcMemref().getType());
      if (!dstType || !srcType)
        return;
      if (getMemorySpace(dstType) != air::MemorySpace::L2 ||
          getMemorySpace(srcType) != air::MemorySpace::L3)
        return;
      if (!dmaOp.getDstSizes().empty())
        return;

      Value dstBuf = dmaOp.getDstMemref();
      auto segIt = segArgToLaunchL2.find(dstBuf);
      Value launchBuf =
          (segIt != segArgToLaunchL2.end()) ? segIt->second : dstBuf;
      auto argIt = l2BufToArgIdx.find(launchBuf);
      if (argIt == l2BufToArgIdx.end())
        return;
      unsigned argIdx = argIt->second;
      if (skipInput(argIdx, padM, padN))
        return;
      bool isA = (argIdx == 0);

      auto memrefType = dyn_cast<MemRefType>(dmaOp.getDstMemref().getType());
      if (!memrefType || memrefType.getRank() < 2)
        return;
      auto shape = memrefType.getShape();
      int64_t dimActualLast = isA ? mActualLast : nActualLast;
      int64_t shimIdx = 0;

      int64_t padDimInMemref = findPadDimInMemrefShape(
          shape, shimIdx, dimActualLast, /*searchForward=*/isA);
      if (padDimInMemref < 0)
        return;

      int64_t chunkSize = shape[padDimInMemref];
      int64_t shimOffset = shimIdx * chunkSize;
      int64_t actualForShim =
          std::max(int64_t(0), std::min(chunkSize, dimActualLast - shimOffset));
      if (actualForShim >= chunkSize || actualForShim <= 0)
        return;

      dmasToFixup.push_back({dmaOp, padDimInMemref, actualForShim});
    });

    for (auto &info : dmasToFixup) {
      auto dmaOp = info.dmaOp;
      auto memrefType = cast<MemRefType>(dmaOp.getDstMemref().getType());
      auto shape = memrefType.getShape();
      int64_t rank = memrefType.getRank();

      OpBuilder builder(dmaOp);
      Location loc = dmaOp.getLoc();

      SmallVector<Value> offsets, sizes, strides;
      for (int64_t d = 0; d < rank; ++d) {
        offsets.push_back(arith::ConstantIndexOp::create(builder, loc, 0));
        int64_t dimSize =
            (d == info.padDimInMemref) ? info.actualForShim : shape[d];
        sizes.push_back(arith::ConstantIndexOp::create(builder, loc, dimSize));
        int64_t stride = 1;
        for (int64_t j = d + 1; j < rank; ++j)
          stride *= shape[j];
        strides.push_back(arith::ConstantIndexOp::create(builder, loc, stride));
      }

      auto newDma = air::DmaMemcpyNdOp::create(
          builder, loc,
          dmaOp.getAsyncToken() ? dmaOp.getAsyncToken().getType() : Type(),
          dmaOp.getAsyncDependencies(), dmaOp.getDstMemref(), offsets, sizes,
          strides, dmaOp.getSrcMemref(), dmaOp.getSrcOffsets(),
          dmaOp.getSrcSizes(), dmaOp.getSrcStrides(),
          /*pad_before=*/DenseI32ArrayAttr(),
          /*pad_after=*/DenseI32ArrayAttr());
      for (auto attr : dmaOp->getAttrs()) {
        if (attr.getName() != "operandSegmentSizes")
          newDma->setAttr(attr.getName(), attr.getValue());
      }
      if (dmaOp.getAsyncToken())
        dmaOp.getAsyncToken().replaceAllUsesWith(newDma.getAsyncToken());
      dmaOp.erase();
    }
  }

  // ===--------------------------------------------------------------------===//
  // Entry point
  // ===--------------------------------------------------------------------===//

  void runOnOperation() override {
    if (clSplitMode == "multi-launch") {
      runMultiLaunchMode();
    } else if (clSplitMode == "single-launch") {
      if (clPadLocation == "memtile" && !clUseDmaMemcpy) {
        getOperation().emitError(
            "air-split-launch-for-padding: split-mode=single-launch does not "
            "support pad-location=memtile; use pad-location=source");
        return signalPassFailure();
      }
      runSingleLaunchMode();
    } else {
      getOperation().emitError(
          "air-split-launch-for-padding: invalid split-mode '")
          << clSplitMode << "'; expected 'multi-launch' or 'single-launch'";
      return signalPassFailure();
    }
  }
};

std::unique_ptr<mlir::Pass> createAIRSplitLaunchForPadding() {
  return std::make_unique<AIRSplitLaunchForPadding>();
}
std::unique_ptr<OperationPass<ModuleOp>>
createAIRSplitLaunchForPadding(const AIRSplitLaunchForPaddingOptions &options) {
  return std::make_unique<AIRSplitLaunchForPadding>(options);
}

} // namespace air
} // namespace xilinx
