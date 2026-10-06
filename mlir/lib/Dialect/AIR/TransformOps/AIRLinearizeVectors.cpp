//===- AIRLinearizeVectors.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// transform.air.linearize_vectors: rank-1 vector code for the AIE core
// lowerings.
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/AIR/AIRTransformOps.h"

#if AIR_ENABLE_AIE
#include "aie/Dialect/AIEVec/IR/AIEVecOps.h"
#endif

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <numeric>

using namespace mlir;

//===----------------------------------------------------------------------===//
// LinearizeVectorsOp
//===----------------------------------------------------------------------===//

namespace {

// A transfer_read whose permutation map is the identity except for broadcast
// (constant 0) results reads one element along each broadcast dim: memref dim
// i is then not indexed by any vector dim, so the read takes the element at
// indices[i]. Read that with an identity map (extent 1 at the broadcast dims)
// and stretch it with vector.broadcast, which linearization turns into a
// shuffle.
struct UnbroadcastTransferRead
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    VectorType vecType = read.getVectorType();
    AffineMap map = read.getPermutationMap();
    if (read.getMask() || vecType.isScalable() ||
        map.getNumResults() != map.getNumDims() ||
        !isa<MemRefType>(read.getBase().getType()))
      return failure();

    SmallVector<int64_t> readShape(vecType.getShape());
    SmallVector<bool> inBounds = read.getInBoundsValues();
    bool broadcasts = false;
    for (auto [i, expr] : llvm::enumerate(map.getResults())) {
      if (auto dim = dyn_cast<AffineDimExpr>(expr)) {
        if (dim.getPosition() != i)
          return failure();
        continue;
      }
      auto cst = dyn_cast<AffineConstantExpr>(expr);
      if (!cst || cst.getValue() != 0)
        return failure();
      // The original read already took this element, so it is in bounds.
      readShape[i] = 1;
      inBounds[i] = true;
      broadcasts = true;
    }
    if (!broadcasts)
      return failure();

    auto newRead = vector::TransferReadOp::create(
        rewriter, read.getLoc(),
        VectorType::get(readShape, vecType.getElementType()), read.getBase(),
        read.getIndices(), read.getPadding(),
        rewriter.getMultiDimIdentityMap(map.getNumDims()),
        ArrayRef<bool>(inBounds));
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(read, vecType, newRead);
    return success();
  }
};

// A transfer_read whose vector is not one contiguous run of its memref (a
// 4x8 block of rows 64 wide, say), read one innermost row at a time and
// assembled with vector.insert, which linearization turns into shuffles.
// Downstream flattening of n-D transfers (mlir-aie's
// FlattenMultDimTransferReadPattern) checks only that the memref is
// row-major, and reads such a block as consecutive elements.
struct SplitNonContiguousTransferRead
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    VectorType vecType = read.getVectorType();
    auto memrefType = dyn_cast<MemRefType>(read.getBase().getType());
    if (!memrefType || read.getMask() || vecType.getRank() < 2 ||
        vecType.isScalable() || !read.getPermutationMap().isMinorIdentity() ||
        vector::isContiguousSlice(memrefType, vecType))
      return failure();
    // Each row keeps only the innermost dim's bound: the rows themselves
    // must be in bounds.
    SmallVector<bool> inBounds = read.getInBoundsValues();
    if (!llvm::all_of(ArrayRef<bool>(inBounds).drop_back(),
                      [](bool b) { return b; }))
      return failure();

    Location loc = read.getLoc();
    int64_t rank = vecType.getRank();
    int64_t memRank = memrefType.getRank();
    ArrayRef<int64_t> shape = vecType.getShape();
    auto rowType = VectorType::get({shape.back()}, vecType.getElementType());
    Value result = arith::ConstantOp::create(rewriter, loc, vecType,
                                             rewriter.getZeroAttr(vecType));
    SmallVector<int64_t> pos(rank - 1, 0);
    for (int64_t n = 0, e = vecType.getNumElements() / shape.back(); n < e;
         ++n) {
      // Row `pos` of the vector reads memref indices offset by `pos` in the
      // vector's (trailing) dims.
      SmallVector<Value> indices(read.getIndices());
      for (int64_t d = 0; d < rank - 1; ++d)
        if (pos[d] != 0) {
          Value &idx = indices[memRank - rank + d];
          idx = arith::AddIOp::create(
              rewriter, loc, idx,
              arith::ConstantIndexOp::create(rewriter, loc, pos[d]));
        }
      Value row = vector::TransferReadOp::create(
          rewriter, loc, rowType, read.getBase(), indices, read.getPadding(),
          ArrayRef<bool>{inBounds.back()});
      result = vector::InsertOp::create(rewriter, loc, row, result, pos);
      for (int64_t d = rank - 2; d >= 0; --d) {
        if (++pos[d] < shape[d])
          break;
        pos[d] = 0;
      }
    }
    rewriter.replaceOp(read, result);
    return success();
  }
};

// A rank-1 transfer_read along memref dim k, every later dim of extent 1:
// the permutation map `(..., dk, ...) -> (dk)` that folding a rank-reducing
// subview into the read leaves. Collapse the trailing unit dims into dim k
// and read with a minor identity map: the AIE core lowering handles only
// those. (The trailing indices index extent-1 dims, so they are 0.)
struct TrailingUnitDimsTransferRead
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    VectorType vecType = read.getVectorType();
    AffineMap map = read.getPermutationMap();
    auto memrefType = dyn_cast<MemRefType>(read.getBase().getType());
    if (read.getMask() || vecType.getRank() != 1 || !memrefType ||
        map.getNumResults() != 1 || map.isMinorIdentity())
      return failure();
    auto dim = dyn_cast<AffineDimExpr>(map.getResult(0));
    int64_t rank = memrefType.getRank();
    if (!dim)
      return failure();
    // The trailing indices are dropped, so each must be the constant 0.
    for (int64_t d = dim.getPosition() + 1; d < rank; ++d)
      if (memrefType.getDimSize(d) != 1 ||
          getConstantIntValue(read.getIndices()[d]) != 0)
        return failure();
    // Fold dims k.. (all but k of extent 1) into one: the read is then a
    // minor identity read of the collapsed memref.
    int64_t k = dim.getPosition();
    SmallVector<ReassociationIndices> reassoc;
    for (int64_t d = 0; d < k; ++d)
      reassoc.push_back({d});
    ReassociationIndices last;
    for (int64_t d = k; d < rank; ++d)
      last.push_back(d);
    reassoc.push_back(last);
    if (!memref::CollapseShapeOp::isGuaranteedCollapsible(memrefType, reassoc))
      return failure();
    Location loc = read.getLoc();
    Value collapsed =
        memref::CollapseShapeOp::create(rewriter, loc, read.getBase(), reassoc);
    SmallVector<Value> indices(read.getIndices().begin(),
                               read.getIndices().begin() + k + 1);
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        read, vecType, collapsed, indices, read.getPadding(),
        ArrayRef<bool>{read.getInBoundsValues()[0]});
    return success();
  }
};

// A rank-1 transfer_read that is a strided gather, along a memref dim
// other than the innermost (permutation map `(..., dk, ...) -> (dk)`), or
// along a strided innermost dim: the AIE core lowering has no pattern for
// it. Read element by element.
struct UnrollStridedTransferRead
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    VectorType vecType = read.getVectorType();
    AffineMap map = read.getPermutationMap();
    auto memrefType = dyn_cast<MemRefType>(read.getBase().getType());
    if (read.getMask() || vecType.getRank() != 1 || vecType.isScalable() ||
        map.getNumResults() != 1 || !memrefType)
      return failure();
    // memref.load has no padding: only a read known to be in bounds.
    if (read.hasOutOfBoundsDim())
      return failure();
    auto dim = dyn_cast<AffineDimExpr>(map.getResult(0));
    if (!dim)
      return failure();
    // Along the innermost dim it is a gather only if that dim is strided
    // (a rank-reduced subview of a column, say).
    if (map.isMinorIdentity() && vector::isContiguousSlice(memrefType, vecType))
      return failure();
    Location loc = read.getLoc();
    SmallVector<Value> elems;
    for (int64_t i = 0, e = vecType.getNumElements(); i < e; ++i) {
      SmallVector<Value> indices(read.getIndices());
      Value &idx = indices[dim.getPosition()];
      if (i != 0)
        idx = arith::AddIOp::create(
            rewriter, loc, idx,
            arith::ConstantIndexOp::create(rewriter, loc, i));
      elems.push_back(
          memref::LoadOp::create(rewriter, loc, read.getBase(), indices));
    }
    rewriter.replaceOpWithNewOp<vector::FromElementsOp>(read, vecType, elems);
    return success();
  }
};

// vector.broadcast of a vector, linearized as a vector.shuffle of the
// flattened source: result element i takes the source element at i's
// coordinates, with the stretched (extent 1) and the new leading dims
// pinned to 0.
struct LinearizeBroadcastToShuffle
    : public OpConversionPattern<vector::BroadcastOp> {
  using OpConversionPattern<vector::BroadcastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(vector::BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = dyn_cast<VectorType>(op.getSourceType());
    VectorType resType = op.getResultVectorType();
    if (!srcType || srcType.isScalable() || resType.isScalable())
      return failure();
    Type flatType = getTypeConverter()->convertType(resType);
    if (!flatType)
      return failure();

    ArrayRef<int64_t> resShape = resType.getShape();
    ArrayRef<int64_t> srcShape = srcType.getShape();
    int64_t lead = resShape.size() - srcShape.size();
    SmallVector<int64_t> srcStrides = computeStrides(srcShape);

    SmallVector<int64_t> mask;
    mask.reserve(resType.getNumElements());
    SmallVector<int64_t> coord(resShape.size(), 0);
    for (int64_t n = 0, e = resType.getNumElements(); n < e; ++n) {
      int64_t src = 0;
      for (size_t d = 0; d < srcShape.size(); ++d)
        if (srcShape[d] != 1)
          src += coord[lead + d] * srcStrides[d];
      mask.push_back(src);
      for (int64_t d = static_cast<int64_t>(resShape.size()) - 1; d >= 0; --d) {
        if (++coord[d] < resShape[d])
          break;
        coord[d] = 0;
      }
    }

    Value src = adaptor.getSource();
    if (cast<VectorType>(src.getType()).getRank() != 1)
      src = vector::ShapeCastOp::create(
          rewriter, op.getLoc(),
          VectorType::get({srcType.getNumElements()}, srcType.getElementType()),
          src);
    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(op, flatType, src, src,
                                                   mask);
    return success();
  }
};

// The lanes of a rank-1 `v` as lanes of the value it is a static permutation
// of, through vector.shuffle (of one source), vector.shape_cast and
// vector.transpose: `lanes[l]` is the source lane of lane l.
static Value traceLanes(Value v, SmallVector<int64_t> &lanes) {
  auto vt = cast<VectorType>(v.getType());
  lanes.resize(vt.getNumElements());
  std::iota(lanes.begin(), lanes.end(), 0);
  while (true) {
    Operation *def = v.getDefiningOp();
    if (auto sc = dyn_cast_if_present<vector::ShapeCastOp>(def)) {
      v = sc.getSource();
      continue;
    }
    if (auto tr = dyn_cast_if_present<vector::TransposeOp>(def)) {
      // Result lane at coords c reads source coords c' with c'[perm[i]] =
      // c[i].
      VectorType rt = tr.getResultVectorType();
      VectorType st = tr.getSourceVectorType();
      ArrayRef<int64_t> perm = tr.getPermutation();
      SmallVector<int64_t> rStrides = computeStrides(rt.getShape());
      SmallVector<int64_t> sStrides = computeStrides(st.getShape());
      for (int64_t &l : lanes) {
        SmallVector<int64_t> c = delinearize(l, rStrides);
        int64_t src = 0;
        for (size_t i = 0; i < perm.size(); ++i)
          src += c[i] * sStrides[perm[i]];
        l = src;
      }
      v = tr.getVector();
      continue;
    }
    if (auto sh = dyn_cast_if_present<vector::ShuffleOp>(def)) {
      int64_t n1 = sh.getV1VectorType().getNumElements();
      ArrayRef<int64_t> mask = sh.getMask();
      bool fromV1 =
          llvm::all_of(lanes, [&](int64_t l) { return mask[l] < n1; });
      bool fromV2 =
          llvm::all_of(lanes, [&](int64_t l) { return mask[l] >= n1; });
      if (!fromV1 && !fromV2 && sh.getV1() != sh.getV2())
        return v;
      for (int64_t &l : lanes)
        l = mask[l] % n1;
      v = (fromV2 && !fromV1) ? sh.getV2() : sh.getV1();
      continue;
    }
    return v;
  }
}

// `(X >> [0, 4, 8, ...]) & 15` where lane l of X is word l / k of a vector
// W (k = 4-bit fields per word): the 4-bit fields of W in order. Rewritten
// as `extsi(extui(bitcast(W) to i4) to i8)`, which AIE lowers to its
// unpack instruction; neither the per-lane right shift nor the
// replication of W has a lowering there. Triton, having no i4, spells a
// 4-bit unpack this way.
struct NibbleUnpackFromShifts : public OpRewritePattern<arith::AndIOp> {
  NibbleUnpackFromShifts(MLIRContext *ctx, bool aie2p)
      : OpRewritePattern<arith::AndIOp>(ctx), aie2p(aie2p) {}

  bool aie2p;

  LogicalResult matchAndRewrite(arith::AndIOp andOp,
                                PatternRewriter &rewriter) const override {
    auto vt = dyn_cast<VectorType>(andOp.getType());
    if (!vt || vt.getRank() != 1)
      return failure();
    auto elemTy = dyn_cast<IntegerType>(vt.getElementType());
    if (!elemTy || elemTy.getWidth() <= 8 || elemTy.getWidth() % 4 != 0)
      return failure();
    int64_t k = elemTy.getWidth() / 4, n = vt.getNumElements();
    // Constants may come reshaped or replicated (defined outside the
    // linearized region, say): look through lane permutations.
    auto constantOf = [](Value v, DenseIntElementsAttr &attr) {
      SmallVector<int64_t> unused;
      return matchPattern(traceLanes(v, unused), m_Constant(&attr));
    };
    DenseIntElementsAttr maskAttr, shiftAttr;
    Value shifted = andOp.getLhs();
    if (!constantOf(andOp.getRhs(), maskAttr)) {
      shifted = andOp.getRhs();
      if (!constantOf(andOp.getLhs(), maskAttr))
        return failure();
    }
    if (!maskAttr.isSplat() || maskAttr.getSplatValue<APInt>() != 15)
      return failure();
    Operation *shOp = shifted.getDefiningOp();
    if (!isa_and_present<arith::ShRSIOp, arith::ShRUIOp>(shOp))
      return failure();
    // The shift amounts: a constant, possibly replicated by shuffles.
    SmallVector<int64_t> shiftLanes;
    Value shiftBase = traceLanes(shOp->getOperand(1), shiftLanes);
    if (!matchPattern(shiftBase, m_Constant(&shiftAttr)))
      return failure();
    auto shifts = llvm::to_vector(shiftAttr.getValues<APInt>());
    for (int64_t l = 0; l < n; ++l)
      if (shifts[shiftLanes[l]].getZExtValue() !=
          static_cast<uint64_t>(4 * (l % k)))
        return failure();
    SmallVector<int64_t> lanes;
    Value words = traceLanes(shOp->getOperand(0), lanes);
    auto wt = dyn_cast<VectorType>(words.getType());
    if (!wt || wt.isScalable() || wt.getElementType() != elemTy)
      return failure();
    for (int64_t l = 0; l < n; ++l)
      if (lanes[l] != l / k)
        return failure();
    // The unpack instruction takes 64 or 128 4-bit fields.
    int64_t fields = n <= 64 ? 64 : 128;
    if (n > 128 || n % k != 0)
      return failure();

    Location loc = andOp.getLoc();
    Type i8 = rewriter.getI8Type();
    if (wt.getRank() != 1) {
      wt = VectorType::get({wt.getNumElements()}, elemTy);
      words = vector::ShapeCastOp::create(rewriter, loc, wt, words);
    }
    int64_t wordBytes = elemTy.getWidth() / 8;
#if AIR_ENABLE_AIE
    // Exactly one unpack's worth of bytes: emit aievec.unpack itself. The
    // standard spelling (bitcast to i4 + extui to i8) is folded apart by
    // canonicalization (bitcast chains merge, extsi(extui) becomes one
    // extui) before the AIE lowering gets to match it.
    if (aie2p && wt.getNumElements() * wordBytes * 2 == n &&
        (n == 64 || n == 128)) {
      Value bytes = vector::BitCastOp::create(
          rewriter, loc, VectorType::get({n / 2}, i8), words);
      Value unpacked = xilinx::aievec::UnpackOp::create(
          rewriter, loc, VectorType::get({n}, i8), bytes);
      rewriter.replaceOpWithNewOp<arith::ExtSIOp>(andOp, vt, unpacked);
      if (shOp->use_empty())
        rewriter.eraseOp(shOp);
      return success();
    }
#endif
    Value bytes = vector::BitCastOp::create(
        rewriter, loc, VectorType::get({wt.getNumElements() * wordBytes}, i8),
        words);
    int64_t haveBytes = wt.getNumElements() * wordBytes;
    SmallVector<int64_t> pick(fields / 2);
    for (int64_t b = 0; b < fields / 2; ++b)
      pick[b] = b < haveBytes ? b : haveBytes; // padding: lane 0 of `zero`
    Value zero = arith::ConstantOp::create(
        rewriter, loc, cast<VectorType>(bytes.getType()),
        rewriter.getZeroAttr(bytes.getType()));
    Value padded = vector::ShuffleOp::create(rewriter, loc, bytes, zero, pick);
    Value nibbles = vector::BitCastOp::create(
        rewriter, loc, VectorType::get({fields}, rewriter.getIntegerType(4)),
        padded);
    Value unpacked = arith::ExtUIOp::create(
        rewriter, loc, VectorType::get({fields}, i8), nibbles);
    Value lanesN =
        vector::ShuffleOp::create(rewriter, loc, unpacked, unpacked,
                                  llvm::to_vector(llvm::seq<int64_t>(0, n)));
    // 0..15 in i8: sign and zero extension agree.
    rewriter.replaceOpWithNewOp<arith::ExtSIOp>(andOp, vt, lanesN);
    if (shOp->use_empty())
      rewriter.eraseOp(shOp);
    return success();
  }
};

// A rank-1 elementwise op wider than the AIE vector lowering handles, split
// into native-width pieces: 32 lanes of bf16 (add/sub/mul), 16 lanes when
// f32 is involved (vector.fma, extf to f32, truncf from f32, f32
// add/sub/mul), one 512-bit register or accumulator each. The pieces are
// assembled with vector.insert_strided_slice so that the next split op's
// vector.extract_strided_slice folds onto them, keeping producer and
// consumer pieces adjacent (mul+add for the FMA lowering, extf feeding an
// fma for mac_elem).
struct SplitWideElementwise : public RewritePattern {
  SplitWideElementwise(MLIRContext *ctx, int64_t f32Lanes)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx),
        f32Lanes(f32Lanes) {}

  int64_t f32Lanes;

  int64_t nativeLanes(Operation *op) const {
    auto isF32 = [](Type t) {
      auto vt = dyn_cast<VectorType>(t);
      return vt && vt.getElementType().isF32();
    };
    bool f32 = llvm::any_of(op->getOperandTypes(), isF32) ||
               llvm::any_of(op->getResultTypes(), isF32);
    if (isa<vector::FMAOp, arith::ExtFOp, arith::TruncFOp>(op))
      return f32 ? f32Lanes : 0;
    if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp>(op)) {
      if (f32)
        return f32Lanes;
      auto vt = dyn_cast<VectorType>(op->getResult(0).getType());
      return vt && vt.getElementType().isBF16() ? 32 : 0;
    }
    return 0;
  }

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    int64_t lanes = nativeLanes(op);
    if (!lanes || op->getNumResults() != 1)
      return failure();
    auto vt = dyn_cast<VectorType>(op->getResult(0).getType());
    if (!vt || vt.getRank() != 1 || vt.getNumElements() <= lanes ||
        vt.getNumElements() % lanes != 0)
      return failure();
    for (Value v : op->getOperands()) {
      auto ot = dyn_cast<VectorType>(v.getType());
      if (!ot || ot.getRank() != 1 ||
          ot.getNumElements() != vt.getNumElements())
        return failure();
    }
    // Halve, and let the pattern apply again to the halves: the AIE
    // lowering extracts halves of a register, not arbitrary sub-ranges.
    if (!llvm::isPowerOf2_64(vt.getNumElements() / lanes))
      return failure();
    lanes = vt.getNumElements() / 2;
    Location loc = op->getLoc();
    auto pieceType = VectorType::get({lanes}, vt.getElementType());
    Value result = ub::PoisonOp::create(rewriter, loc, vt);
    for (int64_t off = 0; off < vt.getNumElements(); off += lanes) {
      SmallVector<Value> pieces;
      for (Value v : op->getOperands())
        pieces.push_back(vector::ExtractStridedSliceOp::create(rewriter, loc, v,
                                                               off, lanes, 1));
      OperationState state(loc, op->getName().getIdentifier(), pieces,
                           TypeRange{pieceType}, op->getAttrs());
      Value r = rewriter.create(state)->getResult(0);
      result = vector::InsertStridedSliceOp::create(rewriter, loc, r, result,
                                                    off, 1);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// `shuffle(extf(a), _)` reading only its first operand, as
// `extf(shuffle(a, a))`: the widening reaches the op that consumes the
// replicated value, where the AIE multiply-accumulate lowering matches a
// bf16 operand widened to f32.
struct SinkExtFBelowShuffle : public OpRewritePattern<vector::ShuffleOp> {
  using OpRewritePattern<vector::ShuffleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShuffleOp shuffle,
                                PatternRewriter &rewriter) const override {
    auto ext = shuffle.getV1().getDefiningOp<arith::ExtFOp>();
    int64_t n1 = shuffle.getV1VectorType().getNumElements();
    if (!ext ||
        llvm::any_of(shuffle.getMask(), [&](int64_t m) { return m >= n1; }))
      return failure();
    Value src = ext.getIn();
    Value narrow = vector::ShuffleOp::create(rewriter, shuffle.getLoc(), src,
                                             src, shuffle.getMask());
    rewriter.replaceOpWithNewOp<arith::ExtFOp>(shuffle, shuffle.getType(),
                                               narrow);
    return success();
  }
};

// A vector.extract_strided_slice of a register (not of an insert it would
// fold with) that is not a half of it, as a vector.shuffle: the AIE lowering
// extracts only register halves; LLVM lowers the shuffle.
struct ExtractSliceToShuffle
    : public OpRewritePattern<vector::ExtractStridedSliceOp> {
  using OpRewritePattern<vector::ExtractStridedSliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ExtractStridedSliceOp ex,
                                PatternRewriter &rewriter) const override {
    VectorType st = ex.getSourceVectorType(), rt = ex.getType();
    if (st.getRank() != 1 ||
        ex.getSource().getDefiningOp<vector::InsertStridedSliceOp>())
      return failure();
    int64_t off = cast<IntegerAttr>(ex.getOffsets()[0]).getInt();
    int64_t stride = cast<IntegerAttr>(ex.getStrides()[0]).getInt();
    // A slice of a shuffle is a narrower shuffle of the same inputs: a
    // replicated scale or base then never exists at the full width.
    if (auto sh = ex.getSource().getDefiningOp<vector::ShuffleOp>()) {
      ArrayRef<int64_t> full = sh.getMask();
      SmallVector<int64_t> sub;
      for (int64_t i = 0; i < rt.getNumElements(); ++i)
        sub.push_back(full[off + i * stride]);
      rewriter.replaceOpWithNewOp<vector::ShuffleOp>(ex, sh.getV1(), sh.getV2(),
                                                     sub);
      return success();
    }
    if (stride == 1 && rt.getNumElements() * 2 == st.getNumElements())
      return failure();
    SmallVector<int64_t> mask;
    for (int64_t i = 0; i < rt.getNumElements(); ++i)
      mask.push_back(off + i * stride);
    rewriter.replaceOpWithNewOp<vector::ShuffleOp>(ex, ex.getSource(),
                                                   ex.getSource(), mask);
    return success();
  }
};

// The bf16 value `v` is widened from by arith.extf, or null.
static Value bf16Source(Value v) {
  auto ext = v.getDefiningOp<arith::ExtFOp>();
  if (!ext || !getElementTypeOrSelf(ext.getIn().getType()).isBF16())
    return nullptr;
  return ext.getIn();
}

// `addf(mulf(a, b), c)` on f32 vectors, the product used only there, as
// vector.fma: with a and b widened from bf16 the AIE lowering makes it one
// bf16 x bf16 + f32 multiply-accumulate.
struct MulAddToFMA : public OpRewritePattern<arith::AddFOp> {
  MulAddToFMA(MLIRContext *ctx, bool contract, PatternBenefit benefit)
      : OpRewritePattern<arith::AddFOp>(ctx, benefit), contract(contract) {}

  LogicalResult matchAndRewrite(arith::AddFOp add,
                                PatternRewriter &rewriter) const override {
    auto vt = dyn_cast<VectorType>(add.getType());
    if (!vt || vt.getRank() != 1 || !vt.getElementType().isF32())
      return failure();
    // Contraction drops the product's rounding: only where allowed.
    auto mayContract = [&](Operation *op) {
      if (contract)
        return true;
      auto fm = dyn_cast<arith::ArithFastMathInterface>(op);
      return fm && bitEnumContainsAll(fm.getFastMathFlagsAttr().getValue(),
                                      arith::FastMathFlags::contract);
    };
    if (!mayContract(add))
      return failure();
    // Only a product of widened bf16 values (possibly replicated by a shuffle):
    // that is the multiply-accumulate the AIE lowering has; an f32 x f32 fma
    // has none.
    auto widened = [](Value v) {
      if (auto sh = v.getDefiningOp<vector::ShuffleOp>())
        v = sh.getV1();
      return bf16Source(v) != nullptr;
    };
    for (auto [mulSide, other] : {std::make_pair(add.getLhs(), add.getRhs()),
                                  std::make_pair(add.getRhs(), add.getLhs())}) {
      auto mul = mulSide.getDefiningOp<arith::MulFOp>();
      if (!mul || !mul->hasOneUse() || !mayContract(mul) ||
          !widened(mul.getLhs()) || !widened(mul.getRhs()))
        continue;
      rewriter.replaceOpWithNewOp<vector::FMAOp>(add, mul.getLhs(),
                                                 mul.getRhs(), other);
      return success();
    }
    return failure();
  }

private:
  bool contract;
};

#if AIR_ENABLE_AIE
// A 32-lane f32 `vector.fma` of two bf16 values widened by arith.extf, as
// aievec.mac_elem, and a 32-lane f32 -> bf16 arith.truncf, as aievec.srs.
// AIE2P multiplies 32 bf16 lanes into a 32-lane f32 accumulator and converts
// it back in one instruction, and its LLVM lowering takes both, but the
// vector-to-aievec conversion only matches the 16-lane forms.
struct WideFMAToMacElem : public OpRewritePattern<vector::FMAOp> {
  using OpRewritePattern<vector::FMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::FMAOp fma,
                                PatternRewriter &rewriter) const override {
    auto vt = dyn_cast<VectorType>(fma.getType());
    if (!vt || vt.getRank() != 1 || vt.getNumElements() != 32 ||
        !vt.getElementType().isF32())
      return failure();
    Value lhs = bf16Source(fma.getLhs()), rhs = bf16Source(fma.getRhs());
    if (!lhs || !rhs)
      return failure();
    rewriter.replaceOpWithNewOp<xilinx::aievec::FMAElemOp>(
        fma, vt, lhs, rhs, fma.getAcc(), /*fmsub=*/false);
    return success();
  }
};

struct WideTruncFToSRS : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern<arith::TruncFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp trunc,
                                PatternRewriter &rewriter) const override {
    auto vt = dyn_cast<VectorType>(trunc.getType());
    auto st = dyn_cast<VectorType>(trunc.getIn().getType());
    // Only the default rounding, to nearest even, which is what srs does.
    if (!vt || !st || vt.getRank() != 1 || vt.getNumElements() != 32 ||
        !vt.getElementType().isBF16() || !st.getElementType().isF32() ||
        trunc.getRoundingmodeAttr())
      return failure();
    Value shift = arith::ConstantOp::create(rewriter, trunc.getLoc(),
                                            rewriter.getI32IntegerAttr(0));
    rewriter.replaceOpWithNewOp<xilinx::aievec::SRSOp>(trunc, vt, trunc.getIn(),
                                                       shift);
    return success();
  }
};

// f32_lanes = 64: the AIE2P instruction forms a hand-written dequant kernel
// uses, called as LLVM intrinsics because the aievec lowering has no route to
// them (aievec.shuffle lowers to the AIE2 vshuffle; mac_elem stops at 32
// lanes).

// Calls an AIE2P LLVM intrinsic through a private func.func named after it,
// the way mlir-aie declares its own (llvm.aie2p.acquire, ...). An LLVM-dialect
// op would not do: aie-standard-lowering silently drops a core that holds one.
static ModuleOp outermostModule(Operation *op) {
  ModuleOp mod = op->getParentOfType<ModuleOp>();
  while (auto parent = mod->getParentOfType<ModuleOp>())
    mod = parent;
  return mod;
}

// Whether `name` can be declared (or is already declared) with type `ty` in
// the module holding `op`. A symbol of that name with another type would
// make the call ill-typed, so the patterns check this before rewriting.
static bool canDeclareAIE2p(Operation *op, StringRef name, FunctionType ty) {
  Operation *sym = SymbolTable::lookupSymbolIn(outermostModule(op), name);
  if (!sym)
    return true;
  auto fn = dyn_cast<func::FuncOp>(sym);
  return fn && fn.getFunctionType() == ty;
}

static Value callAIE2p(PatternRewriter &rewriter, Location loc, Type resTy,
                       StringRef name, ValueRange args) {
  ModuleOp mod = outermostModule(rewriter.getInsertionBlock()->getParentOp());
  auto fnTy = rewriter.getFunctionType(TypeRange(ValueRange(args)), {resTy});
  auto fn = mod.lookupSymbol<func::FuncOp>(name);
  if (!fn) {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToStart(mod.getBody());
    fn = func::FuncOp::create(rewriter, loc, name, fnTy);
    fn.setPrivate();
  }
  return func::CallOp::create(rewriter, loc, fn, args).getResult(0);
}

// `bf16(0x4300 | zext(q))` on 64 lanes (exactly 128 + q) as two byte
// interleaves of q with 0x43 (AIE2P vshuffle modes 20/21), replacing an
// upshift, two shift-round-saturates and an or.
struct ByteInterleave4300 : public OpRewritePattern<arith::OrIOp> {
  using OpRewritePattern<arith::OrIOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::OrIOp orOp,
                                PatternRewriter &rewriter) const override {
    auto vt = dyn_cast<VectorType>(orOp.getType());
    if (!vt || vt.getRank() != 1 || vt.getNumElements() != 64 ||
        !vt.getElementType().isInteger(16))
      return failure();
    for (auto [extSide, cstSide] :
         {std::make_pair(orOp.getLhs(), orOp.getRhs()),
          std::make_pair(orOp.getRhs(), orOp.getLhs())}) {
      auto ext = extSide.getDefiningOp<arith::ExtSIOp>();
      if (!ext)
        ext = nullptr;
      Value src = ext ? ext.getIn() : Value();
      if (!src) {
        if (auto zext = extSide.getDefiningOp<arith::ExtUIOp>())
          src = zext.getIn();
      }
      DenseIntElementsAttr cst;
      if (!src || !matchPattern(cstSide, m_Constant(&cst)) || !cst.isSplat() ||
          cst.getSplatValue<APInt>().getZExtValue() != 0x4300 ||
          !getElementTypeOrSelf(src.getType()).isInteger(8))
        continue;
      // Sign extension equals zero extension only for bytes below 0x80: the
      // source must be an unpacked nibble.
      if (ext && !src.getDefiningOp<xilinx::aievec::UnpackOp>())
        continue;
      auto i32 = rewriter.getI32Type();
      auto v16 = VectorType::get({16}, i32);
      if (!canDeclareAIE2p(orOp, "llvm.aie2p.vshuffle",
                           rewriter.getFunctionType({v16, v16, i32}, {v16})))
        return failure();
      Location loc = orOp.getLoc();
      Value words = vector::BitCastOp::create(rewriter, loc, v16, src);
      Value bias = arith::ConstantOp::create(
          rewriter, loc, DenseElementsAttr::get(v16, APInt(32, 0x43434343)));
      SmallVector<Value> halves;
      for (int32_t mode : {20, 21}) {
        Value m = arith::ConstantOp::create(rewriter, loc,
                                            rewriter.getI32IntegerAttr(mode));
        halves.push_back(callAIE2p(rewriter, loc, v16, "llvm.aie2p.vshuffle",
                                   {words, bias, m}));
      }
      Value both =
          vector::ShuffleOp::create(rewriter, loc, halves[0], halves[1],
                                    llvm::to_vector(llvm::seq<int64_t>(0, 32)));
      rewriter.replaceOpWithNewOp<vector::BitCastOp>(orOp, vt, both);
      return success();
    }
    return failure();
  }
};

// A 64-lane f32 vector.fma of two bf16 values widened by arith.extf, as one
// 64-lane bf16 multiply-accumulate (I1024.I1024.ACC2048, aie_api's config).
struct FMA64ToMac : public OpRewritePattern<vector::FMAOp> {
  using OpRewritePattern<vector::FMAOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::FMAOp fma,
                                PatternRewriter &rewriter) const override {
    auto vt = dyn_cast<VectorType>(fma.getType());
    if (!vt || vt.getRank() != 1 || vt.getNumElements() != 64 ||
        !vt.getElementType().isF32())
      return failure();
    Value lhs = bf16Source(fma.getLhs()), rhs = bf16Source(fma.getRhs());
    if (!lhs || !rhs)
      return failure();
    if (!canDeclareAIE2p(fma, "llvm.aie2p.I1024.I1024.ACC2048.bf.mac.conf",
                         rewriter.getFunctionType({lhs.getType(), rhs.getType(),
                                                   vt, rewriter.getI32Type()},
                                                  {vt})))
      return failure();
    Location loc = fma.getLoc();
    Value conf = arith::ConstantOp::create(rewriter, loc,
                                           rewriter.getI32IntegerAttr(828));
    Value mac = callAIE2p(rewriter, loc, vt,
                          "llvm.aie2p.I1024.I1024.ACC2048.bf.mac.conf",
                          {lhs, rhs, fma.getAcc(), conf});
    rewriter.replaceOp(fma, mac);
    return success();
  }
};

// A 64-lane f32 -> bf16 arith.truncf as two 32-lane aievec.srs (srs is also
// what makes aie-standard-lowering set the rounding mode). When the result is
// only stored, through a shape_cast and an in-bounds minor-identity
// transfer_write, the halves are stored separately: the backend then fuses
// each conversion into its store (vst.conv), which a re-concatenated 64-lane
// store prevents.
struct TruncF64ToSRS : public OpRewritePattern<arith::TruncFOp> {
  using OpRewritePattern<arith::TruncFOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::TruncFOp trunc,
                                PatternRewriter &rewriter) const override {
    auto vt = dyn_cast<VectorType>(trunc.getType());
    if (!vt || vt.getRank() != 1 || vt.getNumElements() != 64 ||
        !vt.getElementType().isBF16() ||
        !getElementTypeOrSelf(trunc.getIn().getType()).isF32() ||
        trunc.getRoundingmodeAttr())
      return failure();
    Location loc = trunc.getLoc();
    Value shift =
        arith::ConstantOp::create(rewriter, loc, rewriter.getI32IntegerAttr(0));
    // Halve the accumulator with vector.shuffle, which reaches LLVM as a
    // shufflevector (a register half); extract_strided_slice would become
    // aievec shifts.
    auto half = VectorType::get({32}, rewriter.getBF16Type());
    Value parts[2];
    for (int k = 0; k < 2; ++k) {
      Value acc = vector::ShuffleOp::create(
          rewriter, loc, trunc.getIn(), trunc.getIn(),
          llvm::to_vector(llvm::seq<int64_t>(k * 32, k * 32 + 32)));
      parts[k] = xilinx::aievec::SRSOp::create(rewriter, loc, half, acc, shift);
    }
    vector::ShapeCastOp cast;
    vector::TransferWriteOp write;
    if (trunc->hasOneUse())
      cast = dyn_cast<vector::ShapeCastOp>(*trunc->user_begin());
    if (cast && cast->hasOneUse())
      write = dyn_cast<vector::TransferWriteOp>(*cast->user_begin());
    VectorType wt = cast ? cast.getResultVectorType() : VectorType();
    // Memref writes only: a tensor write's result would need rethreading.
    if (!write || write.getMask() ||
        !isa<MemRefType>(write.getBase().getType()) || wt.getRank() < 2 ||
        wt.getDimSize(0) % 2 != 0 ||
        !write.getPermutationMap().isMinorIdentity() ||
        write.getPermutationMap().getNumResults() != wt.getRank() ||
        write.hasOutOfBoundsDim()) {
      rewriter.replaceOpWithNewOp<vector::ShuffleOp>(
          trunc, parts[0], parts[1],
          llvm::to_vector(llvm::seq<int64_t>(0, 64)));
      return success();
    }
    SmallVector<int64_t> hs(wt.getShape());
    hs[0] /= 2;
    auto hvt = VectorType::get(hs, wt.getElementType());
    unsigned d0 = write.getIndices().size() - wt.getRank();
    rewriter.setInsertionPoint(write);
    for (int k = 0; k < 2; ++k) {
      Value v = vector::ShapeCastOp::create(rewriter, loc, hvt, parts[k]);
      SmallVector<Value> idx(write.getIndices());
      if (k)
        idx[d0] = arith::AddIOp::create(
            rewriter, loc, idx[d0],
            arith::ConstantIndexOp::create(rewriter, loc, hs[0]));
      vector::TransferWriteOp::create(rewriter, loc, v, write.getBase(), idx,
                                      write.getPermutationMapAttr(),
                                      write.getInBoundsAttr());
    }
    rewriter.eraseOp(write);
    rewriter.eraseOp(cast);
    rewriter.eraseOp(trunc);
    return success();
  }
};
#endif

// A transfer_read that fixes the memref's innermost dim at a constant k and
// reads the dims just above it (every s-th element of a contiguous run, e.g.
// one slot of the interleaved pairs a fused gate/up epilogue reads) as a read
// of the whole run and a shuffle taking slot k. The AIE lowering takes
// contiguous reads only.
struct DeinterleaveTransferRead
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern<vector::TransferReadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp read,
                                PatternRewriter &rewriter) const override {
    auto mt = dyn_cast<MemRefType>(read.getBase().getType());
    VectorType vt = read.getVectorType();
    AffineMap map = read.getPermutationMap();
    if (!mt || read.getMask() || !mt.hasStaticShape() ||
        !mt.getLayout().isIdentity())
      return failure();
    int64_t r = mt.getRank(), vr = vt.getRank();
    if (vr < 1 || map.getNumResults() != vr || r < vr + 1)
      return failure();
    for (int64_t i = 0; i < vr; ++i) {
      auto e = dyn_cast<AffineDimExpr>(map.getResult(i));
      if (!e || e.getPosition() != r - 1 - vr + i)
        return failure();
    }
    int64_t s = mt.getDimSize(r - 1);
    std::optional<int64_t> k = getConstantIntValue(read.getIndices()[r - 1]);
    if (s < 2 || s > 8 || !k)
      return failure();
    if (read.hasOutOfBoundsDim())
      return failure();
    Location loc = read.getLoc();
    SmallVector<Value> idx(read.getIndices());
    idx[r - 1] = arith::ConstantIndexOp::create(rewriter, loc, 0);
    SmallVector<int64_t> wshape(vt.getShape());
    wshape.push_back(s);
    auto wvt = VectorType::get(wshape, vt.getElementType());
    Value whole;
    // Through an expand_shape that only splits the innermost dim into [n, s],
    // with the read covering all n: read the unexpanded rows instead, so the
    // view dies (memref analyses, e.g. L1 shrinking, do not see through it).
    auto ex = read.getBase().getDefiningOp<memref::ExpandShapeOp>();
    if (ex && ex.getSrcType().getRank() == r - 1 &&
        ex.getReassociationIndices().back() ==
            ReassociationIndices{r - 2, r - 1} &&
        getConstantIntValue(idx[r - 2]) == 0 &&
        vt.getShape().back() == mt.getDimSize(r - 2)) {
      SmallVector<Value> sidx(idx.begin(), idx.begin() + r - 1);
      SmallVector<int64_t> rshape(vt.getShape());
      rshape.back() *= s;
      auto rvt = VectorType::get(rshape, vt.getElementType());
      Value rows = vector::TransferReadOp::create(
          rewriter, loc, rvt, ex.getSrc(), sidx,
          AffineMapAttr::get(
              AffineMap::getMinorIdentityMap(r - 1, vr, rewriter.getContext())),
          read.getPadding(), Value(),
          rewriter.getBoolArrayAttr(SmallVector<bool>(vr, true)));
      whole = vector::ShapeCastOp::create(rewriter, loc, wvt, rows);
    } else {
      SmallVector<bool> inb(vr + 1, true);
      whole = vector::TransferReadOp::create(
          rewriter, loc, wvt, read.getBase(), idx,
          AffineMapAttr::get(
              AffineMap::getMinorIdentityMap(r, vr + 1, rewriter.getContext())),
          read.getPadding(), Value(), rewriter.getBoolArrayAttr(inb));
    }
    int64_t n = vt.getNumElements();
    auto flat = VectorType::get({n * s}, vt.getElementType());
    Value lin = vector::ShapeCastOp::create(rewriter, loc, flat, whole);
    SmallVector<int64_t> mask;
    for (int64_t i = 0; i < n; ++i)
      mask.push_back(i * s + *k);
    Value picked = vector::ShuffleOp::create(rewriter, loc, lin, lin, mask);
    rewriter.replaceOpWithNewOp<vector::ShapeCastOp>(read, vt, picked);
    return success();
  }
};

// The ops whose n-D vector types get linearized: elementwise computation and
// what feeds it. Transfers, contractions and loops keep their types.
static bool isLinearizedOp(Operation *op) {
  return OpTrait::hasElementwiseMappableTraits(op) ||
         op->hasTrait<OpTrait::ConstantLike>() ||
         isa<vector::BitCastOp, vector::BroadcastOp, vector::InsertOp>(op);
}

} // namespace

DiagnosedSilenceableFailure
transform::LinearizeVectorsOp::apply(transform::TransformRewriter &rewriter,
                                     transform::TransformResults &results,
                                     transform::TransformState &state) {
  bool aie2p = getArch() == "aie2p";
#if !AIR_ENABLE_AIE
  if (aie2p)
    return emitSilenceableError()
           << "arch = \"aie2p\" needs a build with AIE support";
#endif
  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));
  for (Operation *target : targets) {
    MLIRContext *ctx = target->getContext();

    // Each phase rewrites only the ops it collects, and the ops it creates.
    GreedyRewriteConfig config;
    config.setStrictness(GreedyRewriteStrictness::ExistingAndNewOps);
    auto *listener =
        static_cast<RewriterBase::Listener *>(rewriter.getListener());
    config.setListener(listener);
    SmallVector<Operation *> reads;
    target->walk([&](vector::TransferReadOp op) { reads.push_back(op); });
    RewritePatternSet readPatterns(ctx);
    readPatterns.add<UnbroadcastTransferRead, SplitNonContiguousTransferRead,
                     TrailingUnitDimsTransferRead>(ctx);
    readPatterns.add<DeinterleaveTransferRead>(ctx, /*benefit=*/2);
    if (failed(applyOpPatternsGreedily(reads, std::move(readPatterns), config)))
      return emitDefiniteFailure()
             << "failed to unbroadcast vector.transfer_read ops";

    TypeConverter typeConverter;
    ConversionTarget convTarget(*ctx);
    vector::populateForVectorLinearize(typeConverter, convTarget);
    // Replaces the upstream predicate, which would make every n-D vector
    // dialect op (transfers, contract) illegal with no pattern to legalize it.
    convTarget.markUnknownOpDynamicallyLegal(
        [&](Operation *op) -> std::optional<bool> {
          if (!isLinearizedOp(op))
            return true;
          return typeConverter.isLegal(op);
        });
    RewritePatternSet patterns(ctx);
    vector::populateVectorLinearizeBasePatterns(typeConverter, convTarget,
                                                patterns);
    vector::populateVectorLinearizeShuffleLikeOpsPatterns(typeConverter,
                                                          convTarget, patterns);
    patterns.add<LinearizeBroadcastToShuffle>(typeConverter, ctx,
                                              /*benefit=*/2);
    ConversionConfig convConfig;
    convConfig.listener = listener;
    if (failed(applyPartialConversion(target, convTarget, std::move(patterns),
                                      convConfig)))
      return emitDefiniteFailure() << "failed to linearize vector ops";

    SmallVector<Operation *> casts;
    target->walk([&](Operation *op) {
      if (isa<vector::ShapeCastOp, vector::BroadcastOp>(op))
        casts.push_back(op);
    });
    RewritePatternSet canonPatterns(ctx);
    vector::ShapeCastOp::getCanonicalizationPatterns(canonPatterns, ctx);
    vector::BroadcastOp::getCanonicalizationPatterns(canonPatterns, ctx);
    if (failed(
            applyOpPatternsGreedily(casts, std::move(canonPatterns), config)))
      return emitDefiniteFailure()
             << "failed to canonicalize after linearization";

    SmallVector<Operation *> ands;
    target->walk([&](Operation *op) {
      if (isa<arith::AndIOp, vector::TransferReadOp, arith::AddFOp,
              arith::SubFOp, arith::MulFOp, arith::ExtFOp, arith::TruncFOp,
              vector::FMAOp, vector::ShuffleOp>(op))
        ands.push_back(op);
    });
    RewritePatternSet unpackPatterns(ctx);
    unpackPatterns.add<NibbleUnpackFromShifts>(ctx, aie2p);
    unpackPatterns.add<UnrollStridedTransferRead>(ctx, /*benefit=*/0);
    unpackPatterns.add<SplitWideElementwise>(ctx, getF32Lanes());
    unpackPatterns.add<MulAddToFMA>(ctx, getContract(), /*benefit=*/2);
    unpackPatterns.add<SinkExtFBelowShuffle>(ctx, /*benefit=*/2);
    vector::ExtractStridedSliceOp::getCanonicalizationPatterns(unpackPatterns,
                                                               ctx);
    vector::ShuffleOp::getCanonicalizationPatterns(unpackPatterns, ctx);
    arith::TruncIOp::getCanonicalizationPatterns(unpackPatterns, ctx);
    if (failed(
            applyOpPatternsGreedily(ands, std::move(unpackPatterns), config)))
      return emitDefiniteFailure() << "failed to rewrite 4-bit unpacks";

    // Once the pieces have settled (extracts of inserts folded), extracts
    // that remain read real registers.
    SmallVector<Operation *> extracts;
    target->walk(
        [&](vector::ExtractStridedSliceOp op) { extracts.push_back(op); });
    RewritePatternSet extractPatterns(ctx);
    extractPatterns.add<ExtractSliceToShuffle>(ctx);
    if (failed(applyOpPatternsGreedily(extracts, std::move(extractPatterns),
                                       config)))
      return emitDefiniteFailure() << "failed to rewrite register extracts";

#if AIR_ENABLE_AIE
    if (getF32Lanes() == 64) {
      SmallVector<Operation *> wide;
      target->walk([&](Operation *op) {
        if (isa<vector::FMAOp, arith::TruncFOp, arith::OrIOp>(op))
          wide.push_back(op);
      });
      RewritePatternSet widePatterns(ctx);
      widePatterns.add<FMA64ToMac, TruncF64ToSRS, ByteInterleave4300>(ctx);
      if (failed(
              applyOpPatternsGreedily(wide, std::move(widePatterns), config)))
        return emitDefiniteFailure()
               << "failed to emit 64-lane AIE2P intrinsics";
    }
    if (getF32Lanes() == 32) {
      SmallVector<Operation *> wide;
      target->walk([&](Operation *op) {
        if (isa<vector::FMAOp, arith::TruncFOp>(op))
          wide.push_back(op);
      });
      RewritePatternSet widePatterns(ctx);
      widePatterns.add<WideFMAToMacElem, WideTruncFToSRS>(ctx);
      if (failed(
              applyOpPatternsGreedily(wide, std::move(widePatterns), config)))
        return emitDefiniteFailure() << "failed to emit 32-lane aievec ops";
    }
#endif
  }
  results.set(llvm::cast<OpResult>(getResult()), targets);
  return DiagnosedSilenceableFailure::success();
}

LogicalResult transform::LinearizeVectorsOp::verify() {
  StringRef arch = getArch();
  if (arch != "aie2" && arch != "aie2p")
    return emitOpError() << "arch must be \"aie2\" or \"aie2p\"";
  int64_t lanes = getF32Lanes();
  if (lanes != 16 && lanes != 32 && lanes != 64)
    return emitOpError() << "f32_lanes must be 16, 32 or 64";
  if (lanes != 16 && arch != "aie2p")
    return emitOpError() << "f32_lanes = " << lanes
                         << " needs arch = \"aie2p\"";
  return success();
}
