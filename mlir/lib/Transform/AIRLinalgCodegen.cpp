//===- AIRLinalgCodegen.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRLinalgCodegen.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIR/AIRTransformOps.h"
#if AIR_ENABLE_AIE
#include "air/Transform/AIRDependencyScheduleOpt.h"
#endif
#include "air/Util/CostModel.h"
#include "air/Util/Outliner.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Utils/MemRefUtils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ValueBoundsOpInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"

#include <numeric>
#include <optional>

#define DEBUG_TYPE "air-linalg-codegen"

using namespace mlir;

namespace xilinx {
namespace air {

struct FoldSubViewOpsPattern : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    if (!dyn_cast_or_null<memref::SubViewOp>(op.getSource().getDefiningOp()))
      return failure();

    auto source_subview =
        cast<memref::SubViewOp>(op.getSource().getDefiningOp());

    // FIXME: do we still need this?
    // for (auto m : llvm::zip(source_subview.getType().getLayout(),
    //                         op.getType().getLayout()))
    //   if (std::get<0>(m) != std::get<1>(m))
    //     return failure();

    auto offsets = op.getOffsets().begin();
    auto source_offsets = source_subview.getOffsets().begin();
    SmallVector<Value, 4> result_offsets;

    auto static_offsets = op.getStaticOffsets();
    auto source_static_offsets = source_subview.getStaticOffsets();
    SmallVector<int64_t, 4> result_static_offsets;

    for (auto p : llvm::zip(static_offsets, source_static_offsets)) {
      auto op_offset = std::get<0>(p);
      auto source_offset = std::get<1>(p);
      if (op_offset >= 0 && source_offset >= 0) {
        result_static_offsets.push_back(op_offset + source_offset);
      } else if (op_offset < 0 && source_offset >= 0) {
        result_static_offsets.push_back(op_offset);
        if (source_offset == 0) {
          result_offsets.push_back(*offsets++);
        } else {
          Value a = *offsets++;
          Value b = arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                                   source_offset);
          result_offsets.push_back(
              arith::AddIOp::create(rewriter, op.getLoc(), a.getType(), a, b));
        }
      } else if (op_offset >= 0 && source_offset < 0) {
        result_static_offsets.push_back(source_offset);
        if (op_offset == 0) {
          result_offsets.push_back(*source_offsets++);
        } else {
          Value a = *source_offsets++;
          Value b =
              arith::ConstantIndexOp::create(rewriter, op.getLoc(), op_offset);
          result_offsets.push_back(
              arith::AddIOp::create(rewriter, op.getLoc(), a.getType(), a, b));
        }
      } else if (op_offset < 0 && source_offset < 0) {
        Value a = *source_offsets++;
        Value b = *offsets++;
        result_offsets.push_back(
            arith::AddIOp::create(rewriter, op.getLoc(), a.getType(), a, b));
        result_static_offsets.push_back(source_offset);
      }
    }

    rewriter.replaceOpWithNewOp<memref::SubViewOp>(
        op.getOperation(), op.getType(), source_subview.getSource(),
        result_offsets, op.getSizes(), op.getStrides(),
        rewriter.getDenseI64ArrayAttr(result_static_offsets),
        op.getStaticSizes(), op.getStaticStrides());

    return success();
  }
};

struct MemrefsPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {
    auto ty = op.getType();
    if (ty.hasStaticShape())
      return failure();

    std::vector<int64_t> shape = ty.getShape();
    if (op.getNumOperands() != shape.size())
      return failure();

    int dim = 0;
    for (auto oper : op.getOperands()) {
      if (auto c = oper.getDefiningOp<arith::ConstantIndexOp>())
        shape[dim] = c.value();
      else
        return failure();
      dim++;
    }
    Value newOp = rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op, MemRefType::get(shape, ty.getElementType(), nullptr,
                            ty.getMemorySpace()));
    for (auto use : newOp.getUsers()) {
      if (auto launch = dyn_cast_if_present<air::HerdOp>(use)) {
        for (unsigned int i = 0; i < launch.getNumKernelOperands(); i++) {
          auto arg = launch.getKernelArguments()[i];
          auto oper = launch.getKernelOperand(i);
          if (oper == newOp) {
            Block *b = arg.getOwner();
            auto new_arg = b->insertArgument(arg.getArgNumber(),
                                             newOp.getType(), newOp.getLoc());
            rewriter.setInsertionPointToStart(&*launch.getRegion().begin());
            arg.replaceAllUsesWith(memref::CastOp::create(
                rewriter, op.getLoc(), arg.getType(), new_arg));
            b->eraseArgument(arg.getArgNumber());
          }
        }
      }
    }
    return success();
  }
};

// struct DimPattern
//     : public OpRewritePattern<memref::DimOp> {
//   using OpRewritePattern<memref::DimOp>::OpRewritePattern;

//   LogicalResult matchAndRewrite(memref::DimOp op,
//                                 PatternRewriter &rewriter) const override {
//     auto operTy =
//     llvm::dyn_cast_if_present<ShapedType>(op.memrefOrTensor().getType()); if
//     (!operTy.hasStaticShape())
//       return failure();

//     auto indexOp = op.index().getDefiningOp<arith::ConstantIndexOp>();
//     if (!indexOp)
//       return failure();

//     rewriter.replaceOp(op, indexOp.getResult());
//     return success();
//   }
// };

// Replace a pattern like this:
// %7 = memref.alloc() : memref<20736xi8>
// %8 = memref.view %7[%c0][] : memref<20736xi8> to
// memref<1x16x18x18xf32> With this %7 = memref.alloc() : memref<
// 1x16x18x18xf32, 2>
struct RemoveSubViewOpsPattern : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  RemoveSubViewOpsPattern(MLIRContext *ctx, unsigned int fast_memory_space = 1);

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    auto view = op.getSource().getDefiningOp<memref::ViewOp>();
    if (!view)
      return failure();
    auto alloc = view.getSource().getDefiningOp<memref::AllocOp>();
    if (!alloc)
      return failure();

    // Resolve static shapes from constant size operands to avoid creating
    // dynamic allocs that later canonicalize into static alloc + memref.cast.
    // Note: getSizes() returns only dynamic size operands (one per `?` in the
    // shape), so we iterate over the full shape and consume from getSizes()
    // only for dynamic dimensions.
    SmallVector<int64_t> staticShape;
    SmallVector<Value> dynamicSizes;
    unsigned dynamicIdx = 0;
    for (int64_t dim : op.getType().getShape()) {
      if (ShapedType::isStatic(dim)) {
        staticShape.push_back(dim);
      } else {
        Value size = op.getSizes()[dynamicIdx++];
        APInt constVal;
        if (matchPattern(size, m_ConstantInt(&constVal))) {
          staticShape.push_back(constVal.getSExtValue());
        } else {
          staticShape.push_back(ShapedType::kDynamic);
          dynamicSizes.push_back(size);
        }
      }
    }

    /* Force memory space */
    Value newOp = rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op,
        MemRefType::get(staticShape, op.getType().getElementType(), AffineMap(),
                        rewriter.getI32IntegerAttr(fast_space)),
        dynamicSizes);
    alloc.replaceAllUsesWith(newOp);
    return success();
  }

private:
  unsigned int fast_space;
};

struct RemoveViewOpsPattern : public OpRewritePattern<memref::ViewOp> {
  using OpRewritePattern<memref::ViewOp>::OpRewritePattern;

  RemoveViewOpsPattern(MLIRContext *ctx, unsigned int fast_memory_space = 1);

  LogicalResult matchAndRewrite(memref::ViewOp op,
                                PatternRewriter &rewriter) const override {
    auto alloc = op.getSource().getDefiningOp<memref::AllocOp>();
    if (!alloc)
      return failure();

    // Resolve static shapes from constant size operands to avoid creating
    // dynamic allocs that later canonicalize into static alloc + memref.cast.
    // Note: getSizes() returns only dynamic size operands (one per `?` in the
    // shape), so we iterate over the full shape and consume from getSizes()
    // only for dynamic dimensions.
    SmallVector<int64_t> staticShape;
    SmallVector<Value> dynamicSizes;
    unsigned dynamicIdx = 0;
    for (int64_t dim : op.getType().getShape()) {
      if (ShapedType::isStatic(dim)) {
        staticShape.push_back(dim);
      } else {
        Value size = op.getSizes()[dynamicIdx++];
        APInt constVal;
        if (matchPattern(size, m_ConstantInt(&constVal))) {
          staticShape.push_back(constVal.getSExtValue());
        } else {
          staticShape.push_back(ShapedType::kDynamic);
          dynamicSizes.push_back(size);
        }
      }
    }

    /* Force memory space */
    Value newOp = rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op,
        MemRefType::get(staticShape, op.getType().getElementType(), AffineMap(),
                        rewriter.getI32IntegerAttr(fast_space)),
        dynamicSizes);
    alloc.replaceAllUsesWith(newOp);
    return success();
  }

private:
  unsigned int fast_space;
};

// Replace a pattern like this:
//  linalg.fill %a
//  ...
//  %b = alloc
//  copy %a -> %b
//  linalg.op with init tensor %b
//  ...
//  %c = alloc
//  copy %a -> %c
//  linalg.op with init tensor %c
//  ...
// with this:
//  %b = alloc
//  linalg.fill %b
//  linalg.op with init tensor %b
//  ...
//  %c = alloc
//  linalg.fill %c
//  linalg.op with init tensor %c
struct RemoveFillCopyLinalgPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {

    auto iter = copyOp->getIterator();
    auto linalgOp = dyn_cast_if_present<linalg::LinalgOp>(++iter);
    if (!linalgOp)
      return failure();

    Value copyOper0 = copyOp->getOperand(0);
    Value copyOper1 = copyOp->getOperand(1);

    auto allocOp =
        dyn_cast_if_present<memref::AllocOp>(copyOper0.getDefiningOp());
    if (!allocOp)
      return failure();

    iter = allocOp->getIterator();
    Operation *fillOp = dyn_cast_if_present<linalg::FillOp>(++iter);
    if (!fillOp)
      return failure();

    auto num_uses = 0;
    for (auto &u : copyOper0.getUses()) {
      (void)u;
      num_uses++;
    }

    IRMapping map;
    map.map(copyOper0, copyOper1);
    rewriter.clone(*fillOp, map);
    rewriter.eraseOp(copyOp);
    if (num_uses <= 2)
      rewriter.eraseOp(fillOp);

    return success();
  }
};

// Replace a pattern like this:
//  memref.copy %1, %2 : memref<?xi32> to memref<?xi32>
//  linalg op with write to %1, no use of %2
//  memref.copy %1, %2 : memref<?xi32> to memref<?xi32>
// with this:
//  linalg op with write to %1, no use of %2
//  memref.copy %1, %2 : memref<?xi32> to memref<?xi32>
struct RemoveDeadCopyPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {

    auto iter = op->getIterator();
    auto linalgOp = dyn_cast_if_present<linalg::LinalgOp>(++iter);
    if (!linalgOp)
      return failure();
    auto copyOp = dyn_cast_if_present<memref::CopyOp>(++iter);
    if (!copyOp)
      return failure();

    auto oper0 = copyOp->getOperand(0);
    auto oper1 = copyOp->getOperand(1);
    if (op.getOperand(0) != oper0)
      return failure();
    if (op.getOperand(1) != oper1)
      return failure();

    // no use of %2
    auto lopers = linalgOp->getOperands();
    if (std::find(lopers.begin(), lopers.end(), oper1) != lopers.end())
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

// Replace a pattern like this:
//  %alloc_1 = memref.alloc() : memref<?x?xi32, 2>
//  ...
//  memref.copy %alloc_1, %m : memref<?x?xi32, 2> to memref<?x?xi32, 2>
//  memref.dealloc %alloc_1 : memref<?x?xi32, 2>
//  %alloc_2 = memref.alloc() : memref<?x?xi32, 2>
//  memref.copy %m, %alloc_2 : memref<?x?xi32, 2> to memref<?x?xi32, 2>
// with this:
//  %alloc_1 = memref.alloc() : memref<?x?xi32, 2>
//  ...
//  memref.copy %alloc_1, %m : memref<?x?xi32, 2> to memref<?x?xi32, 2>
//  [ and replace uses of %alloc_2 with %alloc_1 ]
struct RemoveExtraAllocPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {

    auto existingAlloc =
        dyn_cast_if_present<memref::AllocOp>(op.getOperand(0).getDefiningOp());
    if (!existingAlloc)
      return failure();

    auto iter = op->getIterator();
    auto deallocOp = dyn_cast_if_present<memref::DeallocOp>(++iter);
    if (!deallocOp)
      return failure();

    auto allocOp = dyn_cast_if_present<memref::AllocOp>(++iter);
    if (!allocOp)
      return failure();
    if (allocOp.getType() != existingAlloc.getType())
      return failure();

    auto copyOp = dyn_cast_if_present<memref::CopyOp>(++iter);
    if (!copyOp)
      return failure();

    if (op.getOperand(0) != deallocOp.getOperand())
      return failure();

    if (op.getOperand(1) != copyOp.getOperand(0))
      return failure();

    if (allocOp.getResult() != copyOp.getOperand(1))
      return failure();

    rewriter.replaceAllUsesWith(allocOp.getResult(), {op.getOperand(0)});
    rewriter.eraseOp(copyOp);
    rewriter.eraseOp(allocOp);
    rewriter.eraseOp(deallocOp);

    return success();
  }
};

// Replace a pattern like this:
//  %0 = memref.alloc() : memref<4096xi32>
//  linalg.generic with outs(%0 : memref<4096xi32>), does not read %0
//  %1 = memref.cast %0 : memref<4096xi32> to memref<?xi32>
//  memref.copy %1, %2 : memref<?xi32> to memref<?xi32>
// with this:
//  %1 = memref.cast %2 : memref<?xi32> to memref<4096xi32>
//  linalg.generic with outs(%1 : memref<4096xi32>)
struct RemoveAllocLinalgOpCopyPattern
    : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern<memref::AllocOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp op,
                                PatternRewriter &rewriter) const override {

    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    Operation *castOp = nullptr;
    Operation *linalgOp = nullptr;
    Operation *copyOp = nullptr;
    for (auto &u : op->getUses())
      if (auto c = dyn_cast_if_present<memref::CastOp>(u.getOwner()))
        castOp = c;
      else if (auto c = dyn_cast_if_present<memref::CopyOp>(u.getOwner())) {
        if (u.getOperandNumber() == 0)
          copyOp = c;
        else {
          if (auto l = dyn_cast_if_present<linalg::LinalgOp>(u.getOwner())) {
            linalgOp = l;
            if (l.isInitTensor(&u))
              return failure();
          }
        }
      }
    if (castOp && copyOp)
      return failure();

    if (!(castOp || copyOp) || !linalgOp)
      return failure();

    if (!copyOp) {
      if (!castOp->hasOneUse())
        return failure();
      copyOp = dyn_cast_if_present<memref::CopyOp>(*castOp->user_begin());
      if (!copyOp)
        return failure();
    }
    auto copyOperand = copyOp->getOperand(1);
    rewriter.setInsertionPointAfter(copyOperand.getDefiningOp());
    auto newOp = memref::CastOp::create(rewriter, op->getLoc(), op.getType(),
                                        copyOperand);
    rewriter.replaceOp(op, newOp->getResults());
    rewriter.eraseOp(copyOp);

    return success();
  }
};

// Replace a pattern like this:
//  %sv = memref.subview ...
//  %alloc = memref.alloc() : memref<...>
//  memref.copy %sv, %alloc
//  linalg.generic with outs(%alloc : memref<...>), does not read %alloc
// with this:
//  %sv = memref.subview ...
//  %alloc = memref.alloc() : memref<...>
//  linalg.generic with outs(%alloc : memref<...>), does not read %alloc
// that is, remove the no-op copy.
struct RemoveAllocCopyLinalgOpCopyPattern
    : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp op,
                                PatternRewriter &rewriter) const override {

    // the target of the copy is an alloc
    Operation *allocOp =
        dyn_cast_if_present<memref::AllocOp>(op.getTarget().getDefiningOp());
    if (!allocOp)
      return failure();

    // find the next linalg use in this block
    linalg::LinalgOp linalgOp = nullptr;
    for (auto &u : allocOp->getResult(0).getUses()) {
      if (auto l = dyn_cast_if_present<linalg::LinalgOp>(u.getOwner())) {
        // bail without trying to resolve the ordering
        // if there's a linalg use in a different block
        if (l->getBlock() != op->getBlock())
          return failure();
        if (l.payloadUsesValueFromOperand(&u))
          continue;
        // take the earliest use
        if (linalgOp && linalgOp->isBeforeInBlock(l))
          continue;
        linalgOp = l;
      }
    }
    if (!linalgOp)
      return failure();

    for (auto &u : allocOp->getResult(0).getUses()) {
      auto use = u.getOwner();
      if (use == op)
        continue;
      // if there's a use between the copy and the linalg op
      if (!isa<linalg::LinalgOp>(use)) {
        if (use->getBlock() != op->getBlock())
          continue;
        if (use->isBeforeInBlock(linalgOp))
          return failure();
      }
    }

    rewriter.eraseOp(op);
    return success();
  }
};

// Loop invarient code motion pass doesn't move kernel op partial result
// (linalg.copy) memcpy outside loop this patternMatch transform perform kernel
// op partial result accumulation modeling. This transformation model mllib
// kernel library.
//
// Replace a pattern like this:
// scf.for %arg6 = %c0 to %c64 step %c16 {
//  %6 = memref.subview %arg1[0, %4, %arg6, %5] [1, 32, 16, 16] [1, 1, 1, 1] :
//  memref<1x128x64x64xf32> to memref<1x32x16x16xf32, #map1> scf.for %ag7 = %c0
//  to %c64 step %c16 {
//    %7 = memref.subview %2[0, %arg7, %arg6, %5] [1, 16, 18, 18] [1, 1, 1, 1] :
//    memref<1x64x66x66xf32> to memref<1x16x18x18xf32, #map2> %8 =
//    memref.subview %0[%4, %arg7, 0, 0] [32, 16, 3, 3] [1, 1, 1, 1] :
//    memref<128x64x3x3xf32> to memref<32x16x3x3xf32, #map3> %9 = memref.alloc()
//    : memref<1x16x18x18xf32, 2> %10 = memref.alloc() : memref<32x16x3x3xf32,
//    2> %11 = memref.alloc() : memref<1x32x16x16xf32, 2> linalg.copy(%7, %9) :
//    memref<1x16x18x18xf32, #map2>, memref<1x16x18x18xf32, 2> linalg.copy(%8,
//    %10) : memref<32x16x3x3xf32, #map3>, memref<32x16x3x3xf32, 2>
//    linalg.copy(%6, %11) : memref<1x32x16x16xf32, #map1>,
//    memref<1x32x16x16xf32, 2> linalg.conv_2d_nchw_fchw {dilations = dense<1> :
//    vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%9, %10 :
//    memref<1x16x18x18xf32, 2>, memref<32x16x3x3xf32, 2>) outs(%11 :
//    memref<1x32x16x16xf32, 2>) linalg.copy(%11, %6) : memref<1x32x16x16xf32,
//    2>, memref<1x32x16x16xf32, #map1> memref.dealloc %9 :
//    memref<1x16x18x18xf32, 2> memref.dealloc %10 : memref<32x16x3x3xf32, 2>
//    memref.dealloc %11 : memref<1x32x16x16xf32, 2>
//  }
//}
// with this:
// scf.for %arg6 = %c0 to %c64 step %c16 {
//  %6 = memref.subview %arg1[0, %4, %arg6, %5] [1, 32, 16, 16] [1, 1, 1, 1] :
//  memref<1x128x64x64xf32> to memref<1x32x16x16xf32, #map1> %11 =
//  memref.alloc() : memref<1x32x16x16xf32, 2> linalg.copy(%6, %11) :
//  memref<1x32x16x16xf32, #map1>, memref<1x32x16x16xf32, 2> scf.for %ag7 = %c0
//  to %c64 step %c16 {
//    %7 = memref.subview %2[0, %arg7, %arg6, %5] [1, 16, 18, 18] [1, 1, 1, 1] :
//    memref<1x64x66x66xf32> to memref<1x16x18x18xf32, #map2> %8 =
//    memref.subview %0[%4, %arg7, 0, 0] [32, 16, 3, 3] [1, 1, 1, 1] :
//    memref<128x64x3x3xf32> to memref<32x16x3x3xf32, #map3> %9 = memref.alloc()
//    : memref<1x16x18x18xf32, 2> %10 = memref.alloc() : memref<32x16x3x3xf32,
//    2> linalg.copy(%7, %9) : memref<1x16x18x18xf32, #map2>,
//    memref<1x16x18x18xf32, 2> linalg.copy(%8, %10) : memref<32x16x3x3xf32,
//    #map3>, memref<32x16x3x3xf32, 2> linalg.conv_2d_nchw_fchw {dilations =
//    dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%9, %10
//    : memref<1x16x18x18xf32, 2>, memref<32x16x3x3xf32, 2>) outs(%11 :
//    memref<1x32x16x16xf32, 2>) memref.dealloc %9 : memref<1x16x18x18xf32, 2>
//    memref.dealloc %10 : memref<32x16x3x3xf32, 2>
//  }
//  linalg.copy(%11, %6) : memref<1x32x16x16xf32, 2>, memref<1x32x16x16xf32,
//  #map1> memref.dealloc %11 : memref<1x32x16x16xf32, 2>
//}

struct ConvertMemrefCopyToLinalgCopyPattern
    : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value source = copyOp.getSource();
    Value target = copyOp.getTarget();

    // Create linalg.copy operation
    rewriter.replaceOpWithNewOp<linalg::CopyOp>(copyOp, source, target);

    return success();
  }
};

// Eliminate intermediate memref in cascaded DMA operations
// Replace a pattern like this:
//  air.dma_memcpy_nd (%intermediate[] [] [], %source[] [] []) : (memref<...>,
//  memref<...>) air.dma_memcpy_nd (%dest[] [] [], %intermediate[] [] []) :
//  (memref<...>, memref<...>)
// where %intermediate is only used by these two operations and has default
// access patterns with this:
//  air.dma_memcpy_nd (%dest[] [] [], %source[] [] []) : (memref<...>,
//  memref<...>)
struct EliminateIntermediateMemrefPattern
    : public OpRewritePattern<air::DmaMemcpyNdOp> {
  using OpRewritePattern<air::DmaMemcpyNdOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(air::DmaMemcpyNdOp firstMemcpy,
                                PatternRewriter &rewriter) const override {

    // Get the destination of the first memcpy (potential intermediate buffer)
    Value intermediate = firstMemcpy.getDstMemref();

    // Check if the intermediate buffer has exactly two uses
    if (std::distance(intermediate.use_begin(), intermediate.use_end()) != 2)
      return failure();

    // Find the second memcpy that uses the intermediate buffer as source
    air::DmaMemcpyNdOp secondMemcpy = nullptr;
    for (auto user : intermediate.getUsers()) {
      auto memcpyOp = dyn_cast_if_present<air::DmaMemcpyNdOp>(user);
      if (!memcpyOp)
        continue;
      if (memcpyOp.getSrcMemref() != intermediate)
        continue;
      if (air::opOrAncestorIsDominantOver(memcpyOp, firstMemcpy))
        continue;
      secondMemcpy = memcpyOp;
      break;
    }

    if (!secondMemcpy)
      return failure();

    // Check that both operations have default access patterns using existing
    // utility
    auto isDefaultAccess = [](ArrayRef<OpFoldResult> offsets,
                              ArrayRef<OpFoldResult> sizes,
                              ArrayRef<OpFoldResult> strides) {
      return offsets.empty() && sizes.empty() && strides.empty();
    };
    if (!isDefaultAccess(firstMemcpy.getMixedDstOffsets(),
                         firstMemcpy.getMixedDstSizes(),
                         firstMemcpy.getMixedDstStrides()) ||
        !isDefaultAccess(secondMemcpy.getMixedSrcOffsets(),
                         secondMemcpy.getMixedSrcSizes(),
                         secondMemcpy.getMixedSrcStrides()))
      return failure();
    if (firstMemcpy.getDstMemref() != secondMemcpy.getSrcMemref())
      return failure();

    // Create a new memcpy that directly copies from the source of the first
    // memcpy to the destination of the second memcpy
    rewriter.setInsertionPoint(firstMemcpy);

    SmallVector<Value> emptyOffsets, emptySizes, emptyStrides;
    SmallVector<Type> emptyTypes;

    if (!firstMemcpy.getAsyncDependencies().empty()) {
      emptyTypes.push_back(air::AsyncTokenType::get(rewriter.getContext()));
    }

    auto newMemcpy = air::DmaMemcpyNdOp::create(
        rewriter, firstMemcpy.getLoc(),
        emptyTypes,                         // result types
        firstMemcpy.getAsyncDependencies(), // async dependencies
        secondMemcpy.getDstMemref(),        // destination from second memcpy
        emptyOffsets, emptySizes, emptyStrides, // dst access pattern
        firstMemcpy.getSrcMemref(),             // source from first memcpy
        emptyOffsets, emptySizes, emptyStrides, // src access pattern
        /*pad_before=*/nullptr, /*pad_after=*/nullptr);

    // Replace the async token of the second memcpy with the new one if needed
    if (secondMemcpy.getAsyncToken() && newMemcpy.getAsyncToken()) {
      if (!secondMemcpy.getAsyncToken().use_empty()) {
        secondMemcpy.getAsyncToken().replaceAllUsesWith(
            newMemcpy.getAsyncToken());
      }
    }

    // Erase both original memcpy operations
    rewriter.eraseOp(secondMemcpy);
    rewriter.eraseOp(firstMemcpy);

    return success();
  }
};

struct HoistReduceBufferPattern : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern<linalg::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::CopyOp op,
                                PatternRewriter &rewriter) const override {

    Operation *DeallocOp = nullptr;
    Operation *kernelOp = nullptr;
    Operation *subviewOp = nullptr;
    Operation *memrefAllocaOp = nullptr;
    Operation *scfForOp = nullptr;
    Operation *otherCpy = nullptr;

    // Find linalg.copy input uses that has following pattern
    //  a. Uses in kernel op (conv2d, fused_ops etc)
    //  b. Uses in linalg.copy. 1_linalg.copy.input == 2_linalg.copy.output &&
    //  1_linalg.copy.output == 2_linalg.copy.input
    //     and 1_linalg.copy & 2_linalg.copy & kernel op are in same region
    //  c. Uses in Dealloc op && defining op is memref.alloc()
    //  d. 2_linalg.copy input defining op is memref.subview & memref.subview
    //  and kernel op are not in same region.
    //     loop invarient code motion move this subview op outside loop.
    //  e. kernel parent op is a scf.for op
    //  After matching those pattern, memref.alloc() & 2_linalg.copy move before
    //  inner scf.for loop and 1_linalg.copy & memref.dealloc() move after inner
    //  scf.for loop.
    for (Operation *userOp : op.getInputs()[0].getUsers()) {
      if (isa<linalg::CopyOp>(userOp))
        otherCpy = userOp;
      else if (isa<linalg::LinalgOp>(userOp))
        kernelOp = userOp;
      else if (isa<memref::DeallocOp>(userOp))
        DeallocOp = userOp;
      else
        ; // return failure();
    }

    // a. Uses in kernel op (conv2d, fused_ops etc)
    // b. Uses in linalg.copy
    // c. Uses in Dealloc op
    if (!kernelOp || !DeallocOp || !otherCpy)
      return failure();

    auto linalgCpy = dyn_cast_if_present<linalg::CopyOp>(otherCpy);

    // d. 2_linalg.copy input defining op is memref.subview
    if (isa<memref::SubViewOp>(linalgCpy.getInputs()[0].getDefiningOp()))
      subviewOp = linalgCpy.getInputs()[0].getDefiningOp();
    else
      return failure();

    // 1_linalg.copy.input == 2_linalg.copy.output && 1_linalg.copy.output ==
    // 2_linalg.copy.input
    if (op.getInputs()[0] != linalgCpy.getOutputs()[0] ||
        op.getOutputs()[0] != linalgCpy.getInputs()[0])
      return failure();

    // 1_linalg.copy & 2_linalg.copy & kernel op are in same region
    if (kernelOp->getParentRegion() != linalgCpy->getParentRegion() ||
        kernelOp->getParentRegion() != op->getParentRegion())
      return failure();

    // c. defining op is memref.alloc()
    if (isa<memref::AllocOp>(linalgCpy.getOutputs()[0].getDefiningOp()))
      memrefAllocaOp = linalgCpy.getOutputs()[0].getDefiningOp();
    else
      return failure();

    // e. kernel parent op is a scf.for op
    if (isa<scf::ForOp>(kernelOp->getParentOp()))
      scfForOp = kernelOp->getParentOp();
    else
      return failure();

    // memref.subview and kernel op are not in same region.
    // loop invarient code motion move this subview op outside loop.
    if (scfForOp->getParentRegion() != subviewOp->getParentRegion())
      return failure();

    // hoist alloc and copy in
    memrefAllocaOp->moveBefore(scfForOp);
    linalgCpy->moveBefore(scfForOp);

    // hoist copy out and dealloc
    DeallocOp->moveAfter(scfForOp);
    op->moveAfter(scfForOp);

    return success();
  }
};

struct LinalgTransformationFilter {
  using FilterFunction = std::function<LogicalResult(Operation *)>;

  explicit LinalgTransformationFilter(
      ArrayRef<StringAttr> matchDisjunction = {},
      std::optional<StringAttr> replacement = std::nullopt);

  LinalgTransformationFilter(LinalgTransformationFilter &&) = default;
  LinalgTransformationFilter(const LinalgTransformationFilter &) = default;
  LogicalResult checkAndNotify(PatternRewriter &rewriter, Operation *op) const;
  void replaceLinalgTransformationFilter(PatternRewriter &rewriter,
                                         Operation *op) const;

  LinalgTransformationFilter &addFilter(const FilterFunction &f) {
    if (f)
      filters.push_back(f);
    return *this;
  }

  template <typename... OpTypes>
  LinalgTransformationFilter &addOpFilter() {
    return addFilter(
        [](Operation *op) { return success(isa<OpTypes...>(op)); });
  }

  LinalgTransformationFilter &addOpNameFilter(StringRef opName) {
    return addFilter([opName](Operation *op) {
      return success(op->getName().getStringRef() == opName);
    });
  }

  LinalgTransformationFilter &setMatchByDefault() {
    matchByDefault = true;
    return *this;
  }

private:
  SmallVector<FilterFunction> filters;
  SmallVector<StringAttr> matchDisjunction;
  std::optional<StringAttr> replacement;
  /// When set to true, if the attribute is not set, it will be treated as
  /// a match. Default is false.
  bool matchByDefault;
};

LinalgTransformationFilter::LinalgTransformationFilter(
    ArrayRef<StringAttr> matchDisjunction,
    std::optional<StringAttr> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement), matchByDefault(false) {}

LogicalResult
LinalgTransformationFilter::checkAndNotify(PatternRewriter &rewriter,
                                           Operation *op) const {
  if (llvm::any_of(filters,
                   [&](const FilterFunction &f) { return failed(f(op)); }))
    return failure();

  auto attr = op->template getAttrOfType<StringAttr>(
      air::LinalgTransforms::kLinalgTransformMarker);

  if (!attr) {
    // 1. Has no filter case and matchDisjunction is empty.
    if (matchDisjunction.empty() || matchByDefault)
      return success();

    // 2. Has no filter but was expecting a filter.
    return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
      diag << " does not have any filter from list: ";
      interleaveComma(matchDisjunction, diag);
    });
  }

  // 4. Match explicit filter.
  for (auto filter : matchDisjunction)
    if (attr.getValue() == filter)
      return success();

  // 5. Fail to match.
  return rewriter.notifyMatchFailure(op, [&](Diagnostic &diag) {
    diag << " does not have any filter from list: ";
    interleaveComma(matchDisjunction, diag);
  });
}

void LinalgTransformationFilter::replaceLinalgTransformationFilter(
    PatternRewriter &rewriter, Operation *op) const {
  if (replacement)
    op->setAttr(air::LinalgTransforms::kLinalgTransformMarker, *replacement);
  else
    op->removeAttr(
        rewriter.getStringAttr(air::LinalgTransforms::kLinalgTransformMarker));
}

RemoveSubViewOpsPattern::RemoveSubViewOpsPattern(MLIRContext *ctx,
                                                 unsigned int fast_memory_space)
    : OpRewritePattern(ctx), fast_space(fast_memory_space) {}

RemoveViewOpsPattern::RemoveViewOpsPattern(MLIRContext *ctx,
                                           unsigned int fast_memory_space)
    : OpRewritePattern(ctx), fast_space(fast_memory_space) {}

// Custom LinalgOp tiling pattern
struct TileLinalgOpPattern : public RewritePattern {
  TileLinalgOpPattern(
      StringLiteral operation_name, MLIRContext *context,
      linalg::LinalgTilingOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : RewritePattern(operation_name, benefit, context), filter(filter),
        options(options) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    linalg::LinalgOp linalgOp = dyn_cast_if_present<linalg::LinalgOp>(op);
    if (!linalgOp)
      return failure();
    if (failed(filter.checkAndNotify(rewriter, linalgOp)))
      return failure();

    std::optional<linalg::TiledLinalgOp> tiledLinalgOp =
        tileLinalgOp(rewriter, linalgOp, options);
    if (!tiledLinalgOp)
      return failure();

    filter.replaceLinalgTransformationFilter(rewriter, tiledLinalgOp->op);

    if (tiledLinalgOp->tensorResults.empty())
      rewriter.eraseOp(op);
    else
      rewriter.replaceOp(op, tiledLinalgOp->tensorResults);

    return success();
  }

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// Options to control tiling;
  linalg::LinalgTilingOptions options;
};

// Temporary custom LinalgOp promotion pattern,
// copied from mlir before 5a001136
//
struct PromoteLinalgOpPattern : public RewritePattern {
  PromoteLinalgOpPattern(
      MLIRContext *context, linalg::LinalgPromotionOptions options,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context), filter(filter),
        options(std::move(options)) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {

    if (failed(filter.checkAndNotify(rewriter, op)))
      return failure();
    if (failed(promoteSubviewsPrecondition(op, options)))
      return failure();

    // TODO: We cannot use root update here. This pattern is creating other ops,
    // so if the promotion fails, those need to be cleaned up, which doesnt seem
    // to be happening here. So to fail properly, we should be cloning the op
    // and deleting the previous op. This needs more investigation.
    rewriter.startOpModification(op);
    std::optional<linalg::LinalgOp> promotedOp =
        promoteSubViews(rewriter, cast<linalg::LinalgOp>(op), options);
    if (!promotedOp) {
      rewriter.cancelOpModification(op);
      return op->emitError("subview promotion failed");
    }
    rewriter.finalizeOpModification(op);
    filter.replaceLinalgTransformationFilter(rewriter, op);
    return success();
  }

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// Options to control promotion
  linalg::LinalgPromotionOptions options;
};

static std::optional<Value>
allocBufferCallBack(OpBuilder &b, memref::SubViewOp subView,
                    ArrayRef<Value> boundingSubViewSize, DataLayout &layout) {
  MemRefType viewType = subView.getType();
  MemRefType allocType = MemRefType::get(
      viewType.getShape(), viewType.getElementType(), AffineMap(),
      air::MemorySpaceAttr::get(b.getContext(), air::MemorySpace::L1));
  Value buffer = b.createOrFold<memref::AllocOp>(subView.getLoc(), allocType);
  return buffer;
}

static LogicalResult deallocBufferCallBack(OpBuilder &b, Value buffer) {
  // memref::DeallocOp::create(b, buffer.getLoc(), buffer);
  return success();
}

// Create channel name as string
static std::string createChannelName(ModuleOp module) {
  std::string new_cname = "channel_0";
  std::string cname = "channel";
  int which_try = 0;
  while (module.lookupSymbol(new_cname))
    new_cname = cname + "_" + std::to_string(++which_try);
  cname = new_cname;
  return cname;
}

// Split a linalg reduction into 'pipeline_depth' consecutive
// stages, each one feeding partial reductions to the next stage.
// Stages are mapped to Nx1 or Nx1 herd.
FailureOr<linalg::TiledLinalgOp> static pipelineReduceLinalgOp(
    RewriterBase &b, linalg::LinalgOp op, ArrayRef<int64_t> static_tile_sizes,
    unsigned int pipeline_depth, std::string pipeline_direction, bool promote) {

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  auto loc = op.getLoc();
  auto ctx = op.getContext();

  if (!(pipeline_direction == "vert" || pipeline_direction == "horiz"))
    return failure();

  auto iteratorTypes = op.getIteratorTypesArray();
  if (linalg::isParallelIterator(iteratorTypes.back()))
    return failure();

  bool isHoriz = pipeline_direction == "horiz";
  int new_herd_x = isHoriz ? pipeline_depth : 1;
  int new_herd_y = !isHoriz ? pipeline_depth : 1;

  SmallVector<Value, 2> dims{
      arith::ConstantIndexOp::create(b, loc, new_herd_x),
      arith::ConstantIndexOp::create(b, loc, new_herd_y)};

  SmallVector<Value, 4> args;
  for (auto o : op->getOperands())
    args.push_back(o);

  auto herd = air::HerdOp::create(b, loc, dims, args);
  b.setInsertionPointToStart(&herd.getBody().front());

  Value x = herd.getIds()[0];
  Value y = herd.getIds()[1];

  auto nLoops = op.getNumLoops();
  auto tileSizes = static_tile_sizes.take_front(nLoops);

  SmallVector<OpFoldResult, 4> tileSizeVector;
  for (auto s : tileSizes)
    tileSizeVector.push_back(
        arith::ConstantIndexOp::create(b, loc, s).getResult());
  if (tileSizeVector.size() < nLoops) {
    auto zero = arith::ConstantIndexOp::create(b, loc, 0);
    tileSizeVector.append(nLoops - tileSizeVector.size(), zero.getResult());
  }

  auto allShapeSizes = op.createFlatListOfOperandDims(b, loc);
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap)
    return failure();
  SmallVector<OpFoldResult> sizeBounds =
      affine::makeComposedFoldedMultiResultAffineApply(
          b, loc, shapeSizesToLoopsMap, allShapeSizes);

  SmallVector<OpFoldResult> tileIds;
  for (auto s : tileSizes) {
    if (s == 0)
      continue;
    AffineExpr d0 = b.getAffineDimExpr(0);
    auto map = AffineMap::get(1, 0, d0 * s);
    tileIds.push_back(affine::AffineApplyOp::create(b, loc, map,
                                                    isHoriz ? herd.getIds()[0]
                                                            : herd.getIds()[1])
                          .getResult());
  }
  SmallVector<Value, 4> tiledOperands = linalg::makeTiledShapes(
      b, loc, op, args, tileIds, tileSizeVector, sizeBounds, true);

  unsigned int resultIdx = 0;
  for (OpOperand &opOperand : op->getOpOperands())
    if (op.isDpsInit(&opOperand)) {
      resultIdx = opOperand.getOperandNumber();
      break;
    }

  Value firstOutputOperand = tiledOperands[resultIdx];
  SmallVector<air::ChannelOp> channels(pipeline_depth, nullptr);
  for (unsigned int i = 0; i < pipeline_depth; i++) {
    OpBuilder::InsertionGuard pipeline_guard(b);
    bool last_stage = i == pipeline_depth - 1;
    bool first_stage = i == 0;

    SmallVector<AffineExpr, 2> constraints{
        getAffineDimExpr(isHoriz ? 0 : 1, ctx) - getAffineConstantExpr(i, ctx),
        getAffineDimExpr(isHoriz ? 1 : 0, ctx)};
    SmallVector<bool, 2> eqflags{true, false};
    auto int_set = IntegerSet::get(2, 0, constraints, eqflags);
    SmallVector<Value, 2> int_set_args{x, y};
    affine::AffineIfOp aif = affine::AffineIfOp::create(
        b, op->getLoc(), int_set, int_set_args, false);

    Block *stageBlock = aif.getBody();
    b.setInsertionPointToStart(stageBlock);

    if (i) {
      auto ty = llvm::cast<MemRefType>(tiledOperands[resultIdx].getType());
      auto alloc = memref::AllocOp::create(
          b, loc,
          MemRefType::get(
              ty.getShape(), ty.getElementType(), AffineMap(),
              air::MemorySpaceAttr::get(b.getContext(), air::MemorySpace::L1)));
      tiledOperands[resultIdx] = alloc.getResult();
      SmallVector<Value> src_offsets;
      SmallVector<Value> src_sizes;
      SmallVector<Value> src_strides;
      SmallVector<Value> channel_idx;
      SmallVector<Value> deps;
      SmallVector<Type> tys;
      air::ChannelGetOp::create(b, loc, tys, deps, channels[i - 1].getSymName(),
                                channel_idx, tiledOperands[resultIdx],
                                src_offsets, src_sizes, src_strides,
                                /*pad_before=*/nullptr, /*pad_after=*/nullptr);
    }

    linalg::LinalgOp linalgOp = clone(b, op, {}, tiledOperands);

    auto defaultCopyCallBack = [loc](OpBuilder &bldr, Value src,
                                     Value dst) -> LogicalResult {
      memref::CopyOp::create(bldr, loc, src, dst);
      return success();
    };

    if (promote) {
      SmallVector<int64_t, 3> opers_to_promote(linalgOp->getNumOperands() - 1);
      std::iota(opers_to_promote.begin(), opers_to_promote.end(), 0);
      if (first_stage /* || last_stage*/)
        opers_to_promote.push_back(linalgOp->getNumOperands() - 1);

      auto emptyCopyCallBack = [](OpBuilder &bldr, Value src,
                                  Value dst) -> LogicalResult {
        return success();
      };
      b.setInsertionPoint(linalgOp);
      auto options = linalg::LinalgPromotionOptions()
                         .setOperandsToPromote(opers_to_promote)
                         .setAllocationDeallocationFns(allocBufferCallBack,
                                                       deallocBufferCallBack);
      if (first_stage)
        options.setCopyInOutFns(defaultCopyCallBack, emptyCopyCallBack);
      auto res = linalg::promoteSubViews(b, linalgOp, options);
      if (failed(res))
        return failure();
    }

    if (last_stage) {
      b.setInsertionPointAfter(linalgOp);
      (void)defaultCopyCallBack(b, tiledOperands[resultIdx],
                                firstOutputOperand);
      b.setInsertionPoint(stageBlock->getTerminator());
    } else {
      auto mref = tiledOperands[resultIdx];
      if (promote && first_stage) {
        memref::SubViewOp sv = dyn_cast_if_present<memref::SubViewOp>(
            linalgOp.getDpsInitOperand(0)->get().getDefiningOp());
        mref = sv.getSource();
        sv.replaceAllUsesWith(mref);
      }

      auto module = op->getParentOfType<ModuleOp>();
      auto cname = createChannelName(module);
      b.setInsertionPointToStart(module.getBody());
      auto channel_op =
          air::ChannelOp::create(b, loc, cname, b.getI64ArrayAttr({1}),
                                 b.getStringAttr("npu_dma_stream"));
      b.setInsertionPoint(stageBlock->getTerminator());
      SmallVector<Value> src_offsets;
      SmallVector<Value> src_sizes;
      SmallVector<Value> src_strides;
      SmallVector<Value> channel_idx;
      SmallVector<Value> deps;
      SmallVector<Type> tys;
      air::ChannelPutOp::create(b, loc, tys, deps,
                                FlatSymbolRefAttr::get(ctx, cname), channel_idx,
                                mref, src_offsets, src_sizes, src_strides,
                                /*pad_before=*/nullptr, /*pad_after=*/nullptr);
      channels[i] = channel_op;
    }
    // if (erased) erased.erase();
  }

  int i = 0;
  for (auto a : args) {
    replaceAllUsesInRegionWith(a, herd.getKernelArgument(i++), herd.getBody());
  }
  return linalg::TiledLinalgOp{op, {herd}, {}};
}

struct PipelineReducePattern : public RewritePattern {
  PipelineReducePattern(
      MLIRContext *context, linalg::LinalgTilingOptions options,
      ArrayRef<int64_t> tile_size, int pipeline_depth,
      std::string &pipeline_direction, bool promote,
      LinalgTransformationFilter filter = LinalgTransformationFilter(),
      PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context), filter(filter),
        options(options), tile_size(tile_size), pipeline_depth(pipeline_depth),
        pipeline_direction(pipeline_direction), promote(promote) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    linalg::LinalgOp linalgOp = dyn_cast_if_present<linalg::LinalgOp>(op);
    if (!linalgOp)
      return failure();

    if (failed(filter.checkAndNotify(rewriter, linalgOp)))
      return failure();

    if (op->getParentOfType<air::HerdOp>())
      return failure();

    auto result =
        pipelineReduceLinalgOp(rewriter, linalgOp, tile_size, pipeline_depth,
                               pipeline_direction, promote);

    if (failed(result))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  LinalgTransformationFilter filter;
  /// Options to control tiling;
  linalg::LinalgTilingOptions options;
  SmallVector<int64_t, 4> tile_size;
  unsigned int pipeline_depth;
  std::string pipeline_direction;
  bool promote;
};

class AIRPipelineReducePass
    : public air::impl::AIRPipelineReducePassBase<AIRPipelineReducePass> {

public:
  AIRPipelineReducePass() = default;
  AIRPipelineReducePass(const AIRPipelineReducePass &pass){};

  void runOnOperation() override;

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<air::airDialect, bufferization::BufferizationDialect>();
  }

private:
};

void AIRPipelineReducePass::runOnOperation() {
  auto func = getOperation();
  auto ctx = func.getContext();
  RewritePatternSet patterns(ctx);
  SmallVector<int64_t, 4> sizes;
  for (auto &s : clTileSize)
    sizes.push_back(s);

  patterns.add<PipelineReducePattern>(ctx, linalg::LinalgTilingOptions(), sizes,
                                      clPipelineDepth, clPipelineDirection,
                                      clPromoteSubViews);

  (void)applyPatternsGreedily(func, std::move(patterns));
}
class AIRLinalgCodegen
    : public air::impl::AIRLinalgCodegenBase<AIRLinalgCodegen> {

public:
  AIRLinalgCodegen() = default;
  AIRLinalgCodegen(const AIRLinalgCodegen &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect, memref::MemRefDialect,
                    linalg::LinalgDialect, scf::SCFDialect, air::airDialect,
                    func::FuncDialect>();
  }

  void runTestPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<RemoveSubViewOpsPattern, FoldSubViewOpsPattern,
                    RemoveViewOpsPattern, HoistReduceBufferPattern,
                    RemoveAllocLinalgOpCopyPattern, RemoveExtraAllocPattern,
                    RemoveAllocCopyLinalgOpCopyPattern, RemoveDeadCopyPattern,
                    RemoveFillCopyLinalgPattern,
                    EliminateIntermediateMemrefPattern>(ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));
  }

  /// Collect perfectly nested loops starting from `rootForOps`.  Loops are
  /// perfectly nested if each loop is the first and only non-terminator
  /// operation in the parent loop.  Collect at most `maxLoops` loops and append
  /// them to `forOps`.
  template <typename T>
  static void getPerfectlyNestedLoopsImpl(
      SmallVectorImpl<T> &forOps, T rootForOp,
      unsigned maxLoops = std::numeric_limits<unsigned>::max()) {
    for (unsigned i = 0; i < maxLoops; ++i) {
      forOps.push_back(rootForOp);
      Block &body = rootForOp.getRegion().front();
      if (body.begin() != std::prev(body.end(), 2))
        return;

      rootForOp = dyn_cast_if_present<T>(&body.front());
      if (!rootForOp)
        return;
    }
  }

  void getPerfectlyNestedLoops(SmallVectorImpl<scf::ForOp> &nestedLoops,
                               scf::ForOp root) {
    getPerfectlyNestedLoopsImpl(nestedLoops, root);
  }

  static SmallVector<int64_t> getTripCounts(linalg::LinalgOp op) {

    SmallVector<int64_t, 4> tripCounts;
    OpBuilder b(op);
    auto loc = op.getLoc();

    auto allShapeSizes = op.createFlatListOfOperandDims(b, loc);
    AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
    if (!shapeSizesToLoopsMap)
      return {};

    SmallVector<OpFoldResult> shapeSizes =
        affine::makeComposedFoldedMultiResultAffineApply(
            b, loc, shapeSizesToLoopsMap, allShapeSizes);
    for (auto size : shapeSizes) {
      if (auto v = llvm::dyn_cast_if_present<Value>(size)) {
        auto c = dyn_cast_if_present<arith::ConstantIndexOp>(v.getDefiningOp());
        if (!c) {
          LLVM_DEBUG(llvm::outs() << "Found non-constant dim!\n");
          return {};
        }
        tripCounts.push_back(c.value());
      } else {
        auto a = llvm::dyn_cast_if_present<Attribute>(size);
        auto c = llvm::dyn_cast_if_present<IntegerAttr>(a);
        if (!c) {
          LLVM_DEBUG(llvm::outs() << "unhandled addr!\n");
          return {};
        }
        tripCounts.push_back(c.getInt());
      }
    }
    return std::move(tripCounts);
  }

  static void
  adjustToDivisorsOfTripCounts(linalg::LinalgOp op,
                               SmallVectorImpl<int64_t> *tileSizes,
                               SmallVectorImpl<int64_t> &tripCounts) {

    if (op.getNumLoops() != tileSizes->size()) {
      op->emitOpError("invalid tile size count");
      return;
    }
    for (unsigned i = 0, e = op.getNumLoops(); i < e; i++) {
      auto &tFactorAdjusted = (*tileSizes)[i];
      tFactorAdjusted = std::max((int64_t)1, tripCounts[i] / tFactorAdjusted);
      // Adjust the tile size to largest factor of the trip count less than
      // tSize.
      auto constTripCount = tripCounts[i];
      LLVM_DEBUG(llvm::outs() << "adj: " << tFactorAdjusted
                              << " iters: " << constTripCount << "\n");
      if (constTripCount > 1 && tFactorAdjusted > constTripCount / 2)
        tFactorAdjusted = constTripCount / 2;
      while (constTripCount % tFactorAdjusted != 0)
        tFactorAdjusted--;
      LLVM_DEBUG(llvm::outs() << "final adj: " << tFactorAdjusted << "\n");
    }
  }

  // use the algorithm from affine loop tiling pass
  static void getTileSizes(linalg::LinalgOp op, size_t cacheSizeBytes,
                           SmallVectorImpl<int64_t> &tripCounts,
                           SmallVectorImpl<int64_t> *tileSizes) {
    if (!cacheSizeBytes)
      return;

    auto nLoops = op.getNumLoops();
    tileSizes->resize(nLoops);

    uint64_t fp = air::CostModel().getOpCounts(op)["footprint"];
    LLVM_DEBUG(llvm::outs() << "Footprint: " << fp << "\n");
    LLVM_DEBUG(llvm::outs() << "Cache size: " << cacheSizeBytes << "\n");
    uint64_t excessFactor = llvm::divideCeil(fp, cacheSizeBytes);
    if (excessFactor <= 1) {
      *tileSizes = tripCounts;
      return;
    }
    // For an n-d tileable band, compute the n^th root of the excess.
    int64_t tSize =
        static_cast<int64_t>(floorl(std::pow(excessFactor, 1.0 / nLoops)));

    // We'll keep a running product to determine the last tile size better.
    unsigned cumulProductOfTileSizes = 1;
    for (unsigned i = 0, e = nLoops; i < e; i++) {
      if (i < e - 1)
        (*tileSizes)[i] = std::min(tSize, tripCounts[i]);
      else
        // Set last tile size to cover the balance.
        (*tileSizes)[i] = std::max(
            1U, static_cast<unsigned>(excessFactor / cumulProductOfTileSizes));
      cumulProductOfTileSizes *= (*tileSizes)[i];
    }

    adjustToDivisorsOfTripCounts(op, tileSizes, tripCounts);
  }

  static LogicalResult copyCallBack(OpBuilder &b, Value src, Value dst) {
    memref::CopyOp::create(b, b.getUnknownLoc(), src, dst);
    return success();
  }

  void runGenericPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();

    SmallVector<linalg::GenericOp, 4> genericOps;
    funcOp.walk([&](linalg::GenericOp op) { genericOps.push_back(op); });

    // GenericOp
    for (auto genericOp : genericOps) {

      auto attr = genericOp->getAttrOfType<StringAttr>(
          air::LinalgTransforms::kLinalgTransformMarker);
      if (!attr) {
        if (clInputFilter != "")
          continue;
        genericOp->setAttr(air::LinalgTransforms::kLinalgTransformMarker,
                           StringAttr::get(ctx, ""));
        attr = genericOp->getAttrOfType<StringAttr>(
            air::LinalgTransforms::kLinalgTransformMarker);
      } else if (clInputFilter != attr.str())
        continue;
      StringAttr next_match = attr;

      size_t nLoops = genericOp.getNumLoops();
      SmallVector<int64_t, 2> herd_size{2, 2};
      SmallVector<int64_t, 4> l1_tile_size(nLoops, 1);
      SmallVector<unsigned, 4> l1_tile_interchange(nLoops, 0);
      SmallVector<int64_t> l1_promote_operands;
      SmallVector<int64_t, 4> l2_tile_size(nLoops, 1);
      SmallVector<unsigned, 4> l2_tile_interchange(nLoops, 0);
      SmallVector<int64_t> l2_promote_operands;

      for (int i = 0, e = clL1OperandsToPromote.size(); i < e; i++)
        l1_promote_operands.push_back(clL1OperandsToPromote[i]);

      for (int i = 0, e = clL2OperandsToPromote.size(); i < e; i++)
        l2_promote_operands.push_back(clL2OperandsToPromote[i]);

      auto tripCounts = getTripCounts(genericOp);

      bool tileForL2 = true;
      if (clL2TileSize.size())
        for (int i = 0, e = std::min(nLoops, clL2TileSize.size()); i < e; i++)
          l2_tile_size[i] = clL2TileSize[i];
      else if (clL2MaxSize > 0)
        getTileSizes(genericOp, clL2MaxSize, tripCounts, &l2_tile_size);
      else
        tileForL2 = false;

      std::iota(l2_tile_interchange.begin(), l2_tile_interchange.end(), 0);
      for (int i = 0, e = std::min(nLoops, clL2TileInterchange.size()); i < e;
           i++)
        l2_tile_interchange[i] = clL2TileInterchange[i];

      for (int i = 0, e = std::min(2, (int)clHerdSize.size()); i < e; i++)
        herd_size[i] = clHerdSize[i];

      // outline the operation for convenience
      air::AIROutliner olnr;
      func::CallOp call =
          olnr.outline(std::vector<Operation *>{genericOp}, "call_generic_op");
      func::FuncOp called =
          funcOp->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(
              call.getCallee());

      // L2 tiling
      if (tileForL2) {
        RewritePatternSet stageL2Patterns(ctx);
        stageL2Patterns.insert<TileLinalgOpPattern>(
            linalg::GenericOp::getOperationName(), ctx,
            linalg::LinalgTilingOptions()
                .setTileSizes(l2_tile_size)
                .setInterchange(l2_tile_interchange)
                .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops),
            LinalgTransformationFilter(
                next_match,
                StringAttr::get(ctx, clL2Promote ? "L2" : "L2_promoted")));

        linalg::LinalgPromotionOptions l2PromoteOptions;
        if (l2_promote_operands.size())
          l2PromoteOptions.setOperandsToPromote(l2_promote_operands);
        l2PromoteOptions.setCopyInOutFns(copyCallBack, copyCallBack);
        stageL2Patterns.insert<PromoteLinalgOpPattern>(
            ctx, l2PromoteOptions,
            LinalgTransformationFilter(StringAttr::get(ctx, "L2"),
                                       StringAttr::get(ctx, "L2_promoted")));
        stageL2Patterns.insert<RemoveSubViewOpsPattern>(ctx, 1);
        stageL2Patterns.insert<FoldSubViewOpsPattern>(ctx);
        stageL2Patterns.insert<MemrefsPattern>(ctx);
        scf::populateSCFForLoopCanonicalizationPatterns(stageL2Patterns);
        (void)applyPatternsGreedily(called, std::move(stageL2Patterns));

        LLVM_DEBUG(llvm::outs() << "After L2 Tiling\n");
        LLVM_DEBUG(called.print(llvm::outs()));
        for (int i = 0, e = tripCounts.size(); i < e; i++)
          tripCounts[i] = l2_tile_size[i];
        next_match = StringAttr::get(ctx, "L2_promoted");
      }

      // compute L1 tile size

      called.walk([&](linalg::GenericOp l1_op) {
        if (clL1TileSize.size())
          for (int i = 0, e = std::min(nLoops, clL1TileSize.size()); i < e; i++)
            l1_tile_size[i] = clL1TileSize[i];
        else if (clL1MaxSize > 0) {
          getTileSizes(l1_op, clL1MaxSize, tripCounts, &l1_tile_size);
        }
      });

      std::iota(l1_tile_interchange.begin(), l1_tile_interchange.end(), 0);
      for (int i = 0, e = std::min(nLoops, clL1TileInterchange.size()); i < e;
           i++)
        l1_tile_interchange[i] = clL1TileInterchange[i];

      // tile to the herd size

      SmallVector<int64_t, 4> herd_tile_size(tripCounts.size(), -1);
      for (int i = 0, e = std::min(2, (int)l1_tile_size.size()); i < e; i++) {
        if (herd_size[i] > tripCounts[i])
          herd_tile_size[i] = tripCounts[i];
        else if (herd_size[i] < tripCounts[i] / l1_tile_size[i])
          herd_tile_size[i] = tripCounts[i] / herd_size[i];
        else {
          herd_tile_size[i] = l1_tile_size[i];
          l1_tile_size[i] = 0;
        }
        LLVM_DEBUG(llvm::outs() << "herd tile size [" << i
                                << "] = " << herd_tile_size[i] << "\n");
        LLVM_DEBUG(llvm::outs() << "L1 tile size [" << i
                                << "] = " << l1_tile_size[i] << "\n");
      }

      for (auto &s : herd_tile_size)
        if (s == -1)
          s = 0;

      RewritePatternSet patterns(ctx);
      patterns.insert<TileLinalgOpPattern>(
          linalg::GenericOp::getOperationName(), ctx,
          linalg::LinalgTilingOptions()
              .setTileSizes(herd_tile_size)
              .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops),
          LinalgTransformationFilter(next_match,
                                     StringAttr::get(ctx, "herd_tiling")));
      (void)applyPatternsGreedily(called, std::move(patterns));
      next_match = StringAttr::get(ctx, "herd_tiling");

      LLVM_DEBUG(llvm::outs() << "After Herd Tiling\n");
      LLVM_DEBUG(called.print(llvm::outs()));

      bool needL1Tiling = !std::all_of(l1_tile_size.begin(), l1_tile_size.end(),
                                       [](int i) { return i == 0; });
      RewritePatternSet stageL1Patterns(ctx);
      if (needL1Tiling) {
        stageL1Patterns.insert<TileLinalgOpPattern>(
            linalg::GenericOp::getOperationName(), ctx,
            linalg::LinalgTilingOptions()
                .setTileSizes(l1_tile_size)
                .setInterchange(l1_tile_interchange)
                .setLoopType(linalg::LinalgTilingLoopType::Loops),
            LinalgTransformationFilter(
                next_match,
                StringAttr::get(ctx, clL1Promote ? "L1" : "L1_promoted")));
      }

      linalg::LinalgPromotionOptions l1PromoteOptions;
      if (l1_promote_operands.size())
        l1PromoteOptions.setOperandsToPromote(l1_promote_operands);
      l1PromoteOptions.setCopyInOutFns(copyCallBack, copyCallBack);
      stageL1Patterns.insert<PromoteLinalgOpPattern>(
          ctx, l1PromoteOptions,
          LinalgTransformationFilter(needL1Tiling ? StringAttr::get(ctx, "L1")
                                                  : next_match,
                                     StringAttr::get(ctx, "L1_promoted")));
      stageL1Patterns.insert<RemoveSubViewOpsPattern>(ctx, 2);
      stageL1Patterns.insert<FoldSubViewOpsPattern>(ctx);
      scf::populateSCFForLoopCanonicalizationPatterns(stageL1Patterns);
      (void)applyPatternsGreedily(called, std::move(stageL1Patterns));

      RewritePatternSet stage3Patterns(&getContext());
      stage3Patterns.insert<MemrefsPattern>(ctx);
      stage3Patterns.insert<RemoveAllocCopyLinalgOpCopyPattern>(ctx);
      (void)applyPatternsGreedily(called, std::move(stage3Patterns));

      LLVM_DEBUG(llvm::outs() << "After L1 Tiling\n");
      LLVM_DEBUG(called.print(llvm::outs()));

      called.walk([](linalg::LinalgOp op) {
        op->removeAttr(air::LinalgTransforms::kLinalgTransformMarker);
      });

      InlinerInterface interface(&getContext());
      InlinerConfig config;
      (void)inlineCall(interface, config.getCloneCallback(), call, called,
                       &called.getRegion(), true);
      call.erase();
      called.erase();
    }
  }

  void runMatmulPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();

    SmallVector<linalg::MatmulOp, 4> matmulOps;
    funcOp.walk([&](linalg::MatmulOp op) { matmulOps.push_back(op); });

    // MatmulOp
    for (auto matmulOp : matmulOps) {

      auto attr = matmulOp->getAttrOfType<StringAttr>(
          air::LinalgTransforms::kLinalgTransformMarker);
      if (!attr) {
        if (clInputFilter != "")
          continue;
        matmulOp->setAttr(air::LinalgTransforms::kLinalgTransformMarker,
                          StringAttr::get(ctx, ""));
        attr = matmulOp->getAttrOfType<StringAttr>(
            air::LinalgTransforms::kLinalgTransformMarker);
      } else if (clInputFilter != attr.str())
        continue;

      StringAttr next_match = attr;

      air::AIROutliner olnr;
      func::CallOp call =
          olnr.outline(std::vector<Operation *>{matmulOp}, "call_mmult");
      func::FuncOp called =
          funcOp->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(
              call.getCallee());

      SmallVector<int64_t, 3> herd_size{2, 2, 2};
      SmallVector<int64_t, 3> l1_tile_size{32, 32, 32};
      SmallVector<unsigned, 3> l1_tile_interchange{0, 1, 2};
      SmallVector<int64_t, 3> l1_promote_operands;
      SmallVector<int64_t, 3> l2_tile_size{64, 64, 64};
      SmallVector<unsigned, 3> l2_tile_interchange{0, 1, 2};
      SmallVector<int64_t, 3> l2_promote_operands;

      for (int i = 0, e = clL1TileSize.size(); i < e; i++)
        l1_tile_size[i] = clL1TileSize[i];

      for (int i = 0, e = clL1TileInterchange.size(); i < e; i++)
        l1_tile_interchange[i] = clL1TileInterchange[i];

      for (int i = 0, e = clL2TileInterchange.size(); i < e; i++)
        l2_tile_interchange[i] = clL2TileInterchange[i];

      for (int i = 0, e = clL1OperandsToPromote.size(); i < e; i++)
        l1_promote_operands.push_back(clL1OperandsToPromote[i]);

      for (int i = 0, e = clL2OperandsToPromote.size(); i < e; i++)
        l2_promote_operands.push_back(clL2OperandsToPromote[i]);

      bool tileForL2 = false;
      if (clL2TileSize.size()) {
        for (int i = 0, e = clL2TileSize.size(); i < e; i++)
          l2_tile_size[i] = clL2TileSize[i];
        tileForL2 = true;
      }

      if (tileForL2) {
        RewritePatternSet stageL2Patterns(ctx);
        stageL2Patterns.insert<TileLinalgOpPattern>(
            linalg::MatmulOp::getOperationName(), ctx,
            linalg::LinalgTilingOptions()
                .setTileSizes(l2_tile_size)
                .setInterchange(l2_tile_interchange)
                .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops),
            LinalgTransformationFilter(
                next_match,
                StringAttr::get(ctx, clL2Promote ? "L2" : "L2_promoted")));

        linalg::LinalgPromotionOptions l2PromoteOptions;
        if (l2_promote_operands.size())
          l2PromoteOptions.setOperandsToPromote(l2_promote_operands);
        l2PromoteOptions.setCopyInOutFns(copyCallBack, copyCallBack);
        stageL2Patterns.insert<PromoteLinalgOpPattern>(
            ctx, l2PromoteOptions,
            LinalgTransformationFilter(StringAttr::get(ctx, "L2"),
                                       StringAttr::get(ctx, "L2_promoted")));
        stageL2Patterns.insert<RemoveSubViewOpsPattern>(ctx, 1);
        stageL2Patterns.insert<FoldSubViewOpsPattern>(ctx);
        stageL2Patterns.insert<MemrefsPattern>(ctx);
        scf::populateSCFForLoopCanonicalizationPatterns(stageL2Patterns);
        (void)applyPatternsGreedily(called, std::move(stageL2Patterns));
        next_match = StringAttr::get(ctx, "L2_promoted");
      }

      RewritePatternSet stageL1Patterns(ctx);

      stageL1Patterns.insert<TileLinalgOpPattern>(
          linalg::MatmulOp::getOperationName(), ctx,
          linalg::LinalgTilingOptions()
              .setTileSizes(l1_tile_size)
              .setInterchange(l1_tile_interchange)
              .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops),
          LinalgTransformationFilter(
              next_match,
              StringAttr::get(ctx, clL1Promote ? "L1" : "L1_promoted")));

      linalg::LinalgPromotionOptions l1PromoteOptions;
      if (l1_promote_operands.size())
        l1PromoteOptions.setOperandsToPromote(l1_promote_operands);
      l1PromoteOptions.setCopyInOutFns(copyCallBack, copyCallBack);
      stageL1Patterns.insert<PromoteLinalgOpPattern>(
          ctx, l1PromoteOptions,
          LinalgTransformationFilter(StringAttr::get(ctx, "L1"),
                                     StringAttr::get(ctx, "L1_promoted")));

      RewritePatternSet stage3Patterns(&getContext());
      stage3Patterns.insert<RemoveSubViewOpsPattern>(ctx, 2);
      stage3Patterns.insert<FoldSubViewOpsPattern>(ctx);
      stage3Patterns.insert<MemrefsPattern>(ctx);
      scf::populateSCFForLoopCanonicalizationPatterns(stage3Patterns);

      (void)applyPatternsGreedily(called, std::move(stageL1Patterns));
      (void)applyPatternsGreedily(called, std::move(stage3Patterns));
      called.walk([](linalg::LinalgOp op) {
        op->removeAttr(air::LinalgTransforms::kLinalgTransformMarker);
      });

      InlinerInterface interface(&getContext());
      InlinerConfig config;
      (void)inlineCall(interface, config.getCloneCallback(), call, called,
                       &called.getRegion(), true);
      call.erase();
      called.erase();
    }
  }

  void runConv2dPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();

    SmallVector<linalg::Conv2DNchwFchwOp, 4> conv2dOps;
    funcOp.walk([&](linalg::Conv2DNchwFchwOp op) { conv2dOps.push_back(op); });

    // Conv2dOp
    for (auto conv2dOp : conv2dOps) {
      air::AIROutliner olnr;
      func::CallOp call =
          olnr.outline(std::vector<Operation *>{conv2dOp}, "call_conv_2d_nchw");
      func::FuncOp called =
          funcOp->getParentOfType<ModuleOp>().lookupSymbol<func::FuncOp>(
              call.getCallee());

      SmallVector<int64_t, 7> l1_tile_size{1, 32, 32, 32, 32, 3, 3};
      SmallVector<unsigned, 7> l1_tile_interchange{0, 1, 2, 3, 4, 5, 6};

      for (int i = 0, e = clL1TileSize.size(); i < e; i++)
        l1_tile_size[i] = clL1TileSize[i];

      for (int i = 0, e = clL1TileInterchange.size(); i < e; i++)
        l1_tile_interchange[i] = clL1TileInterchange[i];

      RewritePatternSet stage1Patterns(&getContext());

      stage1Patterns.insert<TileLinalgOpPattern>(
          linalg::Conv2DNchwFchwOp::getOperationName(), ctx,
          linalg::LinalgTilingOptions()
              .setTileSizes(l1_tile_size)
              .setInterchange(l1_tile_interchange)
              .setLoopType(linalg::LinalgTilingLoopType::Loops),
          LinalgTransformationFilter(ArrayRef<StringAttr>{},
                                     StringAttr::get(ctx, "promote_HERD")));

      stage1Patterns.insert<PromoteLinalgOpPattern>(
          ctx,
          linalg::LinalgPromotionOptions().setOperandsToPromote(
              std::vector<int64_t>{0, 1, 2}),
          LinalgTransformationFilter(StringAttr::get(ctx, "promote_HERD"),
                                     StringAttr::get(ctx, "HERD")));

      RewritePatternSet stage2Patterns(ctx);
      linalg::populateLinalgTilingCanonicalizationPatterns(stage2Patterns);
      scf::populateSCFForLoopCanonicalizationPatterns(stage2Patterns);

      RewritePatternSet stage3Patterns(&getContext());
      stage3Patterns.insert<RemoveSubViewOpsPattern>(ctx, 2);
      stage3Patterns.insert<FoldSubViewOpsPattern>(ctx);
      stage3Patterns.insert<MemrefsPattern>(ctx);
      stage3Patterns.insert<RemoveViewOpsPattern>(ctx, 2);

      (void)applyPatternsGreedily(called, std::move(stage1Patterns));
      (void)applyPatternsGreedily(called, std::move(stage2Patterns));
      (void)applyPatternsGreedily(called, std::move(stage3Patterns));

      /// scf.parallel transform from herd dimension
      /// Step-1: Capture the perfectly nested scf.for loops
      /// Step-2: Create scf.parallel loop based on herd dimension
      /// Step-3: Replace the scf.for loops IV with scf.parallel loops IV

      /// Capture the perfectly nested loops
      SmallVector<scf::ForOp, 6> loops;
      called.walk([&](Operation *op) {
        if (auto scfForOp = dyn_cast_if_present<scf::ForOp>(op))
          if (!op->getParentOfType<scf::ForOp>())
            getPerfectlyNestedLoops(loops, scfForOp);
      });

      if (clHerdSize.size() == 0) {
        funcOp->emitOpError("AIE tile dimension can't be zero");
        return;
      }
      if (clHerdSize.size() > loops.size()) {
        funcOp->emitOpError(
            "AIE tile dimension must be equal or less than Tiled loops number");
        return;
      }

      scf::ForOp outermost = loops[0];
      OpBuilder builder(outermost);
      Location loc = outermost.getLoc();

      // Create parallel loops for spatial iteration.
      SmallVector<Value, 2> lowerBounds, upperBounds, steps;
      for (unsigned i = 0, e = clHerdSize.size(); i < e; ++i) {
        lowerBounds.push_back(loops[i].getLowerBound());
        upperBounds.push_back(loops[i].getUpperBound());
        steps.push_back(loops[i].getStep());
      }

      auto parallelLoop = scf::ParallelOp::create(builder, loc, lowerBounds,
                                                  upperBounds, steps);

      builder.setInsertionPointToStart(parallelLoop.getBody());

      // Replace the scf.for IV with scf.parallel IV
      auto pLoopIV = parallelLoop.getInductionVars();
      for (unsigned i = 0, e = pLoopIV.size(); i < e; ++i)
        replaceAllUsesInRegionWith(loops[i].getInductionVar(), pLoopIV[i],
                                   loops[loops.size() - 1].getRegion());

      // Move the remaining inner scf.for loops and delete extra
      // terminator and perfectly nested loops.
      loops[clHerdSize.size() - 1].getBody()->back().erase();
      parallelLoop.getBody()->getOperations().splice(
          Block::iterator(parallelLoop.getBody()->back()),
          loops[clHerdSize.size() - 1].getBody()->getOperations());

      outermost.erase();

      // Drop the marker.
      called.walk([](linalg::LinalgOp op) {
        op->removeAttr(air::LinalgTransforms::kLinalgTransformMarker);
      });

      InlinerInterface interface(&getContext());
      InlinerConfig config;
      (void)inlineCall(interface, config.getCloneCallback(), call, called,
                       &called.getRegion(), true);
      call.erase();
      called.erase();
    }
  }

  void runOnFunction(func::FuncOp f) {

    // RewritePatternSet prePatterns(&getContext());
    // prePatterns.insert<RemoveAllocLinalgOpCopyPattern>(&getContext());
    //(void)applyPatternsGreedily(f, std::move(prePatterns));
    if (!clLinalgCodegenTestPatterns) {
      runMatmulPatterns(f);
      runConv2dPatterns(f);
      runGenericPatterns(f);
    } else {
      runTestPatterns(f);
    }
  }

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<func::FuncOp, 4> funcOps;
    module.walk([&](func::FuncOp op) { funcOps.push_back(op); });
    for (auto f : funcOps)
      runOnFunction(f);
  }

private:
};

} // namespace air
} // namespace xilinx

//===----------------------------------------------------------------------===//
// PipelineReduceOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::PipelineReduceOp::applyToOne(
    transform::TransformRewriter &rewriter, linalg::LinalgOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  auto result = xilinx::air::pipelineReduceLinalgOp(
      rewriter, target, extractFromIntegerArrayAttr<int64_t>(getTileSize()),
      getPipelineDepth(), getDirection().str(), getPromote());
  if (failed(result))
    return emitDefiniteFailure() << "Failed";
  results.push_back(result->op);
  rewriter.eraseOp(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// LinalgTileOp
//===----------------------------------------------------------------------===//

void transform::LinalgTileOp::build(OpBuilder &builder, OperationState &result,
                                    Value target,
                                    ArrayRef<int64_t> staticTileSizes) {
  return build(builder, result,
               /*target=*/target,
               /*mixedTileSizes=*/
               getAsOpFoldResult(builder.getI64ArrayAttr(staticTileSizes)));
}

void transform::LinalgTileOp::build(OpBuilder &builder, OperationState &result,
                                    Value target,
                                    ArrayRef<OpFoldResult> mixedTileSizes) {
  SmallVector<int64_t> staticTileSizes;
  SmallVector<Value> dynamicTileSizes;
  dispatchIndexOpFoldResults(mixedTileSizes, dynamicTileSizes, staticTileSizes);
  // Call the default builder which sets up the proper operands segment sizes
  // attributes for multiple variadic operands. In the absence of this, horrible
  // bugs ensue.
  MLIRContext *ctx = builder.getContext();
  auto operationType = transform::AnyOpType::get(ctx);
  auto staticTileSizesAttr = builder.getDenseI64ArrayAttr(staticTileSizes);
  build(builder, result,
        /*resultTypes=*/TypeRange{operationType, operationType},
        /*target=*/target,
        /*dynamic_sizes=*/dynamicTileSizes,
        /*static_sizes=*/staticTileSizesAttr);
}

// Return true if all dimensions are integer divisible by the respective tiles.
static bool validateTilableByInteger(linalg::LinalgOp linalgOp,
                                     SmallVector<OpFoldResult> &tiles) {
  if (tiles.empty())
    return false;

  auto tileOp = cast<TilingInterface>(linalgOp.getOperation());
  OpBuilder builder(tileOp);
  OpBuilder::InsertionGuard guard(builder);
  SmallVector<Range> iterationDomain = tileOp.getIterationDomain(builder);

  auto getConstantRange = [](const Range &range) {
    std::optional<int64_t> output = std::nullopt;
    std::optional<int64_t> stride = getConstantIntValue(range.stride);
    if (!stride || *stride != 1)
      return output;
    std::optional<int64_t> offset = getConstantIntValue(range.offset);
    if (!offset)
      return output;
    std::optional<int64_t> size = getConstantIntValue(range.size);
    if (!size)
      return output;
    output = (*size - *offset);
    return output;
  };

  for (unsigned i = 0; i < tiles.size(); i++) {
    std::optional<int64_t> tileSize = getConstantIntValue(tiles[i]);
    std::optional<int64_t> rangeOnDim = getConstantRange(iterationDomain[i]);

    // If the tile factor or the range are non-constant, the tile size is
    // considered to be invalid.
    if (!tileSize || !rangeOnDim)
      return false;

    // Skip dimension with zero tile size.
    if (*tileSize == 0)
      continue;

    // If tile size is bigger than the range, then set tile size to be equal to
    // range.
    if (*tileSize > *rangeOnDim) {
      tiles[i] = builder.getI64IntegerAttr(*rangeOnDim);
      continue;
    }

    // The dimension must be fully divisible by the tile.
    if (*rangeOnDim % *tileSize != 0)
      return false;
  }

  return true;
}

DiagnosedSilenceableFailure
transform::LinalgTileOp::apply(transform::TransformRewriter &rewriter,
                               transform::TransformResults &transformResults,
                               transform::TransformState &state) {
  auto transformOp = cast<TransformOpInterface>(getOperation());

  // Result payload ops.
  SmallVector<Operation *> tileOps;
  SmallVector<Operation *> tiledOps;

  SmallVector<OpFoldResult> mixedTileSizes = getMixedSizes();

  for (Operation *target : state.getPayloadOps(getTarget())) {
    // Check if tiling sizes lead to integer tiling factors; enfore tiling
    // factor = 1 when tile size is bigger than the problem size.
    linalg::LinalgOp linalgOp = dyn_cast_if_present<linalg::LinalgOp>(target);
    if (!linalgOp) {
      DiagnosedSilenceableFailure diag = transformOp.emitSilenceableError()
                                         << "only Linalg ops are supported";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }
    if (!validateTilableByInteger(linalgOp, mixedTileSizes)) {
      DiagnosedSilenceableFailure diag =
          transformOp.emitSilenceableError()
          << "only support tiling in integer factors";
      diag.attachNote(target->getLoc()) << "target op";
      return diag;
    }

    scf::SCFTilingResult tilingResult;
    DiagnosedSilenceableFailure diag = transform::tileToForallOpImpl(
        rewriter, state, transformOp, target,
        /*mixedNumThreads*/ SmallVector<OpFoldResult>{}, mixedTileSizes,
        /*getMapping()*/ std::nullopt, tilingResult);
    if (!diag.succeeded())
      return diag;
    tileOps.push_back(tilingResult.loops.front());
    tiledOps.append(tilingResult.tiledOps);
  }

  transformResults.set(cast<OpResult>(getTiledLinalgOp()), tiledOps);
  transformResults.set(cast<OpResult>(getLoops()), tileOps);

  return DiagnosedSilenceableFailure::success();
}

SmallVector<OpFoldResult> transform::LinalgTileOp::getMixedSizes() {
  ValueRange dynamic = getDynamicSizes();
  ArrayRef<int64_t> tileSizes = getStaticSizes();
  SmallVector<OpFoldResult> results;
  results.reserve(tileSizes.size());
  unsigned dynamicPos = 0;
  Builder builder(getContext());
  for (int64_t size : tileSizes) {
    if (size == ShapedType::kDynamic) {
      results.push_back(dynamic[dynamicPos++]);
    } else {
      results.push_back(builder.getIndexAttr(size));
    }
  }
  return results;
}

ParseResult transform::LinalgTileOp::parse(OpAsmParser &parser,
                                           OperationState &result) {
  OpAsmParser::UnresolvedOperand target;
  SmallVector<OpAsmParser::UnresolvedOperand> dynamicSizes;
  DenseI64ArrayAttr staticSizes;
  auto operationType = transform::AnyOpType::get(parser.getContext());
  if (parser.parseOperand(target) ||
      parser.resolveOperand(target, operationType, result.operands) ||
      parseDynamicIndexList(parser, dynamicSizes, staticSizes) ||
      parser.resolveOperands(dynamicSizes, operationType, result.operands))
    return ParseResult::failure();

  result.addAttribute(getStaticSizesAttrName(result.name), staticSizes);
  result.addTypes(SmallVector<Type>(2, operationType));
  return success();
}

void transform::LinalgTileOp::print(OpAsmPrinter &p) {
  p << ' ' << getTarget();
  printDynamicIndexList(p, getOperation(), getDynamicSizes(), getStaticSizes());
}

void transform::LinalgTileOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  onlyReadsHandle(getDynamicSizesMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// LinalgPromoteOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::LinalgPromoteOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {

  SmallVector<Operation *> payloadOps =
      llvm::to_vector(state.getPayloadOps(getTarget()));
  if (!payloadOps.size()) {
    results.set(llvm::cast<OpResult>(getResult()),
                ArrayRef(payloadOps.begin(), payloadOps.end()));
    return DiagnosedSilenceableFailure::success();
  }

  linalg::LinalgPromotionOptions promotionOptions;
  auto operandsToPromote =
      extractFromIntegerArrayAttr<int64_t>(getOperandsToPromote());

  if (getUseFullTilesByDefault())
    promotionOptions = promotionOptions.setUseFullTileBuffersByDefault(
        getUseFullTilesByDefault());
  if (getUseAlloca())
    promotionOptions = promotionOptions.setUseAlloca(getUseAlloca());
  if (!getUseFullTileBuffers().empty())
    promotionOptions = promotionOptions.setUseFullTileBuffers(
        llvm::to_vector(getUseFullTileBuffers().getAsValueRange<BoolAttr>()));
  if (getAlignment().has_value())
    promotionOptions = promotionOptions.setAlignment(*getAlignment());

  auto copyCallBack = [](OpBuilder &b, Value src, Value dst) -> LogicalResult {
    memref::CopyOp::create(b, b.getUnknownLoc(), src, dst);
    return success();
  };
  promotionOptions.setCopyInOutFns(copyCallBack, copyCallBack);

  auto memorySpace = xilinx::air::symbolizeMemorySpace(getMemorySpace())
                         .value_or(xilinx::air::MemorySpace::L1);

  SetVector<Operation *> transformed;
  int64_t operandOffset = 0;
  // Shared across ops so that the same Value promoted for one op is reused
  // by another op, avoiding duplicate DMAs.
  DenseMap<Value, Value> promotedValueMap;
  DominanceInfo dominance(payloadOps[0]->getParentOfType<func::FuncOp>());

  uint32_t group_size = getGroupSize();
  uint32_t group = 0;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto linalgOp = dyn_cast_if_present<linalg::LinalgOp>(target);
    if (!linalgOp)
      continue;

    int64_t numOperands = linalgOp->getNumOperands();
    SmallVector<int64_t, 4> opersToPromote;
    if (!operandsToPromote.size()) {
      opersToPromote.resize_for_overwrite(numOperands);
      std::iota(opersToPromote.begin(), opersToPromote.end(), 0);
    } else {
      for (auto &o : operandsToPromote) {
        int64_t operand = o - operandOffset;
        if (operand < 0)
          continue;
        if (operand >= numOperands)
          continue;
        opersToPromote.push_back(operand);
      }
    }
    operandOffset += numOperands;
    if (++group == group_size) {
      group = 0;
      operandOffset = 0;
    }
    if (opersToPromote.empty())
      continue;

    // Partition operands into subview (handled by upstream promoteSubViews)
    // and non-subview (handled manually below). Non-subview operands arise
    // from broadcast indexing maps where tiling doesn't produce a subview.
    SmallVector<int64_t, 4> subviewOperands;
    SmallVector<int64_t, 4> nonSubviewOperands;
    for (int64_t idx : opersToPromote) {
      Value operand = linalgOp->getOperand(idx);
      if (isa_and_nonnull<memref::SubViewOp>(operand.getDefiningOp()))
        subviewOperands.push_back(idx);
      else
        nonSubviewOperands.push_back(idx);
    }

    // For subview input operands already promoted by a previous op,
    // replace the operand directly and remove from the promotion list.
    // This enables cross-op buffer sharing (e.g., weighted_rms_norm where
    // sq and out generics both read from the same X subview).
    SmallVector<int64_t, 4> subviewsNeedingPromotion;
    for (int64_t idx : subviewOperands) {
      Value operand = linalgOp->getOperand(idx);
      bool isInput = (idx < static_cast<int64_t>(linalgOp.getNumDpsInputs()));
      if (isInput) {
        auto it = promotedValueMap.find(operand);
        if (it != promotedValueMap.end()) {
          Value promoted = it->second;
          if (dominance.properlyDominates(promoted, linalgOp)) {
            rewriter.modifyOpInPlace(
                linalgOp, [&]() { linalgOp->setOperand(idx, promoted); });
            // Remove deallocs of the promoted buffer since it's now
            // shared across multiple ops and must stay live.
            // Walk up through view/subview chains to find the alloc.
            Value allocRoot = promoted;
            while (auto *def = allocRoot.getDefiningOp()) {
              if (auto sv = dyn_cast<memref::SubViewOp>(def))
                allocRoot = sv.getSource();
              else if (auto vw = dyn_cast<memref::ViewOp>(def))
                allocRoot = vw.getSource();
              else
                break;
            }
            SmallVector<memref::DeallocOp> toErase;
            for (auto *user : allocRoot.getUsers())
              if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
                toErase.push_back(dealloc);
            for (auto d : toErase)
              rewriter.eraseOp(d);
            continue;
          }
        }
      }
      subviewsNeedingPromotion.push_back(idx);
    }

    // Promote subview operands using upstream LLVM infrastructure.
    if (!subviewsNeedingPromotion.empty()) {
      // Save original operand values before promoteSubViews replaces them.
      DenseMap<int64_t, Value> originalOperands;
      for (int64_t idx : subviewsNeedingPromotion)
        originalOperands[idx] = linalgOp->getOperand(idx);

      promotionOptions.setOperandsToPromote(subviewsNeedingPromotion);

      if (failed(promoteSubviewsPrecondition(target, promotionOptions)))
        return emitDefaultDefiniteFailure(target);

      rewriter.setInsertionPoint(target);
      FailureOr<linalg::LinalgOp> res =
          promoteSubViews(rewriter, linalgOp, promotionOptions);
      if (failed(res))
        return emitDefaultDefiniteFailure(target);

      // Record promoted values for cross-op sharing.
      // Map original subview → new promoted operand so subsequent ops
      // sharing the same subview input can reuse the promoted buffer.
      for (int64_t idx : subviewsNeedingPromotion) {
        Value newOperand = linalgOp->getOperand(idx);
        Value origOperand = originalOperands[idx];
        if (newOperand != origOperand)
          promotedValueMap[origOperand] = newOperand;
      }
    }

    // Manually promote non-subview operands (e.g., broadcast-indexed memrefs).
    // Deduplicate by Value so that if the same memref appears in multiple
    // operand positions, only one promoted buffer and copy is created.
    auto targetMemSpaceAttr =
        rewriter.getI32IntegerAttr(static_cast<int>(memorySpace));
    SmallVector<std::pair<Value, Value>> outputWritebacks;
    for (int64_t operandIdx : nonSubviewOperands) {
      Value operand = linalgOp->getOperand(operandIdx);
      auto operandType = dyn_cast<MemRefType>(operand.getType());
      if (!operandType)
        continue;

      // Skip if already in the target memory space. Compare integer
      // values to handle format mismatch (2 vs 2:i32).
      if (operandType.getMemorySpaceAsInt() ==
          static_cast<unsigned>(memorySpace))
        continue;

      // Reuse an existing promoted buffer if the same Value was already
      // promoted.
      bool isInput =
          (operandIdx < static_cast<int64_t>(linalgOp.getNumDpsInputs()));
      if (isInput) {
        auto it = promotedValueMap.find(operand);
        if (it != promotedValueMap.end()) {
          Value promoted = it->second;
          if (dominance.properlyDominates(promoted, linalgOp)) {
            rewriter.modifyOpInPlace(linalgOp, [&]() {
              linalgOp->setOperand(operandIdx, promoted);
            });
            continue;
          }
        }
      }

      rewriter.setInsertionPoint(linalgOp);

      // Build promoted memref type with target memory space.
      auto promotedType =
          MemRefType::get(operandType.getShape(), operandType.getElementType(),
                          AffineMap(), targetMemSpaceAttr);

      // Handle dynamic dimensions.
      SmallVector<Value> dynamicSizes;
      for (unsigned i = 0; i < operandType.getRank(); ++i) {
        if (operandType.isDynamicDim(i))
          dynamicSizes.push_back(
              memref::DimOp::create(rewriter, linalgOp.getLoc(), operand, i));
      }

      // Choose allocation strategy based on promotion options.
      Value promotedAlloc;
      if (getUseAlloca()) {
        promotedAlloc = memref::AllocaOp::create(rewriter, linalgOp.getLoc(),
                                                 promotedType, dynamicSizes);
      } else {
        promotedAlloc = memref::AllocOp::create(rewriter, linalgOp.getLoc(),
                                                promotedType, dynamicSizes);
      }

      // Copy data into the promoted buffer for input operands, or for
      // init operands whose values are read by the linalg payload.
      OpOperand *opOperand = &linalgOp->getOpOperand(operandIdx);
      bool needsCopyIn =
          isInput || linalgOp.payloadUsesValueFromOperand(opOperand);
      if (needsCopyIn)
        memref::CopyOp::create(rewriter, linalgOp.getLoc(), operand,
                               promotedAlloc);

      // Replace operand on the linalg op.
      rewriter.modifyOpInPlace(
          linalgOp, [&]() { linalgOp->setOperand(operandIdx, promotedAlloc); });

      promotedValueMap[operand] = promotedAlloc;

      // For output (init) operands, record for copy-back after the linalg op.
      if (!isInput)
        outputWritebacks.emplace_back(promotedAlloc, operand);
    }

    // Emit copy-backs for promoted output operands.
    if (!outputWritebacks.empty()) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointAfter(linalgOp);
      for (auto &[promoted, original] : outputWritebacks)
        memref::CopyOp::create(rewriter, linalgOp.getLoc(), promoted, original);
    }

    transformed.insert(linalgOp);
  }

  auto ctx = payloadOps[0]->getContext();
  RewritePatternSet patterns(ctx);
  // promoteSubViews generates extra copies and subviews, these patterns try to
  // simplify them.
  patterns.insert<xilinx::air::RemoveSubViewOpsPattern>(ctx, (int)memorySpace);
  patterns.insert<xilinx::air::FoldSubViewOpsPattern,
                  xilinx::air::RemoveViewOpsPattern>(ctx);
  patterns.insert<xilinx::air::RemoveExtraAllocPattern,
                  xilinx::air::RemoveDeadCopyPattern,
                  xilinx::air::RemoveAllocCopyLinalgOpCopyPattern>(ctx);
  // canonicalize allocs like:
  //  memref.alloc(%c32, %c32) : memref<?x?xi32, 2>
  // to:
  //  memref.alloc() : memref<32x32xi32, 2>
  memref::AllocOp::getCanonicalizationPatterns(patterns, ctx);
  (void)applyPatternsGreedily(payloadOps[0]->getParentOfType<func::FuncOp>(),
                              std::move(patterns));

  if (!transformed.size())
    return emitDefaultDefiniteFailure(payloadOps[0]);

  results.set(llvm::cast<OpResult>(getResult()), transformed.getArrayRef());
  return DiagnosedSilenceableFailure::success();
}

void transform::LinalgPromoteOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// FuseIntoContainingMemrefOp
//===----------------------------------------------------------------------===//

void transform::FuseIntoContainingMemrefOp::build(OpBuilder &builder,
                                                  OperationState &result,
                                                  Value producerOp,
                                                  Value containingOp) {
  result.addOperands({producerOp, containingOp});
  result.addTypes(transform::AnyOpType::get(builder.getContext()));
}

void transform::FuseIntoContainingMemrefOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getProducerOpMutable(), effects);
  onlyReadsHandle(getContainingOpMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

/// Find the first subview user of `producerOp` and tile it right before its
/// use. The tiled op is fused under the `containingOp`.
/// Return this fused op on success or nullptr if anything fails.
static Operation *tileAndFuseFirstExtractUse(RewriterBase &rewriter,
                                             Diagnostic &diag,
                                             Operation *producerOp,
                                             Operation *containingOp) {
  LLVM_DEBUG(llvm::dbgs() << "Try to fuse a direct extract use\n");
  auto tileableProducer = dyn_cast_if_present<TilingInterface>(producerOp);
  if (!tileableProducer) {
    diag.attachNote(producerOp->getLoc())
        << "producer is not a TileableInterface: " << *producerOp;
    return nullptr;
  }

  linalg::LinalgOp producerLinalgOp = cast<linalg::LinalgOp>(producerOp);
  auto users = producerLinalgOp.getDpsInits()[0].getUsers();
  auto it = llvm::find_if(users, [&](Operation *user) {
    auto sliceOp = dyn_cast_if_present<memref::SubViewOp>(user);
    return sliceOp && containingOp->isProperAncestor(sliceOp);
  });

  // Find a fusion opportunity.
  if (it == users.end()) {
    diag.attachNote(tileableProducer->getLoc())
        << "could not find fusion opportunity for: " << *tileableProducer;
    return nullptr;
  }
  auto sliceOpToTile = cast<memref::SubViewOp>(*it);

  // Try to fuse the producer in-place.
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Tile the producer.
  SmallVector<OpFoldResult> offsets = sliceOpToTile.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = sliceOpToTile.getMixedSizes();

  FailureOr<TilingResult> tileAndFuseResult =
      tileableProducer.getTiledImplementation(rewriter, offsets, sizes);

  if (failed(tileAndFuseResult)) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to tile producer op: " << *tileableProducer;
    return {};
  }
  if (tileAndFuseResult->tiledOps.size() != 1) {
    diag.attachNote(tileableProducer->getLoc())
        << "producer op should tile to generate only one op, but got: "
        << tileAndFuseResult->tiledOps.size();
    return {};
  }
  LLVM_DEBUG(llvm::dbgs() << "tiled producer: "
                          << tileAndFuseResult->tiledOps.front() << "\n");

  // Replace the subview op.
  rewriter.replaceOp(sliceOpToTile,
                     cast<linalg::LinalgOp>(tileAndFuseResult->tiledOps[0])
                         .getDpsInitOperand(0)
                         ->get());
  return tileAndFuseResult->tiledOps.front();
}

DiagnosedSilenceableFailure transform::FuseIntoContainingMemrefOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  SmallVector<Operation *> fusedOps;
  SmallVector<Operation *> producerOps =
      llvm::to_vector(state.getPayloadOps(getProducerOp()));
  // If nothing to fuse, propagate success.
  if (producerOps.empty()) {
    results.set(llvm::cast<OpResult>(getFusedOp()),
                SmallVector<mlir::Operation *>{});
    return DiagnosedSilenceableFailure::success();
  }
  if (producerOps.size() != 1) {
    return emitDefiniteFailure()
           << "requires exactly one producer_op handle (got "
           << producerOps.size() << ")";
  }
  Operation *producerOp = producerOps.front();

  SmallVector<Operation *> containingOps =
      llvm::to_vector(state.getPayloadOps(getContainingOp()));
  if (containingOps.size() != 1) {
    return emitDefiniteFailure()
           << "requires exactly one containing_op handle (got "
           << containingOps.size() << ")";
  }
  Operation *containingOp = containingOps.front();

  linalg::LinalgOp producerLinalgOp =
      dyn_cast_if_present<linalg::LinalgOp>(producerOp);
  if (!producerLinalgOp) {
    return emitDefiniteFailure() << "requires producer_op to be LinalgOp";
  }
  if (producerLinalgOp.getNumDpsInits() != 1) {
    return emitDefiniteFailure()
           << "requires producer_op to have exactly one init operand (got "
           << producerLinalgOp.getNumDpsInits() << ")";
  }

  auto initOperand = producerLinalgOp.getDpsInits()[0];
  // The containing op may be a user of producerOp: use isAncestor.
  int64_t numUsesInContainingOp =
      llvm::count_if(initOperand.getUsers(), [&](Operation *op) {
        return containingOp->isAncestor(op);
      });
  if (numUsesInContainingOp == 0) {
    results.set(llvm::cast<OpResult>(getFusedOp()), ArrayRef<Operation *>());
    Diagnostic diag(containingOp->getLoc(), DiagnosticSeverity::Remark);
    diag << "producer_op does not have uses in the container";
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }

  // Default diagnostic, to be complemented with more failure information.
  Diagnostic diag(producerOp->getLoc(), DiagnosticSeverity::Remark);
  diag << "could not fuse " << *producerOp << " into " << *containingOp;

  Operation *tiled =
      tileAndFuseFirstExtractUse(rewriter, diag, producerOp, containingOp);
  if (tiled) {
    LLVM_DEBUG(llvm::dbgs() << "\nFused a direct extract use\n"
                            << *containingOp);
    fusedOps.push_back(tiled);
    rewriter.eraseOp(producerOp);

    results.set(llvm::cast<OpResult>(getFusedOp()), fusedOps);
    return DiagnosedSilenceableFailure::success();
  }

  results.set(llvm::cast<OpResult>(getFusedOp()), ArrayRef<Operation *>());
  return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
}

//===----------------------------------------------------------------------===//
// HoistLoopInvariantTransfersOp / HoistAllAccumulatorTransfersOp
//===----------------------------------------------------------------------===//

// Forward declaration (defined in EliminateRedundantVectorTransfersOp section)
static bool areEquivalentIndices(Value idx1, Value idx2);

/// Check if a value depends on the given loop induction variable
static bool dependsOnLoopIV(Value val, Value loopIV) {
  if (val == loopIV)
    return true;

  // Check if the value is defined by an affine.apply that uses the loop IV
  if (auto affineOp = val.getDefiningOp<affine::AffineApplyOp>()) {
    for (Value operand : affineOp.getMapOperands()) {
      if (dependsOnLoopIV(operand, loopIV))
        return true;
    }
  }

  // Check for arithmetic operations
  if (auto defOp = val.getDefiningOp()) {
    for (Value operand : defOp->getOperands()) {
      if (dependsOnLoopIV(operand, loopIV))
        return true;
    }
  }

  return false;
}

/// Recursively clone an operation and its operands, using current insertion
/// point. Only clones operations that are inside the loop being hoisted from.
static Value cloneOpAndOperands(Operation *op, Value loopIV, scf::ForOp loopOp,
                                RewriterBase &rewriter, IRMapping &mapping) {
  // If already mapped, return the mapped value
  if (!op->getResults().empty())
    if (mapping.contains(op->getResult(0)))
      return mapping.lookup(op->getResult(0));

  // Clone operand-producing operations first
  for (Value operand : op->getOperands()) {
    if (operand == loopIV)
      continue; // Can't clone loop IV

    if (mapping.contains(operand))
      continue; // Already cloned

    // BlockArguments from enclosing loops are still in scope after hoisting -
    // use directly
    if (isa<BlockArgument>(operand) && operand != loopIV)
      continue; // BlockArguments from outer loops are still accessible

    Operation *defOp = operand.getDefiningOp();
    if (!defOp)
      continue;

    // If the defining operation is outside the loop we're hoisting from,
    // it's already in scope - use directly without cloning
    if (!loopOp->isAncestor(defOp))
      continue;

    if (!dependsOnLoopIV(operand, loopIV)) {
      Value clonedOperand =
          cloneOpAndOperands(defOp, loopIV, loopOp, rewriter, mapping);
      mapping.map(operand, clonedOperand);
    }
  }

  // Clone this operation at the current insertion point (don't reset it!)
  Operation *cloned = rewriter.clone(*op, mapping);
  if (cloned->getResults().empty())
    return nullptr;
  else
    return cloned->getResult(0);
}

/// Hoist a single transfer read/write pair out of a loop. The read is cloned
/// before the loop, the write is cloned after the loop, and an iter_arg is
/// added to carry the accumulator value through the loop body. The write goes
/// after `lastHoistedWrite` when that is set, so that writes hoisted one after
/// another keep their order in the loop body; it is updated to the new write.
/// Returns the new ForOp on success.
static FailureOr<scf::ForOp> hoistTransferPairFromLoop(
    vector::TransferReadOp readOp, vector::TransferWriteOp writeOp,
    scf::ForOp loopOp, Operation *&lastHoistedWrite, RewriterBase &rewriter) {
  Value loopIV = loopOp.getInductionVar();

  // Clone the read and its operands before the loop
  rewriter.setInsertionPoint(loopOp);
  IRMapping readMapping;
  Value clonedReadResult =
      cloneOpAndOperands(readOp, loopIV, loopOp, rewriter, readMapping);

  // Capture writeVector before replaceWithAdditionalYields
  Value writeVector = writeOp.getVector();
  auto yieldValuesFn =
      [&](OpBuilder &b, Location loc,
          ArrayRef<BlockArgument> newBbArgs) -> SmallVector<Value> {
    BlockArgument readIterArg = newBbArgs.back();
    rewriter.replaceAllUsesWith(readOp.getResult(), readIterArg);
    SmallVector<Value> yieldValues;
    yieldValues.push_back(writeVector);
    return yieldValues;
  };

  FailureOr<LoopLikeOpInterface> newLoopResult =
      cast<LoopLikeOpInterface>(loopOp.getOperation())
          .replaceWithAdditionalYields(rewriter, ValueRange{clonedReadResult},
                                       true, yieldValuesFn);
  if (failed(newLoopResult))
    return failure();

  auto newLoop = cast<scf::ForOp>(newLoopResult->getOperation());
  rewriter.eraseOp(readOp);

  // Clone the write operation after the loop using the yielded value
  Value valueToWrite = newLoop.getResults().back();
  IRMapping writeMapping;
  writeMapping.map(writeVector, valueToWrite);

  if (lastHoistedWrite)
    rewriter.setInsertionPointAfter(lastHoistedWrite);
  else
    rewriter.setInsertionPointAfter(newLoop);

  for (Value index : writeOp.getIndices()) {
    Operation *defOp = index.getDefiningOp();
    if (!defOp || dependsOnLoopIV(index, loopIV))
      continue;
    if (!newLoop->isProperAncestor(defOp))
      continue;
    if (!writeMapping.contains(index)) {
      Value clonedIndex =
          cloneOpAndOperands(defOp, loopIV, newLoop, rewriter, writeMapping);
      if (clonedIndex)
        writeMapping.map(index, clonedIndex);
    }
  }

  lastHoistedWrite = rewriter.clone(*writeOp.getOperation(), writeMapping);
  rewriter.eraseOp(writeOp);

  return newLoop;
}

DiagnosedSilenceableFailure transform::HoistLoopInvariantTransfersOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {

  SmallVector<Operation *> scopeOps =
      llvm::to_vector(state.getPayloadOps(getScopeOp()));
  SmallVector<Operation *> loopOps =
      llvm::to_vector(state.getPayloadOps(getLoopOp()));

  if (scopeOps.size() != 1 || loopOps.size() != 1) {
    return emitDefiniteFailure()
           << "requires exactly one scope_op and one loop_op handle";
  }

  auto scopeOp = scopeOps[0];
  auto loopOp = dyn_cast_if_present<scf::ForOp>(loopOps[0]);
  if (!loopOp) {
    return emitDefiniteFailure() << "loop_op must be an scf.for";
  }

  if (!scopeOp->isProperAncestor(loopOp)) {
    return emitDefiniteFailure() << "loop must be inside the scope operation";
  }

  // Iteratively discover and hoist one loop-invariant transfer pair at a time.
  // After each hoist, the loop is replaced with a new loop, so we re-discover
  // pairs in the new loop to avoid stale Operation* pointers.
  scf::ForOp currentLoop = loopOp;
  Operation *lastHoistedWrite = nullptr;

  while (true) {
    Value loopIV = currentLoop.getInductionVar();

    // Find one loop-invariant write and its paired read
    vector::TransferWriteOp foundWrite = nullptr;
    vector::TransferReadOp foundRead = nullptr;

    currentLoop->walk([&](vector::TransferWriteOp writeOp) {
      if (foundWrite)
        return;
      if (writeOp->getParentOfType<scf::ForOp>() != currentLoop)
        return;

      // Check all write indices are loop-invariant
      bool allInvariant = true;
      for (Value index : writeOp.getIndices()) {
        if (dependsOnLoopIV(index, loopIV)) {
          allInvariant = false;
          break;
        }
      }
      if (!allInvariant)
        return;

      // Find paired read with same memref and matching loop-invariant indices
      currentLoop->walk([&](vector::TransferReadOp readOp) {
        if (foundRead)
          return;
        if (readOp->getParentOfType<scf::ForOp>() != currentLoop)
          return;
        if (readOp.getBase() != writeOp.getBase())
          return;

        for (Value index : readOp.getIndices()) {
          if (dependsOnLoopIV(index, loopIV))
            return;
        }

        if (readOp.getIndices().size() != writeOp.getIndices().size())
          return;
        for (auto [ri, wi] :
             llvm::zip(readOp.getIndices(), writeOp.getIndices())) {
          if (!areEquivalentIndices(ri, wi))
            return;
        }

        foundRead = readOp;
      });

      if (foundRead)
        foundWrite = writeOp;
    });

    if (!foundWrite || !foundRead)
      break; // No more pairs to hoist

    FailureOr<scf::ForOp> newLoop = hoistTransferPairFromLoop(
        foundRead, foundWrite, currentLoop, lastHoistedWrite, rewriter);
    if (failed(newLoop)) {
      return emitDefiniteFailure() << "failed to hoist transfer pair";
    }
    currentLoop = *newLoop;
  }

  SmallVector<Operation *> resultOps = {currentLoop.getOperation()};
  results.set(llvm::cast<OpResult>(getResult()), resultOps);
  return DiagnosedSilenceableFailure::success();
}

void transform::HoistLoopInvariantTransfersOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getScopeOpMutable(), effects);
  onlyReadsHandle(getLoopOpMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// RemoveUninitializedMemrefCopyOp
//===----------------------------------------------------------------------===//

/// Trace a value back through subview operations to find the original
/// allocation
static memref::AllocOp traceToAlloc(Value value) {
  Value current = value;
  while (current) {
    if (auto allocOp = current.getDefiningOp<memref::AllocOp>()) {
      return allocOp;
    }
    if (auto subviewOp = current.getDefiningOp<memref::SubViewOp>()) {
      current = subviewOp.getSource();
      continue;
    }
    // Handle other view-like operations if needed
    break;
  }
  return nullptr;
}

/// Check if an operation has a write effect on the given value or any value
/// derived from it
static bool hasWriteEffectOn(Operation *op, Value allocResult) {
  SmallVector<MemoryEffects::EffectInstance> effects;
  auto memInterface = dyn_cast_if_present<MemoryEffectOpInterface>(op);
  if (!memInterface) {
    // If the operation doesn't implement the memory effect interface,
    // conservatively assume it might have side effects unless it's pure
    return !op->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
           !op->hasTrait<OpTrait::HasRecursiveMemoryEffects>();
  }

  memInterface.getEffects(effects);

  for (auto &effect : effects) {
    // Check if this is a write effect
    if (!isa<MemoryEffects::Write>(effect.getEffect())) {
      continue;
    }

    // Check if the effect is on the value we're interested in
    Value effectValue = effect.getValue();
    if (!effectValue) {
      // Effect on unknown memory - conservatively assume it could affect our
      // value
      return true;
    }

    // Check if the effect is on our allocation or a derived value
    if (effectValue == allocResult ||
        traceToAlloc(effectValue) == traceToAlloc(allocResult)) {
      return true;
    }
  }

  return false;
}

/// Helper functions to extract source and target from different copy operation
/// types
static Value getCopySource(memref::CopyOp copyOp) { return copyOp.getSource(); }

static Value getCopySource(linalg::CopyOp copyOp) {
  return copyOp.getInputs()[0];
}

/// Get the target/destination of a copy operation
static Value getCopyTarget(memref::CopyOp copyOp) { return copyOp.getTarget(); }

static Value getCopyTarget(linalg::CopyOp copyOp) {
  return copyOp.getOutputs()[0];
}

/// Enum to represent different write patterns for copy optimization
enum class WritePattern {
  NoWrites,      // Uninitialized - can remove copy
  SingleFill,    // Single fill - can replace copy with fill
  MultipleWrites // Multiple writes - cannot optimize
};

/// Structure to hold the result of write analysis
struct WriteAnalysis {
  WritePattern pattern;
  linalg::FillOp fillOp; // Only valid if pattern == SingleFill
};

/// Unified analysis function that examines writes between allocation and copy
/// This replaces the redundant logic in hasWritesBetween() and
/// findSingleFillBetween()
static WriteAnalysis analyzeWritesBetween(memref::AllocOp allocOp,
                                          Operation *beforeOp) {
  Value allocResult = allocOp.getResult();

  // Get the function containing both operations
  auto funcOp = allocOp->getParentOfType<func::FuncOp>();
  if (!funcOp || funcOp != beforeOp->getParentOfType<func::FuncOp>()) {
    // If they're in different functions, conservatively assume multiple writes
    return {WritePattern::MultipleWrites, nullptr};
  }

  linalg::FillOp foundFill = nullptr;
  int writeCount = 0;

  // Single walk through all operations to analyze write pattern
  funcOp.walk([&](Operation *op) {
    // Skip if we've already found multiple writes
    if (writeCount > 1)
      return;

    // Skip the allocation itself
    if (op == allocOp)
      return;

    // Only consider operations that are dominated by the allocation
    // and that dominate the beforeOp
    if (!xilinx::air::opOrAncestorIsDominantOver(allocOp.getOperation(), op))
      return;
    if (!xilinx::air::opOrAncestorIsDominantOver(op, beforeOp))
      return;

    // Check if this operation writes to our allocation
    if (hasWriteEffectOn(op, allocResult)) {
      bool isWriteToAlloc = false;

      // Check if this operation writes to our allocation
      if (auto fillOp = dyn_cast_if_present<linalg::FillOp>(op)) {
        // Verify this fill writes to our allocation
        if (traceToAlloc(fillOp.getOutputs()[0]) == allocOp) {
          isWriteToAlloc = true;
          // If this is the first write, record the fill
          if (writeCount == 0)
            foundFill = fillOp;
        }
      } else {
        // For other ops, check if any operand/result traces to allocOp
        for (Value operand : op->getOperands()) {
          if (traceToAlloc(operand) == allocOp) {
            isWriteToAlloc = true;
            break;
          }
        }
      }

      if (isWriteToAlloc) {
        writeCount++;
      }
    }
  });

  // Determine the pattern based on analysis
  if (writeCount == 0) {
    return {WritePattern::NoWrites, nullptr};
  } else if (writeCount == 1 && foundFill) {
    return {WritePattern::SingleFill, foundFill};
  } else {
    return {WritePattern::MultipleWrites, nullptr};
  }
}

/// Unified pattern to optimize copy operations based on write analysis
/// This replaces RemoveUninitializedCopyOpPattern and
/// ReplaceCopyWithFillPattern
template <typename CopyOpType>
struct OptimizeCopyOpPattern : public OpRewritePattern<CopyOpType> {
  using OpRewritePattern<CopyOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOpType copyOp,
                                PatternRewriter &rewriter) const override {
    Value source = getCopySource(copyOp);
    Value target = getCopyTarget(copyOp);

    // Trace the source back to its allocation
    memref::AllocOp allocOp = traceToAlloc(source);
    if (!allocOp) {
      return failure();
    }

    // Perform unified write analysis
    WriteAnalysis analysis = analyzeWritesBetween(allocOp, copyOp);

    switch (analysis.pattern) {
    case WritePattern::NoWrites:
      // Source is uninitialized - remove the copy
      rewriter.eraseOp(copyOp);
      return success();

    case WritePattern::SingleFill:
      // Source is only written by a single fill - replace copy with fill
      {
        Value fillValue = analysis.fillOp.getInputs()[0];
        rewriter.setInsertionPoint(copyOp);
        linalg::FillOp::create(rewriter, copyOp.getLoc(), fillValue, target);
        rewriter.eraseOp(copyOp);
        return success();
      }

    case WritePattern::MultipleWrites:
      // Cannot optimize - multiple writes to source
      return failure();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// RaiseLoadBufferToTensorOp
//===----------------------------------------------------------------------===//

// Returns the op that fills all of `alloc` if `op` is one: a linalg.fill of
// the whole buffer, or an scf.if that holds only such a fill. Returns null
// otherwise.
static linalg::FillOp asWholeFill(Operation *op, Value alloc) {
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (ifOp.getNumResults() || !ifOp.getElseRegion().empty())
      return nullptr;
    Block &then = ifOp.getThenRegion().front();
    if (then.getOperations().size() != 2)
      return nullptr;
    op = &then.front();
  }
  auto fill = dyn_cast<linalg::FillOp>(op);
  if (!fill || !fill.hasPureBufferSemantics() || fill.getDpsInits()[0] != alloc)
    return nullptr;
  return fill;
}

static bool raiseLoadBuffer(RewriterBase &rewriter, memref::AllocOp alloc) {
  MemRefType ty = alloc.getType();
  if (!ty.hasStaticShape() || !ty.getLayout().isIdentity())
    return false;
  Block *block = alloc->getBlock();

  struct Box {
    memref::CopyOp copy;
    memref::SubViewOp view; // null when the copy writes the whole buffer
  };
  linalg::FillOp fill;
  Operation *fillRoot = nullptr; // the fill, or the scf.if around it
  SmallVector<Box> boxes;
  bufferization::ToTensorOp read;
  SmallVector<memref::DeallocOp> deallocs;

  for (OpOperand &use : alloc->getUses()) {
    Operation *user = use.getOwner();
    if (auto f = dyn_cast<linalg::FillOp>(user)) {
      Operation *root = f->getParentOp() != alloc->getParentOp() &&
                                isa<scf::IfOp>(f->getParentOp())
                            ? f->getParentOp()
                            : f.getOperation();
      if (fill || !asWholeFill(root, alloc))
        return false;
      fill = f;
      fillRoot = root;
    } else if (auto copy = dyn_cast<memref::CopyOp>(user)) {
      if (copy.getTarget() != alloc.getResult() ||
          copy.getSource() == alloc.getResult())
        return false;
      boxes.push_back({copy, nullptr});
    } else if (auto view = dyn_cast<memref::SubViewOp>(user)) {
      if (!view->hasOneUse())
        return false;
      auto copy = dyn_cast<memref::CopyOp>(*view->user_begin());
      if (!copy || copy.getTarget() != view.getResult() ||
          view.getType().getRank() != ty.getRank() ||
          llvm::any_of(view.getMixedStrides(), [](OpFoldResult s) {
            return getConstantIntValue(s) != 1;
          }))
        return false;
      boxes.push_back({copy, view});
    } else if (auto toTensor = dyn_cast<bufferization::ToTensorOp>(user)) {
      if (read || !toTensor.getRestrict())
        return false;
      read = toTensor;
    } else if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      deallocs.push_back(dealloc);
    } else {
      return false;
    }
  }
  if (!read || boxes.empty() || read->getBlock() != block)
    return false;
  // Every write runs once, in the buffer's block, before the read; the fill
  // (or the scf.if around it) comes before every copy.
  auto before = [&](Operation *op) {
    return op->getBlock() == block && op->isBeforeInBlock(read);
  };
  if ((fillRoot && !before(fillRoot)) ||
      llvm::any_of(boxes,
                   [&](const Box &b) {
                     return !before(b.copy) ||
                            (b.view && b.view->getBlock() != block) ||
                            (fillRoot && b.copy->isBeforeInBlock(fillRoot));
                   }) ||
      llvm::any_of(deallocs, [&](memref::DeallocOp d) {
        Operation *a = block->findAncestorOpInBlock(*d);
        return !a || a->isBeforeInBlock(read);
      }))
    return false;
  // Insert the boxes in program order, so a later box overwrites an earlier
  // one.
  llvm::sort(boxes, [&](const Box &a, const Box &b) {
    return a.copy->isBeforeInBlock(b.copy);
  });

  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = read.getLoc();
  // Each box's source is copied where the original copy was, so it is read
  // at the same point; the parts are assembled at the read.
  SmallVector<Value> bufs;
  for (Box &b : boxes) {
    rewriter.setInsertionPoint(b.copy);
    auto viewTy = cast<MemRefType>(b.copy.getTarget().getType());
    SmallVector<Value> dynSizes;
    if (b.view)
      for (OpFoldResult sz : b.view.getMixedSizes())
        if (auto v = dyn_cast<Value>(sz))
          dynSizes.push_back(v);
    auto boxTy =
        MemRefType::get(viewTy.getShape(), ty.getElementType(),
                        MemRefLayoutAttrInterface(), ty.getMemorySpace());
    Value buf = memref::AllocOp::create(rewriter, b.copy.getLoc(), boxTy,
                                        dynSizes, alloc.getAlignmentAttr());
    memref::CopyOp::create(rewriter, b.copy.getLoc(), b.copy.getSource(), buf);
    bufs.push_back(buf);
  }
  rewriter.setInsertionPoint(read);
  Value result = tensor::EmptyOp::create(rewriter, loc, ty.getShape(),
                                         ty.getElementType());
  if (fill)
    result =
        linalg::FillOp::create(rewriter, loc, ValueRange{fill.getInputs()[0]},
                               ValueRange{result})
            .getResult(0);
  for (auto [b, buf] : llvm::zip_equal(boxes, bufs)) {
    SmallVector<OpFoldResult> offsets, sizes, strides;
    if (b.view) {
      offsets = b.view.getMixedOffsets();
      sizes = b.view.getMixedSizes();
      strides = b.view.getMixedStrides();
    } else {
      for (int64_t d : ty.getShape()) {
        offsets.push_back(rewriter.getIndexAttr(0));
        sizes.push_back(rewriter.getIndexAttr(d));
        strides.push_back(rewriter.getIndexAttr(1));
      }
    }
    auto shape = cast<MemRefType>(buf.getType()).getShape();
    Value part = bufferization::ToTensorOp::create(
        rewriter, loc, RankedTensorType::get(shape, ty.getElementType()), buf,
        /*restrict=*/true, /*writable=*/true);
    result = tensor::InsertSliceOp::create(rewriter, loc, part, result, offsets,
                                           sizes, strides);
  }
  rewriter.replaceAllUsesWith(read.getResult(), result);

  rewriter.eraseOp(read);
  for (Box &b : boxes) {
    rewriter.eraseOp(b.copy);
    if (b.view)
      rewriter.eraseOp(b.view);
  }
  if (fillRoot)
    rewriter.eraseOp(fillRoot);
  for (memref::DeallocOp d : deallocs)
    rewriter.eraseOp(d);
  rewriter.eraseOp(alloc);
  return true;
}

DiagnosedSilenceableFailure transform::RaiseLoadBufferToTensorOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  SmallVector<Operation *> targets;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    // The target itself stays: the result handle holds it.
    SmallVector<memref::AllocOp> allocs;
    target->walk([&](memref::AllocOp alloc) {
      if (alloc.getOperation() != target)
        allocs.push_back(alloc);
    });
    for (memref::AllocOp alloc : allocs)
      raiseLoadBuffer(rewriter, alloc);
    targets.push_back(target);
  }
  results.set(llvm::cast<OpResult>(getResult()), targets);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// FoldExtractSliceOfInsertSliceOp
//===----------------------------------------------------------------------===//

namespace {
struct FoldExtractSliceOfInsertSlice
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extract,
                                PatternRewriter &rewriter) const override {
    auto insert = extract.getSource().getDefiningOp<tensor::InsertSliceOp>();
    if (!insert)
      return failure();
    // The same box read back with the source's type is folded by
    // tensor.extract_slice's own folder.
    if (extract.getType() == insert.getSourceType() &&
        extract.isSameAs(insert,
                         [](OpFoldResult a, OpFoldResult b) { return a == b; }))
      return failure();
    // Disjoint boxes, possibly at offsets that are only bounded: the extract
    // reads the insert's destination.
    FailureOr<bool> overlap = ValueBoundsConstraintSet::areOverlappingSlices(
        extract.getContext(), HyperrectangularSlice(extract),
        HyperrectangularSlice(insert));
    if (succeeded(overlap) && !*overlap) {
      rewriter.modifyOpInPlace(extract, [&] {
        extract.getSourceMutable().assign(insert.getDest());
      });
      return success();
    }
    // Per dim, the range [lo, hi) the slice can cover. A non-constant offset
    // must be bounded (a forall induction variable, say); the range is then
    // the union of the positions the slice can take.
    auto box = [](ArrayRef<OpFoldResult> offs, ArrayRef<OpFoldResult> sizes,
                  ArrayRef<OpFoldResult> strides)
        -> std::optional<SmallVector<std::pair<int64_t, int64_t>>> {
      SmallVector<std::pair<int64_t, int64_t>> b;
      for (auto [o, sz, st] : llvm::zip(offs, sizes, strides)) {
        auto sc = getConstantIntValue(sz), tc = getConstantIntValue(st);
        if (!sc || !tc || *tc != 1)
          return std::nullopt;
        int64_t lo, hi;
        if (auto c = getConstantIntValue(o)) {
          lo = hi = *c;
        } else {
          FailureOr<int64_t> l = ValueBoundsConstraintSet::computeConstantBound(
              presburger::BoundType::LB, {o, std::nullopt});
          FailureOr<int64_t> u = ValueBoundsConstraintSet::computeConstantBound(
              presburger::BoundType::UB, {o, std::nullopt},
              /*stopCondition=*/nullptr, ValueBoundsOptions{/*closedUB=*/true});
          if (failed(l) || failed(u))
            return std::nullopt;
          lo = *l;
          hi = *u;
        }
        b.push_back({lo, hi + *sc});
      }
      return b;
    };
    auto e = box(extract.getMixedOffsets(), extract.getMixedSizes(),
                 extract.getMixedStrides());
    auto i = box(insert.getMixedOffsets(), insert.getMixedSizes(),
                 insert.getMixedStrides());
    if (!e || !i || e->size() != i->size())
      return failure();
    bool inside = true;
    for (auto [eb, ib] : llvm::zip(*e, *i))
      if (eb.first < ib.first || eb.second > ib.second)
        inside = false;
    if (!inside)
      return failure();
    // The insert's offsets must be constants: with a bounded offset the range
    // is a union of positions, not where the source is.
    if (!llvm::all_of(insert.getMixedOffsets(), [](OpFoldResult o) {
          return getConstantIntValue(o).has_value();
        }))
      return failure();
    // A rank-reducing insert drops unit dims that the extract keeps at size
    // 1, so the slice of the source is expanded back to the extract's rank.
    // An extract that drops dims as well is left alone.
    llvm::SmallBitVector dropped = insert.getDroppedDims();
    auto rt = extract.getType();
    if (dropped.any() &&
        (rt.getRank() != extract.getSourceType().getRank() || dropped.all()))
      return failure();
    SmallVector<OpFoldResult> offs, sizes, strides;
    for (auto [d, eo, ib] : llvm::enumerate(extract.getMixedOffsets(), *i)) {
      if (dropped.test(d))
        continue;
      AffineExpr d0 = rewriter.getAffineDimExpr(0);
      offs.push_back(affine::makeComposedFoldedAffineApply(
          rewriter, extract.getLoc(), d0 - ib.first, {eo}));
      sizes.push_back(extract.getMixedSizes()[d]);
      strides.push_back(extract.getMixedStrides()[d]);
    }
    if (dropped.none()) {
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          extract, extract.getType(), insert.getSource(), offs, sizes, strides);
      return success();
    }
    // Each kept dim is grouped with the dropped unit dims before it; trailing
    // dropped dims join the last group.
    SmallVector<ReassociationIndices> groups;
    ReassociationIndices cur;
    for (int64_t d = 0; d < rt.getRank(); ++d) {
      cur.push_back(d);
      if (!dropped.test(d)) {
        groups.push_back(cur);
        cur.clear();
      }
    }
    groups.back().append(cur.begin(), cur.end());
    SmallVector<int64_t> shape;
    for (int64_t d = 0; d < rt.getRank(); ++d)
      if (!dropped.test(d))
        shape.push_back(rt.getDimSize(d));
    auto part = tensor::ExtractSliceOp::create(
        rewriter, extract.getLoc(),
        RankedTensorType::get(shape, rt.getElementType()), insert.getSource(),
        offs, sizes, strides);
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(extract, rt, part,
                                                       groups);
    return success();
  }
};

// insert(x at S, insert(y at T, insert(w at S, z))) with S and T disjoint
// boxes overwrites w, so it is insert(y at T, insert(x at S, z)). A value
// assembled from disjoint parts that are each written more than once then
// writes each part once.
struct DropInsertOverwrittenPastDisjointInsert
    : public OpRewritePattern<tensor::InsertSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::InsertSliceOp outer,
                                PatternRewriter &rewriter) const override {
    auto mid = outer.getDest().getDefiningOp<tensor::InsertSliceOp>();
    if (!mid || !mid->hasOneUse())
      return failure();
    auto inner = mid.getDest().getDefiningOp<tensor::InsertSliceOp>();
    if (!inner || !inner->hasOneUse())
      return failure();
    MLIRContext *ctx = outer.getContext();
    HyperrectangularSlice s(outer), t(mid), w(inner);
    FailureOr<bool> same =
        ValueBoundsConstraintSet::areEquivalentSlices(ctx, s, w);
    FailureOr<bool> overlap =
        ValueBoundsConstraintSet::areOverlappingSlices(ctx, s, t);
    if (failed(same) || !*same || failed(overlap) || *overlap)
      return failure();
    auto x = tensor::InsertSliceOp::create(
        rewriter, outer.getLoc(), outer.getSource(), inner.getDest(),
        outer.getMixedOffsets(), outer.getMixedSizes(),
        outer.getMixedStrides());
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        outer, mid.getSource(), x.getResult(), mid.getMixedOffsets(),
        mid.getMixedSizes(), mid.getMixedStrides());
    return success();
  }
};

// A slice of a tensor.expand_shape that only adds unit dims (each group is
// one source dim plus unit dims the slice keeps at offset 0 or drops) reads
// the source directly.
struct FoldExtractSliceOfUnitExpand
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp extract,
                                PatternRewriter &rewriter) const override {
    auto expand = extract.getSource().getDefiningOp<tensor::ExpandShapeOp>();
    if (!expand || !extract.hasUnitStride())
      return failure();
    auto outType = expand.getResultType();
    auto srcType = expand.getSrcType();
    SmallVector<OpFoldResult> offs, sizes, strides;
    for (auto [g, group] : llvm::enumerate(expand.getReassociationIndices())) {
      std::optional<int64_t> kept;
      for (int64_t d : group) {
        if (outType.getDimSize(d) != 1) {
          if (kept)
            return failure();
          kept = d;
          continue;
        }
        if (!isConstantIntValue(extract.getMixedOffsets()[d], 0))
          return failure();
      }
      if (!kept) {
        // A group of only unit dims: the source dim has size 1.
        if (srcType.getDimSize(g) != 1)
          return failure();
        offs.push_back(rewriter.getIndexAttr(0));
        sizes.push_back(rewriter.getIndexAttr(1));
      } else {
        offs.push_back(extract.getMixedOffsets()[*kept]);
        sizes.push_back(extract.getMixedSizes()[*kept]);
      }
      strides.push_back(rewriter.getIndexAttr(1));
    }
    SmallVector<int64_t> staticSizes;
    for (OpFoldResult sz : sizes) {
      auto c = getConstantIntValue(sz);
      staticSizes.push_back(c ? *c : ShapedType::kDynamic);
    }
    // The new slice drops the source dims of size 1 that the old one drops,
    // including those of all-unit groups; its type must be the old result
    // type.
    auto full = RankedTensorType::get(staticSizes, srcType.getElementType());
    if (!full.hasStaticShape() || !extract.getType().hasStaticShape() ||
        full.getNumElements() != extract.getType().getNumElements())
      return failure();
    SmallVector<int64_t> nonUnitFull, nonUnitRes;
    for (int64_t v : full.getShape())
      if (v != 1)
        nonUnitFull.push_back(v);
    for (int64_t v : extract.getType().getShape())
      if (v != 1)
        nonUnitRes.push_back(v);
    if (nonUnitFull != nonUnitRes ||
        extract.getType().getRank() > full.getRank())
      return failure();
    // Reshape back to the old result's unit dims.
    std::optional<SmallVector<ReassociationIndices>> reassoc;
    if (full != extract.getType()) {
      reassoc = getReassociationIndicesForReshape(full, extract.getType());
      if (!reassoc)
        return failure();
    }
    auto direct =
        tensor::ExtractSliceOp::create(rewriter, extract.getLoc(), full,
                                       expand.getSrc(), offs, sizes, strides);
    if (!reassoc) {
      rewriter.replaceOp(extract, direct.getResult());
      return success();
    }
    if (full.getRank() > extract.getType().getRank())
      rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
          extract, extract.getType(), direct.getResult(), *reassoc);
    else
      rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
          extract, extract.getType(), direct.getResult(), *reassoc);
    return success();
  }
};
} // namespace

// Applies `patterns` greedily within each payload op of `target`; the result
// handle is the payload ops.
static DiagnosedSilenceableFailure
applyPatternsToTargets(transform::TransformOpInterface op, Value target,
                       Value result, transform::TransformResults &results,
                       transform::TransformState &state,
                       function_ref<void(RewritePatternSet &)> populate) {
  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(target));
  for (Operation *payload : targets) {
    RewritePatternSet patterns(op->getContext());
    populate(patterns);
    if (failed(applyPatternsGreedily(payload, std::move(patterns))))
      return emitDefiniteFailure(op) << "patterns did not converge";
  }
  results.set(llvm::cast<OpResult>(result), targets);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::FoldExtractSliceOfInsertSliceOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  return applyPatternsToTargets(
      *this, getTarget(), getResult(), results, state,
      [](RewritePatternSet &patterns) {
        patterns.add<FoldExtractSliceOfInsertSlice,
                     DropInsertOverwrittenPastDisjointInsert,
                     FoldExtractSliceOfUnitExpand>(patterns.getContext());
      });
}

//===----------------------------------------------------------------------===//
// ShareAllocOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ShareAllocOp::apply(transform::TransformRewriter &rewriter,
                               transform::TransformResults &results,
                               transform::TransformState &state) {
  SmallVector<memref::AllocOp> allocs;
  for (Operation *op : state.getPayloadOps(getTarget())) {
    auto alloc = dyn_cast<memref::AllocOp>(op);
    if (!alloc)
      return emitDefiniteFailure() << "target must be a memref.alloc";
    allocs.push_back(alloc);
  }
  if (allocs.empty()) {
    results.set(llvm::cast<OpResult>(getResult()), {});
    return DiagnosedSilenceableFailure::success();
  }
  Block *block = allocs.front()->getBlock();
  for (auto a : allocs)
    if (a->getBlock() != block)
      return emitDefiniteFailure() << "targets must be in one block";
  // The shared buffer is freed once, after the last free of a target, which
  // needs every target freed directly, or none.
  int freed = 0;
  for (auto a : allocs) {
    bool direct = llvm::any_of(
        a->getUsers(), [](Operation *u) { return isa<memref::DeallocOp>(u); });
    freed += direct;
    SmallVector<Value> aliases;
    for (Operation *u : a->getUsers())
      if (isa<ViewLikeOpInterface, CastOpInterface>(u))
        llvm::append_range(aliases, u->getResults());
    while (!aliases.empty()) {
      Value v = aliases.pop_back_val();
      for (Operation *u : v.getUsers()) {
        if (isa<memref::DeallocOp>(u))
          return emitDefiniteFailure()
                 << "targets must not be freed through a view or cast";
        if (isa<ViewLikeOpInterface, CastOpInterface>(u))
          llvm::append_range(aliases, u->getResults());
      }
    }
  }
  if (freed != 0 && freed != (int)allocs.size())
    return emitDefiniteFailure() << "either every target or none must be freed";
  // Keep the first target, moved ahead of the others in the block.
  memref::AllocOp kept = allocs.front();
  for (auto a : allocs)
    if (a->isBeforeInBlock(kept))
      rewriter.moveOpBefore(kept, a);
  MemRefType keptTy = kept.getType();
  auto plain = [](MemRefType t) {
    return t.hasStaticShape() && t.getLayout().isIdentity() && t.getRank() > 0;
  };
  if (!plain(keptTy))
    return emitDefiniteFailure()
           << "targets must have static shapes and identity layouts";
  for (auto a : ArrayRef(allocs).drop_front()) {
    MemRefType t = a.getType();
    if (!plain(t) || t.getElementType() != keptTy.getElementType() ||
        t.getMemorySpace() != keptTy.getMemorySpace() ||
        t.getNumElements() != keptTy.getNumElements())
      return emitDefiniteFailure()
             << "targets must have one element type, count and memory space";
  }
  // The kept buffer takes the largest alignment asked for, and is freed
  // where the last of the buffers was.
  std::optional<uint64_t> align;
  for (auto a : allocs)
    if (auto al = a.getAlignment())
      align = std::max(align.value_or(0), *al);
  if (align)
    rewriter.modifyOpInPlace(kept, [&] { kept.setAlignment(*align); });
  SmallVector<memref::DeallocOp> deallocs;
  for (auto a : allocs)
    for (Operation *user : a->getUsers())
      if (auto d = dyn_cast<memref::DeallocOp>(user))
        deallocs.push_back(d);
  Operation *lastFree = nullptr;
  for (memref::DeallocOp d : deallocs) {
    if (d->getBlock() != block)
      return emitDefiniteFailure() << "targets must be freed in their block";
    if (!lastFree || lastFree->isBeforeInBlock(d))
      lastFree = d;
  }
  if (lastFree) {
    rewriter.setInsertionPointAfter(lastFree);
    memref::DeallocOp::create(rewriter, lastFree->getLoc(), kept.getResult());
    for (memref::DeallocOp d : deallocs)
      rewriter.eraseOp(d);
  }
  for (auto a : ArrayRef(allocs).drop_front()) {
    MemRefType t = a.getType();
    Value v = kept.getResult();
    rewriter.setInsertionPoint(a);
    if (t != keptTy) {
      // Go through one dimension: any two shapes with the same number of
      // elements are reshapes of it.
      if (keptTy.getRank() > 1) {
        SmallVector<ReassociationIndices> toFlat(1);
        for (int64_t i = 0; i < keptTy.getRank(); i++)
          toFlat[0].push_back(i);
        v = memref::CollapseShapeOp::create(rewriter, a.getLoc(), v, toFlat);
      }
      if (t.getRank() > 1) {
        SmallVector<ReassociationIndices> fromFlat(1);
        for (int64_t i = 0; i < t.getRank(); i++)
          fromFlat[0].push_back(i);
        v = memref::ExpandShapeOp::create(rewriter, a.getLoc(), t.getShape(), v,
                                          fromFlat);
      }
    }
    rewriter.replaceOp(a, v);
  }
  results.set(llvm::cast<OpResult>(getResult()), {kept.getOperation()});
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// UnswitchLoopOp
//===----------------------------------------------------------------------===//

// Returns the scalar i1 condition of `sel` if it is defined outside `loop`,
// directly or broadcast to a vector; null otherwise.
static Value loopInvariantSelectCondition(arith::SelectOp sel,
                                          scf::ForOp loop) {
  Value c = sel.getCondition();
  if (auto b = c.getDefiningOp<vector::BroadcastOp>())
    c = b.getSource();
  if (!c.getType().isInteger(1) || !loop.isDefinedOutsideOfLoop(c))
    return Value();
  return c;
}

DiagnosedSilenceableFailure
transform::UnswitchLoopOp::apply(transform::TransformRewriter &rewriter,
                                 transform::TransformResults &results,
                                 transform::TransformState &state) {
  // Unswitching a loop clones the loops inside it, so targets must not nest.
  auto targets = llvm::to_vector(state.getPayloadOps(getTarget()));
  for (Operation *a : targets)
    for (Operation *b : targets)
      if (a != b && a->isAncestor(b))
        return emitDefiniteFailure() << "targets must not be nested";
  SmallVector<Operation *> transformed;
  for (Operation *target : targets) {
    auto loop = dyn_cast<scf::ForOp>(target);
    if (!loop)
      return emitDefiniteFailure() << "target must be an scf.for";
    Value cond;
    loop.walk([&](arith::SelectOp sel) {
      if (!cond)
        cond = loopInvariantSelectCondition(sel, loop);
    });
    if (!cond) {
      transformed.push_back(loop);
      continue;
    }
    Location loc = loop.getLoc();
    rewriter.setInsertionPoint(loop);
    auto ifOp = scf::IfOp::create(rewriter, loc, loop.getResultTypes(), cond,
                                  /*withElseRegion=*/true);
    for (bool branch : {true, false}) {
      Block *block = branch ? ifOp.thenBlock() : ifOp.elseBlock();
      if (block->mightHaveTerminator())
        rewriter.setInsertionPoint(block->getTerminator());
      else
        rewriter.setInsertionPointToEnd(block);
      auto copy = cast<scf::ForOp>(rewriter.clone(*loop));
      SmallVector<arith::SelectOp> sels;
      copy.walk([&](arith::SelectOp sel) {
        if (loopInvariantSelectCondition(sel, copy) == cond)
          sels.push_back(sel);
      });
      for (arith::SelectOp sel : sels)
        rewriter.replaceOp(sel,
                           branch ? sel.getTrueValue() : sel.getFalseValue());
      if (!block->mightHaveTerminator())
        scf::YieldOp::create(rewriter, loc, copy.getResults());
    }
    rewriter.replaceOp(loop, ifOp.getResults());
    transformed.push_back(ifOp);
  }
  results.set(llvm::cast<OpResult>(getResult()), transformed);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// SwapTransposeWithCollapseOp
//===----------------------------------------------------------------------===//

namespace {
// transpose(collapse_shape(x)) -> collapse_shape(transpose'(x)), where
// transpose' moves each reassociation group of x the way the transpose moves
// the group's collapsed dim.
struct SwapTransposeWithCollapse
    : public OpRewritePattern<linalg::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::TransposeOp transpose,
                                PatternRewriter &rewriter) const override {
    if (!transpose.hasPureTensorSemantics())
      return failure();
    auto collapse =
        transpose.getInput().getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapse)
      return failure();
    auto srcType = collapse.getSrcType();
    if (!srcType.hasStaticShape())
      return failure();
    SmallVector<ReassociationIndices> groups =
        collapse.getReassociationIndices();
    ArrayRef<int64_t> perm = transpose.getPermutation();
    SmallVector<int64_t> srcPerm, shape;
    SmallVector<ReassociationIndices> newGroups;
    for (int64_t d : perm) {
      ReassociationIndices g;
      for (int64_t s : groups[d]) {
        g.push_back(srcPerm.size());
        srcPerm.push_back(s);
        shape.push_back(srcType.getDimSize(s));
      }
      newGroups.push_back(g);
    }
    Location loc = transpose.getLoc();
    Value init =
        tensor::EmptyOp::create(rewriter, loc, shape, srcType.getElementType());
    auto moved = linalg::TransposeOp::create(rewriter, loc, collapse.getSrc(),
                                             init, srcPerm);
    moved->setDiscardableAttrs(transpose->getDiscardableAttrDictionary());
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        transpose, transpose.getResult()[0].getType(), moved.getResult()[0],
        newGroups);
    return success();
  }
};
} // namespace

DiagnosedSilenceableFailure transform::SwapTransposeWithCollapseOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  return applyPatternsToTargets(*this, getTarget(), getResult(), results, state,
                                [](RewritePatternSet &patterns) {
                                  patterns.add<SwapTransposeWithCollapse>(
                                      patterns.getContext());
                                });
}

//===----------------------------------------------------------------------===//
// BubbleSliceThroughCollapseOp
//===----------------------------------------------------------------------===//

namespace {
// Whether `v` is provably a multiple of `m`.
static bool isMultipleOf(Value v, int64_t m) {
  if (m == 1)
    return true;
  if (auto c = getConstantIntValue(v))
    return *c % m == 0;
  // Integer arithmetic wraps, which keeps divisibility only by a power of two.
  // For other divisors, `v` must provably stay far inside the index range.
  if (!llvm::isPowerOf2_64(m)) {
    auto bound = [&](presburger::BoundType t) {
      return ValueBoundsConstraintSet::computeConstantBound(
          t, {v, std::nullopt}, /*stopCondition=*/nullptr,
          ValueBoundsOptions{/*closedUB=*/true});
    };
    FailureOr<int64_t> lo = bound(presburger::BoundType::LB);
    FailureOr<int64_t> hi = bound(presburger::BoundType::UB);
    constexpr int64_t kSafe = int64_t(1) << 31;
    if (failed(lo) || failed(hi) || *lo < -kSafe || *hi > kSafe)
      return false;
  }
  if (auto arg = dyn_cast<BlockArgument>(v))
    if (auto loop = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp()))
      if (arg == loop.getInductionVar())
        return isMultipleOf(loop.getLowerBound(), m) &&
               isMultipleOf(loop.getStep(), m);
  if (auto add = v.getDefiningOp<arith::AddIOp>())
    return isMultipleOf(add.getLhs(), m) && isMultipleOf(add.getRhs(), m);
  if (auto mul = v.getDefiningOp<arith::MulIOp>())
    return isMultipleOf(mul.getLhs(), m) || isMultipleOf(mul.getRhs(), m);
  return false;
}

// extract_slice(collapse_shape(x)) -> collapse_shape(extract_slice'(x)) when,
// for each collapsed dim, the slice covers whole rows of the group's dims after
// some dim `at` and does not cross an index boundary of the dims before `at`.
// The new slice takes one index of each dim before `at`, a range of `at`, and
// all of each dim after it.
struct BubbleSliceThroughCollapse
    : public OpRewritePattern<tensor::ExtractSliceOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::ExtractSliceOp slice,
                                PatternRewriter &rewriter) const override {
    auto collapse = slice.getSource().getDefiningOp<tensor::CollapseShapeOp>();
    if (!collapse)
      return failure();
    RankedTensorType srcType = collapse.getSrcType();
    if (!srcType.hasStaticShape() ||
        slice.getType().getRank() != collapse.getType().getRank())
      return failure();
    // Constant offsets are handled by the upstream bubble-up patterns.
    if (llvm::all_of(slice.getMixedOffsets(), [](OpFoldResult o) {
          return getConstantIntValue(o).has_value();
        }))
      return failure();
    if (srcType.getNumElements() == 0)
      return failure();
    // For each group: the position of dim `at` and the number of elements in
    // the dims after it. All groups are checked before any IR is created.
    SmallVector<std::pair<size_t, int64_t>> splits;
    for (auto [i, group] :
         llvm::enumerate(collapse.getReassociationIndices())) {
      OpFoldResult off = slice.getMixedOffsets()[i];
      if (!isConstantIntValue(slice.getMixedStrides()[i], 1))
        return failure();
      auto s = getConstantIntValue(slice.getMixedSizes()[i]);
      if (!s)
        return failure();
      // Find the outermost dim `at` such that the slice covers whole rows of
      // the dims after it. For `at` > 0 the number of indices the slice takes
      // in dim `at` must also divide its extent, so that an offset aligned to
      // the slice size never crosses into the next index of the dims before
      // it.
      std::optional<std::pair<size_t, int64_t>> split;
      for (size_t at = 0; at < group.size() && !split; ++at) {
        int64_t inner = 1;
        for (int64_t d : ArrayRef(group).drop_front(at + 1))
          inner *= srcType.getDimSize(d);
        int64_t dim = srcType.getDimSize(group[at]);
        if (*s % inner != 0 || *s / inner > dim)
          continue;
        if (at > 0 && dim % (*s / inner) != 0)
          continue;
        split = {at, inner};
      }
      if (!split)
        return failure();
      int64_t step = split->first > 0 ? *s : split->second;
      if (auto c = getConstantIntValue(off)) {
        if (*c % step != 0)
          return failure();
      } else if (!isMultipleOf(cast<Value>(off), step)) {
        return failure();
      }
      splits.push_back(*split);
    }
    Location loc = slice.getLoc();
    SmallVector<OpFoldResult> offsets, sizes, strides;
    for (auto [i, group] :
         llvm::enumerate(collapse.getReassociationIndices())) {
      OpFoldResult off = slice.getMixedOffsets()[i];
      auto [at, inner] = splits[i];
      int64_t s = *getConstantIntValue(slice.getMixedSizes()[i]);
      // Offset of dim k: the slice offset divided by the number of elements
      // after dim k, modulo the size of dim k for all but the first dim.
      for (size_t k = 0; k < group.size(); ++k) {
        int64_t after = 1;
        for (int64_t d : ArrayRef(group).drop_front(k + 1))
          after *= srcType.getDimSize(d);
        int64_t dim = srcType.getDimSize(group[k]);
        if (k > at) {
          offsets.push_back(rewriter.getIndexAttr(0));
          sizes.push_back(rewriter.getIndexAttr(dim));
        } else {
          if (auto c = getConstantIntValue(off)) {
            int64_t idx = *c / after;
            offsets.push_back(rewriter.getIndexAttr(k ? idx % dim : idx));
          } else {
            AffineExpr d0 = rewriter.getAffineDimExpr(0);
            AffineExpr e = d0.floorDiv(after);
            if (k)
              e = e % dim;
            offsets.push_back(
                affine::makeComposedFoldedAffineApply(rewriter, loc, e, {off}));
          }
          sizes.push_back(rewriter.getIndexAttr(k == at ? s / inner : 1));
        }
        strides.push_back(rewriter.getIndexAttr(1));
      }
    }
    auto moved = tensor::ExtractSliceOp::create(
        rewriter, loc, collapse.getSrc(), offsets, sizes, strides);
    rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
        slice, slice.getType(), moved.getResult(),
        collapse.getReassociationIndices());
    return success();
  }
};
} // namespace

DiagnosedSilenceableFailure transform::BubbleSliceThroughCollapseOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  return applyPatternsToTargets(
      *this, getTarget(), getResult(), results, state,
      [](RewritePatternSet &patterns) {
        tensor::populateBubbleUpExtractSliceOpPatterns(patterns);
        patterns.add<BubbleSliceThroughCollapse>(patterns.getContext());
      });
}

DiagnosedSilenceableFailure transform::RemoveUninitializedCopyOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {

  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  if (targets.empty()) {
    results.set(llvm::cast<OpResult>(getResult()), ArrayRef<Operation *>());
    return DiagnosedSilenceableFailure::success();
  }

  SmallVector<Operation *> transformedOps;

  for (Operation *target : targets) {
    auto funcOp = dyn_cast_if_present<func::FuncOp>(target);
    if (!funcOp) {
      return emitDefiniteFailure() << "target must be a func.func operation";
    }

    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);

    // Apply unified copy optimization pattern that:
    // 1. Removes copy operations with uninitialized sources
    // 2. Replaces copy operations with fill when source is only filled
    patterns.insert<OptimizeCopyOpPattern<memref::CopyOp>,
                    OptimizeCopyOpPattern<linalg::CopyOp>>(ctx);
    (void)applyPatternsGreedily(funcOp, std::move(patterns));

    transformedOps.push_back(funcOp);
  }

  results.set(llvm::cast<OpResult>(getResult()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// EliminateCascadeMemcpyOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::EliminateCascadeMemcpyOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {

  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  if (targets.empty()) {
    results.set(llvm::cast<OpResult>(getResult()), ArrayRef<Operation *>());
    return DiagnosedSilenceableFailure::success();
  }

  SmallVector<Operation *> transformedOps;

  for (Operation *target : targets) {
    MLIRContext *ctx = target->getContext();
    RewritePatternSet patterns(ctx);

    // Use the existing EliminateIntermediateMemrefPattern
    patterns.insert<xilinx::air::EliminateIntermediateMemrefPattern>(ctx);

    // Apply the pattern to eliminate cascade memcpy operations
    (void)applyPatternsGreedily(target, std::move(patterns));

    transformedOps.push_back(target);
  }

  results.set(llvm::cast<OpResult>(getResult()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ConvertMemrefCopyToLinalgCopyOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::ConvertMemrefCopyToLinalgCopyOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {

  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  if (targets.empty()) {
    results.set(llvm::cast<OpResult>(getResult()), ArrayRef<Operation *>());
    return DiagnosedSilenceableFailure::success();
  }

  SmallVector<Operation *> transformedOps;

  for (Operation *target : targets) {
    MLIRContext *ctx = target->getContext();
    RewritePatternSet patterns(ctx);

    // Use the ConvertMemrefCopyToLinalgCopyPattern
    patterns.insert<xilinx::air::ConvertMemrefCopyToLinalgCopyPattern>(ctx);

    // Apply the pattern to convert memref.copy to linalg.copy operations
    (void)applyPatternsGreedily(target, std::move(patterns));

    transformedOps.push_back(target);
  }

  results.set(llvm::cast<OpResult>(getResult()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ConvertDivfSqrtToRsqrtOp
//===----------------------------------------------------------------------===//

/// Check if a value is a constant 1.0 (scalar or vector splat)
static bool isConstantOne(Value val) {
  auto constOp = val.getDefiningOp<arith::ConstantOp>();
  if (!constOp)
    return false;

  auto attr = constOp.getValue();

  // Check for scalar float constant
  if (auto floatAttr = dyn_cast_if_present<FloatAttr>(attr)) {
    return floatAttr.getValue().isExactlyValue(1.0);
  }

  // Check for dense vector constant (splat)
  if (auto denseAttr = dyn_cast_if_present<DenseFPElementsAttr>(attr)) {
    if (denseAttr.isSplat()) {
      return denseAttr.getSplatValue<FloatAttr>().getValue().isExactlyValue(
          1.0);
    }
  }

  return false;
}

DiagnosedSilenceableFailure transform::ConvertDivfSqrtToRsqrtOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {

  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  if (targets.empty()) {
    results.set(llvm::cast<OpResult>(getResult()), ArrayRef<Operation *>());
    return DiagnosedSilenceableFailure::success();
  }

  SmallVector<Operation *> transformedOps;
  int numTransformations = 0;

  for (Operation *target : targets) {
    // Verify target has IsolatedFromAbove trait
    if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      return emitDefiniteFailure()
             << "target operation must have IsolatedFromAbove trait";
    }

    // Collect divf operations to transform
    SmallVector<arith::DivFOp> divfOpsToTransform;

    target->walk([&](arith::DivFOp divfOp) {
      // Check if LHS is constant 1.0
      if (!isConstantOne(divfOp.getLhs()))
        return;

      // Check if RHS is a math.sqrt operation
      auto sqrtOp = divfOp.getRhs().getDefiningOp<math::SqrtOp>();
      if (!sqrtOp)
        return;

      // Check if sqrt result has exactly one use (this divf)
      if (!sqrtOp.getResult().hasOneUse())
        return;

      // Check type consistency - both must be floating-point
      Type divfType = divfOp.getType();
      Type sqrtType = sqrtOp.getType();
      if (divfType != sqrtType)
        return;

      Type elementType = getElementTypeOrSelf(divfType);
      if (!isa<FloatType>(elementType))
        return;

      // This divf op matches the pattern
      divfOpsToTransform.push_back(divfOp);
    });

    // Apply transformations
    for (arith::DivFOp divfOp : divfOpsToTransform) {
      auto sqrtOp = divfOp.getRhs().getDefiningOp<math::SqrtOp>();

      // Create math.rsqrt operation
      rewriter.setInsertionPoint(divfOp);
      auto rsqrtOp =
          math::RsqrtOp::create(rewriter, divfOp.getLoc(), sqrtOp.getOperand());

      // Replace the divf operation with the rsqrt result
      rewriter.replaceOp(divfOp, rsqrtOp.getResult());

      // Erase the sqrt operation (it has no other uses)
      rewriter.eraseOp(sqrtOp);

      numTransformations++;
    }

    transformedOps.push_back(target);
  }

  LLVM_DEBUG(llvm::dbgs() << "Converted " << numTransformations
                          << " divf(1.0, sqrt(x)) patterns to rsqrt(x)\n");

  results.set(llvm::cast<OpResult>(getResult()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// BroadcastBeforeUnaryOp
//===----------------------------------------------------------------------===//

/// Check if an operation is a supported element-wise unary operation
/// Uses trait-based checking to automatically support all qualifying operations
static bool isSupportedBroadcastableUnaryOp(Operation *op) {
  // Must be Pure (no side effects) - required for safe broadcast reordering
  if (!isPure(op))
    return false;

  // Must have exactly one operand and one result
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return false;

  // Must have same input and output types (element-wise property)
  // This ensures op(broadcast(x)) == broadcast(op(x))
  if (op->getOperand(0).getType() != op->getResult(0).getType())
    return false;

  // Restrict to math and arith dialects for safety
  // These dialects contain well-defined element-wise operations
  StringRef dialectName = op->getDialect()->getNamespace();
  if (dialectName != "math" && dialectName != "arith")
    return false;

  return true;
}

DiagnosedSilenceableFailure
transform::BroadcastBeforeUnaryOp::apply(transform::TransformRewriter &rewriter,
                                         transform::TransformResults &results,
                                         transform::TransformState &state) {

  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  if (targets.empty()) {
    results.set(llvm::cast<OpResult>(getResult()), ArrayRef<Operation *>());
    return DiagnosedSilenceableFailure::success();
  }

  SmallVector<Operation *> transformedOps;
  int numTransformations = 0;

  // Get the optional op_name filter
  std::optional<StringRef> opNameFilter = std::nullopt;
  if (getOpName())
    opNameFilter = getOpName().value();

  for (Operation *target : targets) {
    // Verify target has IsolatedFromAbove trait
    if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      return emitDefiniteFailure()
             << "target operation must have IsolatedFromAbove trait";
    }

    // Collect broadcast operations to transform
    SmallVector<vector::BroadcastOp> broadcastOpsToTransform;

    target->walk([&](vector::BroadcastOp broadcastOp) {
      // Check if the source is a supported unary operation
      Operation *unaryOp = broadcastOp.getSource().getDefiningOp();
      if (!unaryOp)
        return;

      // If op_name is specified, check if it matches
      if (opNameFilter.has_value()) {
        if (unaryOp->getName().getStringRef() != opNameFilter.value())
          return;
      } else {
        // Otherwise use trait-based checking
        if (!isSupportedBroadcastableUnaryOp(unaryOp))
          return;
      }

      // Check that operation has exactly one operand and one result
      if (unaryOp->getNumOperands() != 1 || unaryOp->getNumResults() != 1)
        return;

      // Get the unary op's operand type - can be scalar or vector<1xT>
      Type unaryOperandType = unaryOp->getOperand(0).getType();
      Type unaryElementType;

      if (auto unaryVecType = dyn_cast<VectorType>(unaryOperandType)) {
        // Vector case: must be single-element vector (e.g., vector<1xf32>)
        int64_t numElements = 1;
        for (int64_t dim : unaryVecType.getShape()) {
          numElements *= dim;
        }
        if (numElements != 1)
          return;
        unaryElementType = unaryVecType.getElementType();
      } else {
        // Scalar case (e.g., f32) - inherently single-element
        unaryElementType = unaryOperandType;
      }

      // Check if unary op result has exactly one use (this broadcast)
      if (!unaryOp->getResult(0).hasOneUse())
        return;

      // Check type consistency: element type must match broadcast element type
      auto broadcastType =
          dyn_cast_if_present<VectorType>(broadcastOp.getType());
      if (!broadcastType)
        return;

      if (unaryElementType != broadcastType.getElementType())
        return;

      // This broadcast op matches the pattern
      broadcastOpsToTransform.push_back(broadcastOp);
    });

    // Apply transformations
    for (vector::BroadcastOp broadcastOp : broadcastOpsToTransform) {
      Operation *unaryOp = broadcastOp.getSource().getDefiningOp();
      Value unaryInput = unaryOp->getOperand(0);

      // Clone the original broadcast operation to preserve its exact semantics
      rewriter.setInsertionPoint(unaryOp);
      IRMapping mapping;
      mapping.map(unaryOp->getResult(0), unaryInput);
      auto *clonedBroadcastOp =
          rewriter.clone(*broadcastOp.getOperation(), mapping);
      Value newBroadcastResult = clonedBroadcastOp->getResult(0);

      // Create a new unary operation with the broadcast result as input
      // The result type will match the broadcast result type (larger vector)
      OperationState newState(unaryOp->getLoc(), unaryOp->getName());
      newState.addOperands(newBroadcastResult);
      newState.addTypes(newBroadcastResult.getType());
      newState.addAttributes(unaryOp->getAttrs());
      auto *newUnaryOp = rewriter.create(newState);

      // Replace the broadcast operation with the new unary op result
      rewriter.replaceOp(broadcastOp, newUnaryOp->getResult(0));

      // Erase the old unary operation (it has no other uses)
      rewriter.eraseOp(unaryOp);

      numTransformations++;
    }

    transformedOps.push_back(target);
  }

  LLVM_DEBUG(llvm::dbgs() << "Moved " << numTransformations
                          << " vector.broadcast operations before unary ops\n");

  results.set(llvm::cast<OpResult>(getResult()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// FuseMultiOpLinalgOp
//===----------------------------------------------------------------------===//

/// Check if an operation is element-wise and pure
static bool isElementwisePureOp(Operation *op) {
  // Must be pure (no side effects)
  if (!isPure(op))
    return false;

  // Skip linalg.yield (terminator) - it's always valid
  if (isa<linalg::YieldOp>(op))
    return true;

  // Allow operations from arith and math dialects (typically element-wise)
  StringRef dialectName = op->getDialect()->getNamespace();
  if (dialectName != "arith" && dialectName != "math")
    return false;

  return true;
}

/// Check if a linalg op contains only element-wise pure operations
static bool containsOnlyElementwisePureOps(linalg::LinalgOp linalgOp) {
  Block *body = linalgOp.getBlock();
  if (!body)
    return false;

  // Check all operations in the body
  for (Operation &op : body->getOperations()) {
    if (!isElementwisePureOp(&op))
      return false;
  }

  return true;
}

/// Check if a linalg op has only parallel iterators (no reductions)
static bool hasOnlyParallelIterators(linalg::LinalgOp linalgOp) {
  auto iteratorTypes = linalgOp.getIteratorTypesArray();

  // Check that all iterator types are parallel
  for (auto iterType : iteratorTypes) {
    if (!linalg::isParallelIterator(iterType)) {
      return false;
    }
  }

  return true;
}

/// Check if the second op directly consumes the result of the first op
static bool directlyConsumesResultMultiOp(linalg::LinalgOp firstOp,
                                          linalg::LinalgOp secondOp) {
  // Get the result of the first op (assuming it has one DPS init)
  if (firstOp.getNumDpsInits() != 1)
    return false;

  Value firstOpResult = firstOp.getTiedOpResult(firstOp.getDpsInitOperand(0));

  // Check if the second op uses this result as an input
  return llvm::is_contained(secondOp.getDpsInputs(), firstOpResult);
}

/// Fuse multi-op linalg operations by cloning first op's body into second op
static FailureOr<linalg::GenericOp>
fuseMultiOpLinalgOps(RewriterBase &rewriter, linalg::LinalgOp firstOp,
                     linalg::LinalgOp secondOp) {

  // Get the result of the first op
  Value firstOpResult = firstOp.getTiedOpResult(firstOp.getDpsInitOperand(0));

  // Find which input of the second op corresponds to the first op's result
  int64_t targetInputIndex = -1;
  for (auto [index, input] : llvm::enumerate(secondOp.getDpsInputs())) {
    if (input == firstOpResult) {
      targetInputIndex = static_cast<int64_t>(index);
      break;
    }
  }

  if (targetInputIndex == -1)
    return failure();

  // Get all original inputs from the first op
  SmallVector<Value> firstOpInputs;
  for (Value input : firstOp.getDpsInputs()) {
    firstOpInputs.push_back(input);
  }

  // Create new input list for the fused operation
  // Replace the first op's result in second op's inputs with first op's inputs
  SmallVector<Value> newInputs;
  for (auto [index, input] : llvm::enumerate(secondOp.getDpsInputs())) {
    if (static_cast<int64_t>(index) == targetInputIndex) {
      // Insert all inputs from the first op at this position
      newInputs.append(firstOpInputs.begin(), firstOpInputs.end());
    } else {
      newInputs.push_back(input);
    }
  }

  // Create new indexing maps for the fused operation
  SmallVector<AffineMap> newIndexingMaps;
  auto secondOpIndexingMaps = secondOp.getIndexingMapsArray();
  auto firstOpIndexingMaps = firstOp.getIndexingMapsArray();

  // Build the new indexing maps: replace the target input's map with maps from
  // first op's inputs
  for (auto [index, map] : llvm::enumerate(secondOpIndexingMaps)) {
    if (static_cast<int64_t>(index) == targetInputIndex) {
      // Insert indexing maps from the first op's inputs
      for (size_t i = 0; i < firstOpInputs.size(); ++i) {
        newIndexingMaps.push_back(firstOpIndexingMaps[i]);
      }
    } else if (static_cast<int64_t>(index) < secondOp.getNumDpsInputs()) {
      // This is another input - keep its map
      newIndexingMaps.push_back(map);
    } else {
      // This is an output map - keep it
      newIndexingMaps.push_back(map);
    }
  }

  // Create the new fused operation
  rewriter.setInsertionPoint(secondOp);
  auto fusedOp = linalg::GenericOp::create(
      rewriter, secondOp.getLoc(), secondOp->getResultTypes(), newInputs,
      secondOp.getDpsInits(), newIndexingMaps,
      secondOp.getIteratorTypesArray());

  // Clone the body from the second operation and modify it
  Block *newBody = &fusedOp.getRegion().emplaceBlock();

  // Create block arguments with updated types
  SmallVector<Type> blockArgTypes;

  // Get the input types from the first op (before any transformations)
  SmallVector<Type> firstOpInputElementTypes;
  for (Value input : firstOpInputs) {
    firstOpInputElementTypes.push_back(
        cast<ShapedType>(input.getType()).getElementType());
  }

  // Build block argument types for the fused operation
  size_t firstOpInputIdx = 0;
  for (auto [index, input] : llvm::enumerate(newInputs)) {
    Type elementType;
    // Check if this index corresponds to one of the first op's inputs
    if (static_cast<int64_t>(index) >= targetInputIndex &&
        static_cast<int64_t>(index) <
            targetInputIndex + static_cast<int64_t>(firstOpInputs.size())) {
      elementType = firstOpInputElementTypes[firstOpInputIdx++];
    } else {
      elementType = cast<ShapedType>(input.getType()).getElementType();
    }
    blockArgTypes.push_back(elementType);
  }

  // Add output argument types
  for (Value output : secondOp.getDpsInits()) {
    blockArgTypes.push_back(
        cast<ShapedType>(output.getType()).getElementType());
  }

  for (Type argType : blockArgTypes) {
    newBody->addArgument(argType, fusedOp.getLoc());
  }

  // Clone operations from the first op's body (except terminator)
  IRMapping firstOpMapping;
  Block *firstOpBody = firstOp.getBlock();

  // Map all of the first op's input block arguments to the new fused inputs
  for (auto [idx, inputArg] : llvm::enumerate(firstOpBody->getArguments())) {
    // Skip output arguments (only map inputs)
    if (static_cast<int64_t>(idx) >= firstOp.getNumDpsInputs())
      break;
    firstOpMapping.map(inputArg, newBody->getArgument(targetInputIndex + idx));
  }

  rewriter.setInsertionPointToStart(newBody);

  // Clone all operations from first op (except yield)
  for (Operation &op : firstOpBody->without_terminator()) {
    rewriter.clone(op, firstOpMapping);
  }

  // Determine the yield value from the actual yield operation
  Value firstOpYieldValue;
  if (auto firstYield =
          dyn_cast_if_present<linalg::YieldOp>(firstOpBody->getTerminator())) {
    if (firstYield.getNumOperands() > 0) {
      firstOpYieldValue = firstOpMapping.lookup(firstYield.getOperand(0));
    }
  }

  // Clone operations from the second op's body
  IRMapping secondOpMapping;
  Block *secondOpBody = secondOp.getBlock();

  // Map block arguments from second op
  // Account for the fact that we inserted (firstOpInputs.size() - 1) extra
  // arguments
  int64_t indexShift = firstOpInputs.size() - 1;

  for (auto [index, oldArg] : llvm::enumerate(secondOpBody->getArguments())) {
    if (static_cast<int64_t>(index) == targetInputIndex) {
      // Map to the result of the first op's computation
      secondOpMapping.map(oldArg, firstOpYieldValue);
    } else {
      // Calculate the new index accounting for inserted inputs
      int64_t newIndex = index;
      if (static_cast<int64_t>(index) > targetInputIndex) {
        newIndex += indexShift;
      }
      secondOpMapping.map(oldArg, newBody->getArgument(newIndex));
    }
  }

  // Clone operations from the second op's body
  for (Operation &op : secondOpBody->getOperations()) {
    rewriter.clone(op, secondOpMapping);
  }

  // Replace the second operation with the fused operation
  rewriter.replaceOp(secondOp, fusedOp);

  return fusedOp;
}

DiagnosedSilenceableFailure
transform::FuseMultiOpLinalgOp::apply(transform::TransformRewriter &rewriter,
                                      transform::TransformResults &results,
                                      transform::TransformState &state) {

  SmallVector<Operation *> firstOps =
      llvm::to_vector(state.getPayloadOps(getFirstOp()));
  SmallVector<Operation *> secondOps =
      llvm::to_vector(state.getPayloadOps(getSecondOp()));

  if (firstOps.size() != 1 || secondOps.size() != 1) {
    return emitDefiniteFailure()
           << "requires exactly one first_op and one second_op handle";
  }

  auto firstLinalgOp = dyn_cast_if_present<linalg::LinalgOp>(firstOps[0]);
  auto secondLinalgOp = dyn_cast_if_present<linalg::LinalgOp>(secondOps[0]);

  if (!firstLinalgOp || !secondLinalgOp) {
    return emitDefiniteFailure() << "both operations must be linalg operations";
  }

  // Check condition 1: first op contains only element-wise pure operations
  if (!containsOnlyElementwisePureOps(firstLinalgOp)) {
    return emitDefiniteFailure() << "first operation must contain only "
                                    "element-wise pure operations in its body";
  }

  // Check condition 2: first op has only parallel iterators (no reductions)
  if (!hasOnlyParallelIterators(firstLinalgOp)) {
    return emitDefiniteFailure() << "first operation must have only parallel "
                                    "iterator types (no reduction dimensions)";
  }

  // Check condition 3: second op directly consumes result of first op
  if (!directlyConsumesResultMultiOp(firstLinalgOp, secondLinalgOp)) {
    return emitDefiniteFailure() << "second operation must directly consume "
                                    "the result of the first operation";
  }

  // Perform the fusion
  FailureOr<linalg::GenericOp> fusedOp =
      fuseMultiOpLinalgOps(rewriter, firstLinalgOp, secondLinalgOp);
  if (failed(fusedOp)) {
    return emitDefiniteFailure() << "failed to fuse the operations";
  }

  SmallVector<Operation *> resultOps = {*fusedOp};
  results.set(llvm::cast<OpResult>(getFusedOp()), resultOps);
  return DiagnosedSilenceableFailure::success();
}

void transform::FuseMultiOpLinalgOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getFirstOpMutable(), effects);
  onlyReadsHandle(getSecondOpMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// TransposeReduceOp
//===----------------------------------------------------------------------===//

/// Check if reduction dimensions are innermost in the given linalg.reduce op
static bool areReductionDimensionsInnermost(linalg::ReduceOp reduceOp) {
  ArrayRef<int64_t> reductionDims = reduceOp.getDimensions();
  if (reductionDims.empty())
    return true;

  // Get the input tensor rank
  auto inputType = llvm::cast<ShapedType>(reduceOp.getInputs()[0].getType());
  int64_t rank = inputType.getRank();

  // Check if all reduction dimensions are at the end (innermost)
  SmallVector<int64_t> sortedReductionDims(reductionDims.begin(),
                                           reductionDims.end());
  llvm::sort(sortedReductionDims);

  // The reduction dimensions should be consecutive and end at rank-1
  for (size_t i = 0; i < sortedReductionDims.size(); ++i) {
    if (sortedReductionDims[i] !=
        (int64_t)(rank - sortedReductionDims.size() + i)) {
      return false;
    }
  }
  return true;
}

/// Create a transpose operation to move reduction dimensions to the end
static Value
createTransposeToMakeReductionInnermost(OpBuilder &builder, Location loc,
                                        Value input,
                                        ArrayRef<int64_t> reductionDims) {

  auto inputType = llvm::cast<ShapedType>(input.getType());
  int64_t rank = inputType.getRank();

  // Create permutation: non-reduction dims first, then reduction dims
  SmallVector<int64_t> permutation;
  SmallVector<bool> isReductionDim(rank, false);

  // Mark reduction dimensions
  for (int64_t dim : reductionDims) {
    isReductionDim[dim] = true;
  }

  // Add non-reduction dimensions first
  for (int64_t i = 0; i < rank; ++i) {
    if (!isReductionDim[i]) {
      permutation.push_back(i);
    }
  }

  // Add reduction dimensions at the end
  for (int64_t dim : reductionDims) {
    permutation.push_back(dim);
  }

  // Create the transpose operation
  SmallVector<int64_t> transposedShape;
  for (int64_t dim : permutation) {
    transposedShape.push_back(inputType.getDimSize(dim));
  }

  // Create linalg.transpose operation
  auto transposeOp = linalg::TransposeOp::create(
      builder, loc, input,
      tensor::EmptyOp::create(builder, loc, transposedShape,
                              inputType.getElementType()),
      builder.getDenseI64ArrayAttr(permutation));

  return transposeOp.getResult()[0];
}

/// Update reduction dimensions after transpose
static SmallVector<int64_t>
updateReductionDimsAfterTranspose(ArrayRef<int64_t> originalReductionDims,
                                  int64_t rank) {

  SmallVector<int64_t> newReductionDims;
  int64_t numReductionDims = originalReductionDims.size();

  // After transpose, reduction dimensions are at the end
  for (int64_t i = 0; i < numReductionDims; ++i) {
    newReductionDims.push_back(rank - numReductionDims + i);
  }

  return newReductionDims;
}

DiagnosedSilenceableFailure
transform::TransposeReduceOp::apply(transform::TransformRewriter &rewriter,
                                    transform::TransformResults &results,
                                    transform::TransformState &state) {

  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  if (targets.empty()) {
    results.set(llvm::cast<OpResult>(getResult()), ArrayRef<Operation *>());
    return DiagnosedSilenceableFailure::success();
  }

  SmallVector<Operation *> transformedOps;

  for (Operation *target : targets) {
    auto reduceOp = dyn_cast_if_present<linalg::ReduceOp>(target);
    if (!reduceOp) {
      return emitDefiniteFailure()
             << "target must be a linalg.reduce operation";
    }

    // Check if reduction dimensions are already innermost
    if (areReductionDimensionsInnermost(reduceOp)) {
      // No transformation needed
      transformedOps.push_back(target);
      continue;
    }

    // Get reduction dimensions and input
    ArrayRef<int64_t> reductionDims = reduceOp.getDimensions();
    Value input = reduceOp.getInputs()[0];
    auto inputType = llvm::cast<ShapedType>(input.getType());
    int64_t rank = inputType.getRank();

    // Create transpose operation to move reduction dimensions to the end
    rewriter.setInsertionPoint(reduceOp);
    Value transposedInput = createTransposeToMakeReductionInnermost(
        rewriter, reduceOp.getLoc(), input, reductionDims);

    // Update reduction dimensions for the new layout
    SmallVector<int64_t> newReductionDims =
        updateReductionDimsAfterTranspose(reductionDims, rank);

    // Create new reduce operation with transposed input and updated dimensions
    auto newReduceOp = linalg::ReduceOp::create(
        rewriter, reduceOp.getLoc(), reduceOp.getResultTypes(),
        ValueRange{transposedInput}, reduceOp.getInits(),
        rewriter.getDenseI64ArrayAttr(newReductionDims));

    // Copy the reduction body from the original operation
    rewriter.cloneRegionBefore(reduceOp.getCombiner(),
                               newReduceOp.getCombiner(),
                               newReduceOp.getCombiner().begin());

    // Replace the original operation
    rewriter.replaceOp(reduceOp, newReduceOp.getResults());
    transformedOps.push_back(newReduceOp);
  }

  results.set(llvm::cast<OpResult>(getResult()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// FuseElementwiseLinalgOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::FuseElementwiseLinalgOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {

  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  if (targets.empty()) {
    results.set(llvm::cast<OpResult>(getResult()), ArrayRef<Operation *>());
    return DiagnosedSilenceableFailure::success();
  }

  SmallVector<Operation *> transformedOps;

  for (Operation *target : targets) {
    auto funcOp = dyn_cast_if_present<func::FuncOp>(target);
    if (!funcOp) {
      return emitDefiniteFailure() << "target must be a func.func operation";
    }

    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);

    // Populate the elementwise fusion patterns using the linalg utilities
    // With no_recompute, a producer is computed once: one read through a
    // broadcast is left alone (fusing would recompute it for every element it
    // is broadcast to), as is one with other users (it stays for them) and
    // one feeding a contraction, so the contraction is still recognized as
    // one.
    bool noRecompute = getNoRecompute();
    linalg::ControlFusionFn controlFn = [noRecompute](OpOperand *fusedOperand) {
      auto consumer = dyn_cast<linalg::LinalgOp>(fusedOperand->getOwner());
      if (noRecompute && consumer &&
          (linalg::isaContractionOpInterface(consumer) ||
           !consumer.getMatchingIndexingMap(fusedOperand).isPermutation() ||
           !fusedOperand->get().hasOneUse()))
        return false;
      return true;
    };

    linalg::populateElementwiseOpsFusionPatterns(patterns, controlFn);
    if (getFoldReshapes()) {
      // Only reshapes between two elementwise (all-parallel) generics: a
      // contraction's operands, packed or not, are left alone.
      linalg::ControlFusionFn reshapeControl = [](OpOperand *fusedOperand) {
        Operation *producer = fusedOperand->get().getDefiningOp();
        Operation *consumer = fusedOperand->getOwner();
        auto isParallelGeneric = [](Operation *op) {
          auto generic = dyn_cast_if_present<linalg::GenericOp>(op);
          return generic &&
                 generic.getNumLoops() == generic.getNumParallelLoops();
        };
        auto isGenericOrReshape = [&](Operation *op) {
          return isParallelGeneric(op) ||
                 isa_and_present<tensor::ExpandShapeOp,
                                 tensor::CollapseShapeOp>(op);
        };
        if (!isGenericOrReshape(producer) || !isGenericOrReshape(consumer))
          return false;
        auto otherSide = [&](Operation *reshape, bool up) -> Operation * {
          if (up)
            return reshape->getOperand(0).getDefiningOp();
          if (!reshape->hasOneUse())
            return nullptr;
          return *reshape->getUsers().begin();
        };
        if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(producer))
          return isParallelGeneric(otherSide(producer, true));
        if (isa<tensor::ExpandShapeOp, tensor::CollapseShapeOp>(consumer))
          return isParallelGeneric(otherSide(consumer, false));
        return false;
      };
      linalg::populateFoldReshapeOpsByExpansionPatterns(patterns,
                                                        reshapeControl);
    }

    // Apply the patterns greedily
    (void)applyPatternsGreedily(funcOp, std::move(patterns));

    transformedOps.push_back(funcOp);
  }

  results.set(llvm::cast<OpResult>(getResult()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// FuseTruncfLinalgOp
//===----------------------------------------------------------------------===//

/// Check if a linalg op contains only arith.truncf as its body operation
/// (apart from terminator)
static bool containsOnlyTruncfOp(linalg::LinalgOp linalgOp) {
  Block *body = linalgOp.getBlock();
  if (!body)
    return false;

  // Should have exactly one non-terminator operation, and it should be truncf
  if (llvm::range_size(body->without_terminator()) != 1)
    return false;
  return llvm::any_of(body->without_terminator(),
                      [](Operation &o) { return isa<arith::TruncFOp>(o); });
}

/// Check if the first op (producer) produces a result that is consumed by the
/// second op (truncf consumer)
static bool producesResultForOp(linalg::LinalgOp producerOp,
                                linalg::LinalgOp truncfOp) {
  // Get the result of the producer op (assuming it has one DPS init)
  if (producerOp.getNumDpsInits() != 1)
    return false;

  Value producerResult =
      producerOp.getTiedOpResult(producerOp.getDpsInitOperand(0));

  // Check if the truncf op uses this result as an input
  return llvm::is_contained(truncfOp.getDpsInputs(), producerResult);
}

/// Get the output type of the truncf op (after arith.truncf type change)
static Type getOutputTypeAfterTruncf(linalg::LinalgOp linalgOp) {
  Block *body = linalgOp.getBlock();
  if (!body)
    return nullptr;

  for (Operation &op : body->getOperations()) {
    if (auto truncfOp = dyn_cast_if_present<arith::TruncFOp>(op)) {
      return truncfOp.getOut().getType();
    }
  }

  return nullptr;
}

/// Fuse truncf linalg op into its producer by creating a new fused operation
static FailureOr<linalg::GenericOp>
fuseTruncfIntoProducer(RewriterBase &rewriter, linalg::LinalgOp producerOp,
                       linalg::LinalgOp truncfOp) {
  // Get the output type after truncf conversion
  Type truncatedType = getOutputTypeAfterTruncf(truncfOp);
  if (!truncatedType)
    return failure();

  // Get the output of the truncf op
  Value truncfOutput = truncfOp.getTiedOpResult(truncfOp.getDpsInitOperand(0));

  // Create new output for the fused operation (with truncated type)
  // Use the producer's init, type-converted to the truncated element type
  rewriter.setInsertionPoint(truncfOp);

  // Get the producer's init and create a new one with the truncated type
  Value producerInit = producerOp.getDpsInits()[0];
  auto producerInitType = cast<RankedTensorType>(producerInit.getType());
  auto fusedInitType =
      RankedTensorType::get(producerInitType.getShape(), truncatedType);

  // Create a type cast for the producer's init to the truncated type
  // If producer init is a linalg.fill, create a new fill with truncated type
  // Otherwise, create a tensor.empty with the truncated type
  Value fusedInit;
  if (auto fillOp = producerInit.getDefiningOp<linalg::FillOp>()) {
    // Create new empty tensor with truncated element type
    auto newEmptyOp =
        tensor::EmptyOp::create(rewriter, truncfOp.getLoc(),
                                producerInitType.getShape(), truncatedType);
    // Truncate the fill value to the new type
    Value fillValue = fillOp.getInputs()[0];
    Value truncatedFillValue = arith::TruncFOp::create(
        rewriter, truncfOp.getLoc(), truncatedType, fillValue);
    // Create a new fill with the truncated fill value
    auto newFillOp = linalg::FillOp::create(rewriter, truncfOp.getLoc(),
                                            ValueRange{truncatedFillValue},
                                            ValueRange{newEmptyOp.getResult()});
    fusedInit = newFillOp.getResult(0);
  } else {
    // For other cases (e.g., tensor.empty), create a new empty tensor
    fusedInit =
        tensor::EmptyOp::create(rewriter, truncfOp.getLoc(),
                                producerInitType.getShape(), truncatedType);
  }

  auto fusedOp = linalg::GenericOp::create(
      rewriter, producerOp.getLoc(), TypeRange{fusedInitType},
      producerOp.getDpsInputs(), ValueRange{fusedInit},
      producerOp.getIndexingMapsArray(), producerOp.getIteratorTypesArray());

  // Clone the producer region directly - this preserves all types and
  // operations
  rewriter.cloneRegionBefore(producerOp->getRegion(0), fusedOp.getRegion(),
                             fusedOp.getRegion().begin());

  Block *clonedBody = &fusedOp.getRegion().front();

  // Get the original type of the output argument before we change it
  size_t numInputs = producerOp.getDpsInputs().size();
  BlockArgument outputArg = clonedBody->getArgument(numInputs);
  Type originalOutputType = outputArg.getType();

  // Change the output block argument type to match the truncated output
  outputArg.setType(truncatedType);

  // If the producer uses its output argument (e.g., for accumulation in
  // matmul), we need to insert extf operations where the argument is used
  if (producerOp.payloadUsesValueFromOperand(producerOp.getDpsInitOperand(0))) {
    // Collect all uses of the output argument before we modify them
    SmallVector<OpOperand *> outputArgUses;
    for (OpOperand &use : outputArg.getUses()) {
      outputArgUses.push_back(&use);
    }

    // For each use of the output argument, insert an extf to cast it back to
    // original type
    for (OpOperand *use : outputArgUses) {
      Operation *user = use->getOwner();
      // Skip if the user is the yield operation we're about to modify
      if (isa<linalg::YieldOp>(user))
        continue;

      rewriter.setInsertionPoint(user);
      auto extfOp = arith::ExtFOp::create(rewriter, fusedOp.getLoc(),
                                          originalOutputType, outputArg);
      use->set(extfOp.getResult());
    }
  }

  // Now add the truncf before the terminator and update the yield
  auto yieldOp = cast<linalg::YieldOp>(clonedBody->getTerminator());
  Value yieldValue = yieldOp.getValues()[0];

  rewriter.setInsertionPoint(yieldOp);
  auto truncfOpInBody = arith::TruncFOp::create(rewriter, fusedOp.getLoc(),
                                                truncatedType, yieldValue);

  // Update the yield to use the truncated value
  yieldOp.getValuesMutable().assign(truncfOpInBody.getResult());

  // Replace uses of truncf op's output with the fused op's output
  rewriter.replaceAllUsesWith(truncfOutput, fusedOp.getResult(0));

  // Erase both original operations
  rewriter.eraseOp(truncfOp);
  rewriter.eraseOp(producerOp);

  return fusedOp;
}

DiagnosedSilenceableFailure
transform::FuseTruncfLinalgOp::apply(transform::TransformRewriter &rewriter,
                                     transform::TransformResults &results,
                                     transform::TransformState &state) {

  SmallVector<Operation *> truncfOps =
      llvm::to_vector(state.getPayloadOps(getTruncfOp()));
  SmallVector<Operation *> producerOps =
      llvm::to_vector(state.getPayloadOps(getProducerOp()));

  if (truncfOps.size() != 1 || producerOps.size() != 1) {
    return emitDefiniteFailure()
           << "requires exactly one truncf_op and one producer_op handle";
  }

  auto truncfLinalgOp = dyn_cast_if_present<linalg::LinalgOp>(truncfOps[0]);
  auto producerLinalgOp = dyn_cast_if_present<linalg::LinalgOp>(producerOps[0]);

  if (!truncfLinalgOp || !producerLinalgOp) {
    return emitDefiniteFailure() << "both operations must be linalg operations";
  }

  // Check condition 1: truncf op contains only arith.truncf
  if (!containsOnlyTruncfOp(truncfLinalgOp)) {
    return emitDefiniteFailure()
           << "truncf_op must contain only arith.truncf in its body";
  }

  // Check condition 2: producer op produces result consumed by truncf op
  if (!producesResultForOp(producerLinalgOp, truncfLinalgOp)) {
    return emitDefiniteFailure() << "producer_op must produce a result that "
                                    "is consumed by truncf_op";
  }

  // Perform the fusion: create a fused generic, then replace it with a
  // linalg.matmul that has the fused output type (bf16). LLVM 23's
  // specialize rejects generics with output casts, so we bypass it by
  // directly creating the matmul with the fused type.
  FailureOr<linalg::GenericOp> fusedOp =
      fuseTruncfIntoProducer(rewriter, producerLinalgOp, truncfLinalgOp);
  if (failed(fusedOp)) {
    return emitDefiniteFailure() << "failed to fuse the operations";
  }

  // LLVM 23: specialize rejects generics with output casts (truncf→yield).
  // If the fused op has 2D+ inputs (matmul-compatible), replace with a
  // linalg.matmul directly, bypassing specialize. The matmul body auto-
  // generates in the output element type (bf16), and Phase 12 adds
  // extf/truncf pairs for f32 accumulation during vectorization.
  auto inputType =
      dyn_cast<RankedTensorType>(fusedOp->getDpsInputs()[0].getType());
  if (inputType && inputType.getRank() >= 2) {
    rewriter.setInsertionPoint(*fusedOp);
    auto matmulOp = linalg::MatmulOp::create(
        rewriter, fusedOp->getLoc(), fusedOp->getResultTypes(),
        ValueRange{fusedOp->getDpsInputs()[0], fusedOp->getDpsInputs()[1]},
        ValueRange{fusedOp->getDpsInits()[0]});
    rewriter.replaceOp(*fusedOp, matmulOp->getResults());

    SmallVector<Operation *> resultOps = {matmulOp.getOperation()};
    results.set(llvm::cast<OpResult>(getFusedOp()), resultOps);
    return DiagnosedSilenceableFailure::success();
  }

  // For non-matmul cases (1D, etc.), return the generic as-is.
  SmallVector<Operation *> resultOps = {*fusedOp};
  results.set(llvm::cast<OpResult>(getFusedOp()), resultOps);
  return DiagnosedSilenceableFailure::success();
}

void transform::FuseTruncfLinalgOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTruncfOpMutable(), effects);
  consumesHandle(getProducerOpMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// ReuseInputAsInitOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::ReuseInputAsInitOp::applyToOne(
    transform::TransformRewriter &rewriter, linalg::LinalgOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  if (!target.hasPureTensorSemantics() || target.getNumDpsInits() != 1)
    return emitDefaultSilenceableFailure(target)
           << "expected a linalg op on tensors with one output";
  if (target.getNumParallelLoops() != target.getNumLoops())
    return emitDefaultSilenceableFailure(target)
           << "expected only parallel iterators";
  OpOperand *out = target.getDpsInitOperand(0);
  if (target.payloadUsesValueFromOperand(out))
    return emitDefaultSilenceableFailure(target) << "the output is read";

  auto fits = [&](OpOperand *in) {
    return in->get().getType() == out->get().getType() &&
           target.getMatchingIndexingMap(in) ==
               target.getMatchingIndexingMap(out);
  };
  OpOperand *in = nullptr;
  if (std::optional<uint64_t> i = getInput()) {
    if (*i >= static_cast<uint64_t>(target.getNumDpsInputs()))
      return emitDefaultSilenceableFailure(target)
             << "input " << *i << " is out of range";
    in = target.getDpsInputOperand(*i);
    if (!fits(in))
      return emitDefaultSilenceableFailure(target)
             << "input " << *i
             << " differs from the output in type or indexing map";
  } else {
    auto inputs = target.getDpsInputOperands();
    auto it = llvm::find_if(inputs, fits);
    if (it == inputs.end())
      return emitDefaultSilenceableFailure(target)
             << "no input has the output's type and indexing map";
    in = *it;
  }
  // Read the value through the output, so that after tiling the read and the
  // write are the same slice of the same tensor. The input is left unused, for
  // cleanup patterns to drop.
  rewriter.modifyOpInPlace(target, [&] {
    out->set(in->get());
    rewriter.replaceAllUsesWith(target.getMatchingBlockArgument(in),
                                target.getMatchingBlockArgument(out));
  });
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ReadForallTilesFromSharedOutsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::ReadForallTilesFromSharedOutsOp::applyToOne(
    transform::TransformRewriter &rewriter, scf::ForallOp target,
    transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  // The one tile each shared output gets, by its block argument.
  DenseMap<Value, tensor::ParallelInsertSliceOp> tile;
  DenseSet<Value> several;
  for (Operation &op : target.getTerminator().getYieldingOps()) {
    auto insert = dyn_cast<tensor::ParallelInsertSliceOp>(op);
    if (!insert)
      continue;
    Value dest = insert.getDest();
    if (!tile.try_emplace(dest, insert).second)
      several.insert(dest);
  }
  for (auto [init, arg] :
       llvm::zip(target.getOutputs(), target.getRegionIterArgs())) {
    auto it = tile.find(arg);
    if (it == tile.end() || several.contains(arg))
      continue;
    tensor::ParallelInsertSliceOp insert = it->second;
    Value initValue = init, shared = arg;
    SmallVector<tensor::ExtractSliceOp> reads;
    target.getBody()->walk([&](tensor::ExtractSliceOp slice) {
      if (slice.getSource() == initValue &&
          isEqualConstantIntOrValueArray(slice.getMixedOffsets(),
                                         insert.getMixedOffsets()) &&
          isEqualConstantIntOrValueArray(slice.getMixedSizes(),
                                         insert.getMixedSizes()) &&
          isEqualConstantIntOrValueArray(slice.getMixedStrides(),
                                         insert.getMixedStrides()))
        reads.push_back(slice);
    });
    for (tensor::ExtractSliceOp slice : reads)
      rewriter.modifyOpInPlace(
          slice, [&] { slice.getSourceMutable().assign(shared); });
  }
  results.push_back(target);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// VectorTypeCastOp
//===----------------------------------------------------------------------===//

/// Calculate the total number of elements in a vector type
static int64_t getVectorNumElements(VectorType vecType) {
  int64_t numElements = 1;
  for (int64_t dim : vecType.getShape()) {
    numElements *= dim;
  }
  return numElements;
}

/// Helper function to create cast operations for both scalar and vector types
static Value createTypeCast(OpBuilder &builder, Location loc, Value input,
                            Type targetElementType, bool isExtension) {
  Type inputType = input.getType();

  // Determine the source element type and target type
  Type sourceElementType;
  Type targetType;

  if (auto inputVectorType = dyn_cast_if_present<VectorType>(inputType)) {
    // Handle vector types
    sourceElementType = inputVectorType.getElementType();
    targetType = VectorType::get(inputVectorType.getShape(), targetElementType);
  } else {
    // Handle scalar types
    sourceElementType = inputType;
    targetType = targetElementType;
  }

  // Create the appropriate cast operation based on element types and operation
  if (isExtension) {
    // Extension: narrow to wide type
    if (isa<FloatType>(sourceElementType) &&
        isa<FloatType>(targetElementType)) {
      return arith::ExtFOp::create(builder, loc, targetType, input);
    } else if (isa<IntegerType>(sourceElementType) &&
               isa<IntegerType>(targetElementType)) {
      // For integer types, use sign extension
      return arith::ExtSIOp::create(builder, loc, targetType, input);
    }
  } else {
    // Truncation: wide to narrow type
    if (isa<FloatType>(sourceElementType) &&
        isa<FloatType>(targetElementType)) {
      return arith::TruncFOp::create(builder, loc, targetType, input);
    } else if (isa<IntegerType>(sourceElementType) &&
               isa<IntegerType>(targetElementType)) {
      return arith::TruncIOp::create(builder, loc, targetType, input);
    }
  }

  // If no cast is needed or supported, return the original value
  return input;
}

/// Helper function to determine if a cast is an extension (narrow to wide)
static bool isExtensionCast(Type fromType, Type toType) {
  // Use getIntOrFloatBitWidth for both integer and float types
  unsigned fromWidth = fromType.getIntOrFloatBitWidth();
  unsigned toWidth = toType.getIntOrFloatBitWidth();

  // Extension is when we go from smaller to larger bit width
  return fromWidth < toWidth;
}

/// Helper function to apply vector type casting to a single operation
static FailureOr<Operation *> applyVectorTypeCastToOp(
    Operation *op, Type targetElementType, ArrayRef<int64_t> inputIndicesToCast,
    ArrayRef<int64_t> outputIndicesToCast, RewriterBase &rewriter) {

  // Skip if operation doesn't have vector operands or results
  bool hasVectorOperands = false;
  bool hasVectorResults = false;
  bool needsTransformation = false;

  // Determine if we should cast all inputs/outputs (default behavior)
  bool castAllInsAndOuts =
      inputIndicesToCast.empty() && outputIndicesToCast.empty();

  // Create sets for quick lookup of indices to cast
  llvm::SmallDenseSet<int64_t> inputIndicesToCastSet(inputIndicesToCast.begin(),
                                                     inputIndicesToCast.end());
  llvm::SmallDenseSet<int64_t> outputIndicesToCastSet(
      outputIndicesToCast.begin(), outputIndicesToCast.end());

  // Check if ALL vector operands and results have exactly one element
  // If so, skip casting this operation entirely
  bool allVectorsAreSingleElement = true;
  bool hasAnyVectors = false;

  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    if (auto vectorType = dyn_cast_if_present<VectorType>(operand.getType())) {
      hasAnyVectors = true;
      if (getVectorNumElements(vectorType) != 1) {
        allVectorsAreSingleElement = false;
      }
    }
  }

  for (auto [idx, result] : llvm::enumerate(op->getResults())) {
    if (auto vectorType = dyn_cast_if_present<VectorType>(result.getType())) {
      hasAnyVectors = true;
      if (getVectorNumElements(vectorType) != 1) {
        allVectorsAreSingleElement = false;
      }
    }
  }

  // Skip the entire operation if all vectors are single-element
  if (hasAnyVectors && allVectorsAreSingleElement) {
    return failure();
  }

  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    if (auto vectorType = dyn_cast_if_present<VectorType>(operand.getType())) {
      hasVectorOperands = true;
      bool shouldCast =
          castAllInsAndOuts || inputIndicesToCastSet.contains((int64_t)idx);
      if (shouldCast && vectorType.getElementType() != targetElementType) {
        needsTransformation = true;
      }
    }
  }

  for (auto [idx, result] : llvm::enumerate(op->getResults())) {
    if (auto vectorType = dyn_cast_if_present<VectorType>(result.getType())) {
      hasVectorResults = true;
      bool shouldCast =
          castAllInsAndOuts || outputIndicesToCastSet.contains((int64_t)idx);
      if (shouldCast && vectorType.getElementType() != targetElementType) {
        needsTransformation = true;
      }
    }
  }

  if (!hasVectorOperands && !hasVectorResults) {
    return failure();
  }

  if (!needsTransformation) {
    return failure();
  }

  auto loc = op->getLoc();
  rewriter.setInsertionPoint(op);

  // Cast input operands to target type (selectively)
  SmallVector<Value> newOperands;
  SmallVector<Type> originalOperandTypes;

  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    originalOperandTypes.push_back(operand.getType());

    if (auto vectorType = dyn_cast_if_present<VectorType>(operand.getType())) {
      Type currentElementType = vectorType.getElementType();
      bool shouldCast =
          castAllInsAndOuts || inputIndicesToCastSet.contains((int64_t)idx);

      if (shouldCast && currentElementType != targetElementType) {
        bool isExt = isExtensionCast(currentElementType, targetElementType);
        Value castOperand =
            createTypeCast(rewriter, loc, operand, targetElementType, isExt);
        newOperands.push_back(castOperand);
      } else {
        newOperands.push_back(operand);
      }
    } else {
      newOperands.push_back(operand);
    }
  }

  // Determine new result types using target element type (selectively)
  SmallVector<Type> newResultTypes;
  SmallVector<Type> originalResultTypes;

  for (auto [idx, resultType] : llvm::enumerate(op->getResultTypes())) {
    originalResultTypes.push_back(resultType);

    if (auto vectorType = dyn_cast_if_present<VectorType>(resultType)) {
      bool shouldCast =
          castAllInsAndOuts || outputIndicesToCastSet.contains((int64_t)idx);

      // Note: We DO change the result type even for single-element vectors
      // to maintain type consistency with the operation's inputs
      if (shouldCast) {
        auto newVectorType =
            VectorType::get(vectorType.getShape(), targetElementType);
        newResultTypes.push_back(newVectorType);
      } else {
        newResultTypes.push_back(resultType);
      }
    } else {
      // Handle scalar result types (e.g., for vector.reduction)
      // For operations like vector.reduction, the scalar result type must match
      // the input vector's element type
      bool shouldCast =
          castAllInsAndOuts || outputIndicesToCastSet.contains((int64_t)idx);

      if (shouldCast &&
          (isa<FloatType>(resultType) || isa<IntegerType>(resultType))) {
        // Change scalar result type to match target element type
        newResultTypes.push_back(targetElementType);
      } else {
        newResultTypes.push_back(resultType);
      }
    }
  }

  // Clone the operation with new operands and result types
  OperationState newState(loc, op->getName());
  newState.addOperands(newOperands);
  newState.addTypes(newResultTypes);
  newState.addAttributes(op->getAttrs());

  // Clone regions if any
  for (Region &region : op->getRegions()) {
    Region *newRegion = newState.addRegion();
    rewriter.cloneRegionBefore(region, *newRegion, newRegion->begin());
  }

  Operation *newOp = rewriter.create(newState);

  // Cast results back to original types (selectively)
  SmallVector<Value> finalResults;
  for (auto [idx, pair] :
       llvm::enumerate(llvm::zip(originalResultTypes, newOp->getResults()))) {
    auto [originalType, newResult] = pair;

    Type originalElementType = originalType;

    if (auto originalVectorType =
            dyn_cast_if_present<VectorType>(originalType)) {
      originalElementType = originalVectorType.getElementType();
    }

    bool shouldCast =
        castAllInsAndOuts || outputIndicesToCastSet.contains((int64_t)idx);

    if (shouldCast && originalElementType != targetElementType) {
      bool isExt = isExtensionCast(targetElementType, originalElementType);
      Value castResult =
          createTypeCast(rewriter, loc, newResult, originalElementType, isExt);
      finalResults.push_back(castResult);
    } else {
      finalResults.push_back(newResult);
    }
  }

  // Replace the original operation
  rewriter.replaceOp(op, finalResults);
  return newOp;
}

DiagnosedSilenceableFailure
transform::VectorTypeCastOp::apply(transform::TransformRewriter &rewriter,
                                   transform::TransformResults &results,
                                   transform::TransformState &state) {

  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  if (targets.empty()) {
    results.set(llvm::cast<OpResult>(getResult()), ArrayRef<Operation *>());
    return DiagnosedSilenceableFailure::success();
  }

  Type targetElementType = getTargetElementType();

  // Extract input and output indices from attributes
  SmallVector<int64_t> inputIndicesToCast =
      extractFromIntegerArrayAttr<int64_t>(getInputIndices());
  SmallVector<int64_t> outputIndicesToCast =
      extractFromIntegerArrayAttr<int64_t>(getOutputIndices());

  SmallVector<Operation *> transformedOps;

  for (Operation *target : targets) {
    // Check if this operation has vector operands or results
    bool hasVectorTypes = false;
    for (Value operand : target->getOperands()) {
      if (isa<VectorType>(operand.getType())) {
        hasVectorTypes = true;
        break;
      }
    }
    if (!hasVectorTypes) {
      for (Value result : target->getResults()) {
        if (isa<VectorType>(result.getType())) {
          hasVectorTypes = true;
          break;
        }
      }
    }

    if (!hasVectorTypes) {
      return emitDefiniteFailure()
             << "target operation must have vector operands or results, but "
                "operation '"
             << target->getName()
             << "' operates on scalar types. Vector type casting "
             << "can only be applied to operations that work with vector "
                "types.";
    }

    // Check if this operation has vector types that need casting
    bool needsTransformation = false;
    for (Value operand : target->getOperands()) {
      if (auto vectorType =
              dyn_cast_if_present<VectorType>(operand.getType())) {
        if (vectorType.getElementType() != targetElementType) {
          needsTransformation = true;
          break;
        }
      }
    }
    if (!needsTransformation) {
      for (Value result : target->getResults()) {
        if (auto vectorType =
                dyn_cast_if_present<VectorType>(result.getType())) {
          if (vectorType.getElementType() != targetElementType) {
            needsTransformation = true;
            break;
          }
        }
      }
    }

    if (needsTransformation) {
      // Apply transformation directly to the target operation with selective
      // casting
      FailureOr<Operation *> castedOpOnVector =
          applyVectorTypeCastToOp(target, targetElementType, inputIndicesToCast,
                                  outputIndicesToCast, rewriter);
      if (failed(castedOpOnVector)) {
        // Operation was skipped (e.g., all vectors are single-element)
        // This is not an error, just add the original operation unchanged
        transformedOps.push_back(target);
      } else {
        transformedOps.push_back(*castedOpOnVector);
      }
    } else {
      transformedOps.push_back(target);
    }
  }

  results.set(llvm::cast<OpResult>(getResult()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// EliminateRedundantVectorTransfersOp
//===----------------------------------------------------------------------===//

/// Check if two values are semantically equivalent indices
static bool areEquivalentIndices(Value idx1, Value idx2) {
  // Direct SSA value equality
  if (idx1 == idx2)
    return true;

  // Check if both are results of affine.apply with the same map and operands
  auto affineOp1 = idx1.getDefiningOp<affine::AffineApplyOp>();
  auto affineOp2 = idx2.getDefiningOp<affine::AffineApplyOp>();

  if (affineOp1 && affineOp2) {
    // Check if they use the same affine map
    if (affineOp1.getAffineMap() != affineOp2.getAffineMap())
      return false;

    // Check if they have the same number of operands
    if (affineOp1.getMapOperands().size() != affineOp2.getMapOperands().size())
      return false;

    // Check if all operands are identical
    for (auto [op1, op2] :
         llvm::zip(affineOp1.getMapOperands(), affineOp2.getMapOperands())) {
      if (op1 != op2)
        return false;
    }

    return true;
  }

  // Check if both are constants with the same value
  auto constOp1 = idx1.getDefiningOp<arith::ConstantIndexOp>();
  auto constOp2 = idx2.getDefiningOp<arith::ConstantIndexOp>();

  if (constOp1 && constOp2) {
    return constOp1.value() == constOp2.value();
  }

  return false;
}

/// Check if two vector.transfer_read operations read from the same location
static bool areIdenticalReads(vector::TransferReadOp read1,
                              vector::TransferReadOp read2) {
  // Check if they read from the same memref
  if (read1.getBase() != read2.getBase())
    return false;

  // Check if they have the same number of indices
  if (read1.getIndices().size() != read2.getIndices().size())
    return false;

  // Check if all indices are semantically equivalent
  for (auto [idx1, idx2] : llvm::zip(read1.getIndices(), read2.getIndices())) {
    if (!areEquivalentIndices(idx1, idx2))
      return false;
  }

  // Check if they have the same result type
  auto vec1Ty = llvm::cast<VectorType>(read1.getVector().getType());
  auto vec2Ty = llvm::cast<VectorType>(read2.getVector().getType());
  if (vec1Ty != vec2Ty)
    return false;

  return true;
}

/// Check if there are any writes to the memref between two operations
static bool hasWritesBetweenReads(vector::TransferReadOp firstRead,
                                  vector::TransferReadOp secondRead) {
  Value sourceMemref = firstRead.getBase();

  // Get the block containing both reads
  Block *block = firstRead->getBlock();
  if (block != secondRead->getBlock())
    return true; // Conservative: assume writes if in different blocks

  // Find the operations between the two reads
  auto firstIt = firstRead->getIterator();
  auto secondIt = secondRead->getIterator();

  // Iterate from first read to second read
  for (auto it = ++firstIt; it != secondIt; ++it) {
    Operation *op = &(*it);

    // Check if this operation writes to the source memref
    auto memInterface = dyn_cast_if_present<MemoryEffectOpInterface>(op);
    if (!memInterface) {
      // Conservative: if we can't determine effects, assume it might write
      if (!op->hasTrait<OpTrait::HasRecursiveMemoryEffects>())
        continue;
      return true;
    }

    SmallVector<MemoryEffects::EffectInstance> effects;
    memInterface.getEffects(effects);

    for (auto &effect : effects) {
      if (!isa<MemoryEffects::Write>(effect.getEffect()))
        continue;

      Value effectValue = effect.getValue();
      if (!effectValue)
        return true; // Unknown write target, be conservative

      // Check if the write is to the same memref or a view of it
      if (effectValue == sourceMemref)
        return true;

      // Check if the effect value is derived from the same memref
      if (auto subview = effectValue.getDefiningOp<memref::SubViewOp>()) {
        if (subview.getSource() == sourceMemref)
          return true;
      }
    }
  }

  return false;
}

DiagnosedSilenceableFailure
transform::EliminateRedundantVectorTransfersOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {

  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  if (targets.empty()) {
    results.set(llvm::cast<OpResult>(getResult()), ArrayRef<Operation *>());
    return DiagnosedSilenceableFailure::success();
  }

  SmallVector<Operation *> transformedOps;
  int eliminatedCount = 0;

  for (Operation *target : targets) {
    // Collect all vector.transfer_read operations in this target
    SmallVector<vector::TransferReadOp> transferReads;
    target->walk([&](vector::TransferReadOp readOp) {
      transferReads.push_back(readOp);
    });

    // Track which reads have been eliminated
    llvm::SmallDenseSet<Operation *> eliminated;

    // Compare each pair of reads
    for (size_t i = 0; i < transferReads.size(); ++i) {
      if (eliminated.contains(transferReads[i]))
        continue;

      for (size_t j = i + 1; j < transferReads.size(); ++j) {
        if (eliminated.contains(transferReads[j]))
          continue;

        vector::TransferReadOp firstRead = transferReads[i];
        vector::TransferReadOp secondRead = transferReads[j];

        // Check if the reads are identical
        if (!areIdenticalReads(firstRead, secondRead))
          continue;

        // Check if there are writes between them
        if (hasWritesBetweenReads(firstRead, secondRead))
          continue;

        // Replace the second read with the result of the first read
        rewriter.replaceAllUsesWith(secondRead.getResult(),
                                    firstRead.getResult());
        rewriter.eraseOp(secondRead);
        eliminated.insert(secondRead);
        eliminatedCount++;
      }
    }

    transformedOps.push_back(target);
  }

  if (eliminatedCount > 0) {
    LLVM_DEBUG(llvm::dbgs() << "Eliminated " << eliminatedCount
                            << " redundant vector.transfer_read operations\n");
  }

  results.set(llvm::cast<OpResult>(getResult()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// FlattenForIterArgsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::FlattenForIterArgsOp::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
                                       transform::TransformState &state) {

  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  if (targets.empty()) {
    results.set(llvm::cast<OpResult>(getResult()), ArrayRef<Operation *>());
    return DiagnosedSilenceableFailure::success();
  }

  SmallVector<Operation *> transformedOps;

  for (Operation *target : targets) {
    auto forOp = dyn_cast_if_present<scf::ForOp>(target);
    if (!forOp) {
      return emitDefiniteFailure() << "target must be an scf.for operation";
    }

    Location loc = forOp.getLoc();

    // Collect vector-typed iter_args
    SmallVector<unsigned> vectorIterArgIndices;
    SmallVector<VectorType> originalVectorTypes;
    SmallVector<VectorType> flattenedVectorTypes;

    for (auto [idx, iterArg] : llvm::enumerate(forOp.getInitArgs())) {
      if (auto vecType = dyn_cast_if_present<VectorType>(iterArg.getType())) {
        vectorIterArgIndices.push_back(idx);
        originalVectorTypes.push_back(vecType);

        // Create flattened vector type
        int64_t numElements = getVectorNumElements(vecType);
        VectorType flatType =
            VectorType::get({numElements}, vecType.getElementType());
        flattenedVectorTypes.push_back(flatType);
      }
    }

    // If no vector iter_args, nothing to do
    if (vectorIterArgIndices.empty()) {
      transformedOps.push_back(target);
      continue;
    }

    // Step 1: Insert vector.shape_cast operations before the loop to flatten
    // init values
    rewriter.setInsertionPoint(forOp);
    SmallVector<Value> newInitArgs(forOp.getInitArgs().begin(),
                                   forOp.getInitArgs().end());

    for (auto [idx, vecIdx] : llvm::enumerate(vectorIterArgIndices)) {
      Value initArg = forOp.getInitArgs()[vecIdx];
      auto shapeCast = vector::ShapeCastOp::create(
          rewriter, loc, flattenedVectorTypes[idx], initArg);
      newInitArgs[vecIdx] = shapeCast.getResult();
    }

    // Step 2: Create new result types (flattened for vector types)
    SmallVector<Type> newResultTypes;
    for (auto [idx, resultType] : llvm::enumerate(forOp.getResultTypes())) {
      auto it = llvm::find(vectorIterArgIndices, idx);
      if (it != vectorIterArgIndices.end()) {
        size_t vecIdx = std::distance(vectorIterArgIndices.begin(), it);
        newResultTypes.push_back(flattenedVectorTypes[vecIdx]);
      } else {
        newResultTypes.push_back(resultType);
      }
    }

    // Step 3: Create new scf.for with flattened iter_args
    auto newForOp =
        scf::ForOp::create(rewriter, loc, forOp.getLowerBound(),
                           forOp.getUpperBound(), forOp.getStep(), newInitArgs);

    // Step 4: Clone the loop body and insert shape_cast operations
    Block *oldBody = forOp.getBody();
    Block *newBody = newForOp.getBody();

    rewriter.setInsertionPointToStart(newBody);
    IRMapping mapping;

    // Map the induction variable
    mapping.map(oldBody->getArgument(0), newBody->getArgument(0));

    // For vector iter_args, insert shape_cast to convert back to original shape
    for (auto [idx, vecIdx] : llvm::enumerate(vectorIterArgIndices)) {
      BlockArgument newArg = newBody->getArgument(vecIdx + 1);
      auto shapeCast = vector::ShapeCastOp::create(
          rewriter, loc, originalVectorTypes[idx], newArg);
      mapping.map(oldBody->getArgument(vecIdx + 1), shapeCast.getResult());
    }

    // Map non-vector iter_args directly
    for (auto [idx, arg] :
         llvm::enumerate(oldBody->getArguments().drop_front(1))) {
      if (llvm::find(vectorIterArgIndices, idx) == vectorIterArgIndices.end()) {
        mapping.map(arg, newBody->getArgument(idx + 1));
      }
    }

    // Clone operations from old body (except the terminator)
    for (Operation &op : oldBody->without_terminator()) {
      rewriter.clone(op, mapping);
    }

    // Step 5: Handle the yield operation
    auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
    SmallVector<Value> newYieldOperands;

    for (auto [idx, yieldValue] : llvm::enumerate(oldYield.getOperands())) {
      auto it = llvm::find(vectorIterArgIndices, idx);
      if (it != vectorIterArgIndices.end()) {
        // Flatten the yielded vector value
        size_t vecIdx = std::distance(vectorIterArgIndices.begin(), it);
        Value mappedValue = mapping.lookup(yieldValue);
        auto shapeCast = vector::ShapeCastOp::create(
            rewriter, loc, flattenedVectorTypes[vecIdx], mappedValue);
        newYieldOperands.push_back(shapeCast.getResult());
      } else {
        newYieldOperands.push_back(mapping.lookup(yieldValue));
      }
    }

    scf::YieldOp::create(rewriter, loc, newYieldOperands);

    // Step 6: Insert shape_cast operations after the loop to convert results
    // back
    rewriter.setInsertionPointAfter(newForOp);
    SmallVector<Value> finalResults;

    for (auto [idx, result] : llvm::enumerate(newForOp.getResults())) {
      auto it = llvm::find(vectorIterArgIndices, idx);
      if (it != vectorIterArgIndices.end()) {
        size_t vecIdx = std::distance(vectorIterArgIndices.begin(), it);
        auto shapeCast = vector::ShapeCastOp::create(
            rewriter, loc, originalVectorTypes[vecIdx], result);
        finalResults.push_back(shapeCast.getResult());
      } else {
        finalResults.push_back(result);
      }
    }

    // Replace uses of the old loop's results
    rewriter.replaceOp(forOp, finalResults);

    transformedOps.push_back(newForOp.getOperation());
  }

  results.set(llvm::cast<OpResult>(getResult()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// HoistVectorTransferPointersOp
//===----------------------------------------------------------------------===//

namespace {
/// Check if a value depends on the given loop induction variable
bool dependsOnLoopIVForHoist(Value val, Value loopIV);

// The coefficient of `forOp`'s induction variable in `v`, when `v` is that
// variable times a constant plus values defined outside the loop;
// std::nullopt otherwise.
std::optional<int64_t> linearIVCoefficient(Value v, scf::ForOp forOp) {
  Value iv = forOp.getInductionVar();
  if (!v.getType().isIndex())
    return std::nullopt;
  AffineMap map;
  ValueDimList operands;
  auto stop = [&](Value x, std::optional<int64_t>, ValueBoundsConstraintSet &) {
    return x == iv || forOp.isDefinedOutsideOfLoop(x);
  };
  if (failed(ValueBoundsConstraintSet::computeBound(
          map, operands, presburger::BoundType::EQ,
          ValueBoundsConstraintSet::Variable(v), stop)))
    return std::nullopt;
  // Linear: one column per operand and the constant, no mod or div columns.
  std::vector<SmallVector<int64_t, 8>> flat;
  if (map.getNumResults() != 1 || failed(getFlattenedAffineExprs(map, &flat)) ||
      flat[0].size() != map.getNumInputs() + 1)
    return std::nullopt;
  int64_t coefficient = 0;
  for (auto [pos, operand] : llvm::enumerate(operands))
    if (operand.first == iv)
      coefficient += flat[0][pos];
  return coefficient;
}

// Whether `v` can be computed before `forOp` for its first iteration: every
// value it is computed from inside the loop is the induction variable or
// produced by ops (not loop-carried block arguments).
bool isComputableAtLowerBound(Value v, scf::ForOp forOp) {
  if (v == forOp.getInductionVar())
    return true;
  Operation *def = v.getDefiningOp();
  if (!def)
    return !forOp->isAncestor(cast<BlockArgument>(v).getOwner()->getParentOp());
  if (!forOp->isAncestor(def))
    return true;
  // Cloned before the loop, it runs even when the loop does not: only pure
  // ops.
  if (def->getNumRegions() != 0 || !isPure(def))
    return false;
  return llvm::all_of(def->getOperands(), [&](Value operand) {
    return isComputableAtLowerBound(operand, forOp);
  });
}

// `v` as of the first iteration of `forOp`: its computation cloned before the
// loop (at the rewriter's insertion point) with the induction variable
// replaced by the lower bound. Requires isComputableAtLowerBound.
Value cloneAtLowerBound(Value v, scf::ForOp forOp, RewriterBase &rewriter,
                        IRMapping &mapping) {
  if (v == forOp.getInductionVar())
    return forOp.getLowerBound();
  if (Value mapped = mapping.lookupOrNull(v))
    return mapped;
  Operation *def = v.getDefiningOp();
  if (!def || !forOp->isAncestor(def))
    return v;
  for (Value operand : def->getOperands())
    mapping.map(operand, cloneAtLowerBound(operand, forOp, rewriter, mapping));
  Operation *clone = rewriter.clone(*def, mapping);
  return clone->getResult(cast<OpResult>(v).getResultNumber());
}

// `base`, a contiguous row-major memref, collapsed to rank 1.
Value collapseToRank1(RewriterBase &rewriter, Location loc, Value base) {
  auto type = cast<MemRefType>(base.getType());
  if (type.getRank() <= 1)
    return base;
  ReassociationIndices all =
      llvm::to_vector(llvm::seq<int64_t>(0, type.getRank()));
  return memref::CollapseShapeOp::create(rewriter, loc, base,
                                         ArrayRef<ReassociationIndices>{all});
}

// Indices of `type` to the element offset they address, from its strides.
AffineMap linearIndexMap(MemRefType type) {
  MLIRContext *ctx = type.getContext();
  AffineExpr e = getAffineConstantExpr(0, ctx);
  for (auto [i, stride] : llvm::enumerate(type.getStridesAndOffset().first))
    e = e + getAffineDimExpr(i, ctx) * stride;
  return AffineMap::get(type.getRank(), 0, e);
}

// The distance, in elements of `type`, from the address of `b` to that of
// `a` on the first trip of `forOp`, if it is a constant.
std::optional<int64_t> constantOffsetBetween(MemRefType type, ArrayRef<Value> a,
                                             ArrayRef<Value> b,
                                             scf::ForOp forOp) {
  auto lb = getConstantIntValue(forOp.getLowerBound());
  if (!lb)
    return std::nullopt;
  AffineMap lin = linearIndexMap(type);
  unsigned n = lin.getNumDims();
  MLIRContext *ctx = forOp.getContext();
  AffineExpr ea = lin.getResult(0);
  AffineExpr eb = lin.getResult(0).shiftDims(n, n);
  AffineMap map = AffineMap::get(2 * n, 0, ea - eb, ctx);
  SmallVector<Value> operands(a.begin(), a.end());
  operands.append(b.begin(), b.end());
  affine::fullyComposeAffineMapAndOperands(&map, &operands);
  affine::canonicalizeMapAndOperands(&map, &operands);
  Value iv = forOp.getInductionVar();
  SmallVector<AffineExpr> dims, syms;
  for (unsigned d = 0; d < map.getNumDims(); ++d)
    dims.push_back(operands[d] == iv ? getAffineConstantExpr(*lb, ctx)
                                     : getAffineDimExpr(d, ctx));
  for (unsigned k = 0; k < map.getNumSymbols(); ++k)
    syms.push_back(operands[map.getNumDims() + k] == iv
                       ? getAffineConstantExpr(*lb, ctx)
                       : getAffineSymbolExpr(k, ctx));
  auto c = dyn_cast<AffineConstantExpr>(
      simplifyAffineExpr(map.getResult(0).replaceDimsAndSymbols(dims, syms),
                         map.getNumDims(), map.getNumSymbols()));
  if (!c)
    return std::nullopt;
  return c.getValue();
}

// How many elements of `type` the address of `indices` moves per trip of
// `forOp`, if that is the same constant on every trip even though no single
// index is linear in the induction variable (as with a delinearized induction
// variable); std::nullopt otherwise. Needs constant loop bounds.
std::optional<int64_t> constantOffsetPerTrip(MemRefType type,
                                             ArrayRef<Value> indices,
                                             scf::ForOp forOp) {
  auto lb = getConstantIntValue(forOp.getLowerBound());
  auto ub = getConstantIntValue(forOp.getUpperBound());
  auto step = getConstantIntValue(forOp.getStep());
  if (!lb || !ub || !step || *step <= 0 || *ub <= *lb ||
      (*ub - *lb) / *step > 4096)
    return std::nullopt;
  AffineMap map = linearIndexMap(type);
  SmallVector<Value> operands(indices.begin(), indices.end());
  affine::fullyComposeAffineMapAndOperands(&map, &operands);
  affine::canonicalizeMapAndOperands(&map, &operands);
  Value iv = forOp.getInductionVar();
  for (Value o : operands)
    if (o != iv && !forOp.isDefinedOutsideOfLoop(o))
      return std::nullopt;
  MLIRContext *ctx = forOp.getContext();
  unsigned nd = map.getNumDims(), ns = map.getNumSymbols();
  auto at = [&](int64_t v) {
    SmallVector<AffineExpr> dims, syms;
    for (unsigned d = 0; d < nd; ++d)
      dims.push_back(operands[d] == iv ? getAffineConstantExpr(v, ctx)
                                       : getAffineDimExpr(d, ctx));
    for (unsigned k = 0; k < ns; ++k)
      syms.push_back(operands[nd + k] == iv ? getAffineConstantExpr(v, ctx)
                                            : getAffineSymbolExpr(k, ctx));
    return map.getResult(0).replaceDimsAndSymbols(dims, syms);
  };
  AffineExpr first = at(*lb);
  std::optional<int64_t> perTrip;
  int64_t trip = 1;
  for (int64_t v = *lb + *step; v < *ub; v += *step, ++trip) {
    auto delta =
        dyn_cast<AffineConstantExpr>(simplifyAffineExpr(at(v) - first, nd, ns));
    if (!delta)
      return std::nullopt;
    if (!perTrip)
      perTrip = delta.getValue();
    if (delta.getValue() != trip * *perTrip)
      return std::nullopt;
  }
  return perTrip.value_or(0);
}

bool dependsOnLoopIVForHoist(Value val, Value loopIV) {
  if (val == loopIV)
    return true;

  // Check if the value is defined by an affine.apply that uses the loop IV
  if (auto affineOp = val.getDefiningOp<affine::AffineApplyOp>()) {
    for (Value operand : affineOp.getMapOperands()) {
      if (dependsOnLoopIVForHoist(operand, loopIV))
        return true;
    }
  }

  // Check for arithmetic operations
  if (auto defOp = val.getDefiningOp()) {
    for (Value operand : defOp->getOperands()) {
      if (dependsOnLoopIVForHoist(operand, loopIV))
        return true;
    }
  }

  return false;
}
} // namespace

DiagnosedSilenceableFailure transform::HoistVectorTransferPointersOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {

  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  if (targets.empty()) {
    results.set(llvm::cast<OpResult>(getResult()), ArrayRef<Operation *>());
    return DiagnosedSilenceableFailure::success();
  }

  SmallVector<Operation *> transformedOps;

  for (Operation *target : targets) {
    auto forOp = dyn_cast_if_present<scf::ForOp>(target);
    if (!forOp) {
      return emitDefiniteFailure() << "target must be an scf.for operation";
    }

    Value loopIV = forOp.getInductionVar();
    Location loc = forOp.getLoc();
    OpBuilder::InsertionGuard guard(rewriter);

    // Collect all vector transfer operations with IV-dependent indices
    struct TransferOpInfo {
      Operation *op;
      Value base;
      MemRefType memrefType;
      VectorType vectorType;
      SmallVector<Value> indices;
      int64_t constantStride; // Total constant stride per iteration
      bool hasIVDependentIndices;
      bool flattenable;
    };

    SmallVector<TransferOpInfo> transferOps;
    std::optional<int64_t> loopStep = getConstantIntValue(forOp.getStep());
    bool unsupported = false;

    for (Operation &op : forOp.getBody()->without_terminator()) {
      auto transferOp = dyn_cast_if_present<VectorTransferOpInterface>(&op);
      if (!transferOp)
        continue;

      Value base = transferOp.getBase();
      auto memrefType = dyn_cast_if_present<MemRefType>(base.getType());
      if (!memrefType)
        continue;

      VectorType vectorType;
      if (auto readOp = dyn_cast_if_present<vector::TransferReadOp>(&op)) {
        vectorType = readOp.getVectorType();
      } else if (auto writeOp =
                     dyn_cast_if_present<vector::TransferWriteOp>(&op)) {
        vectorType = writeOp.getVectorType();
      } else {
        continue;
      }

      SmallVector<Value> indices(transferOp.getIndices().begin(),
                                 transferOp.getIndices().end());

      // The pointer walk, and the flattening of a transfer the loop does not
      // move, address the collapsed memref directly and mark the access in
      // bounds. That only reproduces an unmasked, in-bounds transfer with a
      // minor identity map whose vector is one contiguous run of a contiguous
      // row-major memref.
      bool flattenable =
          !transferOp.getMask() &&
          transferOp.getPermutationMap().isMinorIdentity() &&
          !transferOp.hasOutOfBoundsDim() && memrefType.getRank() >= 1 &&
          vectorType.getRank() <= memrefType.getRank() &&
          memref::isStaticShapeAndContiguousRowMajor(memrefType) &&
          vector::isContiguousSlice(memrefType, vectorType);

      // Whether any index depends on the IV, and how far the transfer moves
      // per iteration.
      bool hasIVDependentIndices = false;
      int64_t constantStride = 0;
      SmallVector<int64_t> strides;
      if (flattenable)
        strides = memrefType.getStridesAndOffset().first;
      bool perIndexLinear = true;
      for (size_t dimIdx = 0; dimIdx < indices.size(); ++dimIdx) {
        Value idx = indices[dimIdx];
        if (!dependsOnLoopIVForHoist(idx, loopIV))
          continue;
        hasIVDependentIndices = true;
        // The index advances by (its coefficient on the IV) * (the loop
        // step) per iteration. An index not linear in the IV cannot be
        // walked by a constant increment.
        std::optional<int64_t> coefficient = linearIVCoefficient(idx, forOp);
        if (!flattenable || !coefficient || !loopStep ||
            !isComputableAtLowerBound(idx, forOp)) {
          perIndexLinear = false;
          break;
        }
        constantStride += strides[dimIdx] * *coefficient * *loopStep;
      }
      if (!perIndexLinear) {
        if (!flattenable || !llvm::all_of(indices, [&](Value idx) {
              return isComputableAtLowerBound(idx, forOp);
            })) {
          unsupported = true;
          break;
        }
        // The indices together may still move the access by a constant on
        // every trip; if they do not, the access is left as it is.
        std::optional<int64_t> perTrip =
            constantOffsetPerTrip(memrefType, indices, forOp);
        if (!perTrip)
          continue;
        constantStride = *perTrip;
      }
      if (unsupported)
        break;

      transferOps.push_back({&op, base, memrefType, vectorType, indices,
                             constantStride, hasIVDependentIndices,
                             flattenable});
    }
    // Leave the loop as it is rather than walk a pointer that is wrong.
    if (unsupported) {
      transformedOps.push_back(forOp);
      continue;
    }

    // Accesses to one memref that move by the same stride and lie a constant
    // apart share one pointer: a follower addresses its leader's pointer plus
    // that constant. One pointer per stream, rather than per access, keeps
    // the loop within the core's pointer registers.
    SmallVector<int64_t> leaderOf(transferOps.size(), -1);
    SmallVector<int64_t> offsetFromLeader(transferOps.size(), 0);
    for (size_t i = 0; i < transferOps.size(); ++i) {
      const auto &info = transferOps[i];
      if (!info.hasIVDependentIndices)
        continue;
      for (size_t j = 0; j < i; ++j) {
        const auto &lead = transferOps[j];
        if (!lead.hasIVDependentIndices || leaderOf[j] != -1 ||
            lead.base != info.base ||
            lead.constantStride != info.constantStride)
          continue;
        if (auto delta = constantOffsetBetween(info.memrefType, info.indices,
                                               lead.indices, forOp)) {
          leaderOf[i] = j;
          offsetFromLeader[i] = *delta;
          break;
        }
      }
    }
    // The index of the iter_arg holding each leader's pointer.
    SmallVector<int64_t> iterArgOf(transferOps.size(), -1);

    // Prepare to add iter_args for each transfer operation with IV-dependent
    // indices
    SmallVector<Value> newInitArgs;
    SmallVector<Value> flatMemrefs;

    for (const auto &[infoIdx, info] : llvm::enumerate(transferOps)) {
      if (!info.hasIVDependentIndices || leaderOf[infoIdx] != -1)
        continue;
      iterArgOf[infoIdx] = newInitArgs.size();

      rewriter.setInsertionPoint(forOp);
      flatMemrefs.push_back(collapseToRank1(rewriter, loc, info.base));

      // The base pointer: the offset on the first iteration.
      AffineMap linearMap = linearIndexMap(info.memrefType);
      SmallVector<Value> baseIndices;
      IRMapping indexMapping;
      for (Value idx : info.indices) {
        if (!dependsOnLoopIVForHoist(idx, loopIV)) {
          if (auto defOp = idx.getDefiningOp()) {
            Value clonedIdx = cloneOpAndOperands(defOp, loopIV, forOp, rewriter,
                                                 indexMapping);
            if (clonedIdx)
              baseIndices.push_back(clonedIdx);
            else
              baseIndices.push_back(idx);
          } else {
            baseIndices.push_back(idx);
          }
        } else {
          // The index's value on the first iteration.
          baseIndices.push_back(
              cloneAtLowerBound(idx, forOp, rewriter, indexMapping));
        }
      }

      Value basePointer =
          affine::AffineApplyOp::create(rewriter, loc, linearMap, baseIndices);

      newInitArgs.push_back(basePointer);
    }

    // If there are no IV-dependent transfers, just process them normally
    if (newInitArgs.empty()) {
      // Process all transfers without using iter_args
      for (const auto &info : transferOps) {
        if (!info.flattenable)
          continue;
        rewriter.setInsertionPoint(info.op);

        // Flatten vector type
        int64_t numElements = getVectorNumElements(info.vectorType);
        VectorType flatVectorType =
            VectorType::get({numElements}, info.vectorType.getElementType());

        rewriter.setInsertionPoint(forOp);
        Value flatMemref = collapseToRank1(rewriter, loc, info.base);
        AffineMap linearMap = linearIndexMap(info.memrefType);

        rewriter.setInsertionPoint(info.op);
        Value currentPointer = affine::AffineApplyOp::create(
            rewriter, loc, linearMap, info.indices);

        // Transform the transfer operation
        AffineMap identityMap1D = AffineMap::get(
            1, 0, rewriter.getAffineDimExpr(0), rewriter.getContext());
        auto inBoundsAttr = rewriter.getBoolArrayAttr({true});

        if (auto readOp =
                dyn_cast_if_present<vector::TransferReadOp>(info.op)) {
          Value flatRead = vector::TransferReadOp::create(
              rewriter, loc, flatVectorType, flatMemref,
              ValueRange{currentPointer}, AffineMapAttr::get(identityMap1D),
              readOp.getPadding(),
              /*mask=*/Value(), inBoundsAttr);
          Value shapedRead = vector::ShapeCastOp::create(
              rewriter, loc, info.vectorType, flatRead);
          rewriter.replaceOp(readOp, shapedRead);
        } else if (auto writeOp =
                       dyn_cast_if_present<vector::TransferWriteOp>(info.op)) {
          Value flatValue = vector::ShapeCastOp::create(
              rewriter, loc, flatVectorType, writeOp.getVector());
          rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
              writeOp, flatValue, flatMemref, ValueRange{currentPointer},
              AffineMapAttr::get(identityMap1D), /*mask=*/Value(),
              inBoundsAttr);
        }
      }
      transformedOps.push_back(forOp);
      continue;
    }

    // Use replaceWithAdditionalYields to add pointer iter_args
    auto yieldValuesFn =
        [&](OpBuilder &b, Location yieldLoc,
            ArrayRef<BlockArgument> newBbArgs) -> SmallVector<Value> {
      SmallVector<Value> yieldValues;
      // The pointer updates go at the end of the body, after the last access
      // (the builder is positioned there when this is called). Updating a
      // pointer right after its first access would make the later accesses
      // address back from the updated value.
      OpBuilder::InsertPoint atYield = b.saveInsertionPoint();

      // Process each transfer operation with IV-dependent indices
      for (size_t i = 0; i < transferOps.size(); ++i) {
        const auto &info = transferOps[i];
        if (!info.hasIVDependentIndices)
          continue;

        bool follower = leaderOf[i] != -1;
        int64_t iterArgIdx = follower ? iterArgOf[leaderOf[i]] : iterArgOf[i];
        BlockArgument leaderPtr =
            newBbArgs[newBbArgs.size() - newInitArgs.size() + iterArgIdx];
        Value flatMemref = flatMemrefs[iterArgIdx];
        Value ptrIterArg = leaderPtr;
        if (follower && offsetFromLeader[i] != 0) {
          b.setInsertionPoint(info.op);
          ptrIterArg = arith::AddIOp::create(
              b, loc, leaderPtr,
              arith::ConstantIndexOp::create(b, loc, offsetFromLeader[i]));
        }

        // Flatten vector type
        int64_t numElements = getVectorNumElements(info.vectorType);
        VectorType flatVectorType =
            VectorType::get({numElements}, info.vectorType.getElementType());

        // Transform the transfer operation to use the iter_arg pointer
        b.setInsertionPoint(info.op);

        AffineMap identityMap1D =
            AffineMap::get(1, 0, b.getAffineDimExpr(0), b.getContext());
        auto inBoundsAttr = b.getBoolArrayAttr({true});

        if (auto readOp =
                dyn_cast_if_present<vector::TransferReadOp>(info.op)) {
          Value flatRead = vector::TransferReadOp::create(
              b, loc, flatVectorType, flatMemref, ValueRange{ptrIterArg},
              AffineMapAttr::get(identityMap1D), readOp.getPadding(),
              /*mask=*/Value(), inBoundsAttr);
          Value shapedRead =
              vector::ShapeCastOp::create(b, loc, info.vectorType, flatRead);
          rewriter.replaceOp(readOp, shapedRead);
        } else if (auto writeOp =
                       dyn_cast_if_present<vector::TransferWriteOp>(info.op)) {
          Value flatValue = vector::ShapeCastOp::create(b, loc, flatVectorType,
                                                        writeOp.getVector());
          rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
              writeOp, flatValue, flatMemref, ValueRange{ptrIterArg},
              AffineMapAttr::get(identityMap1D), /*mask=*/Value(),
              inBoundsAttr);
        }

        if (follower)
          continue;
        // Compute next pointer value: current_ptr + constant_stride
        OpBuilder::InsertionGuard atYieldGuard(b);
        b.restoreInsertionPoint(atYield);
        Value strideConst =
            arith::ConstantIndexOp::create(b, yieldLoc, info.constantStride);
        Value nextPtr =
            arith::AddIOp::create(b, yieldLoc, leaderPtr, strideConst);
        yieldValues.push_back(nextPtr);
      }

      return yieldValues;
    };

    // Create new loop with additional iter_args for pointers
    FailureOr<LoopLikeOpInterface> newLoopResult =
        cast<LoopLikeOpInterface>(forOp.getOperation())
            .replaceWithAdditionalYields(
                rewriter, newInitArgs, // new init operands (base pointers)
                true,                  // replace uses in loop
                yieldValuesFn);

    if (failed(newLoopResult)) {
      return emitDefiniteFailure() << "failed to add pointer iter_args to loop";
    }

    transformedOps.push_back(newLoopResult->getOperation());
  }

  results.set(llvm::cast<OpResult>(getResult()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// HoistCastPairOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::HoistCastPairOp::apply(transform::TransformRewriter &rewriter,
                                  transform::TransformResults &results,
                                  transform::TransformState &state) {

  SmallVector<Operation *> extensionOps =
      llvm::to_vector(state.getPayloadOps(getExtensionOp()));
  SmallVector<Operation *> truncationOps =
      llvm::to_vector(state.getPayloadOps(getTruncationOp()));
  SmallVector<Operation *> loopOps =
      llvm::to_vector(state.getPayloadOps(getLoopOp()));

  if (extensionOps.size() != 1 || truncationOps.size() != 1 ||
      loopOps.size() != 1) {
    return emitDefiniteFailure() << "requires exactly one extension_op, "
                                    "truncation_op, and loop_op handle";
  }

  Operation *extensionOp = extensionOps[0];
  Operation *truncationOp = truncationOps[0];
  auto loopOp = dyn_cast_if_present<scf::ForOp>(loopOps[0]);

  if (!loopOp) {
    return emitDefiniteFailure() << "loop_op handle must be scf.for";
  }

  // Determine extension/truncation operation types and get input/output values
  Value extensionInput, extensionOutput;
  Value truncationInput, truncationOutput;
  bool isFloatingPoint = false;

  if (auto extsiOp = dyn_cast_if_present<arith::ExtSIOp>(extensionOp)) {
    extensionInput = extsiOp.getIn();
    extensionOutput = extsiOp.getOut();
    auto trunciOp = dyn_cast_if_present<arith::TruncIOp>(truncationOp);
    if (!trunciOp) {
      return emitDefiniteFailure()
             << "arith.extsi must be paired with arith.trunci";
    }
    truncationInput = trunciOp.getIn();
    truncationOutput = trunciOp.getOut();
  } else if (auto extuiOp = dyn_cast_if_present<arith::ExtUIOp>(extensionOp)) {
    extensionInput = extuiOp.getIn();
    extensionOutput = extuiOp.getOut();
    auto trunciOp = dyn_cast_if_present<arith::TruncIOp>(truncationOp);
    if (!trunciOp) {
      return emitDefiniteFailure()
             << "arith.extui must be paired with arith.trunci";
    }
    truncationInput = trunciOp.getIn();
    truncationOutput = trunciOp.getOut();
  } else if (auto extfOp = dyn_cast_if_present<arith::ExtFOp>(extensionOp)) {
    extensionInput = extfOp.getIn();
    extensionOutput = extfOp.getOut();
    auto truncfOp = dyn_cast_if_present<arith::TruncFOp>(truncationOp);
    if (!truncfOp) {
      return emitDefiniteFailure()
             << "arith.extf must be paired with arith.truncf";
    }
    truncationInput = truncfOp.getIn();
    truncationOutput = truncfOp.getOut();
    isFloatingPoint = true;
  } else {
    return emitDefiniteFailure() << "extension operation must be arith.extsi, "
                                    "arith.extui, or arith.extf";
  }

  // Verify extension and truncation are in the loop
  if (!loopOp->isProperAncestor(extensionOp) ||
      !loopOp->isProperAncestor(truncationOp)) {
    return emitDefiniteFailure()
           << "extension and truncation operations must be inside the loop";
  }

  // Find which iter_arg the extension operates on
  BlockArgument iterArg = nullptr;
  int64_t iterArgIndex = -1;
  vector::ShapeCastOp shapeCastBeforeExtension = nullptr;

  // The extension input might be the iter_arg directly, or derived from it
  // through shape_cast
  if (auto blockArg = dyn_cast_if_present<BlockArgument>(extensionInput)) {
    if (blockArg.getOwner() == loopOp.getBody() &&
        blockArg.getArgNumber() > 0) {
      iterArg = blockArg;
      iterArgIndex = blockArg.getArgNumber() - 1;
    }
  } else if (auto shapeCastOp =
                 extensionInput.getDefiningOp<vector::ShapeCastOp>()) {
    Value shapeCastSource = shapeCastOp.getSource();
    if (auto blockArg = dyn_cast_if_present<BlockArgument>(shapeCastSource)) {
      if (blockArg.getOwner() == loopOp.getBody() &&
          blockArg.getArgNumber() > 0) {
        iterArg = blockArg;
        iterArgIndex = blockArg.getArgNumber() - 1;
        shapeCastBeforeExtension = shapeCastOp;
      }
    }
  }

  if (!iterArg) {
    return emitDefiniteFailure() << "extension must operate on a loop iter_arg "
                                    "(directly or through shape_cast)";
  }

  // Find the value that gets yielded (should come from truncation, possibly
  // through shape_cast)
  vector::ShapeCastOp shapeCastAfterTruncation = nullptr;

  auto yieldOp = cast<scf::YieldOp>(loopOp.getBody()->getTerminator());
  bool truncationIsYielded = false;
  int64_t yieldIndex = -1;

  for (auto [idx, yieldValue] : llvm::enumerate(yieldOp.getOperands())) {
    if (yieldValue == truncationOutput) {
      truncationIsYielded = true;
      yieldIndex = idx;
      break;
    } else if (auto shapeCast =
                   yieldValue.getDefiningOp<vector::ShapeCastOp>()) {
      if (shapeCast.getSource() == truncationOutput) {
        truncationIsYielded = true;
        yieldIndex = idx;
        shapeCastAfterTruncation = shapeCast;
        break;
      }
    }
  }

  if (!truncationIsYielded || yieldIndex != iterArgIndex) {
    return emitDefiniteFailure() << "truncation result must be yielded at the "
                                    "same position as the extension iter_arg";
  }

  Location loc = loopOp.getLoc();

  // Step 1: Hoist extension before the loop (don't hoist shape_cast yet)
  rewriter.setInsertionPoint(loopOp);
  Value initValue = loopOp.getInitArgs()[iterArgIndex];

  // Get the wide element type from the extension output
  Type wideElemType =
      cast<VectorType>(extensionOutput.getType()).getElementType();
  Type wideInitType = VectorType::get(
      cast<VectorType>(initValue.getType()).getShape(), wideElemType);

  // Extend the init value directly (in narrow flat form)
  Value extendedInit;
  if (isFloatingPoint) {
    extendedInit =
        arith::ExtFOp::create(rewriter, loc, wideInitType, initValue);
  } else if (isa<arith::ExtSIOp>(extensionOp)) {
    extendedInit =
        arith::ExtSIOp::create(rewriter, loc, wideInitType, initValue);
  } else {
    extendedInit =
        arith::ExtUIOp::create(rewriter, loc, wideInitType, initValue);
  }

  // Step 2: Create new loop with wide type for this iter_arg
  SmallVector<Value> newInitArgs(loopOp.getInitArgs().begin(),
                                 loopOp.getInitArgs().end());
  newInitArgs[iterArgIndex] = extendedInit;

  auto newLoopOp =
      scf::ForOp::create(rewriter, loc, loopOp.getLowerBound(),
                         loopOp.getUpperBound(), loopOp.getStep(), newInitArgs);

  // Step 3: Clone the loop body with proper type adjustments
  Block *oldBody = loopOp.getBody();
  Block *newBody = newLoopOp.getBody();

  rewriter.setInsertionPointToStart(newBody);
  IRMapping mapping;

  // Map the induction variable
  mapping.map(oldBody->getArgument(0), newBody->getArgument(0));

  // Map iter_args
  for (auto [idx, oldArg] :
       llvm::enumerate(oldBody->getArguments().drop_front(1))) {
    mapping.map(oldArg, newBody->getArgument(idx + 1));
  }

  // Clone operations from old body, adjusting types as needed
  for (Operation &op : oldBody->without_terminator()) {
    // Skip extension - its result will be mapped to the wide iter_arg or wide
    // shape_cast
    if (&op == extensionOp) {
      if (shapeCastBeforeExtension) {
        // Map extension result to the shape_cast result (which we'll create
        // below) Don't map yet - we'll map it when we encounter the shape_cast
      } else {
        // No shape_cast: map extension result directly to the wide iter_arg
        mapping.map(extensionOutput, newBody->getArgument(iterArgIndex + 1));
      }
      continue;
    }

    // Skip truncation - we'll handle the yielded value specially
    if (&op == truncationOp) {
      continue;
    }

    // Handle shape_cast before extension - clone it with wide element type
    if (shapeCastBeforeExtension &&
        &op == shapeCastBeforeExtension.getOperation()) {
      auto narrowVecType =
          cast<VectorType>(shapeCastBeforeExtension.getResult().getType());
      auto wideVecType =
          VectorType::get(narrowVecType.getShape(), wideElemType);

      Value mappedSource = mapping.lookup(shapeCastBeforeExtension.getSource());
      auto newShapeCast =
          vector::ShapeCastOp::create(rewriter, loc, wideVecType, mappedSource);
      mapping.map(shapeCastBeforeExtension.getResult(),
                  newShapeCast.getResult());
      mapping.map(extensionOutput, newShapeCast.getResult());
      continue;
    }

    // Handle shape_cast after truncation - clone it with wide element type for
    // the yield
    if (shapeCastAfterTruncation &&
        &op == shapeCastAfterTruncation.getOperation()) {
      // We'll handle this in the yield processing
      continue;
    }

    // Clone all other operations normally
    rewriter.clone(op, mapping);
  }

  // Step 4: Update the yield to yield the wide value
  auto oldYield = cast<scf::YieldOp>(oldBody->getTerminator());
  SmallVector<Value> newYieldOperands;

  for (auto [idx, yieldValue] : llvm::enumerate(oldYield.getOperands())) {
    if ((int64_t)idx == iterArgIndex) {
      // Get the wide value (truncation input)
      Value wideValue = mapping.lookup(truncationInput);

      // If there was a shape_cast after truncation, we need to create a wide
      // version of it
      if (shapeCastAfterTruncation) {
        auto narrowVecType =
            cast<VectorType>(shapeCastAfterTruncation.getResult().getType());
        auto wideVecType =
            VectorType::get(narrowVecType.getShape(), wideElemType);

        auto newShapeCast =
            vector::ShapeCastOp::create(rewriter, loc, wideVecType, wideValue);
        newYieldOperands.push_back(newShapeCast.getResult());
      } else {
        newYieldOperands.push_back(wideValue);
      }
    } else {
      newYieldOperands.push_back(mapping.lookup(yieldValue));
    }
  }

  scf::YieldOp::create(rewriter, loc, newYieldOperands);

  // Step 5: Hoist truncation after the loop
  rewriter.setInsertionPointAfter(newLoopOp);
  Value wideResult = newLoopOp.getResults()[iterArgIndex];

  // Get the narrow element type from the original init value
  auto narrowElemType =
      cast<VectorType>(loopOp.getInitArgs()[iterArgIndex].getType())
          .getElementType();
  auto narrowResultType = VectorType::get(
      cast<VectorType>(wideResult.getType()).getShape(), narrowElemType);

  // Create the appropriate truncation operation based on type
  Value narrowResult;
  if (isFloatingPoint) {
    narrowResult =
        arith::TruncFOp::create(rewriter, loc, narrowResultType, wideResult);
  } else {
    narrowResult =
        arith::TruncIOp::create(rewriter, loc, narrowResultType, wideResult);
  }

  // Step 6: Replace uses of the old loop
  SmallVector<Value> finalResults;
  for (auto [idx, result] : llvm::enumerate(newLoopOp.getResults())) {
    if ((int64_t)idx == iterArgIndex) {
      finalResults.push_back(narrowResult);
    } else {
      finalResults.push_back(result);
    }
  }

  rewriter.replaceOp(loopOp, finalResults);

  SmallVector<Operation *> resultOps = {newLoopOp.getOperation()};
  results.set(llvm::cast<OpResult>(getResult()), resultOps);
  return DiagnosedSilenceableFailure::success();
}

void transform::HoistCastPairOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getExtensionOpMutable(), effects);
  consumesHandle(getTruncationOpMutable(), effects);
  onlyReadsHandle(getLoopOpMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// FoldUnitExtentDimsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::FoldUnitExtentDimsOp::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
                                       transform::TransformState &state) {

  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  SmallVector<Operation *> transformedOps;
  for (Operation *target : targets) {
    auto funcOp = dyn_cast_if_present<func::FuncOp>(target);
    if (!funcOp)
      return emitDefiniteFailure() << "target must be a func.func operation";

    MLIRContext *ctx = funcOp.getContext();

    // LLVM 23's collapseValue rejects memrefs with non-identity layouts
    // (strided memrefs from subview ops). Override collapseFn to use
    // rank-reducing subviews for strided memrefs, allowing the fold to
    // handle linalg ops with subview outputs inside air.herd regions.
    RewritePatternSet foldPatterns(ctx);
    linalg::ControlDropUnitDims options;
    options.collapseFn =
        [](RewriterBase &rewriter, Location loc, Value operand,
           ArrayRef<int64_t> targetShape,
           ArrayRef<ReassociationIndices> reassociation,
           const linalg::ControlDropUnitDims &control) -> FailureOr<Value> {
      if (auto memrefType = dyn_cast<MemRefType>(operand.getType())) {
        if (!memrefType.getLayout().isIdentity()) {
          return memref::SubViewOp::rankReduceIfNeeded(rewriter, loc, operand,
                                                       targetShape);
        }
        MemRefLayoutAttrInterface layout;
        auto targetType =
            MemRefType::get(targetShape, memrefType.getElementType(), layout,
                            memrefType.getMemorySpace());
        return memref::CollapseShapeOp::create(rewriter, loc, targetType,
                                               operand, reassociation)
            .getResult();
      }
      return failure();
    };
    linalg::populateFoldUnitExtentDimsPatterns(foldPatterns, options);
    (void)applyPatternsGreedily(funcOp, std::move(foldPatterns));

    transformedOps.push_back(funcOp);
  }

  results.set(llvm::cast<OpResult>(getResult()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// ConvertSize1VectorToScalarOp
//===----------------------------------------------------------------------===//

/// Check if a type is a size-1 vector type
static bool isSize1VectorType(Type type) {
  auto vecType = dyn_cast_if_present<VectorType>(type);
  if (!vecType)
    return false;

  return vecType.getNumElements() == 1;
}

/// Convert vector.transfer_read of vector<1xT> to memref.load + broadcast
/// Following LLVM's pattern: Load scalar → Broadcast to vector<1xT>
struct ConvertSize1TransferReadToLoad
    : public OpRewritePattern<vector::TransferReadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferReadOp readOp,
                                PatternRewriter &rewriter) const override {
    VectorType vecType = readOp.getVectorType();
    if (!isSize1VectorType(vecType))
      return failure();

    // Create memref.load returning scalar
    Value scalarLoad = memref::LoadOp::create(
        rewriter, readOp.getLoc(), readOp.getBase(), readOp.getIndices());

    // Broadcast scalar to vector<1xT>
    Value vectorResult = vector::BroadcastOp::create(rewriter, readOp.getLoc(),
                                                     vecType, scalarLoad);

    rewriter.replaceOp(readOp, vectorResult);
    return success();
  }
};

/// Convert vector.transfer_write of vector<1xT> to extract + memref.store
/// Following LLVM's pattern: Extract to scalar → Store scalar
struct ConvertSize1TransferWriteToStore
    : public OpRewritePattern<vector::TransferWriteOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransferWriteOp writeOp,
                                PatternRewriter &rewriter) const override {
    VectorType vecType = writeOp.getVectorType();
    if (!isSize1VectorType(vecType))
      return failure();

    // Extract scalar from vector<1xT>
    SmallVector<int64_t> indices(vecType.getRank(), 0);
    Value scalarValue = vector::ExtractOp::create(rewriter, writeOp.getLoc(),
                                                  writeOp.getVector(), indices);

    // Create memref.store with scalar
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        writeOp, scalarValue, writeOp.getBase(), writeOp.getIndices());

    return success();
  }
};

/// Convert vector.load of vector<1xT> to memref.load + broadcast
struct ConvertSize1VectorLoadToLoad : public OpRewritePattern<vector::LoadOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::LoadOp loadOp,
                                PatternRewriter &rewriter) const override {
    VectorType vecType = loadOp.getVectorType();
    if (!isSize1VectorType(vecType))
      return failure();

    // Create memref.load returning scalar
    Value scalarLoad = memref::LoadOp::create(
        rewriter, loadOp.getLoc(), loadOp.getBase(), loadOp.getIndices());

    // Broadcast scalar to vector<1xT>
    Value vectorResult = vector::BroadcastOp::create(rewriter, loadOp.getLoc(),
                                                     vecType, scalarLoad);

    rewriter.replaceOp(loadOp, vectorResult);
    return success();
  }
};

/// Convert vector.store of vector<1xT> to extract + memref.store
struct ConvertSize1VectorStoreToStore
    : public OpRewritePattern<vector::StoreOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    VectorType vecType = storeOp.getVectorType();
    if (!isSize1VectorType(vecType))
      return failure();

    // Extract scalar from vector<1xT>
    SmallVector<int64_t> indices(vecType.getRank(), 0);
    Value scalarValue = vector::ExtractOp::create(
        rewriter, storeOp.getLoc(), storeOp.getValueToStore(), indices);

    // Create memref.store with scalar
    rewriter.replaceOpWithNewOp<memref::StoreOp>(
        storeOp, scalarValue, storeOp.getBase(), storeOp.getIndices());

    return success();
  }
};

/// Convert elementwise operations on vector<1xT> to scalar operations
/// Following LLVM's CastAwayElementwiseLeadingOneDim pattern
struct ConvertSize1VectorOpsToScalar : public RewritePattern {
  ConvertSize1VectorOpsToScalar(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Skip memory ops (handled by specific patterns)
    if (isa<vector::TransferReadOp, vector::TransferWriteOp, vector::LoadOp,
            vector::StoreOp, vector::BroadcastOp, vector::ExtractOp>(op))
      return failure();

    // Check if this is an elementwise operation with size-1 vector result
    if (!op->hasTrait<OpTrait::Elementwise>())
      return failure();

    // Check if any result is a size-1 vector
    bool hasSize1VectorResult = false;
    for (Type resultType : op->getResultTypes()) {
      if (isSize1VectorType(resultType)) {
        hasSize1VectorResult = true;
        break;
      }
    }

    if (!hasSize1VectorResult)
      return failure();

    Location loc = op->getLoc();

    // Extract all vector<1xT> operands to scalars
    SmallVector<Value> scalarOperands;
    for (Value operand : op->getOperands()) {
      if (isSize1VectorType(operand.getType())) {
        auto vecType = cast<VectorType>(operand.getType());
        SmallVector<int64_t> indices(vecType.getRank(), 0);
        Value scalarOperand =
            vector::ExtractOp::create(rewriter, loc, operand, indices);
        scalarOperands.push_back(scalarOperand);
      } else {
        scalarOperands.push_back(operand);
      }
    }

    // Determine scalar result types
    SmallVector<Type> scalarResultTypes;
    for (Type resultType : op->getResultTypes()) {
      if (auto vecType = dyn_cast_if_present<VectorType>(resultType)) {
        if (isSize1VectorType(vecType)) {
          scalarResultTypes.push_back(vecType.getElementType());
        } else {
          scalarResultTypes.push_back(resultType);
        }
      } else {
        scalarResultTypes.push_back(resultType);
      }
    }

    // Create scalar operation
    OperationState newState(loc, op->getName());
    newState.addOperands(scalarOperands);
    newState.addTypes(scalarResultTypes);
    newState.addAttributes(op->getAttrs());

    Operation *scalarOp = rewriter.create(newState);

    // Broadcast scalar results back to vector<1xT>
    SmallVector<Value> finalResults;
    for (auto [idx, resultType] : llvm::enumerate(op->getResultTypes())) {
      if (isSize1VectorType(resultType)) {
        Value broadcastResult = vector::BroadcastOp::create(
            rewriter, loc, cast<VectorType>(resultType),
            scalarOp->getResult(idx));
        finalResults.push_back(broadcastResult);
      } else {
        finalResults.push_back(scalarOp->getResult(idx));
      }
    }

    rewriter.replaceOp(op, finalResults);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// NormalizeLoopToUnitStepOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::NormalizeLoopToUnitStepOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  SmallVector<Operation *> loops;
  // Check every target before rewriting any.
  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto forOp = dyn_cast<scf::ForOp>(target);
    if (!forOp)
      return emitSilenceableError() << "expects scf.for targets";
    // The normalized bounds are built as affine maps over index values,
    // with signed division.
    if (!forOp.getInductionVar().getType().isIndex() || forOp.getUnsignedCmp())
      return emitSilenceableError()
             << "expects a signed scf.for over index values";
  }
  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto forOp = cast<scf::ForOp>(target);
    OpFoldResult lb = forOp.getLowerBound(), ub = forOp.getUpperBound(),
                 step = forOp.getStep();
    if (!isZeroInteger(lb) || !isOneInteger(step)) {
      Location loc = forOp.getLoc();
      rewriter.setInsertionPoint(forOp);
      Range range = emitNormalizedLoopBounds(rewriter, loc, lb, ub, step);
      rewriter.modifyOpInPlace(forOp, [&] {
        forOp.setLowerBound(
            getValueOrCreateConstantIndexOp(rewriter, loc, range.offset));
        forOp.setUpperBound(
            getValueOrCreateConstantIndexOp(rewriter, loc, range.size));
        forOp.setStep(
            getValueOrCreateConstantIndexOp(rewriter, loc, range.stride));
      });
      rewriter.setInsertionPointToStart(forOp.getBody());
      denormalizeInductionVariable(rewriter, loc, forOp.getInductionVar(), lb,
                                   step);
    }
    loops.push_back(forOp);
  }
  results.set(llvm::cast<OpResult>(getResult()), loops);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// MergeSiblingCopiesOp
//===----------------------------------------------------------------------===//

namespace {

// b - a, when it is provably a constant.
static std::optional<int64_t> constantDelta(OpFoldResult a, OpFoldResult b) {
  using Variable = ValueBoundsConstraintSet::Variable;
  MLIRContext *ctx = isa<Value>(a) ? cast<Value>(a).getContext()
                                   : cast<Attribute>(a).getContext();
  AffineMap diff =
      AffineMap::get(2, 0, getAffineDimExpr(1, ctx) - getAffineDimExpr(0, ctx));
  FailureOr<int64_t> d = ValueBoundsConstraintSet::computeConstantBound(
      presburger::BoundType::EQ, Variable(diff, {Variable(a), Variable(b)}));
  if (failed(d))
    return std::nullopt;
  return *d;
}

// A memref.copy of a reinterpret_cast view into a fresh buffer that is only
// read back through bufferization.to_tensor.
struct SiblingCopy {
  memref::CopyOp copy;
  memref::ReinterpretCastOp view;
  memref::AllocOp alloc;
  SmallVector<bufferization::ToTensorOp> reads;
  SmallVector<int64_t> sizes, strides;
};

static std::optional<SiblingCopy> matchSiblingCopy(memref::CopyOp copy) {
  auto view = copy.getSource().getDefiningOp<memref::ReinterpretCastOp>();
  auto alloc = copy.getTarget().getDefiningOp<memref::AllocOp>();
  if (!view || !alloc || !alloc.getType().getLayout().isIdentity() ||
      !alloc.getType().hasStaticShape())
    return std::nullopt;
  SiblingCopy c{copy, view, alloc, {}, {}, {}};
  for (OpFoldResult ofr : view.getMixedSizes()) {
    auto v = getConstantIntValue(ofr);
    if (!v)
      return std::nullopt;
    c.sizes.push_back(*v);
  }
  for (OpFoldResult ofr : view.getMixedStrides()) {
    auto v = getConstantIntValue(ofr);
    if (!v || *v <= 0)
      return std::nullopt;
    c.strides.push_back(*v);
  }
  if (ArrayRef<int64_t>(c.sizes) != alloc.getType().getShape())
    return std::nullopt;
  for (Operation *user : alloc->getUsers()) {
    if (user == copy)
      continue;
    auto read = dyn_cast<bufferization::ToTensorOp>(user);
    // The reads are rewritten to use a copy made where this copy is.
    if (!read || read->getBlock() != copy->getBlock() ||
        !copy->isBeforeInBlock(read))
      return std::nullopt;
    c.reads.push_back(read);
  }
  return c;
}

// The position, per dimension of a view with `strides`, of an element
// `delta` elements past the view's origin; std::nullopt if it is not on the
// view's grid.
static std::optional<SmallVector<int64_t>>
decomposeAlongStrides(int64_t delta, ArrayRef<int64_t> strides) {
  if (delta < 0)
    return std::nullopt;
  SmallVector<int64_t> idx;
  for (int64_t stride : strides) {
    idx.push_back(delta / stride);
    delta %= stride;
  }
  if (delta != 0)
    return std::nullopt;
  return idx;
}

} // namespace

DiagnosedSilenceableFailure
transform::MergeSiblingCopiesOp::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
                                       transform::TransformState &state) {
  SmallVector<Operation *> merged;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    // Group by base, rank, element type and destination memory space,
    // within one block. Strides are reconciled per group below: a dimension
    // of extent 1 may carry any stride.
    llvm::MapVector<std::tuple<Block *, Value, int64_t, Type, Attribute>,
                    SmallVector<SiblingCopy>>
        groups;
    target->walk([&](memref::CopyOp copy) {
      auto c = matchSiblingCopy(copy);
      if (!c)
        return;
      MemRefType allocTy = c->alloc.getType();
      groups[{copy->getBlock(), c->view.getSource(),
              static_cast<int64_t>(c->strides.size()), allocTy.getElementType(),
              allocTy.getMemorySpace()}]
          .push_back(*c);
    });

    for (auto &[key, members] : groups) {
      if (members.size() < 2)
        continue;
      // In program order; the region starts at the lowest offset.
      llvm::sort(members, [](const SiblingCopy &a, const SiblingCopy &b) {
        return a.copy->isBeforeInBlock(b.copy);
      });
      OpFoldResult firstOffset = members.front().view.getMixedOffsets()[0];
      SmallVector<int64_t> deltas;
      for (SiblingCopy &m : members) {
        auto d = constantDelta(firstOffset, m.view.getMixedOffsets()[0]);
        if (!d)
          break;
        deltas.push_back(*d);
      }
      if (deltas.size() != members.size())
        continue;
      int64_t minDelta = *llvm::min_element(deltas);
      // The common view's strides: a dimension's stride is set by the members
      // that span more than one element of it, which must agree.
      size_t rank = members.front().strides.size();
      SmallVector<int64_t> strides(members.front().strides);
      bool stridesAgree = true;
      for (size_t i = 0; i < rank; ++i) {
        std::optional<int64_t> set;
        for (SiblingCopy &m : members) {
          if (m.sizes[i] == 1)
            continue;
          if (set && *set != m.strides[i])
            stridesAgree = false;
          set = m.strides[i];
        }
        if (set)
          strides[i] = *set;
      }
      if (!stridesAgree)
        continue;
      SmallVector<SmallVector<int64_t>> positions;
      SmallVector<int64_t> bound(rank, 0);
      bool ok = true;
      for (auto [m, d] : llvm::zip_equal(members, deltas)) {
        auto idx = decomposeAlongStrides(d - minDelta, strides);
        if (!idx) {
          ok = false;
          break;
        }
        for (size_t i = 0; i < rank; ++i)
          bound[i] = std::max(bound[i], (*idx)[i] + m.sizes[i]);
        positions.push_back(*idx);
      }
      // The bounding region must stay a view of the same grid: in each inner
      // dimension it may not run past the next outer stride.
      for (size_t i = 1; ok && i < rank; ++i)
        if (strides[i - 1] % strides[i] == 0 &&
            bound[i] > strides[i - 1] / strides[i])
          ok = false;
      // It may not be larger than the copies it replaces, nor end past the
      // last element any of them reads, which could be past the source.
      if (ok) {
        int64_t boxElems = 1, memberElems = 0, boxLast = 0, memberLast = 0;
        for (size_t i = 0; i < rank; ++i) {
          boxElems *= bound[i];
          boxLast += (bound[i] - 1) * strides[i];
        }
        for (auto [m, pos] : llvm::zip_equal(members, positions)) {
          int64_t elems = 1, last = 0;
          for (size_t i = 0; i < rank; ++i) {
            elems *= m.sizes[i];
            last += (pos[i] + m.sizes[i] - 1) * strides[i];
          }
          memberElems += elems;
          memberLast = std::max(memberLast, last);
        }
        ok = boxElems <= memberElems && boxLast <= memberLast;
      }
      // The merged copy runs where the first one did, so nothing between the
      // first and the last may write memory: a later copy would otherwise
      // read a stale snapshot of the source.
      llvm::SmallPtrSet<Operation *, 8> memberOps;
      for (SiblingCopy &m : members)
        memberOps.insert(m.copy);
      for (Operation *op = members.front().copy->getNextNode();
           ok && op != members.back().copy.getOperation();
           op = op->getNextNode()) {
        if (memberOps.contains(op))
          continue;
        std::optional<SmallVector<MemoryEffects::EffectInstance>> effects =
            getEffectsRecursively(op);
        if (!effects ||
            llvm::any_of(*effects, [](const MemoryEffects::EffectInstance &e) {
              return isa<MemoryEffects::Write>(e.getEffect());
            }))
          ok = false;
      }
      if (!ok)
        continue;

      SiblingCopy &first = members.front();
      Location loc = first.copy.getLoc();
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(first.copy);
      OpFoldResult offset = firstOffset;
      if (minDelta != 0) {
        if (auto c = getConstantIntValue(firstOffset))
          offset = rewriter.getIndexAttr(*c + minDelta);
        else
          offset = arith::AddIOp::create(
                       rewriter, loc, cast<Value>(firstOffset),
                       arith::ConstantIndexOp::create(rewriter, loc, minDelta))
                       .getResult();
      }
      Type elemTy = first.alloc.getType().getElementType();
      auto viewTy = MemRefType::get(
          bound, elemTy,
          StridedLayoutAttr::get(rewriter.getContext(),
                                 getConstantIntValue(offset)
                                     ? *getConstantIntValue(offset)
                                     : ShapedType::kDynamic,
                                 strides),
          first.view.getType().getMemorySpace());
      auto view = memref::ReinterpretCastOp::create(
          rewriter, loc, viewTy, first.view.getSource(), offset,
          getAsIndexOpFoldResult(rewriter.getContext(), bound),
          getAsIndexOpFoldResult(rewriter.getContext(), strides));
      auto allocTy = MemRefType::get(bound, elemTy, MemRefLayoutAttrInterface{},
                                     first.alloc.getType().getMemorySpace());
      auto alloc = memref::AllocOp::create(rewriter, loc, allocTy);
      auto copy = memref::CopyOp::create(rewriter, loc, view, alloc);
      auto whole = bufferization::ToTensorOp::create(
          rewriter, loc, memref::getTensorTypeFromMemRefType(allocTy), alloc,
          rewriter.getUnitAttr(), rewriter.getUnitAttr());
      merged.push_back(copy);

      for (auto [m, pos] : llvm::zip_equal(members, positions)) {
        for (bufferization::ToTensorOp read : m.reads) {
          rewriter.setInsertionPoint(read);
          SmallVector<OpFoldResult> ones(rank, rewriter.getIndexAttr(1));
          rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
              read, cast<RankedTensorType>(read.getType()), whole,
              getAsIndexOpFoldResult(rewriter.getContext(), pos),
              getAsIndexOpFoldResult(rewriter.getContext(), m.sizes), ones);
        }
        rewriter.eraseOp(m.copy);
        rewriter.eraseOp(m.alloc);
        if (m.view->use_empty())
          rewriter.eraseOp(m.view);
      }
    }
  }
  results.set(llvm::cast<OpResult>(getResult()), merged);
  return DiagnosedSilenceableFailure::success();
}

void transform::MergeSiblingCopiesOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// CoalesceSlicesOp
//===----------------------------------------------------------------------===//

namespace {
// `v` as a simplified affine expression of the leaf values it is computed
// from through affine.apply, affine.delinearize_index and arith add/mul by
// constants. Leaves become dims. ValueBoundsConstraintSet has no model for
// affine.delinearize_index, through which these offsets are computed.
struct AffineOfValue {
  SmallVector<Value> leaves;
  std::optional<AffineExpr> build(Value v, MLIRContext *ctx, int depth = 0) {
    if (auto c = getConstantIntValue(v))
      return getAffineConstantExpr(*c, ctx);
    Operation *def = v.getDefiningOp();
    if (depth < 16 && def) {
      if (auto apply = dyn_cast<affine::AffineApplyOp>(def)) {
        AffineMap map = apply.getAffineMap();
        SmallVector<AffineExpr> dims, syms;
        for (unsigned i = 0; i < map.getNumDims(); ++i) {
          auto e = build(apply.getMapOperands()[i], ctx, depth + 1);
          if (!e)
            return std::nullopt;
          dims.push_back(*e);
        }
        for (unsigned i = 0; i < map.getNumSymbols(); ++i) {
          auto e = build(apply.getMapOperands()[map.getNumDims() + i], ctx,
                         depth + 1);
          if (!e)
            return std::nullopt;
          syms.push_back(*e);
        }
        return map.getResult(0).replaceDimsAndSymbols(dims, syms);
      }
      if (auto delin = dyn_cast<affine::AffineDelinearizeIndexOp>(def)) {
        auto in = build(delin.getLinearIndex(), ctx, depth + 1);
        SmallVector<int64_t> basis(delin.getStaticBasis());
        if (!in || llvm::any_of(basis, ShapedType::isDynamic))
          return std::nullopt;
        unsigned i = cast<OpResult>(v).getResultNumber();
        // With an outer bound the basis has one entry per result.
        bool hasOuter = basis.size() == delin.getNumResults();
        int64_t inner = 1;
        for (size_t k = (hasOuter ? i + 1 : i); k < basis.size(); ++k)
          inner *= basis[k];
        AffineExpr e = in->floorDiv(inner);
        if (i != 0 || hasOuter)
          e = e % basis[hasOuter ? i : i - 1];
        return e;
      }
      if (auto add = dyn_cast<arith::AddIOp>(def)) {
        auto a = build(add.getLhs(), ctx, depth + 1);
        auto b = build(add.getRhs(), ctx, depth + 1);
        if (a && b)
          return *a + *b;
        return std::nullopt;
      }
      if (auto mul = dyn_cast<arith::MulIOp>(def)) {
        auto a = build(mul.getLhs(), ctx, depth + 1);
        auto b = build(mul.getRhs(), ctx, depth + 1);
        if (a && b &&
            (isa<AffineConstantExpr>(*a) || isa<AffineConstantExpr>(*b)))
          return *a * *b;
        return std::nullopt;
      }
    }
    auto it = llvm::find(leaves, v);
    unsigned pos = it - leaves.begin();
    if (it == leaves.end())
      leaves.push_back(v);
    return getAffineDimExpr(pos, ctx);
  }
};

// The value of an offset that simplifies to a constant, such as
// `8 * ((8 * j) mod 4)`.
static std::optional<int64_t> provenConstant(OpFoldResult ofr,
                                             MLIRContext *ctx) {
  if (auto c = getConstantIntValue(ofr))
    return c;
  AffineOfValue builder;
  auto e = builder.build(cast<Value>(ofr), ctx);
  if (!e)
    return std::nullopt;
  AffineExpr simplified =
      simplifyAffineExpr(*e, builder.leaves.size(), /*numSymbols=*/0);
  if (auto c = dyn_cast<AffineConstantExpr>(simplified))
    return c.getValue();
  return std::nullopt;
}
} // namespace

DiagnosedSilenceableFailure
transform::CoalesceSlicesOp::apply(transform::TransformRewriter &rewriter,
                                   transform::TransformResults &results,
                                   transform::TransformState &state) {
  SmallVector<Operation *> copies;
  // The slices feeding the targets' inputs, grouped by source.
  llvm::MapVector<Value, SmallVector<tensor::ExtractSliceOp>> groups;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto dps = dyn_cast<DestinationStyleOpInterface>(target);
    if (!dps)
      return emitSilenceableError() << "expects destination-style targets";
    for (OpOperand *operand : dps.getDpsInputOperands()) {
      // A slice may reach the input through tensor.expand_shape and
      // tensor.collapse_shape ops, which are kept.
      Value in = operand->get();
      while (true) {
        if (auto expand = in.getDefiningOp<tensor::ExpandShapeOp>())
          in = expand.getSrc();
        else if (auto collapse = in.getDefiningOp<tensor::CollapseShapeOp>())
          in = collapse.getSrc();
        else
          break;
      }
      if (auto slice = in.getDefiningOp<tensor::ExtractSliceOp>())
        if (!llvm::is_contained(groups[slice.getSource()], slice))
          groups[slice.getSource()].push_back(slice);
    }
  }

  ArrayRef<int64_t> fullDims = getFullDims();
  for (auto &[source, slices] : groups) {
    if (slices.size() < 2 && fullDims.empty())
      continue;
    auto srcType = slices.front().getSourceType();
    int64_t rank = srcType.getRank();
    Block *block = slices.front()->getBlock();
    bool ok = llvm::all_of(slices, [&](tensor::ExtractSliceOp s) {
      return s.hasUnitStride() && s->getBlock() == block;
    });
    // Per dimension, the slices need either the same offset (the region
    // starts there and is as large as the largest slice) or static offsets
    // and sizes. In a dim listed in `full_dims` the region is the whole
    // source.
    SmallVector<OpFoldResult> lo(rank), size(rank);
    SmallVector<bool> common(rank, false);
    for (int64_t d = 0; ok && d < rank; ++d) {
      if (llvm::is_contained(fullDims, d)) {
        ok = !srcType.isDynamicDim(d) &&
             llvm::all_of(slices, [&](tensor::ExtractSliceOp s) {
               return provenConstant(s.getMixedOffsets()[d],
                                     rewriter.getContext())
                   .has_value();
             });
        lo[d] = rewriter.getIndexAttr(0);
        size[d] = rewriter.getIndexAttr(srcType.getDimSize(d));
        continue;
      }
      OpFoldResult off0 = slices.front().getMixedOffsets()[d];
      bool same = llvm::all_of(slices, [&](tensor::ExtractSliceOp s) {
        return isEqualConstantIntOrValue(s.getMixedOffsets()[d], off0);
      });
      if (same) {
        common[d] = true;
        lo[d] = off0;
        int64_t maxSize = 0;
        for (tensor::ExtractSliceOp s : slices) {
          auto sz = getConstantIntValue(s.getMixedSizes()[d]);
          if (!sz) {
            ok = false;
            break;
          }
          maxSize = std::max(maxSize, *sz);
        }
        size[d] = rewriter.getIndexAttr(maxSize);
        continue;
      }
      int64_t l = std::numeric_limits<int64_t>::max(), h = 0;
      for (tensor::ExtractSliceOp s : slices) {
        auto o = provenConstant(s.getMixedOffsets()[d], rewriter.getContext());
        auto sz = getConstantIntValue(s.getMixedSizes()[d]);
        if (!o || !sz) {
          ok = false;
          break;
        }
        l = std::min(l, *o);
        h = std::max(h, *o + *sz);
      }
      lo[d] = rewriter.getIndexAttr(l);
      size[d] = rewriter.getIndexAttr(h - l);
    }
    // Unless `full_dims` asks for whole dims, the region may not be larger
    // than the slices it replaces.
    if (ok && fullDims.empty()) {
      int64_t regionElems = 1, sliceElems = 0;
      for (OpFoldResult sz : size)
        regionElems *= *getConstantIntValue(sz);
      for (tensor::ExtractSliceOp s : slices)
        sliceElems += s.getType().getNumElements();
      ok = regionElems <= sliceElems;
    }
    if (!ok)
      continue;

    tensor::ExtractSliceOp earliest = slices.front();
    for (tensor::ExtractSliceOp s : slices)
      if (s->isBeforeInBlock(earliest))
        earliest = s;
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(earliest);
    Location loc = earliest.getLoc();
    SmallVector<OpFoldResult> ones(rank, rewriter.getIndexAttr(1));
    auto region =
        tensor::ExtractSliceOp::create(rewriter, loc, source, lo, size, ones);
    auto empty =
        tensor::EmptyOp::create(rewriter, loc, region.getType().getShape(),
                                region.getType().getElementType());
    auto copy = linalg::CopyOp::create(rewriter, loc, region.getResult(),
                                       empty.getResult());
    copies.push_back(copy);
    for (tensor::ExtractSliceOp s : slices) {
      SmallVector<OpFoldResult> offsets;
      for (int64_t d = 0; d < rank; ++d)
        offsets.push_back(
            common[d]
                ? rewriter.getIndexAttr(0)
                : rewriter.getIndexAttr(*provenConstant(s.getMixedOffsets()[d],
                                                        rewriter.getContext()) -
                                        *getConstantIntValue(lo[d])));
      rewriter.setInsertionPoint(s);
      rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(
          s, s.getType(), copy.getResult(0), offsets, s.getMixedSizes(), ones);
    }
  }
  results.set(llvm::cast<OpResult>(getResult()), copies);
  return DiagnosedSilenceableFailure::success();
}

void transform::CoalesceSlicesOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// FoldIterArgsIntoInductionVarOp
//===----------------------------------------------------------------------===//

// `v` converted to `type` (both integer or index).
static Value castIntOrIndex(OpBuilder &b, Location loc, Value v, Type type) {
  if (v.getType() == type)
    return v;
  if (isa<IndexType>(type) || isa<IndexType>(v.getType()))
    return arith::IndexCastOp::create(b, loc, type, v);
  if (type.getIntOrFloatBitWidth() > v.getType().getIntOrFloatBitWidth())
    return arith::ExtSIOp::create(b, loc, type, v);
  return arith::TruncIOp::create(b, loc, type, v);
}

static bool foldIterArgsIntoInductionVar(RewriterBase &rewriter,
                                         scf::ForOp forOp) {
  OpBuilder::InsertionGuard guard(rewriter);
  Location loc = forOp.getLoc();
  Operation *yield = forOp.getBody()->getTerminator();
  bool changed = false;
  for (auto [i, arg] : llvm::enumerate(forOp.getRegionIterArgs())) {
    Type type = arg.getType();
    if (!type.isIntOrIndex())
      continue;
    auto add = yield->getOperand(i).getDefiningOp<arith::AddIOp>();
    if (!add)
      continue;
    Value delta = add.getLhs() == arg   ? add.getRhs()
                  : add.getRhs() == arg ? add.getLhs()
                                        : Value();
    if (!delta || !forOp.isDefinedOutsideOfLoop(delta))
      continue;
    Value init = forOp.getInitArgs()[i];
    Value lb = forOp.getLowerBound(), ub = forOp.getUpperBound(),
          step = forOp.getStep();

    // In the body: init + ((iv - lb) / step) * delta.
    rewriter.setInsertionPointToStart(forOp.getBody());
    Value iters = arith::DivUIOp::create(
        rewriter, loc,
        arith::SubIOp::create(rewriter, loc, forOp.getInductionVar(), lb),
        step);
    Value inBody = arith::AddIOp::create(
        rewriter, loc, init,
        arith::MulIOp::create(
            rewriter, loc, castIntOrIndex(rewriter, loc, iters, type), delta));
    // The add's use too, so the advanced value is right where the body reads
    // it.
    rewriter.replaceAllUsesWith(arg, inBody);

    // After the loop: init + ceil((ub - lb) / step) * delta, or init when
    // the loop runs no iteration.
    rewriter.setInsertionPointAfter(forOp);
    Value span = arith::SubIOp::create(rewriter, loc, ub, lb);
    Value zero = arith::ConstantOp::create(
        rewriter, loc, rewriter.getZeroAttr(span.getType()));
    Value trips = arith::MaxSIOp::create(
        rewriter, loc, zero,
        arith::CeilDivSIOp::create(rewriter, loc, span, step));
    Value after = arith::AddIOp::create(
        rewriter, loc, init,
        arith::MulIOp::create(
            rewriter, loc, castIntOrIndex(rewriter, loc, trips, type), delta));
    rewriter.replaceAllUsesWith(forOp.getResult(i), after);

    // Pass the argument through; canonicalization drops it.
    rewriter.modifyOpInPlace(
        yield, [&, i = i, arg = arg] { yield->setOperand(i, arg); });
    changed = true;
  }
  return changed;
}

DiagnosedSilenceableFailure transform::FoldIterArgsIntoInductionVarOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  SmallVector<Operation *> loops;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto forOp = dyn_cast<scf::ForOp>(target);
    if (!forOp)
      return emitDefiniteFailure() << "target must be an scf.for";
    foldIterArgsIntoInductionVar(rewriter, forOp);
    loops.push_back(forOp);
  }
  results.set(llvm::cast<OpResult>(getResult()), loops);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// HoistLoopCarriedPackOp
//===----------------------------------------------------------------------===//

namespace {
// An iteration argument packed once and unpacked back into itself per
// iteration: by linalg.pack and linalg.unpack, or by tensor.expand_shape and
// tensor.collapse_shape.
struct LoopCarriedPack {
  unsigned index;
  Operation *pack;
  Operation *unpack;
};
} // namespace

// Whether `v` is a fresh tensor (tensor.empty or bufferization.alloc_tensor)
// built from values available before `forOp`, so it can be built again
// outside the loop.
static bool isRecreatableEmptyTensor(Value v, scf::ForOp forOp) {
  Operation *def = v.getDefiningOp();
  if (!isa_and_present<tensor::EmptyOp>(def)) {
    auto alloc = dyn_cast_if_present<bufferization::AllocTensorOp>(def);
    if (!alloc || alloc.getCopy())
      return false;
  }
  return llvm::all_of(def->getOperands(), [&](Value operand) {
    return forOp.isDefinedOutsideOfLoop(operand);
  });
}

// The value a pack-like op reads, and the tensor it writes into (none for a
// reshape).
static OpOperand *packSourceOperand(Operation *op) {
  if (auto p = dyn_cast<linalg::PackOp>(op))
    return &p.getSourceMutable();
  if (auto e = dyn_cast<tensor::ExpandShapeOp>(op))
    return &e.getSrcMutable();
  return nullptr;
}
static Value packDest(Operation *op) {
  if (auto p = dyn_cast<linalg::PackOp>(op))
    return p.getDest();
  return {};
}
static Value unpackSource(Operation *op) {
  if (auto u = dyn_cast<linalg::UnPackOp>(op))
    return u.getSource();
  return cast<tensor::CollapseShapeOp>(op).getSrc();
}
static Value unpackDest(Operation *op) {
  if (auto u = dyn_cast<linalg::UnPackOp>(op))
    return u.getDest();
  return {};
}

// Whether `unpack` undoes `pack`.
static bool areInverseLayouts(Operation *pack, Operation *unpack) {
  if (pack->getResult(0).getType() != unpackSource(unpack).getType())
    return false;
  if (auto p = dyn_cast<linalg::PackOp>(pack)) {
    auto u = dyn_cast<linalg::UnPackOp>(unpack);
    return u && !p.getPaddingValue() &&
           p.getInnerDimsPos() == u.getInnerDimsPos() &&
           p.getOuterDimsPerm() == u.getOuterDimsPerm() &&
           p.getStaticInnerTiles() == u.getStaticInnerTiles() &&
           p.getInnerTiles().empty() && u.getInnerTiles().empty();
  }
  auto e = cast<tensor::ExpandShapeOp>(pack);
  auto c = dyn_cast<tensor::CollapseShapeOp>(unpack);
  return c && e.getReassociationIndices() == c.getReassociationIndices();
}

static std::optional<LoopCarriedPack> matchLoopCarriedPack(scf::ForOp forOp,
                                                           unsigned index) {
  BlockArgument arg = forOp.getRegionIterArgs()[index];
  // Static shapes only: with them, equal types are equal shapes, so the value
  // packed before the loop has the shape the loop packs each trip, and a
  // destination made outside the loop has the carried value's shape.
  auto isStatic = [](Value v) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    return t && t.hasStaticShape();
  };
  if (!isStatic(arg))
    return std::nullopt;
  // The unpacked value goes straight back round the loop. An unpack writes
  // all of its destination, so the destination may be the carried value, any
  // tensor available before the loop, or a fresh tensor that can be built
  // again after it.
  Operation *yield = forOp.getBody()->getTerminator();
  Operation *unpack = yield->getOperand(index).getDefiningOp();
  if (!isa_and_present<linalg::UnPackOp, tensor::CollapseShapeOp>(unpack) ||
      !unpack->getResult(0).hasOneUse() || unpack->getParentOp() != forOp)
    return std::nullopt;
  Value uDest = unpackDest(unpack);
  if (uDest && uDest != arg && !forOp.isDefinedOutsideOfLoop(uDest) &&
      !isRecreatableEmptyTensor(uDest, forOp))
    return std::nullopt;
  Operation *pack = nullptr;
  for (OpOperand &use : arg.getUses()) {
    Operation *user = use.getOwner();
    if (user == unpack)
      continue;
    OpOperand *src = packSourceOperand(user);
    if (!src || user->getParentOp() != forOp || &use != src || pack)
      return std::nullopt;
    pack = user;
  }
  if (!pack || !areInverseLayouts(pack, unpack) ||
      !isStatic(pack->getResult(0)) ||
      (unpackDest(unpack) && !isStatic(unpackDest(unpack))))
    return std::nullopt;
  Value dest = packDest(pack);
  if (dest && !forOp.isDefinedOutsideOfLoop(dest) &&
      !isRecreatableEmptyTensor(dest, forOp))
    return std::nullopt;
  return LoopCarriedPack{index, pack, unpack};
}

static FailureOr<std::pair<scf::ForOp, Operation *>>
hoistLoopCarriedPack(RewriterBase &rewriter, scf::ForOp forOp,
                     LoopCarriedPack m) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forOp);
  IRMapping before;
  before.map(forOp.getRegionIterArgs()[m.index], forOp.getInitArgs()[m.index]);
  Value pDest = packDest(m.pack);
  if (pDest && !forOp.isDefinedOutsideOfLoop(pDest))
    before.map(pDest, rewriter.clone(*pDest.getDefiningOp())->getResult(0));
  Operation *packed = rewriter.clone(*m.pack, before);

  Value computed = unpackSource(m.unpack);
  FailureOr<LoopLikeOpInterface> replaced = forOp.replaceWithAdditionalYields(
      rewriter, packed->getResult(0), /*replaceInitOperandUsesInLoop=*/false,
      [&](OpBuilder &, Location, ArrayRef<BlockArgument>) {
        return SmallVector<Value>{computed};
      });
  if (failed(replaced))
    return failure();
  auto newFor = cast<scf::ForOp>(replaced->getOperation());
  BlockArgument carried = newFor.getRegionIterArgs().back();
  BlockArgument original = newFor.getRegionIterArgs()[m.index];

  // After the loop: unpack the carried value into what the loop returns at
  // the original position, which is now its initial value.
  rewriter.setInsertionPointAfter(newFor);
  IRMapping after;
  after.map(computed, newFor.getResults().back());
  after.map(original, newFor.getResult(m.index));
  Value uDest = unpackDest(m.unpack);
  if (uDest && uDest != original && !newFor.isDefinedOutsideOfLoop(uDest))
    after.map(uDest, rewriter.clone(*uDest.getDefiningOp())->getResult(0));
  Operation *unpacked = rewriter.clone(*m.unpack, after);
  rewriter.replaceAllUsesExcept(newFor.getResult(m.index),
                                unpacked->getResult(0), unpacked);

  // Inside: read the carried value, and pass the original through.
  rewriter.replaceAllUsesWith(m.pack->getResult(0), carried);
  Operation *yield = newFor.getBody()->getTerminator();
  rewriter.modifyOpInPlace(yield,
                           [&] { yield->setOperand(m.index, original); });
  rewriter.eraseOp(m.unpack);
  rewriter.eraseOp(m.pack);
  return std::make_pair(newFor, packed);
}

DiagnosedSilenceableFailure
transform::HoistLoopCarriedPackOp::apply(transform::TransformRewriter &rewriter,
                                         transform::TransformResults &results,
                                         transform::TransformState &state) {
  SmallVector<Operation *> loops, packs;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto forOp = dyn_cast<scf::ForOp>(target);
    if (!forOp)
      return emitDefiniteFailure() << "target must be an scf.for";
    bool hoisted = false;
    for (unsigned i = 0; i < forOp.getNumRegionIterArgs(); ++i) {
      std::optional<LoopCarriedPack> m = matchLoopCarriedPack(forOp, i);
      if (!m)
        continue;
      auto result = hoistLoopCarriedPack(rewriter, forOp, *m);
      if (failed(result))
        return emitDefiniteFailure() << "failed to rebuild the loop";
      forOp = result->first;
      packs.push_back(result->second);
      hoisted = true;
    }
    if (!hoisted)
      return emitSilenceableError()
             << "no iteration argument is packed and unpacked in the loop";
    loops.push_back(forOp);
  }
  results.set(llvm::cast<OpResult>(getLoops()), loops);
  results.set(llvm::cast<OpResult>(getPacks()), packs);
  return DiagnosedSilenceableFailure::success();
}

void transform::HoistLoopCarriedPackOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTargetMutable(), effects);
  producesHandle(getOperation()->getOpResults(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// FoldPackIntoGenericOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::FoldPackIntoGenericOp::apply(transform::TransformRewriter &rewriter,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {
  if (!llvm::all_of(state.getPayloadOps(getTarget()),
                    llvm::IsaPred<linalg::GenericOp>))
    return emitSilenceableError() << "expects linalg.generic targets";
  SmallVector<Operation *> updated;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto generic = cast<linalg::GenericOp>(target);
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    // A pack feeding several inputs is folded into each, and erased once.
    llvm::SetVector<Operation *> folded;
    for (OpOperand *operand : generic.getDpsInputOperands()) {
      auto pack = operand->get().getDefiningOp<linalg::PackOp>();
      if (!pack)
        continue;
      ArrayRef<int64_t> srcShape = pack.getSourceType().getShape();
      ArrayRef<int64_t> innerPos = pack.getInnerDimsPos();
      SmallVector<int64_t> tiles(pack.getStaticInnerTiles());
      // Without padding: every tiled dimension divides into whole tiles.
      bool exact = true;
      for (auto [d, t] : llvm::zip_equal(innerPos, tiles))
        exact &= !ShapedType::isDynamic(t) &&
                 !ShapedType::isDynamic(srcShape[d]) && srcShape[d] % t == 0;
      if (!exact)
        continue;
      int64_t srcRank = srcShape.size();
      AffineMap packed = maps[operand->getOperandNumber()];
      ArrayRef<int64_t> outerPerm = pack.getOuterDimsPerm();
      SmallVector<AffineExpr> exprs(srcRank);
      for (int64_t j = 0; j < srcRank; ++j)
        exprs[outerPerm.empty() ? j : outerPerm[j]] = packed.getResult(j);
      for (auto [i, d] : llvm::enumerate(innerPos))
        exprs[d] = exprs[d] * tiles[i] + packed.getResult(srcRank + i);
      maps[operand->getOperandNumber()] =
          AffineMap::get(packed.getNumDims(), packed.getNumSymbols(), exprs,
                         rewriter.getContext());
      rewriter.modifyOpInPlace(generic,
                               [&] { operand->set(pack.getSource()); });
      folded.insert(pack);
    }
    rewriter.modifyOpInPlace(generic, [&] {
      generic.setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
    });
    for (Operation *pack : folded)
      if (pack->use_empty())
        rewriter.eraseOp(pack);
    updated.push_back(generic);
  }
  results.set(llvm::cast<OpResult>(getResult()), updated);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// PushUnpackThroughSlicesOp
//===----------------------------------------------------------------------===//

namespace {
// An unpack matched by push_unpack_through_slices: its expand_shape splits
// dim `d` (result dim `b`, extent `s`); `ti` indexes `d`'s inner tile.
struct UnpackSplit {
  linalg::UnPackOp unpack;
  tensor::ExpandShapeOp expand;
  int64_t d, b, s, ti;
  SmallVector<tensor::ExtractSliceOp> slices;
};

static FailureOr<UnpackSplit> matchUnpackSplit(Operation *op,
                                               std::string &why) {
  auto unpack = dyn_cast<linalg::UnPackOp>(op);
  if (!unpack || !unpack->hasOneUse()) {
    why = "expected a linalg.unpack with one user";
    return failure();
  }
  auto expand = dyn_cast<tensor::ExpandShapeOp>(*unpack->user_begin());
  if (!expand) {
    why = "the unpack's user is not an expand_shape";
    return failure();
  }
  // The one source dim the expand splits, into [n, s].
  int64_t d = -1, b = -1;
  for (auto [i, group] : llvm::enumerate(expand.getReassociationIndices())) {
    if (group.size() == 1)
      continue;
    if (group.size() != 2 || d >= 0) {
      why = "expand_shape must split one dim in two";
      return failure();
    }
    d = i;
    b = group[1];
  }
  ArrayRef<int64_t> outShape = expand.getResultType().getShape();
  ArrayRef<int64_t> innerPos = unpack.getInnerDimsPos();
  SmallVector<int64_t> tiles(unpack.getStaticInnerTiles());
  auto it = llvm::find(innerPos, d);
  if (d < 0 || it == innerPos.end() || ShapedType::isDynamic(outShape[b])) {
    why = "the split dim is not a packed dim";
    return failure();
  }
  int64_t s = outShape[b];
  int64_t ti = it - innerPos.begin();
  if (ShapedType::isDynamic(tiles[ti]) || s <= 0 || tiles[ti] % s) {
    why = "the inner tile is not a multiple of the split extent";
    return failure();
  }
  UnpackSplit m{unpack, expand, d, b, s, ti, {}};
  for (Operation *u : expand->getUsers()) {
    auto x = dyn_cast<tensor::ExtractSliceOp>(u);
    if (!x || !x.hasUnitStride() ||
        x.getType().getRank() != expand.getResultType().getRank() - 1) {
      why = "expected rank-reducing slices";
      return failure();
    }
    for (int64_t i = 0; i < (int64_t)outShape.size(); ++i) {
      std::optional<int64_t> off = getConstantIntValue(x.getMixedOffsets()[i]);
      std::optional<int64_t> sz = getConstantIntValue(x.getMixedSizes()[i]);
      bool ok = i == b ? (off && sz == 1) : (off == 0 && sz == outShape[i]);
      if (!ok) {
        why = "a slice must take one index of the split dim, all else";
        return failure();
      }
    }
    m.slices.push_back(x);
  }
  return m;
}
} // namespace

DiagnosedSilenceableFailure transform::PushUnpackThroughSlicesOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {
  // Match every target before rewriting any.
  SmallVector<UnpackSplit> matches;
  for (Operation *op : state.getPayloadOps(getTarget())) {
    std::string why;
    FailureOr<UnpackSplit> m = matchUnpackSplit(op, why);
    if (failed(m))
      return emitSilenceableError() << why;
    matches.push_back(*m);
  }

  SmallVector<Operation *> created;
  for (UnpackSplit &m : matches) {
    linalg::UnPackOp unpack = m.unpack;
    tensor::ExpandShapeOp expand = m.expand;
    int64_t b = m.b, s = m.s, ti = m.ti;
    ArrayRef<int64_t> innerPos = unpack.getInnerDimsPos();
    SmallVector<int64_t> tiles(unpack.getStaticInnerTiles());

    // The packed source with d's tile dim split into [t / s, s].
    Value src = unpack.getSource();
    auto srcTy = cast<RankedTensorType>(src.getType());
    int64_t pos = unpack.getDestType().getRank() + ti;
    SmallVector<int64_t> expShape;
    SmallVector<ReassociationIndices> reassoc;
    for (int64_t i = 0, j = 0; i < srcTy.getRank(); ++i) {
      if (i == pos) {
        expShape.append({tiles[ti] / s, s});
        reassoc.push_back({j, j + 1});
        j += 2;
      } else {
        expShape.push_back(srcTy.getDimSize(i));
        reassoc.push_back({j++});
      }
    }
    rewriter.setInsertionPoint(expand);
    Location loc = unpack.getLoc();
    Value srcExp = tensor::ExpandShapeOp::create(
        rewriter, loc, RankedTensorType::get(expShape, srcTy.getElementType()),
        src, reassoc);
    SmallVector<int64_t> newTiles = tiles;
    newTiles[ti] = tiles[ti] / s;
    SmallVector<OpFoldResult> newTileOfr;
    for (int64_t t : newTiles)
      newTileOfr.push_back(rewriter.getIndexAttr(t));
    for (tensor::ExtractSliceOp x : m.slices) {
      int64_t k = *getConstantIntValue(x.getMixedOffsets()[b]);
      SmallVector<OpFoldResult> offs, sizes, strides;
      SmallVector<int64_t> sliceShape;
      for (int64_t i = 0; i < (int64_t)expShape.size(); ++i) {
        bool pair = i == pos + 1;
        offs.push_back(rewriter.getIndexAttr(pair ? k : 0));
        sizes.push_back(rewriter.getIndexAttr(pair ? 1 : expShape[i]));
        strides.push_back(rewriter.getIndexAttr(1));
        if (!pair)
          sliceShape.push_back(expShape[i]);
      }
      rewriter.setInsertionPoint(x);
      Value part = tensor::ExtractSliceOp::create(
          rewriter, loc,
          RankedTensorType::get(sliceShape, srcTy.getElementType()), srcExp,
          offs, sizes, strides);
      Value dest = tensor::EmptyOp::create(
          rewriter, loc, x.getType().getShape(), x.getType().getElementType());
      auto newUnpack =
          linalg::UnPackOp::create(rewriter, loc, part, dest, innerPos,
                                   newTileOfr, unpack.getOuterDimsPerm());
      rewriter.replaceOp(x, newUnpack.getResult());
      created.push_back(newUnpack);
    }
    rewriter.eraseOp(expand);
    rewriter.eraseOp(unpack);
  }

  // An elementwise generic all of whose inputs are such unpacks, of one layout,
  // runs on the packed values instead, with one unpack after it.
  SmallVector<Operation *> result;
  llvm::SmallPtrSet<Operation *, 8> done;
  for (Operation *u : created) {
    auto up = cast<linalg::UnPackOp>(u);
    if (!up->hasOneUse()) {
      result.push_back(u);
      continue;
    }
    auto gen = dyn_cast<linalg::GenericOp>(*up->user_begin());
    if (!gen || done.contains(gen))
      continue;
    auto sameLayout = [&](linalg::UnPackOp o) {
      return o && o.getInnerDimsPos() == up.getInnerDimsPos() &&
             o.getStaticInnerTiles() == up.getStaticInnerTiles() &&
             o.getOuterDimsPerm() == up.getOuterDimsPerm() &&
             o.getSourceType().getShape() == up.getSourceType().getShape();
    };
    bool ok =
        gen.getNumDpsInits() == 1 && !gen.hasIndexSemantics() &&
        llvm::all_of(gen.getIteratorTypesArray(), linalg::isParallelIterator) &&
        llvm::all_of(gen.getIndexingMapsArray(),
                     [](AffineMap m) { return m.isIdentity(); }) &&
        llvm::all_of(gen.getDpsInputs(),
                     [&](Value v) {
                       return sameLayout(v.getDefiningOp<linalg::UnPackOp>());
                     }) &&
        gen.getRegionOutputArgs()[0].use_empty();
    if (!ok) {
      result.push_back(u);
      continue;
    }
    done.insert(gen);
    rewriter.setInsertionPoint(gen);
    Location loc = gen.getLoc();
    SmallVector<Value> ins;
    for (Value v : gen.getDpsInputs())
      ins.push_back(v.getDefiningOp<linalg::UnPackOp>().getSource());
    auto outTy = cast<RankedTensorType>(gen.getResult(0).getType());
    ArrayRef<int64_t> pshape = up.getSourceType().getShape();
    Value pout =
        tensor::EmptyOp::create(rewriter, loc, pshape, outTy.getElementType());
    int64_t rank = pshape.size();
    SmallVector<AffineMap> maps(ins.size() + 1,
                                rewriter.getMultiDimIdentityMap(rank));
    SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
    auto pgen =
        linalg::GenericOp::create(rewriter, loc, TypeRange{pout.getType()}, ins,
                                  ValueRange{pout}, maps, iters);
    pgen->setDiscardableAttrs(gen->getDiscardableAttrDictionary());
    rewriter.cloneRegionBefore(gen.getRegion(), pgen.getRegion(),
                               pgen.getRegion().begin());
    Value dest = tensor::EmptyOp::create(rewriter, loc, outTy.getShape(),
                                         outTy.getElementType());
    SmallVector<OpFoldResult> tilesOfr;
    for (int64_t t : up.getStaticInnerTiles())
      tilesOfr.push_back(rewriter.getIndexAttr(t));
    auto out = linalg::UnPackOp::create(rewriter, loc, pgen.getResult(0), dest,
                                        up.getInnerDimsPos(), tilesOfr,
                                        up.getOuterDimsPerm());
    // The result is an unpack, not a generic: erase `gen` rather than have
    // its handles track a different kind of op.
    rewriter.replaceAllUsesWith(gen.getResult(0), out.getResult());
    rewriter.eraseOp(gen);
    result.push_back(out);
  }
  results.set(llvm::cast<OpResult>(getResult()), result);
  return DiagnosedSilenceableFailure::success();
}

DiagnosedSilenceableFailure transform::ConvertSize1VectorToScalarOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {

  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  if (targets.empty()) {
    results.set(llvm::cast<OpResult>(getResult()), ArrayRef<Operation *>());
    return DiagnosedSilenceableFailure::success();
  }

  SmallVector<Operation *> transformedOps;

  for (Operation *target : targets) {
    // Verify target has IsolatedFromAbove trait
    if (!target->hasTrait<OpTrait::IsIsolatedFromAbove>()) {
      return emitDefiniteFailure()
             << "target operation must have IsolatedFromAbove trait";
    }

    MLIRContext *ctx = target->getContext();

    // Apply rewrite patterns using applyPatternsGreedily (not conversion)
    RewritePatternSet patterns(ctx);

    // Add patterns for memory operations and elementwise ops
    patterns.add<ConvertSize1TransferReadToLoad,
                 ConvertSize1TransferWriteToStore, ConvertSize1VectorLoadToLoad,
                 ConvertSize1VectorStoreToStore, ConvertSize1VectorOpsToScalar>(
        ctx);

    if (failed(applyPatternsGreedily(target, std::move(patterns)))) {
      return emitDefiniteFailure()
             << "failed to apply size-1 vector-to-scalar rewrite patterns";
    }

    // Run canonicalization to fold extract(broadcast(x)) -> x
    RewritePatternSet canonPatterns(ctx);
    vector::ExtractOp::getCanonicalizationPatterns(canonPatterns, ctx);
    vector::BroadcastOp::getCanonicalizationPatterns(canonPatterns, ctx);
    if (failed(applyPatternsGreedily(target, std::move(canonPatterns)))) {
      return emitDefiniteFailure()
             << "failed to apply canonicalization patterns "
                "after size-1 vector-to-scalar conversion";
    }

    transformedOps.push_back(target);
  }

  results.set(llvm::cast<OpResult>(getResult()), transformedOps);
  return DiagnosedSilenceableFailure::success();
}

//===----------------------------------------------------------------------===//
// NormalizeForBoundsOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::NormalizeForBoundsOp::apply(transform::TransformRewriter &rewriter,
                                       transform::TransformResults &results,
                                       transform::TransformState &state) {
#if AIR_ENABLE_AIE
  SmallVector<Operation *> targets =
      llvm::to_vector(state.getPayloadOps(getTarget()));

  if (targets.empty()) {
    results.set(llvm::cast<OpResult>(getResult()), ArrayRef<Operation *>());
    return DiagnosedSilenceableFailure::success();
  }

  SmallVector<Operation *> transformedOps;

  for (Operation *target : targets) {
    auto forOp = dyn_cast_if_present<scf::ForOp>(target);
    if (!forOp) {
      return emitDefiniteFailure() << "target must be an scf.for operation";
    }

    // Use the utility function from AIRDependencyScheduleOpt to fold
    // affine.apply into loop bounds
    auto newForOp = xilinx::air::foldAffineApplyIntoLoopBounds(forOp, rewriter);
    if (succeeded(newForOp)) {
      // Use the returned ForOp (which may be a new operation)
      transformedOps.push_back(*newForOp);
    } else {
      // No transformation was applied, return the original op
      transformedOps.push_back(forOp);
    }
  }

  results.set(llvm::cast<OpResult>(getResult()), transformedOps);
  return DiagnosedSilenceableFailure::success();
#else
  return emitSilenceableError() << "NormalizeForBoundsOp requires AIE support "
                                   "which is not enabled in this build";
#endif
}

//===----------------------------------------------------------------------===//

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRLinalgCodegenPass() {
  return std::make_unique<AIRLinalgCodegen>();
}

std::unique_ptr<Pass> createAIRPipelineReducePass() {
  return std::make_unique<AIRPipelineReducePass>();
}

} // namespace air
} // namespace xilinx
