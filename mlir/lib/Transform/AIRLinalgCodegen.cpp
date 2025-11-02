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
#include "air/Util/CostModel.h"
#include "air/Util/Outliner.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
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
          Value b = rewriter.create<arith::ConstantIndexOp>(op.getLoc(),
                                                            source_offset);
          result_offsets.push_back(
              rewriter.create<arith::AddIOp>(op.getLoc(), a.getType(), a, b));
        }
      } else if (op_offset >= 0 && source_offset < 0) {
        result_static_offsets.push_back(source_offset);
        if (op_offset == 0) {
          result_offsets.push_back(*source_offsets++);
        } else {
          Value a = *source_offsets++;
          Value b =
              rewriter.create<arith::ConstantIndexOp>(op.getLoc(), op_offset);
          result_offsets.push_back(
              rewriter.create<arith::AddIOp>(op.getLoc(), a.getType(), a, b));
        }
      } else if (op_offset < 0 && source_offset < 0) {
        Value a = *source_offsets++;
        Value b = *offsets++;
        result_offsets.push_back(
            rewriter.create<arith::AddIOp>(op.getLoc(), a.getType(), a, b));
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
      if (auto launch = dyn_cast<air::HerdOp>(use)) {
        for (unsigned int i = 0; i < launch.getNumKernelOperands(); i++) {
          auto arg = launch.getKernelArguments()[i];
          auto oper = launch.getKernelOperand(i);
          if (oper == newOp) {
            Block *b = arg.getOwner();
            auto new_arg = b->insertArgument(arg.getArgNumber(),
                                             newOp.getType(), newOp.getLoc());
            rewriter.setInsertionPointToStart(&*launch.getRegion().begin());
            arg.replaceAllUsesWith(rewriter.create<memref::CastOp>(
                op.getLoc(), arg.getType(), new_arg));
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
//     auto operTy = llvm::dyn_cast<ShapedType>(op.memrefOrTensor().getType());
//     if (!operTy.hasStaticShape())
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

    /* Force memory space */
    Value newOp = rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op,
        MemRefType::get(op.getType().getShape(), op.getType().getElementType(),
                        AffineMap(), rewriter.getI32IntegerAttr(fast_space)),
        op.getSizes());
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

    /* Force memory space */
    Value newOp = rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op,
        MemRefType::get(op.getType().getShape(), op.getType().getElementType(),
                        AffineMap(), rewriter.getI32IntegerAttr(fast_space)),
        op.getSizes());
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
    auto linalgOp = dyn_cast<linalg::LinalgOp>(++iter);
    if (!linalgOp)
      return failure();

    Value copyOper0 = copyOp->getOperand(0);
    Value copyOper1 = copyOp->getOperand(1);

    auto allocOp =
        dyn_cast_if_present<memref::AllocOp>(copyOper0.getDefiningOp());
    if (!allocOp)
      return failure();

    iter = allocOp->getIterator();
    Operation *fillOp = dyn_cast<linalg::FillOp>(++iter);
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
    auto linalgOp = dyn_cast<linalg::LinalgOp>(++iter);
    if (!linalgOp)
      return failure();
    auto copyOp = dyn_cast<memref::CopyOp>(++iter);
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
    auto deallocOp = dyn_cast<memref::DeallocOp>(++iter);
    if (!deallocOp)
      return failure();

    auto allocOp = dyn_cast<memref::AllocOp>(++iter);
    if (!allocOp)
      return failure();
    if (allocOp.getType() != existingAlloc.getType())
      return failure();

    auto copyOp = dyn_cast<memref::CopyOp>(++iter);
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
      if (auto c = dyn_cast<memref::CastOp>(u.getOwner()))
        castOp = c;
      else if (auto c = dyn_cast<memref::CopyOp>(u.getOwner())) {
        if (u.getOperandNumber() == 0)
          copyOp = c;
        else {
          if (auto l = dyn_cast<linalg::LinalgOp>(u.getOwner())) {
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
      copyOp = dyn_cast<memref::CopyOp>(*castOp->user_begin());
      if (!copyOp)
        return failure();
    }
    auto copyOperand = copyOp->getOperand(1);
    rewriter.setInsertionPointAfter(copyOperand.getDefiningOp());
    auto newOp = rewriter.create<memref::CastOp>(op->getLoc(), op.getType(),
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
      if (auto l = dyn_cast<linalg::LinalgOp>(u.getOwner())) {
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
      auto memcpyOp = dyn_cast<air::DmaMemcpyNdOp>(user);
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
    SmallVector<Value> firstDstOffsets(firstMemcpy.getDstOffsets());
    SmallVector<Value> firstDstSizes(firstMemcpy.getDstSizes());
    SmallVector<Value> firstDstStrides(firstMemcpy.getDstStrides());

    SmallVector<Value> secondSrcOffsets(secondMemcpy.getSrcOffsets());
    SmallVector<Value> secondSrcSizes(secondMemcpy.getSrcSizes());
    SmallVector<Value> secondSrcStrides(secondMemcpy.getSrcStrides());

    auto isDefaultAccess = [](SmallVector<Value> offsets,
                              SmallVector<Value> sizes,
                              SmallVector<Value> strides) {
      return offsets.empty() && sizes.empty() && strides.empty();
    };
    if (!isDefaultAccess(firstDstOffsets, firstDstSizes, firstDstStrides) ||
        !isDefaultAccess(secondSrcOffsets, secondSrcSizes, secondSrcStrides))
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

    auto newMemcpy = rewriter.create<air::DmaMemcpyNdOp>(
        firstMemcpy.getLoc(),
        emptyTypes,                         // result types
        firstMemcpy.getAsyncDependencies(), // async dependencies
        secondMemcpy.getDstMemref(),        // destination from second memcpy
        emptyOffsets, emptySizes, emptyStrides, // dst access pattern
        firstMemcpy.getSrcMemref(),             // source from first memcpy
        emptyOffsets, emptySizes, emptyStrides  // src access pattern
    );

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

    auto linalgCpy = dyn_cast<linalg::CopyOp>(otherCpy);

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

    linalg::LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(op);
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
      b.getI32IntegerAttr((int)air::MemorySpace::L1));
  Value buffer = b.createOrFold<memref::AllocOp>(subView.getLoc(), allocType);
  return buffer;
}

static LogicalResult deallocBufferCallBack(OpBuilder &b, Value buffer) {
  // b.create<memref::DeallocOp>(buffer.getLoc(), buffer);
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

  SmallVector<Value, 2> dims{b.create<arith::ConstantIndexOp>(loc, new_herd_x),
                             b.create<arith::ConstantIndexOp>(loc, new_herd_y)};

  SmallVector<Value, 4> args;
  for (auto o : op->getOperands())
    args.push_back(o);

  auto herd = b.create<air::HerdOp>(loc, dims, args);
  b.setInsertionPointToStart(&herd.getBody().front());

  Value x = herd.getIds()[0];
  Value y = herd.getIds()[1];

  auto nLoops = op.getNumLoops();
  auto tileSizes = static_tile_sizes.take_front(nLoops);

  SmallVector<OpFoldResult, 4> tileSizeVector;
  for (auto s : tileSizes)
    tileSizeVector.push_back(
        b.create<arith::ConstantIndexOp>(loc, s).getResult());
  if (tileSizeVector.size() < nLoops) {
    auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
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
    tileIds.push_back(
        b.create<affine::AffineApplyOp>(
             loc, map, isHoriz ? herd.getIds()[0] : herd.getIds()[1])
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
    affine::AffineIfOp aif = b.create<affine::AffineIfOp>(op->getLoc(), int_set,
                                                          int_set_args, false);

    Block *stageBlock = aif.getBody();
    b.setInsertionPointToStart(stageBlock);

    if (i) {
      auto ty = llvm::cast<MemRefType>(tiledOperands[resultIdx].getType());
      auto alloc = b.create<memref::AllocOp>(
          loc, MemRefType::get(ty.getShape(), ty.getElementType(), AffineMap(),
                               b.getI32IntegerAttr((int)air::MemorySpace::L1)));
      tiledOperands[resultIdx] = alloc.getResult();
      SmallVector<Value> src_offsets;
      SmallVector<Value> src_sizes;
      SmallVector<Value> src_strides;
      SmallVector<Value> channel_idx;
      SmallVector<Value> deps;
      SmallVector<Type> tys;
      b.create<air::ChannelGetOp>(loc, tys, deps, channels[i - 1].getSymName(),
                                  channel_idx, tiledOperands[resultIdx],
                                  src_offsets, src_sizes, src_strides);
    }

    linalg::LinalgOp linalgOp = clone(b, op, {}, tiledOperands);

    auto defaultCopyCallBack = [loc](OpBuilder &bldr, Value src,
                                     Value dst) -> LogicalResult {
      bldr.create<memref::CopyOp>(loc, src, dst);
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
        memref::SubViewOp sv = dyn_cast<memref::SubViewOp>(
            linalgOp.getDpsInitOperand(0)->get().getDefiningOp());
        mref = sv.getSource();
        sv.replaceAllUsesWith(mref);
      }

      auto module = op->getParentOfType<ModuleOp>();
      auto cname = createChannelName(module);
      b.setInsertionPointToStart(module.getBody());
      auto channel_op = b.create<air::ChannelOp>(
          loc, cname, b.getI64ArrayAttr({1}), b.getStringAttr("dma_stream"));
      b.setInsertionPoint(stageBlock->getTerminator());
      SmallVector<Value> src_offsets;
      SmallVector<Value> src_sizes;
      SmallVector<Value> src_strides;
      SmallVector<Value> channel_idx;
      SmallVector<Value> deps;
      SmallVector<Type> tys;
      b.create<air::ChannelPutOp>(
          loc, tys, deps, FlatSymbolRefAttr::get(ctx, cname), channel_idx, mref,
          src_offsets, src_sizes, src_strides);
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
    linalg::LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(op);
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

      rootForOp = dyn_cast<T>(&body.front());
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
      if (auto v = llvm::dyn_cast<Value>(size)) {
        auto c = dyn_cast<arith::ConstantIndexOp>(v.getDefiningOp());
        if (!c) {
          LLVM_DEBUG(llvm::outs() << "Found non-constant dim!\n");
          return {};
        }
        tripCounts.push_back(c.value());
      } else {
        auto a = llvm::dyn_cast<Attribute>(size);
        auto c = llvm::dyn_cast<IntegerAttr>(a);
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
    b.create<memref::CopyOp>(b.getUnknownLoc(), src, dst);
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

      RewritePatternSet stage2Patterns =
          linalg::getLinalgTilingCanonicalizationPatterns(ctx);
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
        if (auto scfForOp = dyn_cast<scf::ForOp>(op))
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

      auto parallelLoop =
          builder.create<scf::ParallelOp>(loc, lowerBounds, upperBounds, steps);

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
  auto operationType = pdl::OperationType::get(ctx);
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
  auto pdlOperationType = pdl::OperationType::get(parser.getContext());
  if (parser.parseOperand(target) ||
      parser.resolveOperand(target, pdlOperationType, result.operands) ||
      parseDynamicIndexList(parser, dynamicSizes, staticSizes) ||
      parser.resolveOperands(dynamicSizes, pdlOperationType, result.operands))
    return ParseResult::failure();

  result.addAttribute(getStaticSizesAttrName(result.name), staticSizes);
  result.addTypes(SmallVector<Type>(2, pdlOperationType));
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
    b.create<memref::CopyOp>(b.getUnknownLoc(), src, dst);
    return success();
  };
  promotionOptions.setCopyInOutFns(copyCallBack, copyCallBack);

  auto memorySpace = xilinx::air::MemorySpace::L1;
  if (getMemorySpace() == "L1")
    memorySpace = xilinx::air::MemorySpace::L1;
  else if (getMemorySpace() == "L2")
    memorySpace = xilinx::air::MemorySpace::L2;
  else if (getMemorySpace() == "L3")
    memorySpace = xilinx::air::MemorySpace::L3;

  SetVector<Operation *> transformed;
  int64_t operandOffset = 0;

  uint32_t group_size = getGroupSize();
  uint32_t group = 0;
  for (Operation *target : state.getPayloadOps(getTarget())) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(target);
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

    promotionOptions.setOperandsToPromote(opersToPromote);

    if (failed(promoteSubviewsPrecondition(target, promotionOptions)))
      return emitDefaultDefiniteFailure(target);

    rewriter.setInsertionPoint(target);
    FailureOr<linalg::LinalgOp> res =
        promoteSubViews(rewriter, linalgOp, promotionOptions);
    if (failed(res))
      return emitDefaultDefiniteFailure(target);

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
  result.addTypes(pdl::OperationType::get(builder.getContext()));
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
  auto tileableProducer = dyn_cast<TilingInterface>(producerOp);
  if (!tileableProducer) {
    diag.attachNote(producerOp->getLoc())
        << "producer is not a TileableInterface: " << *producerOp;
    return nullptr;
  }

  linalg::LinalgOp producerLinalgOp = cast<linalg::LinalgOp>(producerOp);
  auto users = producerLinalgOp.getDpsInits()[0].getUsers();
  auto it = llvm::find_if(users, [&](Operation *user) {
    auto sliceOp = dyn_cast<memref::SubViewOp>(user);
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

  linalg::LinalgOp producerLinalgOp = dyn_cast<linalg::LinalgOp>(producerOp);
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
// HoistLoopInvariantTransfersOp
//===----------------------------------------------------------------------===//

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
/// point
static Value cloneOpAndOperands(Operation *op, Value loopIV,
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
    if (defOp && !dependsOnLoopIV(operand, loopIV)) {
      Value clonedOperand =
          cloneOpAndOperands(defOp, loopIV, rewriter, mapping);
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

DiagnosedSilenceableFailure transform::HoistLoopInvariantTransfersOp::apply(
    transform::TransformRewriter &rewriter,
    transform::TransformResults &results, transform::TransformState &state) {

  SmallVector<Operation *> readOps =
      llvm::to_vector(state.getPayloadOps(getReadOp()));
  SmallVector<Operation *> writeOps =
      llvm::to_vector(state.getPayloadOps(getWriteOp()));
  SmallVector<Operation *> loopOps =
      llvm::to_vector(state.getPayloadOps(getLoopOp()));

  if (readOps.size() != 1 || writeOps.size() != 1 || loopOps.size() != 1) {
    return emitDefiniteFailure()
           << "requires exactly one read_op, write_op, and loop_op handle";
  }

  auto readOp = dyn_cast<vector::TransferReadOp>(readOps[0]);
  auto writeOp = dyn_cast<vector::TransferWriteOp>(writeOps[0]);
  auto loopOp = dyn_cast<scf::ForOp>(loopOps[0]);

  if (!readOp || !writeOp || !loopOp) {
    return emitDefiniteFailure() << "handles must be vector.transfer_read, "
                                    "vector.transfer_write, and scf.for";
  }

  // Verify read and write are in the loop
  if (!loopOp->isProperAncestor(readOp) || !loopOp->isProperAncestor(writeOp)) {
    return emitDefiniteFailure()
           << "read and write operations must be inside the loop";
  }

  Value loopIV = loopOp.getInductionVar();

  // Check if read indices are loop-invariant
  for (Value index : readOp.getIndices()) {
    if (dependsOnLoopIV(index, loopIV)) {
      return emitDefiniteFailure()
             << "read operation indices depend on loop induction variable";
    }
  }

  // Check if write indices are loop-invariant
  for (Value index : writeOp.getIndices()) {
    if (dependsOnLoopIV(index, loopIV)) {
      return emitDefiniteFailure()
             << "write operation indices depend on loop induction variable";
    }
  }

  // Check if they operate on the same memref
  if (readOp.getBase() != writeOp.getBase()) {
    return emitDefiniteFailure()
           << "read and write must operate on the same memref";
  }

  // Step 1: Clone the read and its operands before the loop
  rewriter.setInsertionPoint(loopOp);
  IRMapping readMapping;
  Value clonedReadResult =
      cloneOpAndOperands(readOp, loopIV, rewriter, readMapping);

  // Step 2: Get the value that the write op is writing (its vector operand)
  Value writeVector = writeOp.getVector();

  // Step 3: Use replaceWithAdditionalYields to add the read result as iter_arg
  // and yield the value to be written
  auto yieldValuesFn =
      [&](OpBuilder &b, Location loc,
          ArrayRef<BlockArgument> newBbArgs) -> SmallVector<Value> {
    // The new block argument is the last one (the hoisted read result)
    BlockArgument readIterArg = newBbArgs.back();

    // Replace uses of the original read with the iter_arg
    rewriter.replaceAllUsesWith(readOp.getResult(), readIterArg);

    // Return the value to yield (what the write op was writing)
    SmallVector<Value> yieldValues;
    yieldValues.push_back(writeVector);
    return yieldValues;
  };

  // Create new loop with additional iter_arg
  FailureOr<LoopLikeOpInterface> newLoopResult =
      cast<LoopLikeOpInterface>(loopOp.getOperation())
          .replaceWithAdditionalYields(
              rewriter, ValueRange{clonedReadResult}, // new init operand
              true,                                   // replace uses in loop
              yieldValuesFn);

  if (failed(newLoopResult)) {
    return emitDefiniteFailure() << "failed to add iter_args to loop";
  }

  auto newLoop = cast<scf::ForOp>(newLoopResult->getOperation());

  // Step 4: Erase the original read (now passed as iter_arg)
  rewriter.eraseOp(readOp);

  // Step 5: Create the write operation after the loop using the yielded value
  Value valueToWrite = newLoop.getResults().back();

  // Clone the write operation with updated vector value
  IRMapping writeMapping;
  writeMapping.map(writeVector, valueToWrite);

  // Clone ALL index dependencies FIRST, before creating the write
  // Set insertion point after the loop for index cloning
  rewriter.setInsertionPointAfter(newLoop);

  for (Value index : writeOp.getIndices()) {
    Operation *defOp = index.getDefiningOp();
    if (!defOp || dependsOnLoopIV(index, loopIV))
      continue; // Skip loop IV-dependent or non-operation indices

    // Check if this index is already outside the loop (from previous hoisting)
    if (!newLoop->isProperAncestor(defOp)) {
      // Index is already available outside - use it directly
      continue;
    }

    // Index is inside loop and needs to be cloned
    if (!writeMapping.contains(index)) {
      Value clonedIndex =
          cloneOpAndOperands(defOp, loopIV, rewriter, writeMapping);
      if (clonedIndex)
        writeMapping.map(index, clonedIndex);
    }
  }

  // NOW clone the write operation - DON'T reset insertion point, it's already
  // at the end after cloning indices
  rewriter.clone(*writeOp.getOperation(), writeMapping);

  // Step 6: Erase the original write
  rewriter.eraseOp(writeOp);

  SmallVector<Operation *> resultOps = {newLoop.getOperation()};
  results.set(llvm::cast<OpResult>(getResult()), resultOps);
  return DiagnosedSilenceableFailure::success();
}

void transform::HoistLoopInvariantTransfersOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getReadOpMutable(), effects);
  consumesHandle(getWriteOpMutable(), effects);
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
  auto memInterface = dyn_cast<MemoryEffectOpInterface>(op);
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

/// Check if there are any write operations to the memref between allocation and
/// the given operation Uses DominanceInfo for proper dominance analysis
static bool hasWritesBetween(memref::AllocOp allocOp, Operation *beforeOp) {
  Value allocResult = allocOp.getResult();

  // Get the function containing both operations
  auto funcOp = allocOp->getParentOfType<func::FuncOp>();
  if (!funcOp || funcOp != beforeOp->getParentOfType<func::FuncOp>()) {
    // If they're in different functions, conservatively return true
    return true;
  }

  // Create dominance info for the function
  DominanceInfo domInfo(funcOp);

  // Walk through all operations in the function to find writes
  bool foundWrite = false;
  funcOp.walk([&](Operation *op) {
    // Skip if we've already found a write
    if (foundWrite)
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
      foundWrite = true;
      return;
    }
  });

  return foundWrite;
}

/// Helper functions to extract source and target from different copy operation
/// types
static Value getCopySource(memref::CopyOp copyOp) { return copyOp.getSource(); }

static Value getCopySource(linalg::CopyOp copyOp) {
  return copyOp.getInputs()[0];
}

/// Template function to check if a copy operation copies from an uninitialized
/// memref
template <typename CopyOpType>
static bool isUninitializedCopy(CopyOpType copyOp) {
  Value source = getCopySource(copyOp);

  // Trace the source back to its allocation
  memref::AllocOp allocOp = traceToAlloc(source);
  if (!allocOp) {
    return false;
  }

  // Check if there are any writes to the allocated memref before this copy
  return !hasWritesBetween(allocOp, copyOp);
}

template <typename CopyOpType>
struct RemoveUninitializedCopyOpPattern : public OpRewritePattern<CopyOpType> {
  using OpRewritePattern<CopyOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(CopyOpType copyOp,
                                PatternRewriter &rewriter) const override {
    if (isUninitializedCopy(copyOp)) {
      rewriter.eraseOp(copyOp);
      return success();
    }
    return failure();
  }
};

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
    auto funcOp = dyn_cast<func::FuncOp>(target);
    if (!funcOp) {
      return emitDefiniteFailure() << "target must be a func.func operation";
    }

    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);

    // Apply the pattern to remove memcpy operations with uninitialized sources.
    patterns.insert<RemoveUninitializedCopyOpPattern<memref::CopyOp>,
                    RemoveUninitializedCopyOpPattern<linalg::CopyOp>>(ctx);
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
// FuseExtfLinalgOp
//===----------------------------------------------------------------------===//

/// Check if a linalg op contains only arith.extf as its body operation
/// (apart from terminator)
static bool containsOnlyExtfOp(linalg::LinalgOp linalgOp) {
  Block *body = linalgOp.getBlock();
  if (!body)
    return false;

  // Should have exactly one non-terminator operation, and it should be extf
  if (llvm::range_size(body->without_terminator()) != 1)
    return false;
  return llvm::any_of(body->without_terminator(),
                      [](Operation &o) { return isa<arith::ExtFOp>(o); });
}

/// Check if the second op directly consumes the result of the first op
static bool directlyConsumesResult(linalg::LinalgOp firstOp,
                                   linalg::LinalgOp secondOp) {
  // Get the result of the first op (assuming it has one DPS init)
  if (firstOp.getNumDpsInits() != 1)
    return false;

  Value firstOpResult = firstOp.getTiedOpResult(firstOp.getDpsInitOperand(0));

  // Check if the second op uses this result as an input
  return llvm::is_contained(secondOp.getDpsInputs(), firstOpResult);
}

/// Get the input type of the first op (before arith.extf type change)
static Type getInputTypeBeforeExtf(linalg::LinalgOp linalgOp) {
  Block *body = linalgOp.getBlock();
  if (!body)
    return nullptr;

  for (Operation &op : body->getOperations()) {
    if (auto extfOp = dyn_cast<arith::ExtFOp>(op)) {
      return extfOp.getIn().getType();
    }
  }

  return nullptr;
}

/// Fuse two linalg ops by creating a new fused operation
static FailureOr<linalg::GenericOp>
fuseExtfLinalgOps(RewriterBase &rewriter, linalg::LinalgOp firstOp,
                  linalg::LinalgOp secondOp) {
  // Get the input type before extf conversion
  Type originalType = getInputTypeBeforeExtf(firstOp);
  if (!originalType)
    return failure();

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

  // Get the original input from the first op
  Value originalInput = firstOp.getDpsInputs()[0];

  // Create new input list for the fused operation
  SmallVector<Value> newInputs;
  for (auto [index, input] : llvm::enumerate(secondOp.getDpsInputs())) {
    if (static_cast<int64_t>(index) == targetInputIndex) {
      newInputs.push_back(
          originalInput); // Use original input instead of first op's result
    } else {
      newInputs.push_back(input);
    }
  }

  // Create the new fused operation
  rewriter.setInsertionPoint(secondOp);
  auto fusedOp = rewriter.create<linalg::GenericOp>(
      secondOp.getLoc(), secondOp->getResultTypes(), newInputs,
      secondOp.getDpsInits(), secondOp.getIndexingMapsArray(),
      secondOp.getIteratorTypesArray());

  // Clone the body from the second operation and modify it
  Block *newBody = &fusedOp.getRegion().emplaceBlock();

  // Create block arguments with updated types
  SmallVector<Type> blockArgTypes;
  for (auto [index, input] : llvm::enumerate(newInputs)) {
    Type elementType;
    if ((int)index == targetInputIndex) {
      elementType = originalType; // Use original type for fused input
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

  // Clone the body from the second operation
  IRMapping mapping;
  Block *secondOpBody = secondOp.getBlock();

  // Map block arguments, adding extf for the fused input
  for (auto [index, oldArg] : llvm::enumerate(secondOpBody->getArguments())) {
    if ((int)index == targetInputIndex) {
      // For the fused input, we need to add an extf operation
      rewriter.setInsertionPointToStart(newBody);
      auto extfOp = rewriter.create<arith::ExtFOp>(
          fusedOp.getLoc(),
          cast<ShapedType>(firstOpResult.getType()).getElementType(),
          newBody->getArgument(index));
      mapping.map(oldArg, extfOp.getResult());
    } else {
      mapping.map(oldArg, newBody->getArgument(index));
    }
  }

  // Clone operations from the second op's body
  for (Operation &op : secondOpBody->getOperations()) {
    rewriter.clone(op, mapping);
  }

  // Replace the second operation with the fused operation
  rewriter.replaceOp(secondOp, fusedOp);

  return fusedOp;
}

DiagnosedSilenceableFailure
transform::FuseExtfLinalgOp::apply(transform::TransformRewriter &rewriter,
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

  auto firstLinalgOp = dyn_cast<linalg::LinalgOp>(firstOps[0]);
  auto secondLinalgOp = dyn_cast<linalg::LinalgOp>(secondOps[0]);

  if (!firstLinalgOp || !secondLinalgOp) {
    return emitDefiniteFailure() << "both operations must be linalg operations";
  }

  // Check condition 1: first op contains only arith.extf
  if (!containsOnlyExtfOp(firstLinalgOp)) {
    return emitDefiniteFailure()
           << "first operation must contain only arith.extf in its body";
  }

  // Check condition 2: second op directly consumes result of first op
  if (!directlyConsumesResult(firstLinalgOp, secondLinalgOp)) {
    return emitDefiniteFailure() << "second operation must directly consume "
                                    "the result of the first operation";
  }

  // Perform the fusion
  FailureOr<linalg::GenericOp> fusedOp =
      fuseExtfLinalgOps(rewriter, firstLinalgOp, secondLinalgOp);
  if (failed(fusedOp)) {
    return emitDefiniteFailure() << "failed to fuse the operations";
  }

  SmallVector<Operation *> resultOps = {*fusedOp};
  results.set(llvm::cast<OpResult>(getFusedOp()), resultOps);
  return DiagnosedSilenceableFailure::success();
}

void transform::FuseExtfLinalgOp::getEffects(
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
  auto transposeOp = builder.create<linalg::TransposeOp>(
      loc, input,
      builder.create<tensor::EmptyOp>(loc, transposedShape,
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
    auto reduceOp = dyn_cast<linalg::ReduceOp>(target);
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
    auto newReduceOp = rewriter.create<linalg::ReduceOp>(
        reduceOp.getLoc(), reduceOp.getResultTypes(),
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
    if (auto truncfOp = dyn_cast<arith::TruncFOp>(op)) {
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
  rewriter.setInsertionPoint(producerOp);

  // Get the truncf op's init to use directly
  Value truncfInit = truncfOp.getDpsInits()[0];

  auto fusedOp = rewriter.create<linalg::GenericOp>(
      producerOp.getLoc(), truncfOp->getResultTypes(),
      producerOp.getDpsInputs(), ValueRange{truncfInit},
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
      auto extfOp = rewriter.create<arith::ExtFOp>(
          fusedOp.getLoc(), originalOutputType, outputArg);
      use->set(extfOp.getResult());
    }
  }

  // Now add the truncf before the terminator and update the yield
  auto yieldOp = cast<linalg::YieldOp>(clonedBody->getTerminator());
  Value yieldValue = yieldOp.getValues()[0];

  rewriter.setInsertionPoint(yieldOp);
  auto truncfOpInBody = rewriter.create<arith::TruncFOp>(
      fusedOp.getLoc(), truncatedType, yieldValue);

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

  auto truncfLinalgOp = dyn_cast<linalg::LinalgOp>(truncfOps[0]);
  auto producerLinalgOp = dyn_cast<linalg::LinalgOp>(producerOps[0]);

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

  // Perform the fusion
  FailureOr<linalg::GenericOp> fusedOp =
      fuseTruncfIntoProducer(rewriter, producerLinalgOp, truncfLinalgOp);
  if (failed(fusedOp)) {
    return emitDefiniteFailure() << "failed to fuse the operations";
  }

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
// VectorTypeCastOp
//===----------------------------------------------------------------------===//

/// Helper function to create cast operations for both scalar and vector types
static Value createTypeCast(OpBuilder &builder, Location loc, Value input,
                            Type targetElementType, bool isExtension) {
  Type inputType = input.getType();

  // Determine the source element type and target type
  Type sourceElementType;
  Type targetType;

  if (auto inputVectorType = dyn_cast<VectorType>(inputType)) {
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
      return builder.create<arith::ExtFOp>(loc, targetType, input);
    } else if (isa<IntegerType>(sourceElementType) &&
               isa<IntegerType>(targetElementType)) {
      // For integer types, use sign extension
      return builder.create<arith::ExtSIOp>(loc, targetType, input);
    }
  } else {
    // Truncation: wide to narrow type
    if (isa<FloatType>(sourceElementType) &&
        isa<FloatType>(targetElementType)) {
      return builder.create<arith::TruncFOp>(loc, targetType, input);
    } else if (isa<IntegerType>(sourceElementType) &&
               isa<IntegerType>(targetElementType)) {
      return builder.create<arith::TruncIOp>(loc, targetType, input);
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

  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    if (auto vectorType = dyn_cast<VectorType>(operand.getType())) {
      hasVectorOperands = true;
      bool shouldCast =
          castAllInsAndOuts || inputIndicesToCastSet.contains((int64_t)idx);
      if (shouldCast && vectorType.getElementType() != targetElementType) {
        needsTransformation = true;
      }
    }
  }

  for (auto [idx, result] : llvm::enumerate(op->getResults())) {
    if (auto vectorType = dyn_cast<VectorType>(result.getType())) {
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

    if (auto vectorType = dyn_cast<VectorType>(operand.getType())) {
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

    if (auto vectorType = dyn_cast<VectorType>(resultType)) {
      bool shouldCast =
          castAllInsAndOuts || outputIndicesToCastSet.contains((int64_t)idx);

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

    if (auto originalVectorType = dyn_cast<VectorType>(originalType)) {
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
      if (auto vectorType = dyn_cast<VectorType>(operand.getType())) {
        if (vectorType.getElementType() != targetElementType) {
          needsTransformation = true;
          break;
        }
      }
    }
    if (!needsTransformation) {
      for (Value result : target->getResults()) {
        if (auto vectorType = dyn_cast<VectorType>(result.getType())) {
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
        return emitDefiniteFailure()
               << "failed to apply vector type cast to operation: "
               << target->getName();
      }
      transformedOps.push_back(*castedOpOnVector);
    }

    else
      transformedOps.push_back(target);
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
    auto memInterface = dyn_cast<MemoryEffectOpInterface>(op);
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

/// Calculate the total number of elements in a vector type
static int64_t getVectorNumElements(VectorType vecType) {
  int64_t numElements = 1;
  for (int64_t dim : vecType.getShape()) {
    numElements *= dim;
  }
  return numElements;
}

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
    auto forOp = dyn_cast<scf::ForOp>(target);
    if (!forOp) {
      return emitDefiniteFailure() << "target must be an scf.for operation";
    }

    Location loc = forOp.getLoc();

    // Collect vector-typed iter_args
    SmallVector<unsigned> vectorIterArgIndices;
    SmallVector<VectorType> originalVectorTypes;
    SmallVector<VectorType> flattenedVectorTypes;

    for (auto [idx, iterArg] : llvm::enumerate(forOp.getInitArgs())) {
      if (auto vecType = dyn_cast<VectorType>(iterArg.getType())) {
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
      auto shapeCast = rewriter.create<vector::ShapeCastOp>(
          loc, flattenedVectorTypes[idx], initArg);
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
    auto newForOp = rewriter.create<scf::ForOp>(loc, forOp.getLowerBound(),
                                                forOp.getUpperBound(),
                                                forOp.getStep(), newInitArgs);

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
      auto shapeCast = rewriter.create<vector::ShapeCastOp>(
          loc, originalVectorTypes[idx], newArg);
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
        auto shapeCast = rewriter.create<vector::ShapeCastOp>(
            loc, flattenedVectorTypes[vecIdx], mappedValue);
        newYieldOperands.push_back(shapeCast.getResult());
      } else {
        newYieldOperands.push_back(mapping.lookup(yieldValue));
      }
    }

    rewriter.create<scf::YieldOp>(loc, newYieldOperands);

    // Step 6: Insert shape_cast operations after the loop to convert results
    // back
    rewriter.setInsertionPointAfter(newForOp);
    SmallVector<Value> finalResults;

    for (auto [idx, result] : llvm::enumerate(newForOp.getResults())) {
      auto it = llvm::find(vectorIterArgIndices, idx);
      if (it != vectorIterArgIndices.end()) {
        size_t vecIdx = std::distance(vectorIterArgIndices.begin(), it);
        auto shapeCast = rewriter.create<vector::ShapeCastOp>(
            loc, originalVectorTypes[vecIdx], result);
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
