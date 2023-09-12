//===- AIRLinalgCodegen.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIR/AIRTransformOps.h"
#include "air/Transform/AIRLinalgCodegen.h"
#include "air/Util/CostModel.h"
#include "air/Util/Outliner.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SetVector.h"

#include <numeric>
#include <optional>

#define DEBUG_TYPE "air-linalg-codegen"

using namespace mlir;
using namespace xilinx;

namespace {

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

    auto offsets = op.offsets().begin();
    auto source_offsets = source_subview.offsets().begin();
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
        result_offsets, op.sizes(), op.strides(),
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
        assert(launch.getKernelArguments().size() ==
               launch.getOperands().size());
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
//     auto operTy = op.memrefOrTensor().getType().dyn_cast<ShapedType>();
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
                        {}, fast_space),
        op.sizes());
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
                        {}, fast_space),
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
    for (auto &u : copyOper0.getUses())
      num_uses++;

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
        dyn_cast<memref::AllocOp>(op.getOperand(0).getDefiningOp());
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
      else if (auto c = dyn_cast<memref::CopyOp>(u.getOwner()))
        if (u.getOperandNumber() == 0)
          copyOp = c;
        else if (auto l = dyn_cast<linalg::LinalgOp>(u.getOwner())) {
          linalgOp = l;
          if (l.isInitTensor(&u))
            return failure();
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
      Optional<StringAttr> replacement = std::nullopt);

  explicit LinalgTransformationFilter(
      const FilterFunction &f, ArrayRef<StringAttr> matchDisjunction = {},
      Optional<StringAttr> replacement = std::nullopt);

  LinalgTransformationFilter(LinalgTransformationFilter &&) = default;
  LinalgTransformationFilter(const LinalgTransformationFilter &) = default;
  LogicalResult checkAndNotify(PatternRewriter &rewriter, Operation *op) const;
  void replaceLinalgTransformationFilter(PatternRewriter &rewriter,
                                         Operation *op) const;
  bool hasReplacementFilter(Operation *op) const;

  LinalgTransformationFilter &addFilter(const FilterFunction &f) {
    if (f)
      filters.push_back(f);
    return *this;
  }

  template <typename... OpTypes> LinalgTransformationFilter &addOpFilter() {
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
  Optional<StringAttr> replacement;
  /// When set to true, if the attribute is not set, it will be treated as
  /// a match. Default is false.
  bool matchByDefault;
};

LinalgTransformationFilter::LinalgTransformationFilter(
    ArrayRef<StringAttr> matchDisjunction, Optional<StringAttr> replacement)
    : matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement), matchByDefault(false) {}

LinalgTransformationFilter::LinalgTransformationFilter(
    const FilterFunction &f, ArrayRef<StringAttr> matchDisjunction,
    Optional<StringAttr> replacement)
    : filters(),
      matchDisjunction(matchDisjunction.begin(), matchDisjunction.end()),
      replacement(replacement), matchByDefault(false) {
  if (f)
    filters.push_back(f);
}

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

bool LinalgTransformationFilter::hasReplacementFilter(Operation *op) const {
  if (!replacement)
    return false;
  auto attr = op->getAttr(air::LinalgTransforms::kLinalgTransformMarker)
                  .dyn_cast<StringAttr>();
  return attr && attr == *replacement;
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

    Optional<linalg::TiledLinalgOp> tiledLinalgOp =
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
    rewriter.startRootUpdate(op);
    Optional<linalg::LinalgOp> promotedOp =
        promoteSubViews(rewriter, op, options);
    if (!promotedOp) {
      rewriter.cancelRootUpdate(op);
      return op->emitError("subview promotion failed");
    }
    rewriter.finalizeRootUpdate(op);
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
  MemRefType allocType =
      MemRefType::get(viewType.getShape(), viewType.getElementType(), {},
                      (unsigned)air::MemorySpace::L1);
  Value buffer = b.createOrFold<memref::AllocOp>(subView.getLoc(), allocType);
  return buffer;
}

static LogicalResult deallocBufferCallBack(OpBuilder &b, Value buffer) {
  // b.create<memref::DeallocOp>(buffer.getLoc(), buffer);
  return success();
}

FailureOr<linalg::TiledLinalgOp> static pipelineLinalgOp(
    PatternRewriter &b, linalg::LinalgOp op,
    ArrayRef<int64_t> static_tile_sizes, unsigned int pipeline_depth,
    std::string pipeline_direction, bool promote) {

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  auto loc = op.getLoc();

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

  auto launch = b.create<air::HerdOp>(loc, dims, args);
  b.setInsertionPointToStart(&launch.getBody().front());

  auto nLoops = op.getNumLoops();
  SmallVector<OpFoldResult, 4> tileSizeVector;
  auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
  for (auto s : static_tile_sizes)
    tileSizeVector.push_back(
        b.create<arith::ConstantIndexOp>(loc, s).getResult());
  if (tileSizeVector.size() < nLoops)
    tileSizeVector.append(nLoops - tileSizeVector.size(), zero.getResult());

  auto allShapeSizes = op.createFlatListOfOperandDims(b, loc);
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap)
    return failure();
  SmallVector<OpFoldResult> sizeBounds =
      makeComposedFoldedMultiResultAffineApply(b, loc, shapeSizesToLoopsMap,
                                               allShapeSizes);

  SmallVector<OpFoldResult> tileIds;
  for (int i = 0, e = static_tile_sizes.size(); i < e; i++) {
    tileIds.push_back(b.create<arith::MulIOp>(loc,
                                              isHoriz ? launch.getIds()[0]
                                                      : launch.getIds()[1],
                                              tileSizeVector[i].get<Value>())
                          .getResult());
  }
  SmallVector<Value, 4> tiledOperands = linalg::makeTiledShapes(
      b, loc, op, args, tileIds, tileSizeVector, sizeBounds, true);
  SmallVector<Type, 4> stageResultTypes;
  unsigned int resultIdx = 0;
  for (OpOperand *opOperand : op.getDpsInitOperands()) {
    resultIdx = opOperand->getOperandNumber();
    auto memrefType = tiledOperands[resultIdx].getType().cast<MemRefType>();
    auto tensorType = RankedTensorType::get(memrefType.getShape(),
                                            memrefType.getElementType());
    stageResultTypes.push_back(tensorType);
  }

  auto pipe = b.create<air::HerdPipelineOp>(loc);
  pipe->setAttr("direction",
                StringAttr::get(op->getContext(), pipeline_direction));
  Block *pipelineBlock = new Block();
  pipe.getBody().push_back(pipelineBlock);
  b.setInsertionPointToStart(pipelineBlock);

  Value firstOutputOperand = tiledOperands[resultIdx];
  Value inputOperand = nullptr;
  for (unsigned int i = 0; i < pipeline_depth; i++) {
    OpBuilder::InsertionGuard pipeline_guard(b);
    bool last_stage = i == pipeline_depth - 1;
    bool first_stage = i == 0;
    SmallVector<Value, 1> opers;
    if (inputOperand)
      opers.push_back(inputOperand);

    auto stage = b.create<air::PipelineStageOp>(
        loc, last_stage ? SmallVector<Type, 1>() : stageResultTypes, opers);

    Block *stageBlock = new Block();
    stage.getBody().push_back(stageBlock);
    b.setInsertionPointToStart(stageBlock);

    for (auto o : opers) {
      auto a = stageBlock->addArgument(o.getType(), loc);
      if (!a.getType().isa<MemRefType>()) {
        auto ty = tiledOperands[resultIdx].getType().cast<MemRefType>();
        tiledOperands[resultIdx] =
            b.create<bufferization::ToMemrefOp>(
                 loc, MemRefType::get(ty.getShape(), ty.getElementType()), a)
                .getResult();
      }
    }

    if (last_stage)
      tiledOperands[resultIdx] = firstOutputOperand;

    linalg::LinalgOp linalgOp = clone(b, op, {}, tiledOperands);

    if (promote) {
      SmallVector<int64_t, 3> opers_to_promote(linalgOp->getNumOperands() - 1);
      std::iota(opers_to_promote.begin(), opers_to_promote.end(), 0);
      if (first_stage || last_stage)
        opers_to_promote.push_back(linalgOp->getNumOperands() - 1);

      b.setInsertionPoint(linalgOp);
      auto loc = linalgOp->getLoc();
      auto defaultCopyCallBack = [loc](OpBuilder &b, Value src,
                                       Value dst) -> LogicalResult {
        b.create<memref::CopyOp>(loc, src, dst);
        return success();
      };
      auto emptyCopyCallBack = [](OpBuilder &b, Value src,
                                  Value dst) -> LogicalResult {
        return success();
      };
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
      b.setInsertionPointToEnd(stageBlock);
      b.create<air::PipelineYieldOp>(loc, SmallVector<Value, 1>());
    } else {
      auto mref = tiledOperands[resultIdx];
      if (promote && first_stage) {
        memref::SubViewOp sv = dyn_cast<memref::SubViewOp>(
            linalgOp.getDpsInitOperand(0)->get().getDefiningOp());
        mref = sv.getSource();
        sv.replaceAllUsesWith(mref);
      }

      b.setInsertionPointToEnd(stageBlock);
      auto t = b.create<bufferization::ToTensorOp>(loc, mref);
      b.create<air::PipelineYieldOp>(loc, t.getResult());
      inputOperand = stage.getResult(0);
    }
    // if (erased) erased.erase();
  }

  SmallVector<Type, 1> pipeTys;
  SmallVector<Value, 1> pipeArgs;
  b.create<air::PipelineTerminatorOp>(loc, pipeTys, pipeArgs);

  b.setInsertionPointToEnd(&launch.getBody().front());
  b.create<air::HerdTerminatorOp>(loc);
  int i = 0;
  for (auto a : args) {
    replaceAllUsesInRegionWith(a, launch.getKernelArgument(i++),
                               launch.getBody());
  }

  return linalg::TiledLinalgOp{op, {launch}, {}};
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
    PatternRewriter &b, linalg::LinalgOp op,
    ArrayRef<int64_t> static_tile_sizes, unsigned int pipeline_depth,
    std::string pipeline_direction, bool promote) {

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
      makeComposedFoldedMultiResultAffineApply(b, loc, shapeSizesToLoopsMap,
                                               allShapeSizes);

  SmallVector<OpFoldResult> tileIds;
  for (auto s : tileSizes) {
    if (s == 0)
      continue;
    AffineExpr d0 = b.getAffineDimExpr(0);
    auto map = AffineMap::get(1, 0, d0 * s);
    tileIds.push_back(
        b.create<AffineApplyOp>(loc, map,
                                isHoriz ? herd.getIds()[0] : herd.getIds()[1])
            .getResult());
  }
  SmallVector<Value, 4> tiledOperands = linalg::makeTiledShapes(
      b, loc, op, args, tileIds, tileSizeVector, sizeBounds, true);

  unsigned int resultIdx = 0;
  for (OpOperand *opOperand : op.getDpsInitOperands()) {
    resultIdx = opOperand->getOperandNumber();
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
    AffineIfOp aif =
        b.create<AffineIfOp>(op->getLoc(), int_set, int_set_args, false);

    Block *stageBlock = aif.getBody();
    b.setInsertionPointToStart(stageBlock);

    if (i) {
      auto ty = tiledOperands[resultIdx].getType().cast<MemRefType>();
      auto alloc = b.create<memref::AllocOp>(
          loc, MemRefType::get(ty.getShape(), ty.getElementType(), AffineMap(),
                               (int)air::MemorySpace::L1));
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
      auto channel_op =
          b.create<air::ChannelOp>(loc, cname, b.getI64ArrayAttr({1}));
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

  b.setInsertionPointToEnd(&herd.getBody().front());
  b.create<air::HerdTerminatorOp>(loc);
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
    : public air::AIRPipelineReducePassBase<AIRPipelineReducePass> {

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

  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}
class AIRLinalgCodegen : public air::AIRLinalgCodegenBase<AIRLinalgCodegen> {

public:
  AIRLinalgCodegen() = default;
  AIRLinalgCodegen(const AIRLinalgCodegen &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, linalg::LinalgDialect,
                    scf::SCFDialect, air::airDialect, func::FuncDialect>();
  }

  void runTestPatterns(func::FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<RemoveSubViewOpsPattern, FoldSubViewOpsPattern,
                    RemoveViewOpsPattern, HoistReduceBufferPattern,
                    RemoveAllocLinalgOpCopyPattern, RemoveExtraAllocPattern,
                    RemoveAllocCopyLinalgOpCopyPattern, RemoveDeadCopyPattern,
                    RemoveFillCopyLinalgPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
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
        makeComposedFoldedMultiResultAffineApply(b, loc, shapeSizesToLoopsMap,
                                                 allShapeSizes);
    for (auto size : shapeSizes) {
      if (auto v = size.dyn_cast<Value>()) {
        auto c = dyn_cast<arith::ConstantIndexOp>(v.getDefiningOp());
        if (!c) {
          LLVM_DEBUG(llvm::outs() << "Found non-constant dim!\n");
          return {};
        }
        tripCounts.push_back(c.value());
      } else {
        auto a = size.dyn_cast<Attribute>();
        auto c = a.dyn_cast<IntegerAttr>();
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

    assert(op.getNumLoops() == tileSizes->size() && "invalid tile size count");
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
        stageL2Patterns.insert<PromoteLinalgOpPattern>(
            ctx, l2PromoteOptions,
            LinalgTransformationFilter(StringAttr::get(ctx, "L2"),
                                       StringAttr::get(ctx, "L2_promoted")));
        stageL2Patterns.insert<RemoveSubViewOpsPattern>(ctx, 1);
        stageL2Patterns.insert<FoldSubViewOpsPattern>(ctx);
        stageL2Patterns.insert<MemrefsPattern>(ctx);
        scf::populateSCFForLoopCanonicalizationPatterns(stageL2Patterns);
        (void)applyPatternsAndFoldGreedily(called, std::move(stageL2Patterns));

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
      (void)applyPatternsAndFoldGreedily(called, std::move(patterns));
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
      stageL1Patterns.insert<PromoteLinalgOpPattern>(
          ctx, l1PromoteOptions,
          LinalgTransformationFilter(needL1Tiling ? StringAttr::get(ctx, "L1")
                                                  : next_match,
                                     StringAttr::get(ctx, "L1_promoted")));
      stageL1Patterns.insert<RemoveSubViewOpsPattern>(ctx, 2);
      stageL1Patterns.insert<FoldSubViewOpsPattern>(ctx);
      scf::populateSCFForLoopCanonicalizationPatterns(stageL1Patterns);
      (void)applyPatternsAndFoldGreedily(called, std::move(stageL1Patterns));

      RewritePatternSet stage3Patterns(&getContext());
      stage3Patterns.insert<MemrefsPattern>(ctx);
      stage3Patterns.insert<RemoveAllocCopyLinalgOpCopyPattern>(ctx);
      (void)applyPatternsAndFoldGreedily(called, std::move(stage3Patterns));

      LLVM_DEBUG(llvm::outs() << "After L1 Tiling\n");
      LLVM_DEBUG(called.print(llvm::outs()));

      called.walk([](linalg::LinalgOp op) {
        op->removeAttr(air::LinalgTransforms::kLinalgTransformMarker);
      });

      InlinerInterface interface(&getContext());
      (void)inlineCall(interface, call, called, &called.getRegion(), true);
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
        stageL2Patterns.insert<PromoteLinalgOpPattern>(
            ctx, l2PromoteOptions,
            LinalgTransformationFilter(StringAttr::get(ctx, "L2"),
                                       StringAttr::get(ctx, "L2_promoted")));
        stageL2Patterns.insert<RemoveSubViewOpsPattern>(ctx, 1);
        stageL2Patterns.insert<FoldSubViewOpsPattern>(ctx);
        stageL2Patterns.insert<MemrefsPattern>(ctx);
        scf::populateSCFForLoopCanonicalizationPatterns(stageL2Patterns);
        (void)applyPatternsAndFoldGreedily(called, std::move(stageL2Patterns));
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
      stageL1Patterns.insert<PromoteLinalgOpPattern>(
          ctx, l1PromoteOptions,
          LinalgTransformationFilter(StringAttr::get(ctx, "L1"),
                                     StringAttr::get(ctx, "L1_promoted")));

      RewritePatternSet stage3Patterns(&getContext());
      stage3Patterns.insert<RemoveSubViewOpsPattern>(ctx, 2);
      stage3Patterns.insert<FoldSubViewOpsPattern>(ctx);
      stage3Patterns.insert<MemrefsPattern>(ctx);
      scf::populateSCFForLoopCanonicalizationPatterns(stage3Patterns);

      (void)applyPatternsAndFoldGreedily(called, std::move(stageL1Patterns));
      (void)applyPatternsAndFoldGreedily(called, std::move(stage3Patterns));
      called.walk([](linalg::LinalgOp op) {
        op->removeAttr(air::LinalgTransforms::kLinalgTransformMarker);
      });

      InlinerInterface interface(&getContext());
      (void)inlineCall(interface, call, called, &called.getRegion(), true);
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

      (void)applyPatternsAndFoldGreedily(called, std::move(stage1Patterns));
      (void)applyPatternsAndFoldGreedily(called, std::move(stage2Patterns));
      (void)applyPatternsAndFoldGreedily(called, std::move(stage3Patterns));

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

      assert(clHerdSize.size() != 0 && "AIE tile dimension can't be zero");

      assert(
          clHerdSize.size() <= loops.size() &&
          "AIE tile dimension must be equal or less than Tiled loops number");

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
      (void)inlineCall(interface, call, called, &called.getRegion(), true);
      call.erase();
      called.erase();
    }
  }

  void runOnFunction(func::FuncOp f) {

    // RewritePatternSet prePatterns(&getContext());
    // prePatterns.insert<RemoveAllocLinalgOpCopyPattern>(&getContext());
    //(void)applyPatternsAndFoldGreedily(f, std::move(prePatterns));
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

} // namespace

/// A simple pattern rewriter that implements no special logic.
class SimpleRewriter : public PatternRewriter {
public:
  SimpleRewriter(MLIRContext *context) : PatternRewriter(context) {}
};

//===----------------------------------------------------------------------===//
// PipelineReduceOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure transform::PipelineReduceOp::applyToOne(
    linalg::LinalgOp target, transform::ApplyToEachResultList &results,
    transform::TransformState &state) {
  SimpleRewriter rewriter(getContext());
  auto result = pipelineReduceLinalgOp(
      rewriter, target, extractFromI64ArrayAttr(getTileSize()),
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
                                    ArrayRef<int64_t> staticTileSizes,
                                    ArrayRef<int64_t> interchange) {
  return build(builder, result,
               /*target=*/target,
               /*mixedTileSizes=*/
               getAsOpFoldResult(builder.getI64ArrayAttr(staticTileSizes)),
               interchange);
}

void transform::LinalgTileOp::build(OpBuilder &builder, OperationState &result,
                                    Value target,
                                    ArrayRef<OpFoldResult> mixedTileSizes,
                                    ArrayRef<int64_t> interchange) {
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
        /*static_sizes=*/staticTileSizesAttr,
        /*interchange=*/builder.getDenseI64ArrayAttr(interchange));
}

DiagnosedSilenceableFailure
transform::LinalgTileOp::apply(TransformResults &transformResults,
                               TransformState &state) {
  ArrayRef<int64_t> tileSizes = getStaticSizes();

  ArrayRef<Operation *> targets = state.getPayloadOps(getTarget());
  SmallVector<ArrayRef<Operation *>> dynamicSizeProducers;
  dynamicSizeProducers.reserve(getDynamicSizes().size());
  for (Value dynamicSizeProducerHandle : getDynamicSizes()) {
    dynamicSizeProducers.push_back(
        state.getPayloadOps(dynamicSizeProducerHandle));

    if (dynamicSizeProducers.back().size() != targets.size()) {
      DiagnosedSilenceableFailure diag =
          emitSilenceableError()
          << "expected as many dynamic size-producing operations ("
          << dynamicSizeProducers.back().size() << ") as target ops ("
          << targets.size() << ")";
      diag.attachNote(dynamicSizeProducerHandle.getLoc()) << "for this handle";
      return diag;
    }

    for (Operation *op : dynamicSizeProducers.back()) {
      if (op->getNumResults() == 1 &&
          op->getResult(0).getType().isa<IndexType>())
        continue;
      DiagnosedSilenceableFailure diag =
          emitSilenceableError() << "expected sizes to be produced by ops "
                                    "with a single index-type result";
      diag.attachNote(op->getLoc()) << "size producer op";
      diag.attachNote(dynamicSizeProducerHandle.getLoc()) << "for this handle";
      return diag;
    }
  }

  SmallVector<Operation *> tiled;
  SmallVector<SmallVector<Operation *, 4>, 4> loops;
  loops.resize(getLoops().size());
  for (auto &en : llvm::enumerate(targets)) {
    auto linalgOp = dyn_cast<linalg::LinalgOp>(en.value());
    if (!linalgOp) {
      DiagnosedSilenceableFailure diag = emitSilenceableError()
                                         << "only linalg ops are supported";
      diag.attachNote(en.value()->getLoc()) << "target op";
      return diag;
    }

    linalg::LinalgTilingOptions tilingOptions;
    tilingOptions.setLoopType(linalg::LinalgTilingLoopType::ParallelLoops);
    unsigned index = en.index();
    if (!tileSizes.empty()) {
      tilingOptions.setTileSizeComputationFunction(
          [&, index](OpBuilder &b, Operation *) {
            SmallVector<Value, 4> sizes;
            sizes.reserve(tileSizes.size());
            unsigned dynamicIdx = 0;
            for (OpFoldResult ofr : getMixedSizes()) {
              if (auto attr = ofr.dyn_cast<Attribute>()) {
                sizes.push_back(b.create<arith::ConstantIndexOp>(
                    getLoc(), attr.cast<IntegerAttr>().getInt()));
              } else {
                sizes.push_back(
                    dynamicSizeProducers[dynamicIdx++][index]->getResult(0));
              }
            }
            return sizes;
          });
    }

    SmallVector<unsigned int> inter(getInterchange());
    tilingOptions.setInterchange(inter);
    SimpleRewriter rewriter(linalgOp.getContext());
    FailureOr<linalg::TiledLinalgOp> maybeTilingResult =
        linalg::tileLinalgOp(rewriter, linalgOp, tilingOptions);
    if (failed(maybeTilingResult))
      return DiagnosedSilenceableFailure::definiteFailure();

    if (linalgOp.hasBufferSemantics())
      rewriter.eraseOp(linalgOp);
    else
      rewriter.replaceOp(linalgOp,
                         maybeTilingResult->loops.front()->getResults());

    tiled.push_back(maybeTilingResult->op);
    for (const auto &en2 : llvm::enumerate(maybeTilingResult->loops))
      loops[en2.index()].push_back(en2.value());
  }

  transformResults.set(getTiledLinalgOp().cast<OpResult>(), tiled);
  for (const auto &en : llvm::enumerate(loops))
    transformResults.set(getLoops()[en.index()].cast<OpResult>(), en.value());

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

// We want to parse `DenseI64ArrayAttr` using the short form without the
// `array` prefix to be consistent in the IR with `parseDynamicIndexList`.
static ParseResult parseInterchange(OpAsmParser &parser,
                                    OperationState &result) {
  if (succeeded(parser.parseOptionalLBrace())) {
    if (failed(parser.parseKeyword("interchange")))
      return parser.emitError(parser.getNameLoc()) << "expect `interchange`";
    if (failed(parser.parseEqual()))
      return parser.emitError(parser.getNameLoc()) << "expect `=`";
    result.addAttribute("interchange",
                        DenseI64ArrayAttr::parse(parser, Type{}));
    if (failed(parser.parseRBrace()))
      return parser.emitError(parser.getNameLoc()) << "expect `}`";
  }
  return success();
}

static void printInterchange(OpAsmPrinter &p,
                             ArrayRef<int64_t> interchangeVals) {
  if (!interchangeVals.empty()) {
    p << " {interchange = [";
    llvm::interleaveComma(interchangeVals, p,
                          [&](int64_t integer) { p << integer; });
    p << "]}";
  }
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

  // Parse optional interchange.
  if (failed(parseInterchange(parser, result)))
    return ParseResult::failure();

  result.addAttribute(getStaticSizesAttrName(result.name), staticSizes);
  size_t numExpectedLoops =
      staticSizes.size() - llvm::count(staticSizes.asArrayRef(), 0);
  result.addTypes(SmallVector<Type>(numExpectedLoops + 1, pdlOperationType));
  return success();
}

void transform::LinalgTileOp::print(OpAsmPrinter &p) {
  p << ' ' << getTarget();
  printDynamicIndexList(p, getOperation(), getDynamicSizes(), getStaticSizes());
  printInterchange(p, getInterchange());
}

void transform::LinalgTileOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  consumesHandle(getTarget(), effects);
  onlyReadsHandle(getDynamicSizes(), effects);
  producesHandle(getTiledLinalgOp(), effects);
  producesHandle(getLoops(), effects);
  modifiesPayload(effects);
}

//===----------------------------------------------------------------------===//
// LinalgPromoteOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::LinalgPromoteOp::apply(transform::TransformResults &results,
                                  transform::TransformState &state) {

  ArrayRef<Operation *> payloadOps = state.getPayloadOps(getTarget());
  if (!payloadOps.size())
    DiagnosedSilenceableFailure::success();

  linalg::LinalgPromotionOptions promotionOptions;
  auto operandsToPromote = extractFromI64ArrayAttr(getOperandsToPromote());

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

    auto ctx = target->getContext();
    SimpleRewriter rewriter(ctx);
    rewriter.setInsertionPoint(target);
    FailureOr<linalg::LinalgOp> res =
        promoteSubViews(rewriter, target, promotionOptions);
    if (failed(res))
      return emitDefaultDefiniteFailure(target);

    transformed.insert(linalgOp);
  }

  auto ctx = payloadOps[0]->getContext();
  RewritePatternSet patterns(ctx);
  // promoteSubViews generates extra copies and subviews, these patterns try to
  // simplify them.
  patterns.insert<RemoveSubViewOpsPattern>(ctx, (int)memorySpace);
  patterns.insert<FoldSubViewOpsPattern, RemoveViewOpsPattern>(ctx);
  patterns.insert<RemoveExtraAllocPattern, RemoveDeadCopyPattern,
                  RemoveAllocCopyLinalgOpCopyPattern>(ctx);
  // canonicalize allocs like:
  //  memref.alloc(%c32, %c32) : memref<?x?xi32, 2>
  // to:
  //  memref.alloc() : memref<32x32xi32, 2>
  memref::AllocOp::getCanonicalizationPatterns(patterns, ctx);
  (void)applyPatternsAndFoldGreedily(
      payloadOps[0]->getParentOfType<func::FuncOp>(), std::move(patterns));

  if (!transformed.size())
    return emitDefaultDefiniteFailure(payloadOps[0]);

  results.set(getResult().cast<OpResult>(), transformed.getArrayRef());
  return DiagnosedSilenceableFailure::success();
}

void transform::LinalgPromoteOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  onlyReadsHandle(getTarget(), effects);
  producesHandle(getResult(), effects);
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
  consumesHandle(getProducerOp(), effects);
  onlyReadsHandle(getContainingOp(), effects);
  producesHandle(getFusedOp(), effects);
  modifiesPayload(effects);
}

static FailureOr<linalg::LinalgOp>
generateResultTileValue(Operation *op, Operation *forOp, OpBuilder &b,
                        ArrayRef<OpFoldResult> offsets,
                        ArrayRef<OpFoldResult> sizes) {
  auto linalgOp = cast<linalg::LinalgOp>(op);
  auto loc = op->getLoc();
  SmallVector<Value> args;
  for (auto o : op->getOperands())
    args.push_back(o);
  auto allShapeSizes = linalgOp.createFlatListOfOperandDims(b, loc);
  AffineMap shapeSizesToLoopsMap = linalgOp.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap)
    return failure();
  SmallVector<OpFoldResult> sizeBounds =
      makeComposedFoldedMultiResultAffineApply(b, loc, shapeSizesToLoopsMap,
                                               allShapeSizes);
  SmallVector<OpFoldResult, 2> ivs =
      cast<scf::ParallelOp>(forOp).getInductionVars();
  SmallVector<Value> tiledOperands = linalg::makeTiledShapes(
      b, op->getLoc(), linalgOp, args, ivs, sizes, sizeBounds, true);

  SmallVector<Value> operands;
  auto ti = tiledOperands.begin();
  for (auto o : op->getOperands()) {
    if (isa<MemRefType>(o.getType()))
      operands.push_back(*ti);
    else
      operands.push_back(o);
    ti++;
  }

  linalg::LinalgOp newLinalgOp = clone(b, op, {}, operands);
  return newLinalgOp;
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
  auto users = producerLinalgOp.getDpsInitOperands()[0]->get().getUsers();
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

  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(sliceOpToTile);

  // Tile the producer.
  FailureOr<linalg::LinalgOp> tiledProducer = generateResultTileValue(
      producerOp, containingOp, rewriter, sliceOpToTile.getMixedOffsets(),
      sliceOpToTile.getMixedSizes());
  if (failed(tiledProducer)) {
    diag.attachNote(tileableProducer->getLoc())
        << "failed to tile producer op: " << *tileableProducer;
    return nullptr;
  }
  LLVM_DEBUG(llvm::dbgs() << "tiledProducer: " << *tiledProducer << "\n");

  // Replace the extract op.
  rewriter.replaceOp(sliceOpToTile,
                     tiledProducer.value().getDpsInitOperand(0)->get());
  return *tiledProducer;
}

DiagnosedSilenceableFailure transform::FuseIntoContainingMemrefOp::apply(
    transform::TransformResults &results, transform::TransformState &state) {
  SmallVector<Operation *> fusedOps;
  ArrayRef<Operation *> producerOps = state.getPayloadOps(getProducerOp());
  // If nothing to fuse, propagate success.
  if (producerOps.empty()) {
    results.set(getFusedOp().cast<OpResult>(),
                SmallVector<mlir::Operation *>{});
    return DiagnosedSilenceableFailure::success();
  }
  if (producerOps.size() != 1) {
    return emitDefiniteFailure()
           << "requires exactly one producer_op handle (got "
           << producerOps.size() << ")";
  }
  Operation *producerOp = producerOps.front();

  ArrayRef<Operation *> containingOps = state.getPayloadOps(getContainingOp());
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

  auto initOperand = producerLinalgOp.getDpsInitOperands()[0]->get();
  // The containing op may be a user of producerOp: use isAncestor.
  int64_t numUsesInContainingOp =
      llvm::count_if(initOperand.getUsers(), [&](Operation *op) {
        return containingOp->isAncestor(op);
      });
  if (numUsesInContainingOp == 0) {
    results.set(getFusedOp().cast<OpResult>(), ArrayRef<Operation *>());
    Diagnostic diag(containingOp->getLoc(), DiagnosticSeverity::Remark);
    diag << "producer_op does not have uses in the container";
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }

  // Default diagnostic, to be complemented with more failure information.
  Diagnostic diag(producerOp->getLoc(), DiagnosticSeverity::Remark);
  diag << "could not fuse " << *producerOp << " into " << *containingOp;

  IRRewriter rewriter(getContext());
  Operation *tiled =
      tileAndFuseFirstExtractUse(rewriter, diag, producerOp, containingOp);
  if (tiled) {
    LLVM_DEBUG(llvm::dbgs() << "\nFused a direct extract use\n"
                            << *containingOp);
    fusedOps.push_back(tiled);
    rewriter.eraseOp(producerOp);

    results.set(getFusedOp().cast<OpResult>(), fusedOps);
    return DiagnosedSilenceableFailure::success();
  }

  results.set(getFusedOp().cast<OpResult>(), ArrayRef<Operation *>());
  return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
}

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