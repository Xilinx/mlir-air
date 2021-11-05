// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "air/Transform/AIRLinalgCodegen.h"
#include "air/Util/Outliner.h"
#include "air/Dialect/AIR/AIRDialect.h"

#include "PassDetail.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "air-linalg-codegen"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

namespace {

struct FoldSubViewOpsPattern
    : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    if (!dyn_cast_or_null<memref::SubViewOp>(op.source().getDefiningOp()))
      return failure();

    auto source_subview = cast<memref::SubViewOp>(op.source().getDefiningOp());

    for (auto m : llvm::zip(source_subview.getType().getAffineMaps(),
                            op.getType().getAffineMaps()))
      if (std::get<0>(m) != std::get<1>(m))
        return failure();

    auto offsets = op.offsets().begin();
    auto source_offsets = source_subview.offsets().begin();
    SmallVector<Value,4> result_offsets;

    auto static_offsets = extractFromI64ArrayAttr(op.static_offsets());
    auto source_static_offsets = extractFromI64ArrayAttr(source_subview.static_offsets());
    SmallVector<int64_t,4> result_static_offsets;

    for (auto p : llvm::zip(static_offsets, source_static_offsets)) {
      auto op_offset = std::get<0>(p);
      auto source_offset = std::get<1>(p);
      if (op_offset >= 0 && source_offset >= 0) {
        result_static_offsets.push_back(op_offset+source_offset);
      }
      else if (op_offset < 0 && source_offset >= 0) {
        result_static_offsets.push_back(op_offset);
        if (source_offset == 0) {
          result_offsets.push_back(*offsets++);
        }
        else {
          Value a = *offsets++;
          Value b = rewriter.create<ConstantIndexOp>(op.getLoc(), source_offset);
          result_offsets.push_back( 
            rewriter.create<AddIOp>(op.getLoc(), a.getType(), a, b));
        }
      }
      else if (op_offset >= 0 && source_offset < 0) {
        result_static_offsets.push_back(source_offset);
        if (op_offset == 0) {
          result_offsets.push_back(*source_offsets++);
        }
        else {
          Value a = *source_offsets++;
          Value b = rewriter.create<ConstantIndexOp>(op.getLoc(), op_offset);
          result_offsets.push_back( 
            rewriter.create<AddIOp>(op.getLoc(), a.getType(), a, b));
        }
      }
      else if (op_offset < 0 && source_offset < 0) {
        Value a = *source_offsets++;
        Value b = *offsets++;
        result_offsets.push_back( 
          rewriter.create<AddIOp>(op.getLoc(), a.getType(), a, b));
        result_static_offsets.push_back(source_offset);
      }
    }

    rewriter.replaceOpWithNewOp<memref::SubViewOp>(
      op.getOperation(), op.getType(), source_subview.source(),
      result_offsets, op.sizes(), op.strides(),
      rewriter.getI64ArrayAttr(result_static_offsets),
      op.static_sizes(), op.static_strides());

    return success();
  }
};

struct RemoveSubViewOpsPattern
    : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    auto view = op.source().getDefiningOp<memref::ViewOp>();
    if (!view)
      return failure();
    auto alloc = view.source().getDefiningOp<memref::AllocOp>();
    if (!alloc)
      return failure();

    /* Force memory space 2 below */
    Value newOp = rewriter.replaceOpWithNewOp<memref::AllocOp>(
      op,
      MemRefType::get(op.getType().getShape(),
                      op.getType().getElementType(),
                      {}, 2),
      op.sizes());
//                      {}, view.getType().getMemorySpace()));
    alloc.replaceAllUsesWith(newOp);
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

    Value memref;
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return success();
    }
    Operation *castOp = nullptr;
    Operation *linalgOp = nullptr;
    for (auto &u : op->getUses())
      if (auto c = dyn_cast<memref::CastOp>(u.getOwner()))
        castOp = c;
      else if (auto l = dyn_cast<linalg::LinalgOp>(u.getOwner())) {
        linalgOp = l;
        if (l.isInitTensor(&u))
          return failure();
      }
    if (!castOp || !linalgOp)
      return failure();

    if (!castOp->hasOneUse())
      return failure();
    auto copyOp = dyn_cast<memref::CopyOp>(*castOp->user_begin());
    if (!copyOp)
      return failure();

    auto newOp = rewriter.create<memref::CastOp>(op->getLoc(), op.getType(),
                                                 copyOp->getOperand(1));
    rewriter.replaceOp(op, newOp->getResults());
    rewriter.eraseOp(copyOp);
    rewriter.eraseOp(castOp);
    return success();
  }
};

class AIRLinalgCodegen : public AIRLinalgCodegenBase<AIRLinalgCodegen> {

public:
  AIRLinalgCodegen() = default;
  AIRLinalgCodegen(const AIRLinalgCodegen &pass) {}

  Option<bool>
  AIRLinalgCodegenTestPatterns{*this, "test-patterns",
                               llvm::cl::desc("Test canonicalization patterns"),
                               llvm::cl::init(false)};

  ListOption<unsigned> HerdSize{*this, "herd-size",
                                llvm::cl::desc("Herd size to target"),
                                llvm::cl::ZeroOrMore,
                                llvm::cl::CommaSeparated};

  Option<unsigned> L1CacheSize{*this, "L1-size",
                               llvm::cl::desc("Size of L1 Cache"),
                               llvm::cl::init(32*1024)};

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<AffineDialect,
                    memref::MemRefDialect,
                    linalg::LinalgDialect,
                    scf::SCFDialect,
                    air::airDialect,
                    StandardOpsDialect>();
  }

  void runTestPatterns(FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<RemoveSubViewOpsPattern,
                    FoldSubViewOpsPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  void runGenericPatterns(FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();

    SmallVector<linalg::GenericOp, 4> genericOps;
    funcOp.walk([&](linalg::GenericOp op) { genericOps.push_back(op); });

    // GenericOp
    for (auto genericOp : genericOps) {

      xilinx::air::AIROutliner olnr;
      CallOp call = olnr.outline(std::vector<Operation *>{genericOp},
                                 "call_linalg_generic");
      FuncOp called = funcOp->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
          call.getCallee());

      SmallVector<int64_t, 4> l1_tile_size{32, 32};
      SmallVector<int64_t, 4> herd_size{2, 2};

      for (int i = 0, e = HerdSize.size(); i < e; i++) {
        herd_size[i] = HerdSize[i];
      }

      SmallVector<int64_t, 3> tile_sizes{128, 128};

      OwningRewritePatternList stage1Patterns(ctx);

      stage1Patterns.insert<linalg::LinalgTilingPattern<linalg::GenericOp>>(
          ctx,
          linalg::LinalgTilingOptions()
              .setTileSizes(tile_sizes)
              .setLoopType(linalg::LinalgTilingLoopType::Loops),
          linalg::LinalgTransformationFilter(ArrayRef<Identifier>{},
                                             Identifier::get("L1", ctx)));

      // divide it up evenly between tiles
      stage1Patterns.insert<linalg::LinalgTilingPattern<linalg::GenericOp>>(
          ctx,
          linalg::LinalgTilingOptions()
              .setTileSizes(l1_tile_size)
              .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops),
          linalg::LinalgTransformationFilter(Identifier::get("L1", ctx),
                                             Identifier::get("HERD", ctx)));

      stage1Patterns.insert<linalg::LinalgPromotionPattern<linalg::GenericOp>>(
          ctx, linalg::LinalgPromotionOptions(),
          linalg::LinalgTransformationFilter(Identifier::get("HERD", ctx),
                                             Identifier::get("promote", ctx)));

      OwningRewritePatternList stage2Patterns(ctx);
      scf::populateSCFForLoopCanonicalizationPatterns(stage2Patterns);

      OwningRewritePatternList stage3Patterns(&getContext());
      stage3Patterns.insert<RemoveSubViewOpsPattern, FoldSubViewOpsPattern>(
          ctx);

      (void)applyPatternsAndFoldGreedily(called, std::move(stage1Patterns));
      (void)applyPatternsAndFoldGreedily(called, std::move(stage2Patterns));
      (void)applyPatternsAndFoldGreedily(called, std::move(stage3Patterns));
      called.walk([](linalg::LinalgOp op) {
        op->removeAttr(linalg::LinalgTransforms::kLinalgTransformMarker);
      });
    }
  }

  void runMatmulPatterns(FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();

    SmallVector<linalg::MatmulOp, 4> matmulOps;
    funcOp.walk([&](linalg::MatmulOp op) {
      matmulOps.push_back(op);
    });

    // MatmulOp
    for (auto matmulOp : matmulOps) {

      xilinx::air::AIROutliner olnr;
      CallOp call = olnr.outline(std::vector<Operation*>{matmulOp},
                                 "call_mmult");
      FuncOp called = funcOp->getParentOfType<ModuleOp>()
                        .lookupSymbol<FuncOp>(call.getCallee());

      SmallVector<int64_t, 3> l1_tile_size{32, 32, 32};
      SmallVector<int64_t, 3> herd_size{2, 2, 2};

      for (int i=0, e=HerdSize.size(); i<e; i++) {
        herd_size[i] = HerdSize[i];
      }

      SmallVector<int64_t, 3> tile_sizes{l1_tile_size[0]*herd_size[0],
                                          l1_tile_size[1]*herd_size[1],
                                          l1_tile_size[2]*herd_size[2]};

      OwningRewritePatternList stage1Patterns(ctx);

      stage1Patterns.insert<linalg::LinalgTilingPattern<linalg::MatmulOp>>(
        ctx, linalg::LinalgTilingOptions().setTileSizes(tile_sizes)
                                          .setInterchange({2,1,0})
                                          .setLoopType(linalg::LinalgTilingLoopType::Loops),
        linalg::LinalgTransformationFilter(ArrayRef<Identifier>{},
                                           Identifier::get("L1", ctx)));

      // divide it up evenly between tiles
      stage1Patterns.insert<linalg::LinalgTilingPattern<linalg::MatmulOp>>(
        ctx, linalg::LinalgTilingOptions().setTileSizes(l1_tile_size)
                                          .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops),
        linalg::LinalgTransformationFilter(Identifier::get("L1", ctx),
                                          Identifier::get("HERD", ctx)));

      stage1Patterns.insert<linalg::LinalgPromotionPattern<linalg::MatmulOp>>(
        ctx, linalg::LinalgPromotionOptions(),
        linalg::LinalgTransformationFilter(Identifier::get("HERD", ctx),
                                          Identifier::get("promote", ctx)));

      OwningRewritePatternList stage2Patterns(ctx);
      scf::populateSCFForLoopCanonicalizationPatterns(stage2Patterns);
      
      OwningRewritePatternList stage3Patterns(&getContext());
      stage3Patterns.insert<RemoveSubViewOpsPattern, 
                            FoldSubViewOpsPattern>(ctx);
      
      (void)applyPatternsAndFoldGreedily(called, std::move(stage1Patterns));
      (void)applyPatternsAndFoldGreedily(called, std::move(stage2Patterns));
      (void)applyPatternsAndFoldGreedily(called, std::move(stage3Patterns));
      called.walk([](linalg::LinalgOp op) {
        op->removeAttr(linalg::LinalgTransforms::kLinalgTransformMarker);
      });
    }
  }

  void runConv2dPatterns(FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();

    SmallVector<linalg::Conv2DNchwFchwOp, 4> conv2dOps;
    funcOp.walk([&](linalg::Conv2DNchwFchwOp op) {
      conv2dOps.push_back(op);
    });

    // Conv2dOp
    for (auto conv2dOp : conv2dOps) {

      xilinx::air::AIROutliner olnr;
      CallOp call = olnr.outline(std::vector<Operation*>{conv2dOp},
                                 "call_conv_2d_nchw");
      FuncOp called = funcOp->getParentOfType<ModuleOp>()
                        .lookupSymbol<FuncOp>(call.getCallee());

      Value input = conv2dOp.getOperand(0);
      Value weight = conv2dOp.getOperand(1);
      Value result = conv2dOp.getOperand(2);

      auto inputTy = input.getType().cast<ShapedType>();
      auto weightTy = weight.getType().cast<ShapedType>();
      auto resultTy = result.getType().cast<ShapedType>();

      //int64_t batch_sw = inputTy.getDimSize(0);
      int64_t batch_hw = 1;
      int64_t ifm_channels_sw = inputTy.getDimSize(1);
      int64_t ifm_height_sw = inputTy.getDimSize(2);
      int64_t ifm_width_sw = inputTy.getDimSize(3);
      int64_t ofm_channels_sw = resultTy.getDimSize(1);
      //int64_t ofm_height_sw = resultTy.getDimSize(2);
      //int64_t ofm_width_sw = resultTy.getDimSize(3);
      int64_t kernel_h = weightTy.getDimSize(2);
      int64_t kernel_w = weightTy.getDimSize(3);

      OwningRewritePatternList stage1Patterns(&getContext());
      stage1Patterns.insert<linalg::LinalgTilingPattern<linalg::Conv2DNchwFchwOp>>(
        ctx, linalg::LinalgTilingOptions().setTileSizes({batch_hw,
                                                         ofm_channels_sw,
                                                         ifm_height_sw/4,
                                                         ifm_width_sw,
                                                         kernel_h, kernel_w,
                                                         ifm_channels_sw})
                                          .setInterchange({0,2,1,3,4,5,6})
                                          .setLoopType(linalg::LinalgTilingLoopType::Loops),
        linalg::LinalgTransformationFilter(Identifier::get("xten_conv2d", ctx),
                                          Identifier::get("promote_L2", ctx)));

      stage1Patterns.insert<linalg::LinalgPromotionPattern<linalg::Conv2DNchwFchwOp>>(
        ctx, linalg::LinalgPromotionOptions().setOperandsToPromote(std::vector<int64_t>{0,1,2}),
        linalg::LinalgTransformationFilter(Identifier::get("promote_L2", ctx),
                                          Identifier::get("L2", ctx)));

      stage1Patterns.insert<linalg::LinalgTilingPattern<linalg::Conv2DNchwFchwOp>>(
        ctx, linalg::LinalgTilingOptions().setTileSizes({batch_hw,
                                                         ofm_channels_sw/4,
                                                         ifm_height_sw/4,
                                                         ifm_width_sw,
                                                         kernel_h, kernel_w,
                                                         ifm_channels_sw})
                                          .setInterchange({1,0,2,3,4,5,6})
                                          .setLoopType(linalg::LinalgTilingLoopType::Loops),
        linalg::LinalgTransformationFilter(Identifier::get("L2", ctx),
                                          Identifier::get("promote_HERD", ctx)));

      stage1Patterns.insert<linalg::LinalgPromotionPattern<linalg::Conv2DNchwFchwOp>>(
        ctx, linalg::LinalgPromotionOptions().setOperandsToPromote(std::vector<int64_t>{0,1,2}),
        linalg::LinalgTransformationFilter(Identifier::get("promote_HERD", ctx),
                                          Identifier::get("HERD", ctx)));

      OwningRewritePatternList stage2Patterns =
        linalg::getLinalgTilingCanonicalizationPatterns(ctx);
      scf::populateSCFForLoopCanonicalizationPatterns(stage2Patterns);

      OwningRewritePatternList stage3Patterns(&getContext());
      stage3Patterns.insert<RemoveSubViewOpsPattern, 
                            FoldSubViewOpsPattern>(ctx);
      
      (void)applyPatternsAndFoldGreedily(called, std::move(stage1Patterns));
      (void)applyPatternsAndFoldGreedily(called, std::move(stage2Patterns));
      (void)applyPatternsAndFoldGreedily(called, std::move(stage3Patterns));

      // Drop the marker.
      called.walk([](linalg::LinalgOp op) {
        op->removeAttr(linalg::LinalgTransforms::kLinalgTransformMarker);
      });
    }
  }

  void runOnFunction(FuncOp f) {

    OwningRewritePatternList prePatterns(&getContext());
    prePatterns.insert<RemoveAllocLinalgOpCopyPattern>(&getContext());
    (void)applyPatternsAndFoldGreedily(f, std::move(prePatterns));

    if (!AIRLinalgCodegenTestPatterns) {
      runMatmulPatterns(f);
      runConv2dPatterns(f);
      runGenericPatterns(f);
    }
    else {
      runTestPatterns(f);
    }

  }


  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<FuncOp, 4> funcOps;
    module.walk([&](FuncOp op) {
      funcOps.push_back(op);
    });
    for (auto f : funcOps)
      runOnFunction(f);
  }

private:

};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRLinalgCodegenPass() {
  return std::make_unique<AIRLinalgCodegen>();
}

} // namespace air
} // namespace xilinx
