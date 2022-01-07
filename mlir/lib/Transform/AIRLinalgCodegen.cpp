// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "PassDetail.h"

#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/AIRLinalgCodegen.h"
#include "air/Util/CostModel.h"
#include "air/Util/Outliner.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/Transforms.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/SetVector.h"

#define DEBUG_TYPE "air-linalg-codegen"

using namespace mlir;

using namespace xilinx;
using namespace xilinx::air;

namespace {

struct FoldSubViewOpsPattern : public OpRewritePattern<memref::SubViewOp> {
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
    SmallVector<Value, 4> result_offsets;

    auto static_offsets = extractFromI64ArrayAttr(op.static_offsets());
    auto source_static_offsets =
        extractFromI64ArrayAttr(source_subview.static_offsets());
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
          Value b =
              rewriter.create<ConstantIndexOp>(op.getLoc(), source_offset);
          result_offsets.push_back(
              rewriter.create<AddIOp>(op.getLoc(), a.getType(), a, b));
        }
      } else if (op_offset >= 0 && source_offset < 0) {
        result_static_offsets.push_back(source_offset);
        if (op_offset == 0) {
          result_offsets.push_back(*source_offsets++);
        } else {
          Value a = *source_offsets++;
          Value b = rewriter.create<ConstantIndexOp>(op.getLoc(), op_offset);
          result_offsets.push_back(
              rewriter.create<AddIOp>(op.getLoc(), a.getType(), a, b));
        }
      } else if (op_offset < 0 && source_offset < 0) {
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
        rewriter.getI64ArrayAttr(result_static_offsets), op.static_sizes(),
        op.static_strides());

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
      if (auto c = oper.getDefiningOp<ConstantIndexOp>())
        shape[dim] = c.getValue();
      else
        return failure();
      dim++;
    }
    Value newOp = rewriter.replaceOpWithNewOp<memref::AllocOp>(
        op,
        MemRefType::get(shape, ty.getElementType(), {}, ty.getMemorySpace()));
    for (auto use : newOp.getUsers()) {
      if (auto launch = dyn_cast<air::HerdLaunchOp>(use)) {
        assert(launch.getKernelArguments().size() == launch.operands().size());
        for (unsigned int i = 0; i < launch.getNumKernelOperands(); i++) {
          auto arg = launch.getKernelArguments()[i];
          auto oper = launch.getKernelOperand(i);
          if (oper == newOp) {
            Block *b = arg.getOwner();
            auto new_arg =
                b->insertArgument(arg.getArgNumber(), newOp.getType());
            rewriter.setInsertionPointToStart(&*launch.getRegion().begin());
            arg.replaceAllUsesWith(rewriter.create<memref::CastOp>(
                op.getLoc(), new_arg, arg.getType()));
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

//     auto indexOp = op.index().getDefiningOp<ConstantIndexOp>();
//     if (!indexOp)
//       return failure();

//     rewriter.replaceOp(op, indexOp.getResult());
//     return success();
//   }
// };

struct RemoveSubViewOpsPattern : public OpRewritePattern<memref::SubViewOp> {
  using OpRewritePattern<memref::SubViewOp>::OpRewritePattern;

  RemoveSubViewOpsPattern(MLIRContext *ctx, unsigned int fast_memory_space = 1);

  LogicalResult matchAndRewrite(memref::SubViewOp op,
                                PatternRewriter &rewriter) const override {
    auto view = op.source().getDefiningOp<memref::ViewOp>();
    if (!view)
      return failure();
    auto alloc = view.source().getDefiningOp<memref::AllocOp>();
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

RemoveSubViewOpsPattern::RemoveSubViewOpsPattern(MLIRContext *ctx,
                                                 unsigned int fast_memory_space)
    : OpRewritePattern(ctx), fast_space(fast_memory_space) {}

// Custom LinalgOp tiling pattern
//
struct TileLinalgOpPattern : public RewritePattern {
  TileLinalgOpPattern(MLIRContext *context, linalg::LinalgTilingOptions options,
                      linalg::LinalgTransformationFilter filter =
                          linalg::LinalgTransformationFilter(),
                      PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context), filter(filter),
        options(options) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isa<linalg::GenericOp>(op))
      return failure();

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
  linalg::LinalgTransformationFilter filter;
  /// Options to control tiling;
  linalg::LinalgTilingOptions options;
};

class AIRLinalgCodegen : public AIRLinalgCodegenBase<AIRLinalgCodegen> {

public:
  AIRLinalgCodegen() = default;
  AIRLinalgCodegen(const AIRLinalgCodegen &pass) {}

  Option<bool> AIRLinalgCodegenTestPatterns{
      *this, "test-patterns", llvm::cl::desc("Test canonicalization patterns"),
      llvm::cl::init(false)};

  ListOption<unsigned> clHerdSize{
      *this, "herd-size", llvm::cl::desc("Herd size to target"),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated};

  ListOption<unsigned> clL1TileSize{
      *this, "l1-tile-size",
      llvm::cl::desc("Tile factors to pass to L1 tiling"), llvm::cl::ZeroOrMore,
      llvm::cl::CommaSeparated};

  ListOption<unsigned> clL2TileSize{
      *this, "l2-tile-size",
      llvm::cl::desc("Tile factors to pass to L2 tiling"), llvm::cl::ZeroOrMore,
      llvm::cl::CommaSeparated};

  ListOption<unsigned> clL1TileInterchange{
      *this, "l1-tile-permute",
      llvm::cl::desc("Tile permute vector to pass to L1 tiling"),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated};

  ListOption<unsigned> clL2TileInterchange{
      *this, "l2-tile-permute",
      llvm::cl::desc("Tile permute vector to pass to L2 tiling"),
      llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated};

  Option<unsigned> clL1MaxSize{*this, "L1-size",
                               llvm::cl::desc("L1 allocation limit in bytes"),
                               llvm::cl::init(32 * 1024)};

  Option<unsigned> clL2MaxSize{*this, "L2-size",
                               llvm::cl::desc("L2 allocation limit in bytes"),
                               llvm::cl::init(/*256*1024*/ 0)};

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<AffineDialect, memref::MemRefDialect, linalg::LinalgDialect,
                    scf::SCFDialect, air::airDialect, StandardOpsDialect>();
  }

  void runTestPatterns(FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();
    OwningRewritePatternList patterns(&getContext());
    patterns.insert<RemoveSubViewOpsPattern, FoldSubViewOpsPattern>(ctx);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }

  static SmallVector<int64_t> getTripCounts(linalg::LinalgOp op) {

    SmallVector<int64_t, 4> tripCounts;
    OpBuilder b(op);
    auto loc = op.getLoc();

    // use getStaticLoopRanges instead?
    auto allShapeSizes = op.createFlatListOfOperandDims(b, loc);
    AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
    if (!shapeSizesToLoopsMap)
      return {};

    auto shapeSizes =
        linalg::applyMapToValues(b, loc, shapeSizesToLoopsMap, allShapeSizes);
    for (auto size : shapeSizes) {
      auto c = dyn_cast<ConstantIndexOp>(size.getDefiningOp());
      if (!c) {
        LLVM_DEBUG(llvm::outs() << "Found non-constant dim!\n");
        return {};
      }
      tripCounts.push_back(c.getValue());
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
      tFactorAdjusted = std::max(1L, tripCounts[i] / tFactorAdjusted);
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

    uint64_t fp = CostModel().getOpCounts(op)["footprint"];
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

  void runGenericPatterns(FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();

    SmallVector<linalg::GenericOp, 4> genericOps;
    funcOp.walk([&](linalg::GenericOp op) { genericOps.push_back(op); });

    // GenericOp
    for (auto genericOp : genericOps) {

      SmallVector<int64_t, 2> herd_size{2, 2};
      SmallVector<int64_t, 4> l1_tile_size;
      SmallVector<unsigned, 4> l1_tile_interchange;
      SmallVector<int64_t, 4> l2_tile_size;
      SmallVector<unsigned, 4> l2_tile_interchange;

      auto tripCounts = getTripCounts(genericOp);

      bool tileForL2 = true;
      if (clL2TileSize.size())
        for (int i = 0, e = clL2TileSize.size(); i < e; i++)
          l2_tile_size[i] = clL2TileSize[i];
      else if (clL2MaxSize > 0)
        getTileSizes(genericOp, clL2MaxSize, tripCounts, &l2_tile_size);
      else
        tileForL2 = false;

      for (int i = 0, e = clL2TileInterchange.size(); i < e; i++)
        l2_tile_interchange[i] = clL2TileInterchange[i];

      for (int i = 0, e = std::min(2, (int)clHerdSize.size()); i < e; i++)
        herd_size[i] = clHerdSize[i];

      // outline the operation for convenience

      xilinx::air::AIROutliner olnr;
      CallOp call =
          olnr.outline(std::vector<Operation *>{genericOp}, "call_generic_op");
      FuncOp called = funcOp->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
          call.getCallee());

      // L2 tiling

      Identifier next_match = Identifier::get("", ctx);
      if (tileForL2) {
        OwningRewritePatternList stageL2Patterns(ctx);
        stageL2Patterns.insert<TileLinalgOpPattern>(
            ctx,
            linalg::LinalgTilingOptions()
                .setTileSizes(l2_tile_size)
                .setInterchange(l2_tile_interchange)
                .setLoopType(linalg::LinalgTilingLoopType::Loops),
            linalg::LinalgTransformationFilter(ArrayRef<Identifier>{},
                                               Identifier::get("L2", ctx)));

        stageL2Patterns
            .insert<linalg::LinalgPromotionPattern<linalg::GenericOp>>(
                ctx, linalg::LinalgPromotionOptions(),
                linalg::LinalgTransformationFilter(
                    Identifier::get("L2", ctx),
                    Identifier::get("L2_promoted", ctx)));
        stageL2Patterns.insert<RemoveSubViewOpsPattern>(ctx, 1);
        stageL2Patterns.insert<FoldSubViewOpsPattern>(ctx);
        stageL2Patterns.insert<MemrefsPattern>(ctx);
        scf::populateSCFForLoopCanonicalizationPatterns(stageL2Patterns);
        (void)applyPatternsAndFoldGreedily(called, std::move(stageL2Patterns));

        LLVM_DEBUG(llvm::outs() << "After L2 Tiling\n");
        LLVM_DEBUG(called.print(llvm::outs()));
        for (int i = 0, e = tripCounts.size(); i < e; i++)
          tripCounts[i] = l2_tile_size[i];
        next_match = Identifier::get("L2_promoted", ctx);
      }

      // compute L1 tile size

      called.walk([&](linalg::GenericOp l1_op) {
        if (clL1TileSize.size())
          for (int i = 0, e = clL1TileSize.size(); i < e; i++)
            l1_tile_size[i] = clL1TileSize[i];
        else if (clL1MaxSize > 0) {
          getTileSizes(l1_op, clL1MaxSize, tripCounts, &l1_tile_size);
        }
      });

      for (int i = 0, e = clL1TileInterchange.size(); i < e; i++)
        l1_tile_interchange[i] = clL1TileInterchange[i];

      // tile to the herd size

      SmallVector<int64_t, 4> herd_tile_size(tripCounts.size(), -1);
      for (int i = 0, e = l1_tile_size.size(); i < e; i++) {
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

      OwningRewritePatternList patterns(ctx);
      patterns.insert<TileLinalgOpPattern>(
          ctx,
          linalg::LinalgTilingOptions()
              .setTileSizes(herd_tile_size)
              .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops),
          linalg::LinalgTransformationFilter(
              tileForL2 ? next_match : ArrayRef<Identifier>{},
              Identifier::get("herd_tiling", ctx)));
      (void)applyPatternsAndFoldGreedily(called, std::move(patterns));
      next_match = Identifier::get("herd_tiling", ctx);

      LLVM_DEBUG(llvm::outs() << "After Herd Tiling\n");
      LLVM_DEBUG(called.print(llvm::outs()));

      bool needL1Tiling = !std::all_of(l1_tile_size.begin(), l1_tile_size.end(),
                                       [](int i) { return i == 0; });
      OwningRewritePatternList stageL1Patterns(ctx);
      if (needL1Tiling) {
        stageL1Patterns.insert<TileLinalgOpPattern>(
            ctx,
            linalg::LinalgTilingOptions()
                .setTileSizes(l1_tile_size)
                .setInterchange(l1_tile_interchange)
                .setLoopType(linalg::LinalgTilingLoopType::Loops),
            linalg::LinalgTransformationFilter(next_match,
                                               Identifier::get("L1", ctx)));
      }
      stageL1Patterns.insert<linalg::LinalgPromotionPattern<linalg::GenericOp>>(
          ctx, linalg::LinalgPromotionOptions(),
          linalg::LinalgTransformationFilter(
              needL1Tiling ? Identifier::get("L1", ctx) : next_match,
              Identifier::get("L1_promoted", ctx)));
      stageL1Patterns.insert<RemoveSubViewOpsPattern>(ctx, 2);
      stageL1Patterns.insert<FoldSubViewOpsPattern>(ctx);
      scf::populateSCFForLoopCanonicalizationPatterns(stageL1Patterns);
      (void)applyPatternsAndFoldGreedily(called, std::move(stageL1Patterns));

      OwningRewritePatternList stage3Patterns(&getContext());
      stage3Patterns.insert<MemrefsPattern>(ctx);
      (void)applyPatternsAndFoldGreedily(called, std::move(stage3Patterns));

      LLVM_DEBUG(llvm::outs() << "After L1 Tiling\n");
      LLVM_DEBUG(called.print(llvm::outs()));

      called.walk([](linalg::LinalgOp op) {
        op->removeAttr(linalg::LinalgTransforms::kLinalgTransformMarker);
      });

      InlinerInterface interface(&getContext());
      (void)inlineCall(interface, call, called, &called.getRegion(), true);
      call.erase();
      called.erase();
    }
  }

  void runMatmulPatterns(FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();

    SmallVector<linalg::MatmulOp, 4> matmulOps;
    funcOp.walk([&](linalg::MatmulOp op) { matmulOps.push_back(op); });

    // MatmulOp
    for (auto matmulOp : matmulOps) {

      xilinx::air::AIROutliner olnr;
      CallOp call =
          olnr.outline(std::vector<Operation *>{matmulOp}, "call_mmult");
      FuncOp called = funcOp->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
          call.getCallee());

      SmallVector<int64_t, 3> herd_size{2, 2, 2};
      SmallVector<int64_t, 3> l1_tile_size{32, 32, 32};
      SmallVector<unsigned, 3> l1_tile_interchange{0, 1, 2};
      SmallVector<int64_t, 3> l2_tile_size{64, 64, 64};
      SmallVector<unsigned, 3> l2_tile_interchange{0, 1, 2};

      for (int i = 0, e = clL1TileSize.size(); i < e; i++)
        l1_tile_size[i] = clL1TileSize[i];

      for (int i = 0, e = clL1TileInterchange.size(); i < e; i++)
        l1_tile_interchange[i] = clL1TileInterchange[i];

      for (int i = 0, e = clL2TileInterchange.size(); i < e; i++)
        l2_tile_interchange[i] = clL2TileInterchange[i];

      OwningRewritePatternList stageL2Patterns(ctx);

      bool tileForL2 = false;
      if (clL2TileSize.size()) {
        for (int i = 0, e = clL2TileSize.size(); i < e; i++)
          l2_tile_size[i] = clL2TileSize[i];
        tileForL2 = true;
      }

      if (tileForL2) {
        stageL2Patterns.insert<linalg::LinalgTilingPattern<linalg::MatmulOp>>(
            ctx,
            linalg::LinalgTilingOptions()
                .setTileSizes(l2_tile_size)
                .setInterchange(l2_tile_interchange)
                .setLoopType(linalg::LinalgTilingLoopType::Loops),
            linalg::LinalgTransformationFilter(ArrayRef<Identifier>{},
                                               Identifier::get("L2", ctx)));

        stageL2Patterns
            .insert<linalg::LinalgPromotionPattern<linalg::MatmulOp>>(
                ctx, linalg::LinalgPromotionOptions(),
                linalg::LinalgTransformationFilter(
                    Identifier::get("L2", ctx),
                    Identifier::get("L2_promoted", ctx)));
        stageL2Patterns.insert<RemoveSubViewOpsPattern>(ctx, 1);
        stageL2Patterns.insert<FoldSubViewOpsPattern>(ctx);
        stageL2Patterns.insert<MemrefsPattern>(ctx);
        scf::populateSCFForLoopCanonicalizationPatterns(stageL2Patterns);
      }

      OwningRewritePatternList stageL1Patterns(ctx);

      stageL1Patterns.insert<linalg::LinalgTilingPattern<linalg::MatmulOp>>(
          ctx,
          linalg::LinalgTilingOptions()
              .setTileSizes(l1_tile_size)
              .setInterchange(l1_tile_interchange)
              .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops),
          linalg::LinalgTransformationFilter(
              tileForL2 ? Identifier::get("L2_promoted", ctx)
                        : ArrayRef<Identifier>{},
              Identifier::get("L1", ctx)));

      stageL1Patterns.insert<linalg::LinalgPromotionPattern<linalg::MatmulOp>>(
          ctx, linalg::LinalgPromotionOptions(),
          linalg::LinalgTransformationFilter(
              Identifier::get("L1", ctx), Identifier::get("L1_promoted", ctx)));

      OwningRewritePatternList stage3Patterns(&getContext());
      stage3Patterns.insert<RemoveSubViewOpsPattern>(ctx, 2);
      stage3Patterns.insert<FoldSubViewOpsPattern>(ctx);
      if (!tileForL2)
        stage3Patterns.insert<MemrefsPattern>(ctx);
      scf::populateSCFForLoopCanonicalizationPatterns(stage3Patterns);

      (void)applyPatternsAndFoldGreedily(called, std::move(stageL2Patterns));
      (void)applyPatternsAndFoldGreedily(called, std::move(stageL1Patterns));
      (void)applyPatternsAndFoldGreedily(called, std::move(stage3Patterns));
      called.walk([](linalg::LinalgOp op) {
        op->removeAttr(linalg::LinalgTransforms::kLinalgTransformMarker);
      });

      InlinerInterface interface(&getContext());
      (void)inlineCall(interface, call, called, &called.getRegion(), true);
      call.erase();
      called.erase();
    }
  }

  void runConv2dPatterns(FuncOp funcOp) {
    MLIRContext *ctx = funcOp.getContext();

    SmallVector<linalg::Conv2DNchwFchwOp, 4> conv2dOps;
    funcOp.walk([&](linalg::Conv2DNchwFchwOp op) { conv2dOps.push_back(op); });

    // Conv2dOp
    for (auto conv2dOp : conv2dOps) {

      xilinx::air::AIROutliner olnr;
      CallOp call =
          olnr.outline(std::vector<Operation *>{conv2dOp}, "call_conv_2d_nchw");
      FuncOp called = funcOp->getParentOfType<ModuleOp>().lookupSymbol<FuncOp>(
          call.getCallee());

      Value input = conv2dOp.getOperand(0);
      Value weight = conv2dOp.getOperand(1);
      Value result = conv2dOp.getOperand(2);

      auto inputTy = input.getType().cast<ShapedType>();
      auto weightTy = weight.getType().cast<ShapedType>();
      auto resultTy = result.getType().cast<ShapedType>();

      // int64_t batch_sw = inputTy.getDimSize(0);
      int64_t batch_hw = 1;
      int64_t ifm_channels_sw = inputTy.getDimSize(1);
      int64_t ifm_height_sw = inputTy.getDimSize(2);
      int64_t ifm_width_sw = inputTy.getDimSize(3);
      int64_t ofm_channels_sw = resultTy.getDimSize(1);
      // int64_t ofm_height_sw = resultTy.getDimSize(2);
      // int64_t ofm_width_sw = resultTy.getDimSize(3);
      int64_t kernel_h = weightTy.getDimSize(2);
      int64_t kernel_w = weightTy.getDimSize(3);

      OwningRewritePatternList stage1Patterns(&getContext());
      stage1Patterns
          .insert<linalg::LinalgTilingPattern<linalg::Conv2DNchwFchwOp>>(
              ctx,
              linalg::LinalgTilingOptions()
                  .setTileSizes({batch_hw, ofm_channels_sw, ifm_height_sw / 4,
                                 ifm_width_sw, kernel_h, kernel_w,
                                 ifm_channels_sw})
                  .setInterchange({0, 2, 1, 3, 4, 5, 6})
                  .setLoopType(linalg::LinalgTilingLoopType::Loops),
              linalg::LinalgTransformationFilter(
                  Identifier::get("xten_conv2d", ctx),
                  Identifier::get("promote_L2", ctx)));

      stage1Patterns
          .insert<linalg::LinalgPromotionPattern<linalg::Conv2DNchwFchwOp>>(
              ctx,
              linalg::LinalgPromotionOptions().setOperandsToPromote(
                  std::vector<int64_t>{0, 1, 2}),
              linalg::LinalgTransformationFilter(
                  Identifier::get("promote_L2", ctx),
                  Identifier::get("L2", ctx)));

      stage1Patterns
          .insert<linalg::LinalgTilingPattern<linalg::Conv2DNchwFchwOp>>(
              ctx,
              linalg::LinalgTilingOptions()
                  .setTileSizes({batch_hw, ofm_channels_sw / 4,
                                 ifm_height_sw / 4, ifm_width_sw, kernel_h,
                                 kernel_w, ifm_channels_sw})
                  .setInterchange({1, 0, 2, 3, 4, 5, 6})
                  .setLoopType(linalg::LinalgTilingLoopType::Loops),
              linalg::LinalgTransformationFilter(
                  Identifier::get("L2", ctx),
                  Identifier::get("promote_HERD", ctx)));

      stage1Patterns
          .insert<linalg::LinalgPromotionPattern<linalg::Conv2DNchwFchwOp>>(
              ctx,
              linalg::LinalgPromotionOptions().setOperandsToPromote(
                  std::vector<int64_t>{0, 1, 2}),
              linalg::LinalgTransformationFilter(
                  Identifier::get("promote_HERD", ctx),
                  Identifier::get("HERD", ctx)));

      OwningRewritePatternList stage2Patterns =
          linalg::getLinalgTilingCanonicalizationPatterns(ctx);
      scf::populateSCFForLoopCanonicalizationPatterns(stage2Patterns);

      OwningRewritePatternList stage3Patterns(&getContext());
      stage3Patterns.insert<RemoveSubViewOpsPattern>(ctx, 2);
      stage3Patterns.insert<FoldSubViewOpsPattern>(ctx);

      (void)applyPatternsAndFoldGreedily(called, std::move(stage1Patterns));
      (void)applyPatternsAndFoldGreedily(called, std::move(stage2Patterns));
      (void)applyPatternsAndFoldGreedily(called, std::move(stage3Patterns));

      // Drop the marker.
      called.walk([](linalg::LinalgOp op) {
        op->removeAttr(linalg::LinalgTransforms::kLinalgTransformMarker);
      });

      InlinerInterface interface(&getContext());
      (void)inlineCall(interface, call, called, &called.getRegion(), true);
      call.erase();
      called.erase();
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
    } else {
      runTestPatterns(f);
    }
  }

  void runOnOperation() override {
    auto module = getOperation();
    SmallVector<FuncOp, 4> funcOps;
    module.walk([&](FuncOp op) { funcOps.push_back(op); });
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