// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.
//
// ===- AIRMiscPasses.cpp -------------------------------------------------===//
//
// Miscellaneous useful and/or experimental passes
//
// ===---------------------------------------------------------------------===//

#include "air/Transform/AIRMiscPasses.h"
#include "PassDetail.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Transform/AIRTilingUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/Support/Debug.h"

#include <list>

#define DEBUG_TYPE "air-misc-passes"

using namespace mlir;

namespace {

class AIRExamplePass : public xilinx::air::AIRExamplePassBase<AIRExamplePass> {

public:
  AIRExamplePass() = default;
  AIRExamplePass(const AIRExamplePass &pass){};

  void runOnOperation() override;

private:
};

void AIRExamplePass::runOnOperation() {}

class AIRLinalgNamePass
    : public xilinx::air::AIRLinalgNamePassBase<AIRLinalgNamePass> {

public:
  AIRLinalgNamePass() = default;
  AIRLinalgNamePass(const AIRLinalgNamePass &pass){};

  void runOnOperation() override;

private:
};

void AIRLinalgNamePass::runOnOperation() {
  auto module = getOperation();
  auto ctx = module.getContext();

  unsigned id = 0;
  module.walk([&](linalg::LinalgOp op) {
    auto attr = op->getAttrOfType<StringAttr>(
        linalg::LinalgTransforms::kLinalgTransformMarker);
    if (!attr) {
      std::string name =
          op->getName().getStringRef().str() + std::to_string(id++);
      op->setAttr(linalg::LinalgTransforms::kLinalgTransformMarker,
                  StringAttr::get(ctx, name));
    }
  });
}

class AIRRemoveLinalgNamePass
    : public xilinx::air::AIRRemoveLinalgNamePassBase<AIRRemoveLinalgNamePass> {

public:
  AIRRemoveLinalgNamePass() = default;
  AIRRemoveLinalgNamePass(const AIRRemoveLinalgNamePass &pass){};

  void runOnOperation() override;

private:
};

void AIRRemoveLinalgNamePass::runOnOperation() {
  auto module = getOperation();

  module.walk([&](linalg::LinalgOp op) {
    auto attr = op->getAttrOfType<StringAttr>(
        linalg::LinalgTransforms::kLinalgTransformMarker);
    if (attr) {
      op->removeAttr(linalg::LinalgTransforms::kLinalgTransformMarker);
    }
  });
}

// AIRPromoteUniformL1Dma
class AIRPromoteUniformL1Dma
    : public xilinx::air::AIRPromoteUniformL1DmaBase<AIRPromoteUniformL1Dma> {

public:
  AIRPromoteUniformL1Dma() = default;
  AIRPromoteUniformL1Dma(const AIRPromoteUniformL1Dma &pass){};

  void runOnOperation() override;

private:
};

void do_clone(OpBuilder &builder, Operation *op,
              BlockAndValueMapping &mapping) {
  if (!op)
    return;
  for (auto o : op->getOperands()) {
    if (mapping.contains(o))
      continue;
    do_clone(builder, o.getDefiningOp(), mapping);
  }
  builder.clone(*op, mapping);
}

void AIRPromoteUniformL1Dma::runOnOperation() {
  auto module = getOperation();
  // auto ctx = module.getContext();

  std::vector<Operation *> erasedOps;
  module.walk([&](xilinx::air::DmaMemcpyNdOp memcpyOp) {
    auto pipeline = memcpyOp->getParentOfType<xilinx::air::HerdPipelineOp>();
    auto stage = memcpyOp->getParentOfType<xilinx::air::PipelineStageOp>();
    auto launch = memcpyOp->getParentOfType<xilinx::air::HerdLaunchOp>();
    if (!pipeline || !stage || !launch)
      return;

    // auto direction = pipeline->getAttrOfType<StringAttr>("direction");
    auto uniform = stage->getAttrOfType<BoolAttr>("uniform");
    if (!uniform)
      return;

    auto src_type = memcpyOp.src().getType().cast<MemRefType>();
    auto dst_type = memcpyOp.dst().getType().cast<MemRefType>();
    auto src_space = src_type.getMemorySpaceAsInt();
    auto dst_space = dst_type.getMemorySpaceAsInt();

    MemRefType ty = nullptr;
    bool to_l1 = (src_space == 0 && dst_space == 2);
    bool from_l1 = (src_space == 2 && dst_space == 0);
    if (to_l1)
      ty = dst_type;
    else if (from_l1)
      ty = src_type;
    else
      return;

    OpBuilder builder(launch);
    auto loc = memcpyOp->getLoc();
    auto alloc = builder.create<memref::AllocOp>(
        loc, MemRefType::get(ty.getShape(), ty.getElementType(),
                             ty.getLayout().getAffineMap(), 1));
    std::vector<Value> launch_operands;
    BlockAndValueMapping remap;
    for (unsigned int i = 0; i < launch.getNumKernelOperands(); i++) {
      auto arg = launch.getKernelArguments()[i];
      auto oper = launch.getKernelOperand(i);
      remap.map(arg, oper);
    }
    if (to_l1)
      remap.map(memcpyOp.dst(), alloc);
    do_clone(builder, memcpyOp.getOperation(), remap);

    launch_operands.insert(launch_operands.begin(),
                           launch->getOperands().begin(),
                           launch->getOperands().end());
    launch_operands.push_back(alloc.getResult());
    launch->setOperands(launch_operands);
    launch.body().front().addArgument(alloc.getType(), loc);
    auto sizeAttr = launch->getAttr("operand_segment_sizes")
                        .cast<::mlir::DenseIntElementsAttr>();
    const uint32_t *it = &*sizeAttr.value_begin<uint32_t>();
    auto newAttr = DenseIntElementsAttr::get(sizeAttr.getType(),
                                             {it[0], it[1], it[2], it[3] + 1});
    launch->setAttr("operand_segment_sizes", newAttr);

    builder.setInsertionPoint(memcpyOp);
    SmallVector<Value, 2> opers{};
    SmallVector<Value, 2> mt;
    Value a = launch.getKernelArguments()[it[3]];
    builder.create<xilinx::air::DmaMemcpyNdOp>(
        loc, SmallVector<Type, 1>{}, mt, to_l1 ? memcpyOp.dst() : a, mt, mt, mt,
        to_l1 ? a : memcpyOp.src(), mt, mt, mt);
    erasedOps.push_back(memcpyOp);
  });
  for (auto e : erasedOps)
    e->erase();
  // module.dump();
}

// return true if op is a function of v
bool isFuncOf(Operation *op, Value v, std::vector<Operation *> &ops) {
  bool r = false;
  if (!op)
    return r;

  for (auto o : op->getOperands()) {
    if ((o == v) || (isFuncOf(o.getDefiningOp(), v, ops))) {
      if (std::find(std::begin(ops), std::end(ops), op) == std::end(ops))
        ops.push_back(op);
      r = true;
    }
  }
  return r;
}

// AIRSpecializeDma
class AIRSpecializeDma
    : public xilinx::air::AIRSpecializeDmaBase<AIRSpecializeDma> {

public:
  AIRSpecializeDma() = default;
  AIRSpecializeDma(const AIRSpecializeDma &pass){};

  void runOnOperation() override;

private:
};

void AIRSpecializeDma::runOnOperation() {
  auto module = getOperation();
  auto ctx = module.getContext();

  module.walk([&](xilinx::air::HerdLaunchOp launch) {
    launch.walk([&](xilinx::air::DmaMemcpyNdOp memcpyOp) {
      std::vector<Operation *> xOps, yOps;
      bool fn_x = isFuncOf(memcpyOp, launch.getTileIds().x, xOps);
      bool fn_y = isFuncOf(memcpyOp, launch.getTileIds().y, yOps);
      auto herd_size = launch.getHerdSizeOperands();
      int64_t herd_size_x =
          cast<arith::ConstantIndexOp>(herd_size.x.getDefiningOp()).value();
      int64_t herd_size_y =
          cast<arith::ConstantIndexOp>(herd_size.y.getDefiningOp()).value();
      if (fn_x && !fn_y) {
        auto loc = memcpyOp->getLoc();
        OpBuilder builder(memcpyOp);
        auto pipe = builder.create<xilinx::air::HerdPipelineOp>(loc);
        pipe->setAttr("direction", StringAttr::get(ctx, "horiz"));
        auto pipe_bb = new Block();
        pipe.body().push_back(pipe_bb);
        builder.setInsertionPointToEnd(pipe_bb);
        builder.create<xilinx::air::PipelineTerminatorOp>(
            loc, SmallVector<Value, 1>{});
        builder.setInsertionPointToStart(pipe_bb);
        for (int x = 0; x < herd_size_x; x++) {
          auto stage = builder.create<xilinx::air::PipelineStageOp>(
              loc, SmallVector<Type, 1>{}, SmallVector<Value, 1>{});
          stage->setAttr("uniform", BoolAttr::get(ctx, true));
          auto stage_bb = new Block();
          stage.body().push_back(stage_bb);
          auto stage_builder = OpBuilder::atBlockEnd(stage_bb);
          auto c_x = stage_builder.create<arith::ConstantIndexOp>(loc, x);
          BlockAndValueMapping remap;
          remap.map(launch.getTileIds().x, c_x);
          for (auto xop : xOps)
            stage_builder.clone(*xop, remap);
          stage_builder.create<xilinx::air::PipelineYieldOp>(
              loc, SmallVector<Type, 1>{}, SmallVector<Value, 1>{});
        }
        memcpyOp.erase();
      }
      if (fn_y && !fn_x) {
        auto loc = memcpyOp->getLoc();
        OpBuilder builder(memcpyOp);
        auto pipe = builder.create<xilinx::air::HerdPipelineOp>(loc);
        pipe->setAttr("direction", StringAttr::get(ctx, "vert"));
        auto pipe_bb = new Block();
        pipe.body().push_back(pipe_bb);
        builder.setInsertionPointToEnd(pipe_bb);
        builder.create<xilinx::air::PipelineTerminatorOp>(
            loc, SmallVector<Value, 1>{});
        builder.setInsertionPointToStart(pipe_bb);
        for (int y = 0; y < herd_size_y; y++) {
          auto stage = builder.create<xilinx::air::PipelineStageOp>(
              loc, SmallVector<Type, 1>{}, SmallVector<Value, 1>{});
          stage->setAttr("uniform", BoolAttr::get(ctx, true));
          auto stage_bb = new Block();
          stage.body().push_back(stage_bb);
          auto stage_builder = OpBuilder::atBlockEnd(stage_bb);
          auto c_y = stage_builder.create<arith::ConstantIndexOp>(loc, y);
          BlockAndValueMapping remap;
          remap.map(launch.getTileIds().y, c_y);
          for (auto yop : yOps)
            stage_builder.clone(*yop, remap);
          stage_builder.create<xilinx::air::PipelineYieldOp>(
              loc, SmallVector<Type, 1>{}, SmallVector<Value, 1>{});
        }
        memcpyOp.erase();
      }
    });
  });
}

FailureOr<linalg::TiledLinalgOp> static pipelineLinalgOp(
    PatternRewriter &b, linalg::LinalgOp op, unsigned int pipeline_depth, std::string pipeline_direction) {

  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(op);
  auto loc = op.getLoc();

  if (!(pipeline_direction == "vert" || pipeline_direction == "horiz"))
    return failure();

  auto iteratorTypes = llvm::to_vector<4>(op.iterator_types().getValue());
  if (isParallelIterator(iteratorTypes.back()))
    return failure();

  xilinx::air::HerdDim2 dims{
      b.create<arith::ConstantIndexOp>(loc, 4),
      b.create<arith::ConstantIndexOp>(loc, 1)};

  SmallVector<Value, 4> args;
  for (auto o : op.getInputAndOutputOperands())
    args.push_back(o->get());

  auto launch = b.create<xilinx::air::HerdLaunchOp>(loc, dims, args);
  b.setInsertionPointToStart(&launch.body().front());

  auto nLoops = op.getNumLoops();
  SmallVector<Value, 4> tileSizeVector;
  auto zero = b.create<arith::ConstantIndexOp>(loc, 0);
  tileSizeVector.append(nLoops - 1, zero);
  tileSizeVector.push_back(b.create<arith::ConstantIndexOp>(loc, 64));

  auto allShapeSizes = op.createFlatListOfOperandDims(b, loc);
  AffineMap shapeSizesToLoopsMap = op.getShapesToLoopsMap();
  if (!shapeSizesToLoopsMap)
    return failure();
  auto sizeBounds =
        applyMapToValues(b, loc, shapeSizesToLoopsMap, allShapeSizes);

  SmallVector<Value, 2> tileIds = {launch.getTileIds().x};
  SmallVector<Value, 4> tiledOperands = linalg::makeTiledShapes(b, loc, op, args, tileIds, tileSizeVector, sizeBounds, true);
  SmallVector<Type, 4> stageResultTypes;
  unsigned int resultIdx = 0;
  for (OpOperand *opOperand : op.getOutputOperands()) {
    resultIdx = opOperand->getOperandNumber();
    stageResultTypes.push_back(tiledOperands[resultIdx].getType());
  }

  auto pipe = b.create<xilinx::air::HerdPipelineOp>(loc);
  pipe->setAttr("direction", StringAttr::get(op->getContext(), pipeline_direction));
  Block *pipelineBlock = new Block();
  pipe.body().push_back(pipelineBlock);
  b.setInsertionPointToStart(pipelineBlock);

  SmallVector<Value,16> stageTensors;
  stageTensors.push_back(tiledOperands[resultIdx]);
  for (int i=0; i<4; i++) {
    OpBuilder::InsertionGuard pipeline_guard(b);
    SmallVector<Value,1> opers{tiledOperands[resultIdx]};

    auto stage = b.create<xilinx::air::PipelineStageOp>(loc, stageResultTypes, opers);
    Block *stageBlock = new Block();
    stage.body().push_back(stageBlock);
    for (auto t : stageResultTypes)
      stageBlock->addArgument(t, loc);

    b.setInsertionPointToStart(stageBlock);

    tiledOperands[resultIdx] = stageBlock->getArgument(0);
    linalg::LinalgOp tiledLinalgOp = op.clone(b, loc, {}, tiledOperands);
    //b.create<xilinx::air::PipelineYieldOp>(loc, tiledLinalgOp->getResults());
    b.create<xilinx::air::PipelineYieldOp>(loc, tiledOperands[resultIdx]);

    tiledOperands[resultIdx] = stage->getResult(0);
    stageTensors.push_back(tiledOperands[resultIdx]);
  }
  if (auto sliceOp = stageTensors[0].getDefiningOp<tensor::ExtractSliceOp>()) {
    b.create<tensor::InsertSliceOp>(
      loc, sliceOp.source().getType(), stageTensors.back(), sliceOp.source(), sliceOp.offsets(),
      sliceOp.sizes(), sliceOp.strides(), sliceOp.static_offsets(),
      sliceOp.static_sizes(), sliceOp.static_strides());
    tiledOperands[resultIdx] = sliceOp.source();
  }
  SmallVector<Type,1> pipeTys;
  SmallVector<Value,1> pipeArgs;
  b.create<xilinx::air::PipelineTerminatorOp>(loc, pipeTys, pipeArgs);

  b.setInsertionPointToEnd(&launch.body().front());
  b.create<xilinx::air::HerdTerminatorOp>(loc);
  int i = 0;
  for (auto a : args) {
    replaceAllUsesInRegionWith(a, launch.getKernelArgument(i++), launch.body());
  }
  //return linalg::TiledLinalgOp{op, {launch}, {tiledOperands[resultIdx]}};
  return linalg::TiledLinalgOp{op, {launch}, {}};

}

struct PipelineReducePattern : public RewritePattern {
  PipelineReducePattern(MLIRContext *context, linalg::LinalgTilingOptions options,
                        linalg::LinalgTransformationFilter filter =
                          linalg::LinalgTransformationFilter(),
                        PatternBenefit benefit = 1)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context), filter(filter),
        options(options) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    linalg::LinalgOp linalgOp = dyn_cast<linalg::LinalgOp>(op);
    if (!linalgOp)
      return failure();

    if (failed(filter.checkAndNotify(rewriter, linalgOp)))
      return failure();

    // if (!op->getParentOfType<xilinx::air::HerdLaunchOp>())
    //   return failure();

    if (op->getParentOfType<xilinx::air::HerdPipelineOp>())
      return failure();

    auto result = pipelineLinalgOp(rewriter, linalgOp, 4, "horiz");
    if (failed(result))
      return failure();
    //linalgOp->getParentOfType<xilinx::air::HerdLaunchOp>()->dump();

    //rewriter.replaceOp(op, result->tensorResults);
    rewriter.eraseOp(op);
    return success();
  }

private:
  /// LinalgTransformMarker handles special attribute manipulations.
  linalg::LinalgTransformationFilter filter;
  /// Options to control tiling;
  linalg::LinalgTilingOptions options;
};

class AIRPipelineReducePass
    : public xilinx::air::AIRPipelineReducePassBase<AIRPipelineReducePass> {

public:
  AIRPipelineReducePass() = default;
  AIRPipelineReducePass(const AIRPipelineReducePass &pass){};

  void runOnOperation() override;

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<xilinx::air::airDialect>();
  }

private:
};

void AIRPipelineReducePass::runOnOperation() {
  auto func = getOperation();
  auto ctx = func.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<PipelineReducePattern>(ctx, linalg::LinalgTilingOptions());
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));
}

class AIRFuseParallelHerdLaunchPass : public xilinx::air::AIRFuseParallelHerdLaunchPassBase<AIRFuseParallelHerdLaunchPass> {

public:
  AIRFuseParallelHerdLaunchPass() = default;
  AIRFuseParallelHerdLaunchPass(const AIRFuseParallelHerdLaunchPass &pass){};

  void runOnOperation() override;

private:
};

void AIRFuseParallelHerdLaunchPass::runOnOperation() {

  auto module = getOperation();
  auto ctx = module.getContext();

  xilinx::air::HerdLaunchOp launchOp = nullptr;
  scf::ParallelOp parOp = nullptr;

  module.walk([&](xilinx::air::HerdLaunchOp launch) {

    // launch must be enclosed by scf.parallel
    parOp = launch->getParentOfType<scf::ParallelOp>();
    if (!parOp)
      return;

    // launch must be at the top level of the scf.parallel
    if (parOp.getBody() != launch->getBlock())
      return;

    // if the herd launch is size 1 in one dimension
    // and the herd launch is enclosed by a 1-d scf.parallel
    // then we try to fuse the scf.parallel onto the herd launch

    launchOp = launch;
  });

  if (!launchOp || !parOp)
    return;

  OpBuilder b(parOp);

  xilinx::air::HerdDim2 dims = {launchOp.getHerdSizeOperands().x, parOp.getUpperBound()[0]};
  SmallVector<Value, 8> args;
  SmallVector<Value, 4> constants;
  llvm::SetVector<Value> region_args;

  getUsedValuesDefinedAbove(parOp.getRegion(), region_args);
  for (Value v : region_args) {
    if (v.getDefiningOp() && isa<arith::ConstantOp>(v.getDefiningOp()))
      constants.push_back(v);
    else
      args.push_back(v);
  }

  auto newLaunchOp = b.create<xilinx::air::HerdLaunchOp>(parOp.getLoc(), dims, args);

  BlockAndValueMapping remap;
  remap.map(parOp.getInductionVars()[0], newLaunchOp.getHerdSize().x);

  b.setInsertionPointToStart(&newLaunchOp.body().front());

  for (auto &o : *parOp.getBody()) {
    if (isa<xilinx::air::HerdLaunchOp>(o)) {
      int idx = 0;
      remap.map(launchOp.getHerdSize().x, launchOp.getHerdSizeOperands().x);
      remap.map(launchOp.getHerdSize().y, launchOp.getHerdSizeOperands().y);
      remap.map(launchOp.getTileIds().x, newLaunchOp.getTileIds().x);
      remap.map(launchOp.getTileIds().y, launchOp.getHerdSizeOperands().y);
      for (auto &a : launchOp.getKernelArguments()) {
        auto v = launchOp.getKernelOperand(idx++);
        remap.map(a, remap.lookupOrDefault(v));
      }
      for (auto &ho : launchOp.body().front()) {
        if (isa<xilinx::air::HerdTerminatorOp>(ho))
          continue;
        b.clone(ho, remap);
      }
    } else if (isa<scf::YieldOp>(o)) {
      continue;
    } else {
      b.clone(o, remap);
    }
  }
  b.create<xilinx::air::HerdTerminatorOp>(parOp.getLoc());

  b.setInsertionPointToStart(&newLaunchOp.body().front());
  for (auto c : constants) {
    replaceAllUsesInRegionWith(
          c, b.clone(*c.getDefiningOp())->getResult(0),
          newLaunchOp.getRegion());
  }


  int idx = 0;
  auto kernel_args = newLaunchOp.getKernelArguments();
  for (Value v : args)
    replaceAllUsesInRegionWith(v, kernel_args[idx++], newLaunchOp.getRegion());

//  newLaunchOp.dump();
  parOp.erase();
}

} // anonymous namespace

namespace xilinx {
namespace air {

std::unique_ptr<Pass> createAIRExamplePass() {
  return std::make_unique<AIRExamplePass>();
}

std::unique_ptr<Pass> createAIRSpecializeDma() {
  return std::make_unique<AIRSpecializeDma>();
}

std::unique_ptr<Pass> createAIRPromoteUniformL1Dma() {
  return std::make_unique<AIRPromoteUniformL1Dma>();
}

std::unique_ptr<Pass> createAIRLinalgNamePass() {
  return std::make_unique<AIRLinalgNamePass>();
}

std::unique_ptr<Pass> createAIRRemoveLinalgNamePass() {
  return std::make_unique<AIRRemoveLinalgNamePass>();
}

std::unique_ptr<Pass> createAIRPipelineReducePass() {
  return std::make_unique<AIRPipelineReducePass>();
}

std::unique_ptr<Pass> createAIRFuseParallelHerdLaunchPass() {
  return std::make_unique<AIRFuseParallelHerdLaunchPass>();
}


} // namespace air
} // namespace xilinx