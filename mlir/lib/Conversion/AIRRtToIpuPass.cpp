//===- AIRRtToIpuPass.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Conversion/AIRRtToIpuPass.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Transform/AIRDependencyScheduleOpt.h"
#include "air/Util/Util.h"

#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Support/MathExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "airrt-to-ipu-pass"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::airrt;

namespace {
#define GEN_PASS_DEF_AIRRTTOIPU
#include "air/Conversion/Passes.h.inc"

struct DmaToIpuPattern : public OpConversionPattern<DmaMemcpyNdOp> {
  using OpConversionPattern<DmaMemcpyNdOp>::OpConversionPattern;

  DmaToIpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<DmaMemcpyNdOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(DmaMemcpyNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto idOp = adaptor.getOperands().front();
    uint64_t idInt = 0;
    if (auto const_int = getConstantIntValue(idOp))
      idInt = *const_int;
    else
      return failure();
    uint64_t xInt = 0;
    if (auto const_int = getConstantIntValue(adaptor.getX()))
      xInt = *const_int;
    else
      return failure();
    uint64_t yInt = 0;
    if (auto const_int = getConstantIntValue(adaptor.getY()))
      yInt = *const_int;
    else
      return failure();

    SmallVector<Value> offsets;
    SmallVector<int64_t> staticOffsets;
    if (auto const_int = getConstantIntValue(adaptor.getOffset3()))
      staticOffsets.push_back(*const_int);
    else
      offsets.push_back(adaptor.getOffset3());
    if (auto const_int = getConstantIntValue(adaptor.getOffset2()))
      staticOffsets.push_back(*const_int);
    else
      offsets.push_back(adaptor.getOffset2());
    if (auto const_int = getConstantIntValue(adaptor.getOffset1()))
      staticOffsets.push_back(*const_int);
    else
      offsets.push_back(adaptor.getOffset1());
    if (auto const_int = getConstantIntValue(adaptor.getOffset0()))
      staticOffsets.push_back(*const_int);
    else
      offsets.push_back(adaptor.getOffset0());
    SmallVector<Value> sizes;
    SmallVector<int64_t> staticSizes;
    if (auto const_int = getConstantIntValue(adaptor.getLength3()))
      staticSizes.push_back(*const_int);
    else
      sizes.push_back(adaptor.getLength3());
    if (auto const_int = getConstantIntValue(adaptor.getLength2()))
      staticSizes.push_back(*const_int);
    else
      sizes.push_back(adaptor.getLength2());
    if (auto const_int = getConstantIntValue(adaptor.getLength1()))
      staticSizes.push_back(*const_int);
    else
      sizes.push_back(adaptor.getLength1());
    if (auto const_int = getConstantIntValue(adaptor.getLength0()))
      staticSizes.push_back(*const_int);
    else
      sizes.push_back(adaptor.getLength0());
    SmallVector<Value> strides;
    SmallVector<int64_t> staticStrides;
    if (auto const_int = getConstantIntValue(adaptor.getStride3()))
      staticStrides.push_back(*const_int);
    else
      strides.push_back(adaptor.getStride3());
    if (auto const_int = getConstantIntValue(adaptor.getStride2()))
      staticStrides.push_back(*const_int);
    else
      strides.push_back(adaptor.getStride2());
    if (auto const_int = getConstantIntValue(adaptor.getStride1()))
      staticStrides.push_back(*const_int);
    else
      strides.push_back(adaptor.getStride1());

    StringRef metadata;
    if (op->hasAttr("metadata"))
      metadata =
          op->getAttrOfType<mlir::FlatSymbolRefAttr>("metadata").getValue();
    else
      metadata =
          FlatSymbolRefAttr::get(op->getContext(),
                                 rewriter.getStringAttr("MetadataNotFound"))
              .getValue();

    rewriter.replaceOpWithNewOp<AIEX::IpuDmaMemcpyNdOp>(
        op, xInt, yInt, adaptor.getMemref(), offsets, sizes, strides,
        staticOffsets, staticSizes, staticStrides, metadata, idInt);

    return success();
  }
};

struct HerdLoadToIpuPattern : public OpConversionPattern<HerdLoadOp> {
  using OpConversionPattern<HerdLoadOp>::OpConversionPattern;

  HerdLoadToIpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<HerdLoadOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(HerdLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct SegmentLoadToIpuPattern : public OpConversionPattern<SegmentLoadOp> {
  using OpConversionPattern<SegmentLoadOp>::OpConversionPattern;

  SegmentLoadToIpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<SegmentLoadOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(SegmentLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct ModuleMetadataToIpuPattern
    : public OpConversionPattern<ModuleMetadataOp> {
  using OpConversionPattern<ModuleMetadataOp>::OpConversionPattern;

  ModuleMetadataToIpuPattern(MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<ModuleMetadataOp>(context, benefit) {}

  LogicalResult
  matchAndRewrite(ModuleMetadataOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

class L1AffineStoreOpConversion
    : public OpConversionPattern<affine::AffineStoreOp> {
public:
  using OpConversionPattern<affine::AffineStoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(affine::AffineStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = op.getMemref().getType().cast<MemRefType>();
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1)
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

class L1MemRefStoreOpConversion : public OpConversionPattern<memref::StoreOp> {
public:
  using OpConversionPattern<memref::StoreOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = op.getMemref().getType().cast<MemRefType>();
    if (memrefTy.getMemorySpaceAsInt() != (int)xilinx::air::MemorySpace::L1)
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

// This is a hack due to the short-term limited support with lowering host code.
// This should be removed in the future.
class HostMemRefCopyOpConversion : public OpConversionPattern<memref::CopyOp> {
public:
  using OpConversionPattern<memref::CopyOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Operation *> erased;
    if (auto alloc = op.getSource().getDefiningOp()) {
      op.getSource().replaceAllUsesWith(op.getTarget());
      erased.push_back(alloc);
    } else if (auto alloc = op.getTarget().getDefiningOp()) {
      op.getTarget().replaceAllUsesWith(op.getSource());
      erased.push_back(alloc);
    }
    for (auto o : erased)
      rewriter.eraseOp(o);
    rewriter.eraseOp(op);
    return success();
  }
};

AIE::DeviceOp getDeviceForSegmentLoad(SegmentLoadOp s) {
  auto module = s->getParentOfType<ModuleOp>();

  // Use the airrt metadata to lookup the segment associated with each head
  // load operation.
  for (auto d : module.getOps<AIE::DeviceOp>()) {
    if (s.getSymName() ==
        d->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
      return d;
  }
  return nullptr;
}

// Splits an Affine for loop into two for loops, by hoisting target operations
// in for loop to a new for loop located at the same scope.
void hoistTargetOpsToNewAffineFor(OpBuilder builder, affine::AffineForOp for_op,
                                  SmallVector<Operation *> target_ops) {
  // Get loop nest
  SmallVector<affine::AffineForOp> for_loops;
  affine::AffineForOp parent_for =
      target_ops[0]->getParentOfType<affine::AffineForOp>();
  while (parent_for != for_op) {
    for_loops.push_back(parent_for);
    parent_for = parent_for->getParentOfType<affine::AffineForOp>();
  }
  for_loops.push_back(for_op);

  // Clone loop nest
  builder.setInsertionPoint(for_op);
  IRMapping remap;
  for (int i = for_loops.size() - 1; i >= 0; i--) {
    auto new_for_op = builder.create<affine::AffineForOp>(
        for_loops[i].getLoc(), for_loops[i].getConstantLowerBound(),
        for_loops[i].getConstantUpperBound());
    remap.map(for_loops[i].getInductionVar(), new_for_op.getInductionVar());
    builder.setInsertionPointToStart(new_for_op.getBody());
    // Bottom of rabbit hole
    if (i == 0) {
      for (auto op : target_ops) {
        builder.clone(*op, remap);
      }
    }
  }
}

template <typename T> void push_back_if_unique(SmallVector<T> &vec, T entry) {
  if (std::find(vec.begin(), vec.end(), entry) == vec.end()) {
    vec.push_back(entry);
  }
}

void identifyTargetAffineForAndOps(
    func::FuncOp f, SmallVector<SmallVector<Operation *>> &target_ops_vec) {
  // Identify the target for loops and their target child ops
  int index = 0;
  for (auto for_op : f.getBody().getOps<affine::AffineForOp>()) {
    for_op.walk([&](airrt::DmaMemcpyNdOp memcpyOp) {
      // Get for_op's immediate child op
      target_ops_vec.push_back(SmallVector<Operation *>{});
      // Check if any operand's defining ops needs to be hoisted together.
      SmallVector<Operation *> oper_def_ops;
      xilinx::air::getDefiningOpsToOperands(memcpyOp.getOperation(),
                                            oper_def_ops);
      for (auto o : oper_def_ops) {
        if (o->getParentOp() == memcpyOp->getParentOp()) {
          push_back_if_unique<Operation *>(target_ops_vec[index], o);
        }
      }
      push_back_if_unique<Operation *>(target_ops_vec[index],
                                       memcpyOp.getOperation());
      index++;
    });
  }
}

void isolateAIRRtDmaLoopNests(ModuleOp module) {
  // Identify affine.for ops and target child ops for hoisting.
  SmallVector<SmallVector<Operation *>> target_ops_vec;
  SmallVector<func::FuncOp> funcOps;
  module.walk([&](func::FuncOp f) { funcOps.push_back(f); });
  for (auto f : funcOps) {
    f.walk(
        [&](affine::AffineForOp afo) { (void)promoteIfSingleIteration(afo); });
    identifyTargetAffineForAndOps(f, target_ops_vec);
  }

  // Hoist ops out of each scf.for.
  SmallVector<Operation *> erased;
  for (auto vec : target_ops_vec) {
    affine::AffineForOp loop_nest_head =
        vec[0]->getParentOfType<affine::AffineForOp>();
    while (!isa<func::FuncOp>(loop_nest_head->getParentOp())) {
      loop_nest_head = loop_nest_head->getParentOfType<affine::AffineForOp>();
    }
    OpBuilder builder(loop_nest_head);
    hoistTargetOpsToNewAffineFor(builder, loop_nest_head, vec);
    push_back_if_unique<Operation *>(erased, loop_nest_head.getOperation());
  }
  for (auto o : erased)
    o->erase();
}

LogicalResult
specializeAffineForInAIRRtDmaWrapAndStride(OpBuilder builder,
                                           affine::AffineForOp for_op) {
  auto loc = for_op->getLoc();
  auto ctx = for_op->getContext();

  // Declaration of constants
  auto i64Ty = builder.getI64Type();
  auto i64_zero =
      builder.create<arith::ConstantOp>(loc, i64Ty, IntegerAttr::get(i64Ty, 0));
  auto i64_one =
      builder.create<arith::ConstantOp>(loc, i64Ty, IntegerAttr::get(i64Ty, 1));

  // Check if the loop is the outermost loop in a perfect loop nest
  auto hasNElements = [](Block *block, unsigned N) {
    auto op_ptr = block->begin();
    for (unsigned i = 0; i < N; i++)
      op_ptr = std::next(op_ptr);
    return op_ptr != block->end() && &*op_ptr == &block->back();
  };
  if (auto parent_for = dyn_cast<affine::AffineForOp>(for_op->getParentOp()))
    if (hasNElements(parent_for.getBody(), 1))
      return failure();

  // Check if the loop nest contains exactly one memcpy op
  SmallVector<airrt::DmaMemcpyNdOp> memcpy_ops;
  for_op.getBody()->walk(
      [&](airrt::DmaMemcpyNdOp putget) { memcpy_ops.push_back(putget); });
  if (memcpy_ops.size() != 1)
    return failure();

  // Fold for loops into channel op's wrap and stride fields
  SmallVector<affine::AffineForOp> for_loops;
  Operation *parent = memcpy_ops[0].getOperation();
  while (parent != for_op.getOperation()) {
    parent = parent->getParentOp();
    if (auto for_op_in_nest = dyn_cast<affine::AffineForOp>(parent))
      for_loops.push_back(for_op_in_nest);
  }

  auto memref = memcpy_ops[0]->getOperand(3);
  auto memref_shape = xilinx::air::getTensorShape(memref.getType());
  auto oper_begin = memcpy_ops[0].getOperands().begin();
  SmallVector<Value> offsets(oper_begin + (8 - memref_shape.size()),
                             oper_begin + 8);
  SmallVector<Value> wraps(oper_begin + (12 - memref_shape.size()),
                           oper_begin + 12);
  SmallVector<Value> strides(oper_begin + (16 - memref_shape.size()),
                             oper_begin + 15);
  // Stride field implicit last element one
  strides.push_back(i64_one);

  xilinx::air::foldForLoopNestAsExtendedSizesAndStrides(
      builder, for_op.getOperation(), memcpy_ops[0].getOperation(), offsets,
      wraps, strides, memcpy_ops[0]->getOperand(3));

  // Stride field implicit last element one
  strides.pop_back();
  while (offsets.size() < 4) {
    offsets.insert(offsets.begin(), i64_zero);
  }
  while (wraps.size() < 4) {
    wraps.insert(wraps.begin(), i64_one);
  }
  while (strides.size() < 3) {
    strides.insert(strides.begin(), i64_zero);
  }
  assert(offsets.size() == 4);
  assert(wraps.size() == 4);
  assert(strides.size() == 3);

  // Stride = 0 means repeat that dimension. If highest dimension (dim 0) is not
  // used, then move the repeat dimension to dim 0, which is the only dim with
  // repeat capability. Else, NYI. Fall back to unrolling BDs.
  for (unsigned i = 1; i < strides.size(); i++) {
    if (mlir::getConstantIntValue(wraps[i]) &&
        mlir::getConstantIntValue(strides[i])) {
      if (*mlir::getConstantIntValue(wraps[i]) > 1 &&
          !*mlir::getConstantIntValue(strides[i])) {
        // This is a repeat dimension.
        if (mlir::getConstantIntValue(wraps[0]) &&
            *mlir::getConstantIntValue(wraps[0]) == 1) {
          // Move the repeat dimension i to dimension 0.
          auto tmp = wraps[0];
          wraps[0] = wraps[i];
          wraps[i] = tmp;
          tmp = strides[0];
          strides[0] = strides[i];
          strides[i] = tmp;
        } else
          return failure();
      }
    }
  }

  // Create new airrt.dma_memcpy_nd
  SmallVector<Type, 1> tys;
  if (memcpy_ops[0]->getNumResults())
    tys.push_back(airrt::EventType::get(ctx));

  SmallVector<Value, 16> opers;
  auto old_opers = memcpy_ops[0]->getOperands();
  opers.insert(opers.end(), old_opers.begin(), old_opers.begin() + 4);
  opers[1] =
      builder.create<arith::ConstantOp>(loc, i64Ty, IntegerAttr::get(i64Ty, 0));
  opers[2] =
      builder.create<arith::ConstantOp>(loc, i64Ty, IntegerAttr::get(i64Ty, 0));
  opers.insert(opers.end(), offsets.begin(), offsets.end());
  opers.insert(opers.end(), wraps.begin(), wraps.end());
  opers.insert(opers.end(), strides.begin(), strides.end());

  // index_cast
  for (unsigned i = 0; i < opers.size(); i++) {
    if (opers[i].getDefiningOp() &&
        isa<arith::ConstantIndexOp>(opers[i].getDefiningOp())) {
      opers[i] = builder.create<arith::IndexCastOp>(
          loc, IntegerType::get(ctx, 64), opers[i]);
    } else if (opers[i].getDefiningOp() &&
               isa<arith::IndexCastOp>(opers[i].getDefiningOp())) {
      opers[i] = builder.clone(*opers[i].getDefiningOp())->getResult(0);
    }
  }
  auto new_dma = builder.create<airrt::DmaMemcpyNdOp>(loc, tys, opers);
  // If dma op contains shim dma alloc metadata, then inherit this information
  if (memcpy_ops[0]->hasAttr("metadata"))
    new_dma->setAttr(
        "metadata",
        memcpy_ops[0]->getAttrOfType<mlir::SymbolRefAttr>("metadata"));

  return success();
}

void specializeAffineForInAIRRtDmaWrapAndStride(ModuleOp module) {
  SmallVector<func::FuncOp> funcOps;
  module.walk([&](func::FuncOp f) { funcOps.push_back(f); });
  SmallVector<Operation *> erased;
  SmallVector<affine::AffineForOp> unroll_outer_dim;
  auto specialzeAllAffineFors =
      [&](SmallVector<func::FuncOp> funcOps, SmallVector<Operation *> &erased,
          SmallVector<affine::AffineForOp> &unroll_outer_dim) {
        for (auto f : funcOps) {
          for (auto for_op : f.getOps<affine::AffineForOp>()) {
            OpBuilder builder(for_op);
            if (specializeAffineForInAIRRtDmaWrapAndStride(builder, for_op)
                    .succeeded())
              erased.push_back(for_op);
            else {
              // Wait list to be unrolled one outer dimension, and then try
              // specializing the wraps and strides again.
              unroll_outer_dim.push_back(for_op);
            }
          }
        }
      };
  specialzeAllAffineFors(funcOps, erased, unroll_outer_dim);
  for (auto o : erased)
    o->erase();
  erased.clear();
  // In AIE2 BD, there is one single dimension capable of repeating. If
  // unroll_outer_dim isn't empty, then unroll the existing dimension in the
  // repeat dim and repopulate that dimension with a true repeat dimension.
  for (auto o : unroll_outer_dim) {
    int64_t tripCount =
        mlir::ceilDiv(o.getConstantUpperBound() - o.getConstantLowerBound(),
                      o.getStepAsInt());
    (void)loopUnrollByFactor(o, tripCount);
  }
  specialzeAllAffineFors(funcOps, erased, unroll_outer_dim);
  for (auto o : erased)
    o->erase();
}

struct AIRRtToIpuPass : public impl::AIRRtToIpuBase<AIRRtToIpuPass> {
  void runOnOperation() override {

    ModuleOp module = getOperation();

    // Move func op to the end of device op's body
    moveFuncOpToEndOfDeviceOp(module);

    // Purge all wait all ops
    purgeWaitAlls(module);

    // Purge airrt.dma x and y fields, as they are obsolete for AIE2.
    purgeAIRRtDmaXAndY(module);

    // Separate affine for loop nest into loop nests each containing one dma
    // memcpy op
    isolateAIRRtDmaLoopNests(module);

    // Simplify affine apply ops
    auto ctx = &getContext();
    RewritePatternSet canoPatterns_0(ctx);
    xilinx::air::populateAIRLoopIndexCanonicalizationPatterns(canoPatterns_0);
    (void)applyPatternsAndFoldGreedily(module, std::move(canoPatterns_0));

    // Specialize affine for loop nest into wraps and strides
    specializeAffineForInAIRRtDmaWrapAndStride(module);
    unrollAffineFors(module);

    // Simplify arith ops (from airrt)
    RewritePatternSet canoPatterns_1(ctx);
    arith::IndexCastOp::getCanonicalizationPatterns(canoPatterns_1, ctx);
    (void)applyPatternsAndFoldGreedily(module, std::move(canoPatterns_1));
    unrollSCFFors(module);

    // Purge dma ops' async tokens
    purgeDmaAsyncTokens(module);

    ConversionTarget target(getContext());
    target.addIllegalDialect<AIRRtDialect>();
    target.addLegalDialect<arith::ArithDialect, AIEX::AIEXDialect>();

    target.addDynamicallyLegalOp<affine::AffineStoreOp>(
        [&](affine::AffineStoreOp op) {
          if (op->getParentOfType<AIE::CoreOp>())
            return true;
          return (op.getMemref()
                      .getType()
                      .cast<MemRefType>()
                      .getMemorySpaceAsInt() !=
                  (int)xilinx::air::MemorySpace::L1);
        });
    target.addDynamicallyLegalOp<memref::StoreOp>([&](memref::StoreOp op) {
      if (op->getParentOfType<AIE::CoreOp>())
        return true;
      return (
          op.getMemref().getType().cast<MemRefType>().getMemorySpaceAsInt() !=
          (int)xilinx::air::MemorySpace::L1);
    });
    target.addDynamicallyLegalOp<memref::CopyOp>([&](memref::CopyOp op) {
      auto f = op->getParentOfType<func::FuncOp>();
      if (f) {
        for (auto arg : f.getArguments()) {
          if (op.getTarget() == arg)
            return false;
          else if (op.getSource() == arg)
            return false;
        }
      }
      return true;
    });
    RewritePatternSet patterns(ctx);
    patterns
        .insert<DmaToIpuPattern, HerdLoadToIpuPattern, SegmentLoadToIpuPattern,
                ModuleMetadataToIpuPattern, L1MemRefStoreOpConversion,
                L1AffineStoreOpConversion, HostMemRefCopyOpConversion>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();

    // Simplify arith ops (from airrt-to-ipu)
    RewritePatternSet canoPatterns_2(ctx);
    arith::IndexCastOp::getCanonicalizationPatterns(canoPatterns_2, ctx);
    (void)applyPatternsAndFoldGreedily(module, std::move(canoPatterns_2));

    // Unroll any affine for loops
    unrollAffineFors(module);

    // Buffer ipu.dma_memcpy_nd memref to function's argument list.
    BufferMemrefToFuncArgs(module);

    // Insert sync op after copying data out to host
    insertIpuSyncOpForResults(module);

    // Renumber ipu dma ops
    renumberIpuDmaOps(module.getBody());
  }

  void moveFuncOpToEndOfDeviceOp(ModuleOp module) {
    // Move func op to the end of device op's body
    SmallVector<SegmentLoadOp> segs;
    module.walk([&](SegmentLoadOp s) { segs.push_back(s); });
    for (auto s : segs) {
      auto f = s->getParentOfType<func::FuncOp>();
      auto d = getDeviceForSegmentLoad(s);
      if (!f || !d)
        continue;
      f->moveAfter(&d.getBody()->back());
    }
  }

  void purgeDmaAsyncTokens(ModuleOp module) {
    SmallVector<DmaMemcpyNdOp> dmas;
    module.walk([&](DmaMemcpyNdOp dma) { dmas.push_back(dma); });
    for (auto dma : dmas) {
      if (dma->getNumResults()) {
        OpBuilder buider(dma);
        SmallVector<Type, 1> tys = {};
        auto newOp = buider.create<DmaMemcpyNdOp>(dma->getLoc(), tys,
                                                  dma->getOperands());
        if (dma->hasAttr("metadata"))
          newOp->setAttr("metadata",
                         dma->getAttrOfType<mlir::SymbolRefAttr>("metadata"));
        dma->erase();
      }
    }
  }

  void purgeWaitAlls(ModuleOp module) {
    SmallVector<WaitAllOp> waits;
    module.walk([&](WaitAllOp w) { waits.push_back(w); });
    for (auto w : waits) {
      w->eraseOperands(0, w->getNumOperands());
    }
    for (auto w : waits) {
      w.erase();
    }
  }

  void purgeAIRRtDmaXAndY(ModuleOp module) {
    SmallVector<airrt::DmaMemcpyNdOp> dmas;
    module.walk([&](airrt::DmaMemcpyNdOp dma) { dmas.push_back(dma); });
    for (auto dma : dmas) {
      for (unsigned idx = 1; idx <= 2; idx++) {
        auto x_def_op = dma->getOperand(idx).getDefiningOp();
        if (x_def_op && !isa<arith::ConstantOp>(x_def_op)) {
          OpBuilder builder(x_def_op);
          auto i64Ty = builder.getI64Type();
          dma->setOperand(
              idx, builder.create<arith::ConstantOp>(
                       dma->getLoc(), i64Ty, IntegerAttr::get(i64Ty, 0)));
        }
      }
    }
  }

  void unrollAffineFors(ModuleOp module) {
    // Taking into account for loop nests
    SmallVector<affine::AffineForOp> afos;
    module.walk([&](mlir::func::FuncOp f) {
      for (auto op : f.getOps<affine::AffineForOp>()) {
        afos.push_back(op);
      }
      for (auto op : afos) {
        unrollAffineFors(op);
        // Renumber unrolled memcpy ops
        int unrolled_op_id = 0;
        f.walk([&](airrt::DmaMemcpyNdOp dma) {
          if (dma->hasAttr("unrolled")) {
            auto metadata =
                dma->getAttrOfType<mlir::FlatSymbolRefAttr>("metadata")
                    .getValue()
                    .str();
            // Start from unrolled_op_id 1
            if (unrolled_op_id)
              dma->setAttr("metadata", FlatSymbolRefAttr::get(
                                           dma->getContext(),
                                           metadata + "_" +
                                               std::to_string(unrolled_op_id)));
            unrolled_op_id++;
            dma->removeAttr("unrolled");
          }
        });
      }
    });
  }

  void unrollAffineFors(affine::AffineForOp affine_for_op) {
    SmallVector<affine::AffineForOp> afos;
    affine_for_op.walk([&](affine::AffineForOp afo) { afos.push_back(afo); });
    for (auto afo : afos) {
      int64_t tripCount = mlir::ceilDiv(afo.getConstantUpperBound() -
                                            afo.getConstantLowerBound(),
                                        afo.getStepAsInt());
      (void)loopUnrollByFactor(afo, tripCount);
    }
  }

  void getOperandsFromAIRRtDma(airrt::DmaMemcpyNdOp op,
                               SmallVector<uint32_t, 4> &offsets,
                               SmallVector<uint32_t, 4> &lengths,
                               SmallVector<uint32_t, 3> &strides) {
    if (auto c = op.getOffset0().getDefiningOp<arith::ConstantIntOp>())
      offsets[0] = static_cast<uint32_t>(c.value());
    if (auto c = op.getOffset1().getDefiningOp<arith::ConstantIntOp>())
      offsets[1] = static_cast<uint32_t>(c.value());
    if (auto c = op.getOffset2().getDefiningOp<arith::ConstantIntOp>())
      offsets[2] = static_cast<uint32_t>(c.value());
    if (auto c = op.getOffset3().getDefiningOp<arith::ConstantIntOp>())
      offsets[3] = static_cast<uint32_t>(c.value());
    if (auto c = op.getLength0().getDefiningOp<arith::ConstantIntOp>())
      lengths[0] = static_cast<uint32_t>(c.value());
    if (auto c = op.getLength1().getDefiningOp<arith::ConstantIntOp>())
      lengths[1] = static_cast<uint32_t>(c.value());
    if (auto c = op.getLength2().getDefiningOp<arith::ConstantIntOp>())
      lengths[2] = static_cast<uint32_t>(c.value());
    if (auto c = op.getLength3().getDefiningOp<arith::ConstantIntOp>())
      lengths[3] = static_cast<uint32_t>(c.value());
    if (auto c = op.getStride1().getDefiningOp<arith::ConstantIntOp>())
      strides[0] = static_cast<uint32_t>(c.value());
    if (auto c = op.getStride2().getDefiningOp<arith::ConstantIntOp>())
      strides[1] = static_cast<uint32_t>(c.value());
    if (auto c = op.getStride3().getDefiningOp<arith::ConstantIntOp>())
      strides[2] = static_cast<uint32_t>(c.value());
  }

  void unrollSCFFors(ModuleOp module) {
    SmallVector<scf::ForOp> scf_fors;
    module.walk([&](mlir::func::FuncOp f) {
      f.walk([&](scf::ForOp for_op) { scf_fors.push_back(for_op); });
    });
    for (auto for_op : scf_fors) {
      std::optional<int64_t> lbCstOp =
          mlir::getConstantIntValue(for_op.getLowerBound());
      std::optional<int64_t> ubCstOp =
          mlir::getConstantIntValue(for_op.getUpperBound());
      std::optional<int64_t> stepCstOp =
          mlir::getConstantIntValue(for_op.getStep());
      if (lbCstOp && ubCstOp && stepCstOp) {
        int64_t tripCount =
            mlir::ceilDiv(ubCstOp.value() - lbCstOp.value(), stepCstOp.value());
        (void)loopUnrollByFactor(for_op, tripCount);
      }
    }
  }

  std::optional<AIE::ShimDMAAllocationOp>
  getAllocOpForSymbol(AIE::DeviceOp dev, StringRef sym_name) {
    auto sym = dev.lookupSymbol(sym_name);
    if (!sym)
      return std::nullopt;

    auto uses = SymbolTable::getSymbolUses(sym, dev);
    for (auto use : *uses)
      if (auto infoOp = dyn_cast<AIE::ShimDMAAllocationOp>(use.getUser()))
        return infoOp;

    return std::nullopt;
  }

  void insertIpuSyncOpForResults(ModuleOp module) {
    module.walk([&](mlir::func::FuncOp f) {
      SmallVector<AIEX::IpuDmaMemcpyNdOp> dmas;
      f.walk([&](AIEX::IpuDmaMemcpyNdOp dma) { dmas.push_back(dma); });
      auto d = f->getParentOfType<AIE::DeviceOp>();
      for (auto dma : dmas) {
        if (auto infoOp = getAllocOpForSymbol(d, dma.getMetadata())) {
          if (infoOp->getChannelDir() == AIE::DMAChannelDir::S2MM) {
            // Found dma op copying results to host
            OpBuilder builder(dma);
            auto col = builder.getI32IntegerAttr(infoOp->getCol());
            auto row = builder.getI32IntegerAttr(0);
            auto dir = builder.getI32IntegerAttr(0);
            auto chan = builder.getI32IntegerAttr(infoOp->getChannelIndex());
            auto col_num = builder.getI32IntegerAttr(1);
            auto row_num = builder.getI32IntegerAttr(1);
            builder.setInsertionPointAfter(dma);
            builder.create<AIEX::IpuSyncOp>(dma->getLoc(), col, row, dir, chan,
                                            col_num, row_num);
          }
        }
      }
    });
  }

  void renumberIpuDmaOps(Block *blk) {
    unsigned id = 0;
    blk->walk([&](Operation *op) {
      if (isa<AIEX::IpuDmaMemcpyNdOp>(op)) {
        op->setAttr("id",
                    mlir::IntegerAttr::get(
                        mlir::IntegerType::get(op->getContext(), 64), id++));
      } else if (isa<AIEX::IpuSyncOp>(op))
        id = 0;
    });
  }

  // Buffers ipu.dma_memcpy_op memref as function argument
  void BufferMemrefToFuncArgs(ModuleOp module) {
    module.walk([&](mlir::func::FuncOp f) { BufferMemrefToFuncArgs(f); });
  }
  void BufferMemrefToFuncArgs(func::FuncOp funcOp) {
    if (!funcOp)
      return;

    // Collect illegal dma ops whose memrefs are not in function's arguments.
    SmallVector<Type, 6> memrefTypes;
    SmallVector<Value, 6> memrefs;
    funcOp.walk([&](AIEX::IpuDmaMemcpyNdOp dma) {
      // Value test_val = dyn_cast<Value>(dma.getMemref());
      // auto test_val1 = dma.getMemref();
      if (std::find(funcOp.getArguments().begin(), funcOp.getArguments().end(),
                    dma.getMemref()) == funcOp.getArguments().end()) {
        // push back if unique
        if (std::find(memrefs.begin(), memrefs.end(), dma.getMemref()) ==
            memrefs.end()) {
          memrefs.push_back(dma.getMemref());
          memrefTypes.push_back(dma.getMemref().getType());
        }
      }
    });

    // Append memref to function's arguments.
    auto functionType = funcOp.getFunctionType();
    auto newArgTypes = llvm::to_vector<6>(
        llvm::concat<const Type>(functionType.getInputs(), memrefTypes));
    auto newFunctionType = FunctionType::get(funcOp.getContext(), newArgTypes,
                                             functionType.getResults());
    funcOp.setType(newFunctionType);

    // Add the new arguments to the entry block if the function is not external.
    if (!funcOp.isExternal()) {
      Location loc = funcOp.getLoc();
      for (Value v : memrefs) {
        auto newArg = funcOp.front().addArgument(v.getType(), loc);
        v.replaceAllUsesWith(newArg);
      }
    }
  }
};

} // namespace

namespace xilinx {
namespace airrt {

std::unique_ptr<mlir::Pass> createAIRRtToIpuPass() {
  return std::make_unique<AIRRtToIpuPass>();
}

} // namespace airrt
} // namespace xilinx
