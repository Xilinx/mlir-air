// (c) Copyright 2019 Xilinx Inc. All Rights Reserved.

#include "npcomp/Dialect/ATen/IR/ATenDialect.h"
#include "AIRDialect.h"
#include "AffineToAIRPass.h"

#include "mlir/Analysis/Utils.h"
#include "mlir/Dialect/Affine/EDSC/Builders.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/SCF/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Builders.h"
#include "mlir/Dialect/StandardOps/EDSC/Intrinsics.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/MemRef/EDSC/Intrinsics.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/EDSC/Builders.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <sstream>

using namespace mlir;
using namespace xilinx;

#define DEBUG_TYPE "affine-to-air"

namespace {

#include "AffineToAIR.cpp.inc"

static uint64_t DmaMemcpyOpID;

/// Extract int64_t values from the assumed ArrayAttr of IntegerAttr.
static SmallVector<int64_t, 4> extractFromI64ArrayAttr(Attribute attr) {
  return llvm::to_vector<4>(
      llvm::map_range(attr.cast<ArrayAttr>(), [](Attribute a) -> int64_t {
        return a.cast<IntegerAttr>().getInt();
      }));
}

class LinalgCopyToAIRDmaConversion : public OpRewritePattern<linalg::CopyOp> {
  using OpRewritePattern<linalg::CopyOp>::OpRewritePattern;
  LogicalResult matchAndRewrite(linalg::CopyOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto src = op.input();
    auto dst = op.output();
    auto src_type = src.getType().cast<MemRefType>();
    auto dims = src_type.getShape().size();

    SmallVector<Value, 4> src_indices;
    SmallVector<Value, 4> dst_indices;

    if (dims == 2) {
      Value zero = rewriter.create<ConstantIndexOp>(loc,0);
      Value stride = zero;
      Value elem_per_stride = zero;
      SmallVector<Value,1> deps;
      SmallVector<Type,1> tys;
      if (auto alloc = src.getDefiningOp<memref::AllocOp>()) {
        src_indices.push_back(zero);
        src_indices.push_back(zero);
        elem_per_stride = rewriter.create<ConstantIndexOp>(loc,
                            alloc.getType().getShape()[0]);
      }
      else if (auto subview = src.getDefiningOp<memref::SubViewOp>()) {
        auto offsets = subview.offsets().begin();
        auto static_offsets = extractFromI64ArrayAttr(subview.static_offsets());
        for (auto o : static_offsets) {
          if (o >= 0)
            src_indices.push_back(rewriter.create<ConstantIndexOp>(loc, o));
          else
            src_indices.push_back(*offsets++);
        }
        src = subview.source();
        stride = rewriter.create<ConstantIndexOp>(loc,
                   src.getType().cast<MemRefType>().getShape()[1]);
      }
      else
        return failure();

      if (auto alloc = dst.getDefiningOp<memref::AllocOp>()) {
        dst_indices.push_back(zero);
        dst_indices.push_back(zero);
        elem_per_stride = rewriter.create<ConstantIndexOp>(loc,
                            alloc.getType().getShape()[1]);
      }
      else if (auto subview = dst.getDefiningOp<memref::SubViewOp>()) {
        auto offsets = subview.offsets().begin();
        auto static_offsets = extractFromI64ArrayAttr(subview.static_offsets());
        for (auto o : static_offsets) {
          if (o >= 0)
            dst_indices.push_back(rewriter.create<ConstantIndexOp>(loc, o));
          else
            dst_indices.push_back(*offsets++);
        }
        dst = subview.source();
        stride = rewriter.create<ConstantIndexOp>(loc,
                   dst.getType().cast<MemRefType>().getShape()[1]);
      }
      else
        return failure();
  
      auto dma = rewriter.create<air::DmaMemcpy2dOp>(loc, tys,
                                                    deps, dst, src,
                                                    dst_indices[0], dst_indices[1],
                                                    src_indices[0], src_indices[1],
                                                    rewriter.create<ConstantIndexOp>(
                                                      loc,
                                                      src_type.getNumElements()),
                                                    stride, elem_per_stride);
      dma->setAttr("id",
                   mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32),
                                          ++DmaMemcpyOpID));
    }
    else {
      // assert(0 && "dims != 2");
      return failure();
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class AffineCopyToAIRDMAConversion : public ConversionPattern {
public:
  explicit AffineCopyToAIRDMAConversion(MLIRContext *context)
      : ConversionPattern(AffineDmaStartOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value > operands,
                  ConversionPatternRewriter &rewriter) const override
  {
    auto affine_dma_start = cast<AffineDmaStartOp>(op);

    auto src = affine_dma_start.getSrcMemRef();
    auto src_indices = affine_dma_start.getSrcIndices();

    auto dst = affine_dma_start.getDstMemRef();
    auto dst_indices = affine_dma_start.getDstIndices();

    SmallVector<AffineApplyOp, 4> src_applies;
    SmallVector<AffineApplyOp, 4> dst_applies;
    unsigned dims = affine_dma_start.getDstMap().getNumResults();
    for (unsigned i=0; i<dims; i++) {
      auto src_submap = affine_dma_start.getSrcMap().getSubMap({i});
      auto dst_submap = affine_dma_start.getDstMap().getSubMap({i});
      src_applies.push_back(
        rewriter.create<AffineApplyOp>(op->getLoc(), src_submap, src_indices));
      dst_applies.push_back(
        rewriter.create<AffineApplyOp>(op->getLoc(), dst_submap, dst_indices));
    }

    SmallVector<Type,1> tys;
    SmallVector<Value,1> deps;
    Operation *dma = nullptr;
    Value stride;
    Value elem_per_stride;
    if (affine_dma_start.isStrided()) {
      stride = affine_dma_start.getStride();
      elem_per_stride = affine_dma_start.getNumElementsPerStride();
    }
    else {
      stride = elem_per_stride = affine_dma_start.getNumElements();
    }
    if (dims == 1) {
      dma = rewriter.create<air::DmaMemcpyOp>(op->getLoc(), tys,
                                              deps, dst, src,
                                              dst_applies[0],
                                              src_applies[0],
                                              affine_dma_start.getNumElements());
    }
    else if (dims == 2) {
      dma = rewriter.create<air::DmaMemcpy2dOp>(op->getLoc(), tys,
                                                deps, dst, src,
                                                dst_applies[0], dst_applies[1],
                                                src_applies[0], src_applies[1],
                                                affine_dma_start.getNumElements(),
                                                stride, elem_per_stride);
    }
    else if (dims == 4) {
      dma = rewriter.create<air::DmaMemcpy4dOp>(op->getLoc(), tys,
                                                deps, dst, src,
                                                dst_applies[0], dst_applies[1], dst_applies[2], dst_applies[3],
                                                src_applies[0], src_applies[1], src_applies[2], src_applies[3],
                                                affine_dma_start.getNumElements(),
                                                stride, elem_per_stride);
    }
    else {
      llvm::outs() << "unsupported memcpy in affine-to-air";
      op->print(llvm::outs());
      return failure();
    }
    dma->setAttr("id",
                 mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 32),
                                        ++DmaMemcpyOpID));

    rewriter.eraseOp(op);
    return success();
  }
    // mlir::AffineLoadOp load = nullptr;
    // mlir::AffineStoreOp store = nullptr;

    // for (auto &o : afo.getLoopBody().getOps()) {
    //   if (isa<AffineLoadOp>(o)) {
    //     load = cast<AffineLoadOp>(o);
    //   }
    //   else if (isa<AffineStoreOp>(o)) {
    //     store = cast<AffineStoreOp>(o);
    //   }
    //   else if (isa<AffineYieldOp>(o)) {
    //   }
    //   else {
    //     llvm::outs() << "FAIL!\n";
    //     op->print(llvm::outs());
    //     o.print(llvm::outs());
    //     return failure();
    //   }
    // }

    // llvm::outs() << "HERE!\n";
    // op->print(llvm::outs());

    // if (!load || !store)
    //   return failure();

    // if (store.value() != load)
    //   return failure();

    // auto srcTy = load.memref().getType().cast<mlir::MemRefType>();
    // auto dstTy = store.memref().getType().cast<mlir::MemRefType>();

    // if (srcTy.getMemorySpace() == 0 && dstTy.getMemorySpace() == 1) {
    //   // ext -> L2
    //   // #map7 = affine_map<()[s0] -> (s0 + 32)>
    //   // affine.for %arg5 = %arg3 to #map7()[%arg3] {
    //   //   %0 = affine.load %arg1[%arg4, %arg5] : memref<256x256xf32>
    //   //   affine.store %0, %arg2[-%arg0 + %arg4, -%arg3 + %arg5] : memref<32x32xf32, 1>
    //   // }
    //   //air.shim_dma_memcpy(%src,  %dst,  %src_d1, %src_d0, %dst_d1,        %dst_d0, %num)
    //   //air.shim_dma_memcpy(%arg1, %arg2, %arg4,   %arg3,   -%arg0 + %arg4, 0,       32)
    //   llvm::outs() << "L3 to L2!\n";

    //   mlir::AffineMap lbm = afo.getLowerBoundMap();
    //   mlir::AffineMap ubm = afo.getUpperBoundMap();

    //   auto int32Ty = mlir::IntegerType::get(op->getContext(), 32);
    //   auto attr = mlir::IntegerAttr::get(int32Ty, 0);
    //   SmallVector<Attribute, 1> attrs{attr};
    //   SmallVector<Attribute, 2> ints;
    //   lbm.constantFold(attrs, ints);
    //   ubm.constantFold(attrs, ints);
    //   int64_t lower_bound = ints[0].cast<mlir::IntegerAttr>().getInt();
    //   int64_t upper_bound = ints[1].cast<mlir::IntegerAttr>().getInt();

    //   llvm::outs() << "LB: " << lower_bound << " UB: " << upper_bound << "\n";
    //   auto loc = op->getLoc();
    //   auto zero_const = rewriter.create<ConstantIndexOp>(loc, 0);
    //   auto upper_bound_const = rewriter.create<ConstantIndexOp>(loc, upper_bound);
    //   SmallVector<Value, 1> deps;
    //   SmallVector<Type, 1> rets;
    //   /*auto shim_dma_memcpy =*/ rewriter.create<xilinx::air::DmaMemcpy2d>(loc, rets, deps, load.memref(), store.memref(),
    //                                                                    load.indices()[0], afo.getLowerBoundOperands()[0],
    //                                                                    store.indices()[0], zero_const, upper_bound_const);
    //   // rewriter.eraseOp(load);
    //   // rewriter.eraseOp(store);
    //   rewriter.eraseOp(op);

    //   return success();
    // }
    // else if (srcTy.getMemorySpace() == 1 || dstTy.getMemorySpace() == 0) {
    //   // L2 -> ext
    // }
    // else {
    //   return failure();
    // }
    // return failure();
  //}
};

class AffineParToHerdLaunchConversion : public OpRewritePattern<AffineParallelOp> {
public:
  using OpRewritePattern<AffineParallelOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumDims() == 2) {
      auto loc = op.getLoc();
      auto ub0 = op.upperBoundsMap().getResult(0).cast<AffineConstantExpr>();
      auto ub1 = op.upperBoundsMap().getResult(1).cast<AffineConstantExpr>();
      SmallVector<Value, 4> args;
      SmallVector<Value, 4> constants;
      llvm::SetVector<Value> region_args;
      getUsedValuesDefinedAbove(op.getRegion(), region_args);
      for (Value v : region_args) {
        if (v.getDefiningOp() && isa<ConstantOp>(v.getDefiningOp()))
          constants.push_back(v);
        else
          args.push_back(v);
      }
      air::HerdDim2 dims{rewriter.create<ConstantIndexOp>(loc,ub0.getValue()),
                         rewriter.create<ConstantIndexOp>(loc,ub1.getValue())};
      auto launch = rewriter.create<air::HerdLaunchOp>(op.getLoc(), dims, args);

      if (auto attr = op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName()))
        launch->setAttr(SymbolTable::getSymbolAttrName(), attr);

      auto &bb = launch.body().front();
      auto ivs = op.getIVs();
      ivs[0].replaceAllUsesWith(launch.getTileIds().x);
      ivs[1].replaceAllUsesWith(launch.getTileIds().y);
      auto &body = op.getBody()->getOperations();
      bb.getOperations().splice(bb.begin(), body,
                                body.begin(), --body.end());
      rewriter.setInsertionPointToStart(&launch.getRegion().front());
      for (auto c : constants) {
        replaceAllUsesInRegionWith(c,
                                   rewriter.clone(*c.getDefiningOp())->getResult(0),
                                   launch.getRegion());
      }
      auto builder = OpBuilder::atBlockEnd(&bb);
      builder.create<air::HerdTerminatorOp>(loc);

      int i = 0;
      auto kernel_args = launch.getKernelArguments();
      for (Value v : args)
        replaceAllUsesInRegionWith(v, kernel_args[i++], launch.getRegion());

      rewriter.eraseOp(op);
    
      return success();
    }
    return failure();
  }
};

class ScfParToHerdLaunchConversion : public OpRewritePattern<scf::ParallelOp> {
public:
  using OpRewritePattern<scf::ParallelOp>::OpRewritePattern;

  LogicalResult normalizeScfParallel(scf::ParallelOp parOp,
                                     PatternRewriter &rewriter) const
  {
    auto loc = parOp.getLoc();

    // everything must be a constant
    for (auto step : parOp.step()) {
      if (!step.getDefiningOp<ConstantIndexOp>())
        return failure();
    }
    for (auto lowerBound : parOp.lowerBound()) {
      if (!lowerBound.getDefiningOp<ConstantIndexOp>())
        return failure();
    }
    for (auto upperBound : parOp.upperBound()) {
      if (!upperBound.getDefiningOp<ConstantIndexOp>())
        return failure();
    }

    auto ivs = parOp.getInductionVars().begin();
    auto step = parOp.step().begin();
    auto lowerBound = parOp.lowerBound().begin();
    auto upperBound = parOp.upperBound().begin();

    SmallVector<Value, 4> new_step;
    SmallVector<Value, 4> new_ub;
    SmallVector<Value, 4> new_lb;
    
    auto builder = OpBuilder::atBlockBegin(parOp.getBody());
    while (step != parOp.step().end()) {
      Value sv = *step++;
      Value lbv = *lowerBound++;
      float s = sv.getDefiningOp<ConstantIndexOp>().getValue();
      float lb = lbv.getDefiningOp<ConstantIndexOp>().getValue();
      float ub = (*upperBound++).getDefiningOp<ConstantIndexOp>().getValue();
      new_ub.push_back(rewriter.create<ConstantIndexOp>(loc,(uint64_t)ceil((ub - lb) / s)));
      new_lb.push_back(rewriter.create<ConstantIndexOp>(loc,0));
      new_step.push_back(rewriter.create<ConstantIndexOp>(loc,1));
      auto iv = *ivs++;
      auto mul = 
        builder.create<MulIOp>(loc, iv, sv.getDefiningOp<ConstantIndexOp>());
      Value new_iv = builder.create<AddIOp>(loc, mul, lbv);
      SmallPtrSet<Operation *, 1> keep{mul};
      iv.replaceAllUsesExcept(new_iv, keep);
    }

    parOp.lowerBoundMutable().assign(new_lb);
    parOp.upperBoundMutable().assign(new_ub);
    parOp.stepMutable().assign(new_step);

    return success();
  }

  LogicalResult matchAndRewrite(scf::ParallelOp op,
                                PatternRewriter &rewriter) const override {
    if (failed(normalizeScfParallel(op, rewriter)))
      return failure();
  
    if (op.getNumLoops() == 2) {
      auto loc = op.getLoc();
      auto lb0 = dyn_cast<ConstantIndexOp>(op.lowerBound()[0].getDefiningOp());
      auto lb1 = dyn_cast<ConstantIndexOp>(op.lowerBound()[1].getDefiningOp());
      auto ub0 = dyn_cast<ConstantIndexOp>(op.upperBound()[0].getDefiningOp());
      auto ub1 = dyn_cast<ConstantIndexOp>(op.upperBound()[1].getDefiningOp());
      auto step0 = dyn_cast<ConstantIndexOp>(op.step()[0].getDefiningOp());
      auto step1 = dyn_cast<ConstantIndexOp>(op.step()[1].getDefiningOp());

      // lowerBound, upperBound and step must be ConstantIndexOps
      if (!(lb0 && lb1 && step0 && step1 && ub0 && ub1))
        return failure();

      auto ub0_int = ub0.getValue();
      auto ub1_int = ub1.getValue();
      auto lb0_int = lb0.getValue();
      auto lb1_int = lb1.getValue();
      auto step0_int = step0.getValue();
      auto step1_int = step1.getValue();

      // must start at (0,0)
      if (lb0_int || lb1_int)
        return failure();

      // step must divide upper bound evenly
      if ((ub0_int % step0_int) || (ub1_int % step1_int))
        return failure();

      ub0_int = ub0_int / step0_int;
      ub1_int = ub1_int / step1_int;

      // TODO this is code duplicated from the affine version, refactor.
      SmallVector<Value, 4> args;
      SmallVector<Value, 4> constants;
      llvm::SetVector<Value> region_args;
      getUsedValuesDefinedAbove(op.getRegion(), region_args);
      for (Value v : region_args) {
        if (v.getDefiningOp() && isa<ConstantOp>(v.getDefiningOp()))
          constants.push_back(v);
        else
          args.push_back(v);
      }
      air::HerdDim2 dims{rewriter.create<ConstantIndexOp>(loc,ub0_int),
                         rewriter.create<ConstantIndexOp>(loc,ub1_int)};
      auto launch = rewriter.create<air::HerdLaunchOp>(op.getLoc(), dims, args);
      auto &bb = launch.body().front();
      auto ivs = op.getInductionVars();
      ivs[0].replaceAllUsesWith(launch.getTileIds().x);
      ivs[1].replaceAllUsesWith(launch.getTileIds().y);
      auto &body = op.getBody()->getOperations();
      bb.getOperations().splice(bb.begin(), body,
                                body.begin(), --body.end());
      rewriter.setInsertionPointToStart(&launch.getRegion().front());
      for (auto c : constants) {
        replaceAllUsesInRegionWith(c,
                                   rewriter.clone(*c.getDefiningOp())->getResult(0),
                                   launch.getRegion());
      }
      auto builder = OpBuilder::atBlockEnd(&bb);
      builder.create<air::HerdTerminatorOp>(loc);

      int i = 0;
      auto kernel_args = launch.getKernelArguments();
      for (Value v : args)
        replaceAllUsesInRegionWith(v, kernel_args[i++], launch.getRegion());

      rewriter.eraseOp(op);

      return success();
    }
    return failure();
  }
};

struct AffineToAIRPass : public PassWrapper<AffineToAIRPass,
                                            OperationPass<ModuleOp>> {


  LogicalResult lower_dma_to_function(StringRef callee, CallOp dma_callOp)
  {
    auto module = getOperation();
    auto funcOp = module.lookupSymbol<mlir::FuncOp>(callee);
    auto ctx = funcOp.getContext();
    auto loc = dma_callOp.getLoc();

    assert(callee.startswith("air_dma_copy"));
    for (auto &bb : funcOp) {
      for (auto &op : bb) {
        if (auto forOp = dyn_cast<AffineForOp>(op)) {
          mlir::AffineLoadOp load = nullptr;
          mlir::AffineStoreOp store = nullptr;

          for (auto &o : forOp.getLoopBody().getOps()) {
            if (isa<AffineLoadOp>(o)) {
              load = cast<AffineLoadOp>(o);
            }
            else if (isa<AffineStoreOp>(o)) {
              store = cast<AffineStoreOp>(o);
            }
            else if (isa<AffineYieldOp>(o)) {
            }
            else {
              return failure();
            }
          }


          if (!load || !store)
            return failure();

          if (store.value() != load)
            return failure();

          //auto srcTy = load.memref().getType().cast<mlir::MemRefType>();
          //auto dstTy = store.memref().getType().cast<mlir::MemRefType>();
        }
      }
    }

    // for now it's all very much hard coded
    if ( callee.equals("acap_L2_dma_copy_1") ) {
      auto arg_iter = dma_callOp.arg_operand_begin();
      // input and output here are relative to the copy
      auto dim1_idx = *(arg_iter);
      auto input_operand = *(++arg_iter);
      auto output_operand = *(++arg_iter);
      auto dim0_idx = *(++arg_iter);
      std::string dmafn_name = "acap_L2_dma_copy_arg0";
      FuncOp dmafn = module.lookupSymbol<FuncOp>(dmafn_name);
      if (!dmafn) {
        SmallVector<Type, 4> tys{input_operand.getType(),
                                 output_operand.getType(),
                                 dim1_idx.getType(),
                                 dim0_idx.getType()};
        SmallVector<Type, 1> retTy{};
        auto fnTy = FunctionType::get(ctx, tys, retTy);
        dmafn = FuncOp::create(loc, dmafn_name, fnTy);
        dmafn.setPrivate();
        module.push_back(dmafn);
      } 
      OpBuilder builder(dma_callOp);
      SmallVector<Value,4> opers{input_operand, output_operand, dim1_idx, dim0_idx};
      SmallVector<Type, 1> retTy;
      builder.create<CallOp>(loc, retTy, builder.getSymbolRefAttr(dmafn_name), opers);
      dma_callOp.erase();
      //acap_L2_dma_copy_arg1(&weights);
    }
    else if (callee.equals("acap_L2_dma_copy")) {
      auto arg_iter = dma_callOp.arg_operand_begin();
      // input and output here are relative to the copy
      auto dim1_idx = *(arg_iter);
      auto input_operand = *(++arg_iter);
      auto dim0_idx = *(++arg_iter);
      auto output_operand = *(++arg_iter);
      std::string dmafn_name = "acap_L2_dma_copy_arg1";
      FuncOp dmafn = module.lookupSymbol<FuncOp>(dmafn_name);
      if (!dmafn) {
        SmallVector<Type, 4> tys{input_operand.getType(),
                                 output_operand.getType(),
                                 dim1_idx.getType(),
                                 dim0_idx.getType()};
        SmallVector<Type, 1> retTy{};
        auto fnTy = FunctionType::get(ctx, tys, retTy);
        dmafn = FuncOp::create(loc, dmafn_name, fnTy);
        dmafn.setPrivate();
        module.push_back(dmafn);
      } 
      OpBuilder builder(dma_callOp);
      SmallVector<Value,4> opers{input_operand, output_operand, dim1_idx, dim0_idx};
      SmallVector<Type, 1> retTy;
      builder.create<CallOp>(loc, retTy, builder.getSymbolRefAttr(dmafn_name), opers);
      dma_callOp.erase();
    }
    return success();
  }

  LogicalResult lowerDma(StringRef callee, CallOp dma_callOp) {
    //return lowerDma_pad(callee, dma_callOp);
    return lower_dma_to_function(callee, dma_callOp);
  }

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
     registry.insert<xilinx::air::airDialect>();
  }

  void runOnOperation() override {
    auto module = getOperation();
    auto context = module.getContext();

    LLVM_DEBUG(llvm::outs() << "input\n");
    LLVM_DEBUG(module.print(llvm::outs()));

    for (auto f : module.getOps<FuncOp>()) {
      f.walk([&](Operation *op) {
        if (auto co = dyn_cast<CallOp>(op)) {
          if (co.getCallee().startswith("air_dma_copy")) {
            if (failed(lowerDma(co.getCallee(), co)))
              LLVM_DEBUG(llvm::outs() << "Failed to lower 'air_dma_copy'\n");
          }
        }
      });
    }

    // tablegen patterns
    OwningRewritePatternList patterns(context);
    patterns.insert<AffineParToHerdLaunchConversion,
                    ScfParToHerdLaunchConversion>(context);

    populateWithGenerated(patterns);

    ConversionTarget target(*context);

    target.addLegalDialect<LLVM::LLVMDialect,
                           StandardOpsDialect/*,
                           scf::SCFDialect*/>();

    target.addLegalOp<xilinx::air::DmaMemcpyOp>();
    target.addLegalOp<xilinx::air::DmaMemcpy2dOp>();
    target.addLegalOp<xilinx::air::DmaMemcpy4dOp>();
    target.addLegalOp<xilinx::air::HerdLaunchOp>();

    target.addLegalOp<AffineApplyOp,
                      AffineForOp,
                      AffineLoadOp,
                      AffineStoreOp,
                      AffineYieldOp>();

    DmaMemcpyOpID = 0;

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
      assert(0);
    }

    // Simplify all the subviews so we can rewrite them easily.
    // Mostly this is propagating constant sizes into dimensioned memref types.
    OwningRewritePatternList stage2Patterns =
      linalg::getLinalgTilingCanonicalizationPatterns(context);
    memref::AllocOp::getCanonicalizationPatterns(stage2Patterns, context);
    (void)applyPatternsAndFoldGreedily(module, std::move(stage2Patterns));

    OwningRewritePatternList stage3Patterns(context);
    stage3Patterns.insert<AffineCopyToAIRDMAConversion,
                   LinalgCopyToAIRDmaConversion>(context);
    if (failed(applyPartialConversion(module, target, std::move(stage3Patterns)))) {
      emitError(UnknownLoc::get(context), "error\n");
      signalPassFailure();
      assert(0);
    }

    std::vector<Operation*> waits;
    for (auto f : module.getOps<FuncOp>()) {
      f.walk([&](Operation *op) {
        if (auto wo = dyn_cast<AffineDmaWaitOp>(op)) {
          auto memref = wo.getTagMemRef();
          for (auto u : memref.getUsers()) {
              waits.push_back(u);
          }
        }
      });
    }
    for (auto o : waits) o->erase();

    std::vector<std::string> herd_syms;
    for (auto f : module.getOps<FuncOp>()) {
      // record existing symbol names
      f.walk([&](xilinx::air::HerdLaunchOp op) {
        if (auto attr = op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
          std::string name = attr.getValue().str();
          assert( (std::find(herd_syms.begin(), herd_syms.end(), name) == herd_syms.end())
            && "unexpected duplicate symbol");
          herd_syms.push_back(name);
        }
      });
      // generate missing symbol names
      f.walk([&](xilinx::air::HerdLaunchOp op) {
        if (!op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
          unsigned id = 0;
          std::string name;
          do {
            std::stringstream ss;
            ss << "herd_" << id++;
            name = ss.str();
          } while ( std::find(herd_syms.begin(), herd_syms.end(), name) != herd_syms.end() );
          herd_syms.push_back(name);
          op->setAttr(SymbolTable::getSymbolAttrName(), StringAttr::get(op->getContext(), name));
        }
      });
    }
    LLVM_DEBUG(llvm::outs() << "output\n");
    LLVM_DEBUG(module.print(llvm::outs()));

  }
};

}// namespace


namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAffineToAIRPass() {
  return std::make_unique<AffineToAIRPass>();
}

} // namespace air
} // namespace xilinx

void xilinx::air::registerAffineToAIRPass() {
    PassRegistration<AffineToAIRPass>(
      "affine-to-air",
      "Lift affine loops to AIR dialect");
}
