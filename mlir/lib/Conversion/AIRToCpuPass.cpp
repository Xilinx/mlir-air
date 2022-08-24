// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

#include "PassDetail.h"

#include "air/Conversion/AIRPipeline.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <vector>

#define DEBUG_TYPE "air-to-cpu"

using namespace mlir;
using namespace mlir::arith;
using namespace xilinx::air;

namespace {

class AIRHerdToCpuConversion : public ConversionPattern {
public:
  explicit AIRHerdToCpuConversion(MLIRContext *context)
      : ConversionPattern(xilinx::air::HerdOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    xilinx::air::HerdOp launch = cast<xilinx::air::HerdOp>(op);
    std::string herd_name = "herd";
    if (auto attr =
            op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())) {
      herd_name = attr.getValue().str();
    }
    auto herd_size = launch.getSizeOperands();
    int64_t herd_size_x =
        cast<arith::ConstantIndexOp>(herd_size[0].getDefiningOp()).value();
    int64_t herd_size_y =
        cast<arith::ConstantIndexOp>(herd_size[1].getDefiningOp()).value();

    auto outer = rewriter.create<AffineForOp>(launch.getLoc(), 0, herd_size_x);
    auto outer_builder = OpBuilder::atBlockBegin(outer.getBody());
    auto inner =
        outer_builder.create<AffineForOp>(launch.getLoc(), 0, herd_size_y);

    outer->setAttr("air.herd",
                   StringAttr::get(op->getContext(), "outer"));
    inner->setAttr("air.herd",
                   StringAttr::get(op->getContext(), "inner"));

    SmallVector<Value, 16> callops;
    callops.push_back(outer.getInductionVar());
    callops.push_back(inner.getInductionVar());

    launch.getSize()[0].replaceAllUsesWith(herd_size[0]);
    launch.getSize()[1].replaceAllUsesWith(herd_size[1]);

    for (unsigned i = 0, e = launch.getNumKernelOperands(); i < e; i++)
      callops.push_back(launch.getKernelOperand(i));

    auto module = op->getParentOfType<ModuleOp>();
    std::string fname = herd_name + "_body_fn";
    std::string new_fname = fname;
    int which_try = 0;
    while (module.lookupSymbol(new_fname))
      new_fname = fname + "_" + std::to_string(++which_try);
    fname = new_fname;

    std::vector<mlir::Type> ret_types;
    std::vector<mlir::Type> arg_types;
    for (auto o : callops)
      arg_types.push_back(o.getType());

    auto func_type =
        mlir::FunctionType::get(op->getContext(), arg_types, ret_types);
    auto function = mlir::func::FuncOp::create(op->getLoc(), fname, func_type,
                                               /* attrs = */ {});

    auto &entryBlock = *function.addEntryBlock();

    if (1) {
      int i = 0;
      launch.getIds()[0].replaceAllUsesWith(entryBlock.getArgument(i++));
      launch.getIds()[1].replaceAllUsesWith(entryBlock.getArgument(i++));
      for (auto arg : launch.getKernelArguments()) {
        arg.replaceAllUsesWith(entryBlock.getArgument(i++));
      }
    } else {
      launch.getIds()[0].replaceAllUsesWith(outer.getInductionVar());
      launch.getIds()[1].replaceAllUsesWith(inner.getInductionVar());
    }
    int i = 0;
    for (auto arg : launch.getKernelArguments())
      arg.replaceAllUsesWith(launch.getKernelOperand(i++));

    auto &body = launch.body().front().getOperations();
    if (1) {
      entryBlock.getOperations().splice(entryBlock.begin(), body, body.begin(),
                                        --body.end());

      rewriter.setInsertionPointToStart(&inner.getBodyRegion().front());
      rewriter.create<func::CallOp>(op->getLoc(), function, callops);

      rewriter.setInsertionPointToEnd(&entryBlock);
      rewriter.create<func::ReturnOp>(op->getLoc());
      module.push_back(function);
    } else {
      inner.getBody()->getOperations().splice(inner.getBody()->begin(), body,
                                            body.begin(), --body.end());
    }
    rewriter.eraseOp(op);
    return success();
  }
};

class AIRPipelineConversion : public ConversionPattern {
public:
  explicit AIRPipelineConversion(MLIRContext *context)
      : ConversionPattern(xilinx::air::HerdPipelineOp::getOperationName(), 1,
                          context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto pipeOp = cast<xilinx::air::HerdPipelineOp>(op);
    Block &bb = pipeOp.body().front();
    rewriter.eraseOp(pipeOp.body().back().getTerminator());
    bb.getOperations().splice(Block::iterator(op), bb.getOperations());
    rewriter.eraseOp(op);
    return success();
  }
};



static func::CallOp
convertOpToFunctionWithId(Operation *op, ArrayRef<Value> operands,
                          ConversionPatternRewriter &rewriter,
                          StringRef fnName) {
  auto loc = op->getLoc();

  SmallVector<Value, 16> callops;
  SmallVector<Type, 1> retTys{};

  auto idTy = IntegerType::get(op->getContext(), 32);
  if (auto id_attr = op->getAttrOfType<IntegerAttr>("id")) {
    callops.push_back(rewriter.create<arith::ConstantOp>(loc, idTy, id_attr));
  }

  for (auto o : operands) {
    // erase the size to reduce the number of manglings
    if (auto memrefTy = o.getType().dyn_cast<MemRefType>()) {
      auto t = MemRefType::get(std::vector<int64_t>(memrefTy.getRank(), -1),
                               memrefTy.getElementType(), memrefTy.getLayout(),
                               memrefTy.getMemorySpace());
      callops.push_back(rewriter.create<memref::CastOp>(op->getLoc(), t, o));
    } else {
      callops.push_back(o);
    }
  }
  SmallVector<Type, 16> tys;
  for (auto o : callops)
    tys.push_back(o.getType());

  SmallVector<MemRefType, 16> real_result_tys;
  for (auto t : op->getResultTypes()) {
    if (auto memrefTy = t.dyn_cast<MemRefType>()) {
      auto mrt =
          MemRefType::get(std::vector<int64_t>(memrefTy.getRank(), -1),
                          memrefTy.getElementType(), memrefTy.getLayout(),
                          memrefTy.getMemorySpace());
      retTys.push_back(mrt);
      real_result_tys.push_back(memrefTy);
    } else {
      retTys.push_back(t);
    }
  }

  auto fn = xilinx::air::getMangledFunction(op->getParentOfType<ModuleOp>(),
                                            fnName.str(), callops, retTys);
  auto call = rewriter.replaceOpWithNewOp<func::CallOp>(
      op, retTys, SymbolRefAttr::get(fn), callops);
  int real_result_idx = 0;
  int result_idx = 0;
  for (auto r : op->getResults()) {
    if (auto memrefTy = r.getType().dyn_cast<MemRefType>()) {
      auto t = real_result_tys[real_result_idx++];
      auto c = rewriter.create<memref::CastOp>(op->getLoc(), t,
                                               call.getResult(result_idx));
      r.replaceAllUsesWith(c.getResult());
    } else {
      r.replaceAllUsesWith(call.getResult(result_idx));
    }
    result_idx++;
  }
  return call;
}

static func::CallOp
convertOpToFunctionWithTileId(Operation *op, ArrayRef<Value> operands,
                              ConversionPatternRewriter &rewriter,
                              StringRef fnName) {
  auto loc = op->getLoc();
  SmallVector<Value, 16> callops;
  SmallVector<Type, 1> retTys{};

  auto idTy = IntegerType::get(op->getContext(), 32);
  if (auto id_attr = op->getAttrOfType<IntegerAttr>("id")) {
    callops.push_back(rewriter.create<arith::ConstantOp>(loc, idTy, id_attr));
  }

  xilinx::air::HerdOp launch =
      op->getParentOfType<xilinx::air::HerdOp>();
  if (!launch) {
    AffineForOp afo = op->getParentOfType<AffineForOp>();
    while (afo && !afo->getAttr("air.herd"))
      afo = afo->getParentOfType<AffineForOp>();
    if (afo) {
      callops.push_back(afo.getInductionVar());
      afo = afo->getParentOfType<AffineForOp>();
    }
    while (afo && !afo->getAttr("air.herd"))
      afo = afo->getParentOfType<AffineForOp>();
    if (afo)
      callops.push_back(afo.getInductionVar());
  } else {
    auto tileIds = launch.getIds();
    callops.push_back(tileIds[0]);
    callops.push_back(tileIds[1]);
  }

  for (auto o : operands) {
    // erase the size to reduce the number of manglings
    if (auto memrefTy = o.getType().dyn_cast<MemRefType>()) {
      auto t = MemRefType::get(std::vector<int64_t>(memrefTy.getRank(), -1),
                               memrefTy.getElementType(), memrefTy.getLayout(),
                               memrefTy.getMemorySpace());
      callops.push_back(rewriter.create<memref::CastOp>(op->getLoc(), t, o));
    } else {
      callops.push_back(o);
    }
  }

  SmallVector<MemRefType, 16> real_result_tys;
  for (auto t : op->getResultTypes()) {
    if (auto memrefTy = t.dyn_cast<MemRefType>()) {
      auto mrt =
          MemRefType::get(std::vector<int64_t>(memrefTy.getRank(), -1),
                          memrefTy.getElementType(), memrefTy.getLayout(),
                          memrefTy.getMemorySpace());
      retTys.push_back(mrt);
      real_result_tys.push_back(memrefTy);
    } else {
      retTys.push_back(t);
    }
  }

  auto fn = xilinx::air::getMangledFunction(op->getParentOfType<ModuleOp>(),
                                            fnName.str(), callops, retTys);

  auto call = rewriter.create<func::CallOp>(op->getLoc(), retTys,
                                            SymbolRefAttr::get(fn), callops);

  int real_result_idx = 0;
  int result_idx = 0;
  for (auto r : op->getResults()) {
    if (auto memrefTy = r.getType().dyn_cast<MemRefType>()) {
      auto t = real_result_tys[real_result_idx++];
      auto c = rewriter.create<memref::CastOp>(op->getLoc(), t,
                                               call.getResult(result_idx));
      r.replaceAllUsesWith(c.getResult());
    } else {
      r.replaceAllUsesWith(call.getResult(result_idx));
    }
    result_idx++;
  }

  rewriter.eraseOp(op);
  return call;
}

class AIRDmaMemcpyNdToMemcpyConversion
    : public OpConversionPattern<xilinx::air::DmaMemcpyNdOp> {
public:
  using OpConversionPattern<xilinx::air::DmaMemcpyNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xilinx::air::DmaMemcpyNdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 8> operands{adaptor.getOperands()};
    auto call =
        convertOpToFunctionWithTileId(op, operands, rewriter, "air_memcpy_nd");
    if (call)
      return success();
    else
      return failure();
  }
};

class AllocToCpuConversion : public ConversionPattern {
public:
  explicit AllocToCpuConversion(MLIRContext *context)
      : ConversionPattern(memref::AllocOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto call = convertOpToFunctionWithTileId(op, operands, rewriter, "air_alloc");
    if (call)
      return success();
    else
      return failure();
  }
};

class DeallocToCpuConversion : public ConversionPattern {
public:
  explicit DeallocToCpuConversion(MLIRContext *context)
      : ConversionPattern(memref::DeallocOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto call =
        convertOpToFunctionWithTileId(op, operands, rewriter, "air_dealloc");
    if (call)
      return success();
    else
      return failure();
  }
};

// Convert CallOp returns of AsyncTokenType to uint64
class AsyncCallOpConversion : public OpConversionPattern<func::CallOp> {
public:
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Type, 2> retTy;
    for (auto t : op.getResultTypes())
      if (t.isa<xilinx::air::AsyncTokenType>())
        retTy.push_back(mlir::IntegerType::get(op->getContext(), 64));
      else
        retTy.push_back(t);

    auto callOp = rewriter.create<func::CallOp>(
        op->getLoc(), adaptor.getCallee(), retTy, adaptor.getOperands());
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

class WaitAllOpConversion : public OpConversionPattern<xilinx::air::WaitAllOp> {
public:
  using OpConversionPattern<xilinx::air::WaitAllOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(xilinx::air::WaitAllOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 8> operands{adaptor.getOperands()};
    auto call =
        convertOpToFunctionWithId(op, operands, rewriter, "air_wait_all");
    if (call)
      return success();
    else
      return failure();
  }
};

class ScfYieldOpConversion : public OpConversionPattern<scf::YieldOp> {
public:
  using OpConversionPattern<scf::YieldOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 8> operands{adaptor.getOperands()};
    SmallVector<Type, 2> retTys;
    for (auto t : op->getResultTypes()) {
      if (t.isa<xilinx::air::AsyncTokenType>()) {
        retTys.push_back(mlir::IntegerType::get(op->getContext(), 64));
      } else {
        retTys.push_back(t);
      }
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, retTys, operands);
    return success();
  }
};

class ScfForOpConversion : public OpConversionPattern<scf::ForOp> {
public:
  using OpConversionPattern<scf::ForOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<scf::ForOp>(
        op, adaptor.getLowerBound(), adaptor.getUpperBound(), adaptor.getStep(),
        adaptor.getInitArgs());
    auto body = op.getBody();
    auto newBody = newOp.getBody();

    for (int i = 0, e = body->getNumArguments(); i < e; i++) {
      body->getArgument(i).replaceAllUsesWith(newBody->getArgument(i));
    }

    auto &ops = body->getOperations();
    auto &newOps = newBody->getOperations();
    newOps.splice(newOps.begin(), ops, ops.begin(), ops.end());
    return success();
  }
};

class AIRToCpuPass : public AIRToCpuBase<AIRToCpuPass> {

public:
  AIRToCpuPass() = default;
  AIRToCpuPass(const AIRToCpuPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<AffineDialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();

    TypeConverter converter;
    converter.addConversion([&](Type type) -> Optional<Type> {
      // convert event token to unsigned 64 bit integer
      if (auto t = type.dyn_cast<xilinx::air::AsyncTokenType>())
        return mlir::IntegerType::get(context, 64);
      else
        return type;
    });
    auto addUnrealizedCast = [](OpBuilder &builder, Type type,
                                ValueRange inputs, Location loc) {
      auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
      return Optional<Value>(cast.getResult(0));
    };
    converter.addSourceMaterialization(addUnrealizedCast);
    converter.addTargetMaterialization(addUnrealizedCast);

    ConversionTarget target(*context);

    target.addLegalDialect<
        LLVM::LLVMDialect, func::FuncDialect, arith::ArithmeticDialect,
        AffineDialect, scf::SCFDialect, linalg::LinalgDialect,
        memref::MemRefDialect, bufferization::BufferizationDialect,
        xilinx::airrt::AIRRtDialect>();

    // air.memcpy_nd conversion
    RewritePatternSet air_dma_patterns(context);

    air_dma_patterns.add<AIRDmaMemcpyNdToMemcpyConversion>(context);

    if (failed(applyPartialConversion(module, target,
                                      std::move(air_dma_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air dialect\n");
      signalPassFailure();
    }

    // Replace the PipelineStageOps first, followed by the
    // HerdPipelineOps, then run the rest of the patterns.
    // This avoids creating invalid intermediate code with respect
    // to the herd->pipeline->stages nesting requirements.

    // PipelineStageOp conversion
    RewritePatternSet air_pipe_stage_patterns(context);
    air_pipe_stage_patterns.insert<AIRPipeStageConversion>(context, AIRPipeStageConversion::LoweringType::AllocBuffer);
    if (failed(applyPartialConversion(module, target,
                                      std::move(air_pipe_stage_patterns)))) {
      emitError(UnknownLoc::get(context),
                "error lowering air.pipeline.stage\n");
      signalPassFailure();
    }

    // HerdPipelineOp conversion
    RewritePatternSet air_pipe_patterns(context);
    air_pipe_patterns.insert<AIRPipelineConversion>(context);
    if (failed(applyPartialConversion(module, target,
                                      std::move(air_pipe_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air.pipeline\n");
      signalPassFailure();
    }

    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      return (op.getType().getMemorySpaceAsInt() == 0);
    });

    target.addDynamicallyLegalOp<memref::DeallocOp>([&](memref::DeallocOp op) {
      return (op.memref().getType().cast<MemRefType>().getMemorySpaceAsInt() ==
              0);
    });

    RewritePatternSet air_mem_patterns(context);
    air_mem_patterns
        .add<AllocToCpuConversion, DeallocToCpuConversion, WaitAllOpConversion>(
            context);

    if (failed(applyPartialConversion(module, target,
                                      std::move(air_mem_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air dialect\n");
      signalPassFailure();
    }

    RewritePatternSet air_herd_patterns(context);
    air_herd_patterns.add<AIRHerdToCpuConversion>(context);
    if (failed(applyPartialConversion(module, target,
                                      std::move(air_herd_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air.launch_herd\n");
      signalPassFailure();
    }

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      return (!llvm::any_of(op->getResultTypes(), [](Type t) {
        return t.isa<xilinx::air::AsyncTokenType>();
      }) && !llvm::any_of(op->getOperandTypes(), [](Type t) {
        return t.isa<xilinx::air::AsyncTokenType>();
      }));
    });

    target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
      for (auto o : op.getRegionIterArgs()) {
        if (o.getType().isa<xilinx::air::AsyncTokenType>())
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
      for (auto v : op.getResults()) {
        if (v.getType().isa<xilinx::air::AsyncTokenType>())
          return false;
      }
      return true;
    });

    RewritePatternSet typeConversionPatterns(context);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        typeConversionPatterns, converter);

    typeConversionPatterns
        .add<ScfYieldOpConversion, ScfForOpConversion, AsyncCallOpConversion>(
            converter, context);

    if (failed(applyPartialConversion(module, target,
                                      std::move(typeConversionPatterns)))) {
      emitError(UnknownLoc::get(context),
                "error lowering air async token type\n");
      signalPassFailure();
    }

    for (auto func : module.getOps<func::FuncOp>())
      func->setAttr("llvm.emit_c_interface", UnitAttr::get(func.getContext()));
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRToCpuPass() {
  return std::make_unique<AIRToCpuPass>();
}

} // namespace air
} // namespace xilinx
