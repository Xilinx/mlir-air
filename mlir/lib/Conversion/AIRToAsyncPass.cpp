//===- AIRToAsyncPass.cpp ---------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"

#include "air/Conversion/AIRPipeline.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Dialect/AIRRt/AIRRtDialect.h"
#include "air/Dialect/AIRRt/AIRRtOps.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
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
using namespace xilinx;
using namespace xilinx::air;

namespace {

class AIRHerdOpConversion : public ConversionPattern {
public:
  explicit AIRHerdOpConversion(MLIRContext *context)
      : ConversionPattern(air::HerdOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {

    air::HerdOp launch = cast<air::HerdOp>(op);

    auto herd_size = launch.getSizeOperands();
    int64_t herd_size_x =
        cast<arith::ConstantIndexOp>(herd_size[0].getDefiningOp()).value();
    int64_t herd_size_y =
        cast<arith::ConstantIndexOp>(herd_size[1].getDefiningOp()).value();

    SmallVector<Value> empty;
    SmallVector<Type> retTy;
    SmallVector<Value> deps;
    // for (unsigned i=0; i<launch.getAsyncDependencies().size(); ++i)
    //   deps.push_back(
    //     rewriter.create<UnrealizedConversionCastOp>(op->getLoc(),
    //                                                 async::TokenType::get(op->getContext()),
    //                                                 operands[i]).getResult(0));

    auto herdExeOp = rewriter.create<async::ExecuteOp>(
        op->getLoc(), retTy, launch.getAsyncDependencies(), empty,
        [&](OpBuilder &r, Location loc, ValueRange v) {
          auto size =
              r.create<arith::ConstantIndexOp>(loc, herd_size_x * herd_size_y);
          auto group = r.create<async::CreateGroupOp>(loc, size);
          auto outer = r.create<AffineForOp>(loc, 0, herd_size_x);
          r.setInsertionPointToStart(outer.getBody());
          auto inner = r.create<AffineForOp>(loc, 0, herd_size_y);

          outer->setAttr("air.herd",
                         StringAttr::get(op->getContext(), "outer"));
          inner->setAttr("air.herd",
                         StringAttr::get(op->getContext(), "inner"));

          IRMapping mapper;
          mapper.map(launch.getSize()[0], herd_size[0]);
          mapper.map(launch.getSize()[1], herd_size[1]);

          mapper.map(launch.getIds()[0], outer.getInductionVar());
          mapper.map(launch.getIds()[1], inner.getInductionVar());

          int i = launch.getAsyncDependencies().size() + 2;
          for (auto arg : launch.getKernelArguments())
            mapper.map(arg, operands[i++]);

          r.setInsertionPointToStart(inner.getBody());
          auto coreExeOp = r.create<async::ExecuteOp>(
              loc, retTy, empty, empty,
              [&](OpBuilder &b, Location loc, ValueRange v) {
                for (auto &o : launch.getBody().front().getOperations())
                  if (!isa<air::HerdTerminatorOp>(o))
                    b.clone(o, mapper);
                b.create<async::YieldOp>(loc, empty);
              });
          r.create<async::AddToGroupOp>(loc, coreExeOp.getResult(0), group);

          r.setInsertionPointAfter(outer);
          r.create<async::AwaitAllOp>(loc, group);
          r.create<async::YieldOp>(loc, empty);
        });
    rewriter.setInsertionPointAfter(herdExeOp);
    rewriter.create<async::AwaitOp>(op->getLoc(), herdExeOp.getResult(0));

    if (auto t = launch.getAsyncToken())
      t.replaceAllUsesWith(herdExeOp.getResult(0));
    rewriter.eraseOp(op);

    return success();
  }
};

class AIRPipelineConversion : public ConversionPattern {
public:
  explicit AIRPipelineConversion(MLIRContext *context)
      : ConversionPattern(air::HerdPipelineOp::getOperationName(), 1, context) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto pipeOp = cast<air::HerdPipelineOp>(op);
    Block &bb = pipeOp.getBody().front();
    rewriter.eraseOp(pipeOp.getBody().back().getTerminator());
    bb.getOperations().splice(Block::iterator(op), bb.getOperations());
    rewriter.eraseOp(op);
    return success();
  }
};

static func::CallOp convertOpToFunction(Operation *op, ArrayRef<Value> operands,
                                        ConversionPatternRewriter &rewriter,
                                        StringRef fnName) {
  auto loc = op->getLoc();
  SmallVector<Value, 16> callops;
  SmallVector<Type, 1> retTys{};

  SmallVector<Value, 4> dependencies;
  for (auto o : operands) {
    // erase the size to reduce the number of manglings
    if (auto memrefTy = o.getType().dyn_cast<MemRefType>()) {
      auto t = MemRefType::get(
          std::vector<int64_t>(memrefTy.getRank(), ShapedType::kDynamic),
          memrefTy.getElementType(), memrefTy.getLayout(),
          /*memrefTy.getMemorySpace()*/ 0);
      callops.push_back(
          rewriter.create<UnrealizedConversionCastOp>(loc, t, o).getResult(0));
    } else if (o.getType().isa<async::TokenType>()) {
      dependencies.push_back(o);
    } else {
      callops.push_back(o);
    }
  }

  SmallVector<MemRefType, 16> real_result_tys;
  SmallVector<Type, 1> token_result_tys;
  for (auto t : op->getResultTypes()) {
    if (auto memrefTy = t.dyn_cast<MemRefType>()) {
      auto mrt = MemRefType::get(
          std::vector<int64_t>(memrefTy.getRank(), ShapedType::kDynamic),
          memrefTy.getElementType(), memrefTy.getLayout(),
          /*memrefTy.getMemorySpace()*/ 0);
      retTys.push_back(mrt);
      real_result_tys.push_back(memrefTy);
    } else if (t.isa<air::AsyncTokenType>()) {
      token_result_tys.push_back(t);
    } else {
      retTys.push_back(t);
    }
  }

  auto fn = air::getMangledFunction(op->getParentOfType<ModuleOp>(),
                                    fnName.str(), callops, retTys);

  func::CallOp call = nullptr;
  SmallVector<Value, 4> results;
  if (token_result_tys.size()) {
    auto exe = rewriter.create<async::ExecuteOp>(
        op->getLoc(), retTys, dependencies, SmallVector<Value, 1>{},
        [&](OpBuilder &b, Location loc, ValueRange v) {
          call = rewriter.create<func::CallOp>(op->getLoc(), retTys,
                                               SymbolRefAttr::get(fn), callops);
          b.create<async::YieldOp>(loc, call.getResults());
        });
    results = exe.getResults();
  } else {
    for (auto d : dependencies)
      rewriter.create<async::AwaitOp>(op->getLoc(), d);
    call = rewriter.create<func::CallOp>(op->getLoc(), retTys,
                                         SymbolRefAttr::get(fn), callops);
    results = call.getResults();
    for (unsigned i = 0, real_result_idx = 0; i < results.size(); ++i) {
      auto r = results[i];
      if (auto memrefTy = r.getType().dyn_cast<MemRefType>()) {
        auto t = real_result_tys[real_result_idx++];
        auto c =
            rewriter.create<UnrealizedConversionCastOp>(op->getLoc(), t, r);
        results[i] = c.getResult(0);
      }
    }
  }

  rewriter.replaceOp(op, results);
  return call;
}

static func::CallOp
convertOpToFunctionWithTileId(Operation *op, ArrayRef<Value> operands,
                              ConversionPatternRewriter &rewriter,
                              StringRef fnName) {
  auto loc = op->getLoc();
  SmallVector<Value, 16> callops;

  auto idTy = IntegerType::get(op->getContext(), 32);
  if (auto id_attr = op->getAttrOfType<IntegerAttr>("id")) {
    callops.push_back(rewriter.create<arith::ConstantOp>(loc, idTy, id_attr));
  }

  callops.append(operands.begin(), operands.end());
  return convertOpToFunction(op, callops, rewriter, fnName);
}

class AIRDmaMemcpyNdToMemcpyConversion
    : public OpConversionPattern<air::DmaMemcpyNdOp> {
public:
  using OpConversionPattern<air::DmaMemcpyNdOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(air::DmaMemcpyNdOp op, OpAdaptor adaptor,
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

// Convert memref.alloc to a function
class AllocToCpuConversion : public ConversionPattern {
public:
  explicit AllocToCpuConversion(MLIRContext *context)
      : ConversionPattern(memref::AllocOp::getOperationName(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto call =
        convertOpToFunctionWithTileId(op, operands, rewriter, "air_alloc");
    if (call)
      return success();
    else
      return failure();
  }
};

// Convert memref.dealloc to a function
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

// Convert memref.alloc memory space
class AllocOpConversion : public OpConversionPattern<memref::AllocOp> {
public:
  using OpConversionPattern<memref::AllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = op.getType();
    if (op.getType().getMemorySpaceAsInt() == (int)air::MemorySpace::L3)
      return failure();

    auto alloc = rewriter.create<memref::AllocOp>(
        op.getLoc(),
        MemRefType::get(memrefTy.getShape(), memrefTy.getElementType(),
                        memrefTy.getLayout(), 0));
    op.getResult().replaceAllUsesWith(alloc.getResult());
    rewriter.eraseOp(op);
    /// rewriter.replaceOp(op, alloc.getResult());
    return success();
  }
};

// Convert memref.dealloc memory space
class DeallocOpConversion : public OpConversionPattern<memref::DeallocOp> {
public:
  using OpConversionPattern<memref::DeallocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto memrefTy = op.getMemref().getType().cast<MemRefType>();
    if (memrefTy.getMemorySpaceAsInt() == (int)air::MemorySpace::L3)
      return failure();

    rewriter.create<memref::DeallocOp>(op.getLoc(), adaptor.getMemref());
    rewriter.eraseOp(op);
    return success();
  }
};

// Convert CallOp returns of AsyncTokenType to async::TokenType
class AsyncCallOpConversion : public OpConversionPattern<func::CallOp> {
public:
  using OpConversionPattern<func::CallOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Type, 2> retTy;
    for (auto t : op.getResultTypes())
      if (t.isa<air::AsyncTokenType>())
        retTy.push_back(async::TokenType::get(op->getContext()));
      else
        retTy.push_back(t);

    auto callOp = rewriter.create<func::CallOp>(
        op->getLoc(), adaptor.getCallee(), retTy, adaptor.getOperands());
    rewriter.replaceOp(op, callOp.getResults());
    return success();
  }
};

class WaitAllOpConversion : public OpConversionPattern<air::WaitAllOp> {
public:
  using OpConversionPattern<air::WaitAllOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(air::WaitAllOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value, 8> operands{adaptor.getOperands()};

    if (op->getNumResults() == 1) {
      SmallVector<Value, 1> empty;
      SmallVector<Type, 1> retTy;
      auto newOp = rewriter.create<async::ExecuteOp>(
          op->getLoc(), retTy, operands, empty,
          [&](OpBuilder &b, Location loc, ValueRange v) {
            SmallVector<Value, 1> returnValues;
            b.create<async::YieldOp>(loc, returnValues);
          });
      // auto r = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(),
      //                    async::TokenType::get(op->getContext()),
      //                    newOp->getResult(0));
      // op->getResult(0).replaceAllUsesWith(r.getResult(0));
      op->getResult(0).replaceAllUsesWith(newOp->getResult(0));
      rewriter.eraseOp(op);
      return success();
    }

    for (auto o : operands) {
      Value v = o;
      // if (o.getType().isa<air::AsyncTokenType>())
      //   v = rewriter.create<UnrealizedConversionCastOp>(op->getLoc(),
      //                   async::TokenType::get(op->getContext()),
      //                   o).getResult(0);
      rewriter.create<async::AwaitOp>(op->getLoc(), v);
    }
    rewriter.eraseOp(op);
    return success();
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
      if (t.isa<air::AsyncTokenType>()) {
        retTys.push_back(async::TokenType::get(op->getContext()));
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

class ScfParallelOpConversion : public OpConversionPattern<scf::ParallelOp> {
public:
  using OpConversionPattern<scf::ParallelOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::ParallelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Value> empty;
    SmallVector<Type> retTy; // TODO
    SmallVector<Value> deps; // TODO

    // compute the total number of iterations and check that the bounds are
    // constants
    uint64_t total_size = 1;
    auto ivs = op.getInductionVars().begin();
    auto step = op.getStep().begin();
    auto lowerBound = op.getLowerBound().begin();
    auto upperBound = op.getUpperBound().begin();
    for (int i = 0, e = op.getNumLoops(); i < e; i++) {
      Value sv = *step++;
      Value lbv = *lowerBound++;
      Value ubv = *upperBound++;
      auto si = sv.getDefiningOp<arith::ConstantIndexOp>();
      auto lbi = lbv.getDefiningOp<arith::ConstantIndexOp>();
      auto ubi = ubv.getDefiningOp<arith::ConstantIndexOp>();
      if (!si || !lbi || !ubi)
        return failure();
      auto s = si.value();
      auto lb = lbi.value();
      auto ub = ubi.value();
      auto new_ub_int = (ub - lb) / s;
      if ((new_ub_int * s) != (ub - lb))
        return op->emitOpError()
               << "failed to normalize: step '" << s
               << "' does not evenly divide range '" << (ub - lb) << "'";
      total_size *= new_ub_int;
    }

    auto topExeOp = rewriter.create<async::ExecuteOp>(
        op->getLoc(), retTy, deps, empty,
        [&](OpBuilder &r, Location loc, ValueRange v) {
          IRMapping mapper;

          auto size = r.create<arith::ConstantIndexOp>(loc, total_size);
          auto group = r.create<async::CreateGroupOp>(loc, size);

          // create nested for loops
          auto ivs = op.getInductionVars().begin();
          auto step = op.getStep().begin();
          auto lowerBound = op.getLowerBound().begin();
          auto upperBound = op.getUpperBound().begin();
          SmallVector<scf::ForOp> loops;
          for (int i = 0, e = op.getNumLoops(); i < e; i++) {
            Value sv = *step++;
            Value lbv = *lowerBound++;
            Value ubv = *upperBound++;
            Value iv = *ivs++;
            auto l = r.create<scf::ForOp>(loc, lbv, ubv, sv);
            mapper.map(lbv, l.getLowerBound());
            mapper.map(ubv, l.getUpperBound());
            mapper.map(sv, l.getStep());
            mapper.map(iv, l.getInductionVar());
            r.setInsertionPointToStart(l.getBody());
            loops.push_back(l);
          }

          // create an async.execute and clone the scf.parallel body into it
          auto coreExeOp = r.create<async::ExecuteOp>(
              loc, retTy, empty, empty,
              [&](OpBuilder &b, Location loc, ValueRange v) {
                for (auto &o : op.getBody()->getOperations())
                  if (!isa<scf::YieldOp, scf::ReduceOp>(o))
                    b.clone(o, mapper);
                b.create<async::YieldOp>(loc, empty);
              });
          r.create<async::AddToGroupOp>(loc, coreExeOp.getResult(0), group);
          r.setInsertionPointAfter(loops[0]);

          r.create<async::AwaitAllOp>(loc, group);
          r.create<async::YieldOp>(loc, empty);
        });
    rewriter.create<async::AwaitOp>(op->getLoc(), topExeOp.getResult(0));
    if (op.getInitVals().size())
      rewriter.replaceOp(op, topExeOp.getResults());
    else
      rewriter.eraseOp(op);

    return success();
  }
};

class ExecuteOpConversion : public OpConversionPattern<air::ExecuteOp> {
public:
  using OpConversionPattern<air::ExecuteOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(air::ExecuteOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<Type, 4> resultTypes;
    for (unsigned i = 1; i < op->getNumResults(); ++i)
      resultTypes.push_back(op->getResult(i).getType());

    SmallVector<Value, 4> dependencies = adaptor.getAsyncDependencies();
    SmallVector<Value, 4> operands;
    auto newOp = rewriter.create<async::ExecuteOp>(
        op->getLoc(), resultTypes, dependencies, operands,
        [&](OpBuilder &b, Location loc, ValueRange v) {
          IRMapping map;
          for (auto &o : op.getOps()) {
            if (isa<air::ExecuteTerminatorOp>(o)) {
              SmallVector<Value, 4> returnValues;
              for (auto v : o.getOperands())
                returnValues.push_back(map.lookupOrDefault(v));
              b.create<async::YieldOp>(loc, returnValues);
            } else
              b.clone(o, map);
          }
        });

    SmallVector<Value, 4> results{newOp->getResult(0)};
    op.getResult(0).replaceAllUsesWith(newOp->getResult(0));
    for (unsigned i = 1; i < op->getNumResults(); ++i) {
      auto r = newOp.getResult(i);
      auto await = rewriter.create<async::AwaitOp>(op->getLoc(), r);
      // op.getResult(i).replaceAllUsesWith(await.getResult());
      results.push_back(await.getResult());
    }
    // rewriter.eraseOp(op);
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct CallOpConversion : public OpConversionPattern<func::CallOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> retTys;
    if (typeConverter->convertTypes(op.getResultTypes(), retTys).failed())
      return failure();
    rewriter.replaceOpWithNewOp<func::CallOp>(op, adaptor.getCallee(), retTys,
                                              adaptor.getOperands());
    return success();
  }
};

struct ChannelOpConversion : public OpConversionPattern<air::ChannelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(air::ChannelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    SmallVector<int64_t, 2> shape;
    for (auto i : op.getSize()) {
      shape.push_back(i.dyn_cast<IntegerAttr>().getInt());
    }
    // if channel dim < 2, add until dim = 2
    while (shape.size() < 2) {
      shape.push_back(1);
    }

    auto memrefType =
        MemRefType::get(shape, IntegerType::get(op->getContext(), 64));
    auto name = op.getSymName();
    rewriter.eraseOp(op);

    auto ptrType = rewriter.getIntegerType(64);
    auto initialValue = mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get(shape, ptrType),
        rewriter.getIntegerAttr(ptrType, 0));
    auto globalOp = rewriter.create<memref::GlobalOp>(
        op->getLoc(), name.str(), rewriter.getStringAttr("private"), memrefType,
        initialValue, false, nullptr);
    // if op has broadcast attribute, attach it to the global
    if (op->getAttr("broadcast_shape")) {
      globalOp->setAttr("broadcast_shape", op->getAttr("broadcast_shape"));
    }
    return success();
  }
};

class ChannelGetOpConversion : public OpConversionPattern<air::ChannelGetOp> {
public:
  using OpConversionPattern<air::ChannelGetOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(air::ChannelGetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    auto channel_name = op.getChanNameAttr();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto channelOp = moduleOp.lookupSymbol<memref::GlobalOp>(channel_name);
    if (!channelOp)
      return failure();
    auto memrefType = channelOp.getType();
    auto channelPtr = rewriter.create<memref::GetGlobalOp>(
        op->getLoc(), memrefType, op.getChanNameAttr());
    operands.push_back(channelPtr);
    operands.append(adaptor.getOperands().begin(), adaptor.getOperands().end());
    auto call = convertOpToFunction(op, operands, rewriter, "air_channel_get");
    if (call)
      return success();
    else
      return failure();
  }
};

class ChannelPutOpConversion : public OpConversionPattern<air::ChannelPutOp> {
public:
  using OpConversionPattern<air::ChannelPutOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(air::ChannelPutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands;
    auto channel_name = op.getChanNameAttr();
    auto moduleOp = op->getParentOfType<ModuleOp>();
    auto channelOp = moduleOp.lookupSymbol<memref::GlobalOp>(channel_name);
    if (!channelOp)
      return failure();
    auto memrefType = channelOp.getType();
    auto channelPtr = rewriter.create<memref::GetGlobalOp>(
        op->getLoc(), memrefType, op.getChanNameAttr());
    operands.push_back(channelPtr);
    // create constant index op for channel array size
    for (auto i : memrefType.getShape()) {
      operands.push_back(
          rewriter.create<arith::ConstantIndexOp>(op->getLoc(), i));
    }
    // if shape dim < 2, add until dim = 2
    while (operands.size() < 3) {
      operands.push_back(
          rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1));
    }
    // if channel is broadcast, add broadcast shape
    if (channelOp->getAttr("broadcast_shape")) {
      for (auto i : channelOp->getAttr("broadcast_shape").cast<ArrayAttr>()) {
        operands.push_back(rewriter.create<arith::ConstantIndexOp>(
            op->getLoc(), i.cast<IntegerAttr>().getInt()));
      }
    } else {
      // if channel is not broadcast, add 1
      operands.push_back(
          rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1));
      operands.push_back(
          rewriter.create<arith::ConstantIndexOp>(op->getLoc(), 1));
    }
    operands.append(adaptor.getOperands().begin(), adaptor.getOperands().end());
    auto call = convertOpToFunction(op, operands, rewriter, "air_channel_put");
    if (call)
      return success();
    else
      return failure();
  }
};

class AIRToAsyncPass : public AIRToAsyncBase<AIRToAsyncPass> {

public:
  AIRToAsyncPass() = default;
  AIRToAsyncPass(const AIRToAsyncPass &pass) {}

  void getDependentDialects(::mlir::DialectRegistry &registry) const override {
    registry.insert<AffineDialect>();
  }

  void runOnOperation() override {

    auto module = getOperation();
    auto context = module.getContext();

    TypeConverter converter;
    converter.addConversion([&](Type type) -> Optional<Type> {
      // convert air::AsyncTokenType to async::TokenType
      if (auto t = type.dyn_cast<air::AsyncTokenType>())
        return async::TokenType::get(context);
      if (auto t = type.dyn_cast<MemRefType>())
        if (t.getMemorySpaceAsInt() != 0)
          return MemRefType::get(t.getShape(), t.getElementType(),
                                 t.getLayout(), 0);
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

    target.addLegalDialect<LLVM::LLVMDialect, func::FuncDialect,
                           arith::ArithDialect, AffineDialect, scf::SCFDialect,
                           linalg::LinalgDialect, memref::MemRefDialect,
                           bufferization::BufferizationDialect,
                           xilinx::airrt::AIRRtDialect, async::AsyncDialect,
                           mlir::BuiltinDialect>();

    // air.memcpy_nd conversion
    RewritePatternSet air_dma_patterns(context);

    air_dma_patterns.add<AIRDmaMemcpyNdToMemcpyConversion, ExecuteOpConversion,
                         WaitAllOpConversion, ChannelOpConversion>(context);

    if (failed(applyPartialConversion(module, target,
                                      std::move(air_dma_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air dialect\n");
      signalPassFailure();
    }

    RewritePatternSet air_herd_patterns(context);
    air_herd_patterns.add<AIRHerdOpConversion>(context);
    if (failed(applyPartialConversion(module, target,
                                      std::move(air_herd_patterns)))) {
      emitError(UnknownLoc::get(context), "error lowering air.herd\n");
      signalPassFailure();
    }

    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return converter.isSignatureLegal(op.getFunctionType());
    });

    target.addDynamicallyLegalOp<func::CallOp>([&](func::CallOp op) {
      auto isIllegal = [](Type t) {
        if (t.isa<air::AsyncTokenType>())
          return true;
        if (auto mt = t.dyn_cast<MemRefType>())
          return mt.getMemorySpaceAsInt() != 0;
        return false;
      };
      return (!llvm::any_of(op->getResultTypes(), isIllegal) &&
              !llvm::any_of(op->getOperandTypes(), isIllegal));
    });

    target.addDynamicallyLegalOp<scf::ForOp>([&](scf::ForOp op) {
      for (auto o : op.getRegionIterArgs()) {
        if (o.getType().isa<air::AsyncTokenType>())
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<scf::ParallelOp>([&](scf::ParallelOp op) {
      // for (auto o : op.getRegionIterArgs()) {
      //   if (o.getType().isa<air::AsyncTokenType>())
      //     return false;
      // }
      // return true;
      return false;
    });

    target.addDynamicallyLegalOp<scf::YieldOp>([&](scf::YieldOp op) {
      for (auto v : op.getResults()) {
        if (v.getType().isa<air::AsyncTokenType>())
          return false;
      }
      return true;
    });

    target.addDynamicallyLegalOp<memref::AllocOp>([&](memref::AllocOp op) {
      return (op.getType().getMemorySpaceAsInt() == 0);
    });

    target.addDynamicallyLegalOp<memref::DeallocOp>([&](memref::DeallocOp op) {
      return (
          op.getMemref().getType().cast<MemRefType>().getMemorySpaceAsInt() ==
          0);
    });

    RewritePatternSet typeConversionPatterns(context);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        typeConversionPatterns, converter);

    typeConversionPatterns
        .add<ScfYieldOpConversion, ScfForOpConversion, ScfParallelOpConversion,
             AsyncCallOpConversion, AllocOpConversion, DeallocOpConversion,
             CallOpConversion, ChannelGetOpConversion, ChannelPutOpConversion>(
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

std::unique_ptr<mlir::Pass> createAIRToAsyncPass() {
  return std::make_unique<AIRToAsyncPass>();
}

} // namespace air
} // namespace xilinx
