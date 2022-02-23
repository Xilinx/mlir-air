// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "air-util"

using namespace mlir;

namespace xilinx {
namespace air {

namespace {
  
std::string getMangledType(const Type ty) {
  std::stringstream ret;

  if (const MemRefType mrt = ty.dyn_cast<const MemRefType>()) {
    ret << "M";
    ret << mrt.getMemorySpaceAsInt();
    if (mrt.hasStaticShape()) {
      auto shape = mrt.getShape();
      for (auto s : shape)
        ret << s << "x";
    }
    else if (mrt.hasRank()) {
      ret << "D" << mrt.getRank();
    }
    const Type elem = mrt.getElementType();
    ret << getMangledType(elem);
  }
  else if (FloatType ft = ty.dyn_cast<FloatType>()) {
    ret << "F" << ft.getWidth();
  }
  else if (const IntegerType it = ty.dyn_cast<const IntegerType>()) {
    ret << "I" << it.getWidth();
  }
  else if (const IndexType it = ty.dyn_cast<const IndexType>()) {
    ret << "I64";
  }
  else if (ty.dyn_cast<air::AsyncTokenType>()) {
    ret << "E";
  }
  else {
    Type t = ty;
    t.dump();
    assert(0 && "unhandled type in getMangledType");
  }
  return ret.str();
}

std::string getMangledFuncName(ModuleOp module, std::string prefix, FunctionType fnTy) {
  std::string sep = "_";

  auto resultTy = fnTy.getResults();
  auto operTy = fnTy.getInputs();

  std::string ret = prefix;
  for (const Type t : resultTy)
    ret = ret + sep + "r" + getMangledType(t);
  for (const Type t : operTy)
    ret = ret + sep + getMangledType(t);

  return ret;
}
}

void coalesceLoops(AffineForOp outer, AffineForOp inner)
{
  auto ctx = outer.getContext();
  auto loc = outer.getLoc();
  auto builder = OpBuilder::atBlockBegin(outer.getBody());
  // ub_new = ub_inner*ub_outer
  // iv_new = 0...ub_new-1
  // iv_new_inner = mod(iv_new, ub_inner)
  // iv_new_outer = floordiv(iv_new, ub_inner)
  auto ub_inner_expr = inner.getUpperBoundMap().getResult(0);
  auto ub_outer_expr = outer.getUpperBoundMap().getResult(0);
  auto ub_new_expr = ub_inner_expr * ub_outer_expr;
  auto iv_new_inner_expr = getAffineDimExpr(0, ctx) % ub_inner_expr;
  auto iv_new_outer_expr = getAffineDimExpr(0, ctx).floorDiv(ub_inner_expr);

  outer.setUpperBoundMap(AffineMap::get(0, 0, ub_new_expr));
  auto iv_new = outer.getInductionVar();
  auto iv_new_inner = builder.create<AffineApplyOp>(loc,
                                                    AffineMap::get(1, 0, iv_new_inner_expr),
                                                    iv_new);
  auto iv_new_outer = builder.create<AffineApplyOp>(loc,
                                                    AffineMap::get(1, 0, iv_new_outer_expr),
                                                    iv_new);
  SmallPtrSet<Operation *, 2> keep{iv_new_inner,iv_new_outer};
  iv_new.replaceAllUsesExcept(iv_new_outer, keep);
  inner.getInductionVar().replaceAllUsesWith(iv_new_inner);
  // erase terminator from inner loop's body
  inner.getBody()->back().erase();
  // move inner loop's body to outer loop
  outer.getBody()->getOperations().splice(Block::iterator(inner.getOperation()),
                                          inner.getBody()->getOperations());
  inner.erase();
  return;
}

void normalizeLoop(AffineForOp afo)
{
  auto ubMap = afo.getUpperBoundMap();
  auto lbMap = afo.getLowerBoundMap();
  auto ctx = afo.getContext();
  auto loc = afo.getLoc();

  auto step_expr = getAffineConstantExpr(afo.getStep(), ctx);

  auto ub_expr = ubMap.getResult(0);
  auto lb_expr = lbMap.getResult(0);
  auto sub_expr = ub_expr - lb_expr;
  auto new_ub_expr = sub_expr.ceilDiv(step_expr);

  auto iv = afo.getInductionVar();

  afo.setLowerBoundMap(AffineMap::get(0, 0, getAffineConstantExpr(0, ctx)));
  afo.setUpperBoundMap(AffineMap::get(0, 0, new_ub_expr));
  afo.setStep(1);

  auto dim0_expr = getAffineDimExpr(0, ctx);
  auto iv_expr = dim0_expr * step_expr + lb_expr;
  auto iv_map = AffineMap::get(1, 0, iv_expr);
  auto builder = OpBuilder::atBlockBegin(afo.getBody());
  auto new_iv = builder.create<AffineApplyOp>(loc, iv_map, iv);
  SmallPtrSet<Operation *, 1> keep{new_iv};
  iv.replaceAllUsesExcept(new_iv, keep);
  return;
}

FuncOp getMangledFunction(ModuleOp module, std::string prefix, ArrayRef<Value> operands, ArrayRef<Type> retTys)
{
  Builder builder(module);

  SmallVector<Type, 16> tys;
  for (auto o : operands)
    tys.push_back(o.getType());

  auto fnTy = builder.getFunctionType(tys, retTys);

  std::string fnName = getMangledFuncName(module, prefix, fnTy);
  auto fn = module.lookupSymbol<FuncOp>(fnName);

  if (!fn) {
    fn = FuncOp::create(builder.getUnknownLoc(), fnName, fnTy);
    fn.setPrivate();
    module.push_back(fn);
  }

  return fn;
}

uint64_t getTensorVolume(const ShapedType ty) {

  if (!ty.hasRank())
    return 1;

  uint64_t volume = 1;
  for (auto &d : ty.getShape())
    volume *= d;
  return volume;
}

uint64_t getTensorVolume(const Type ty) {
  if (auto t = ty.dyn_cast<ShapedType>()) {
    return getTensorVolume(t);
  }
  else {
    return 1;
  }
}

}
}
