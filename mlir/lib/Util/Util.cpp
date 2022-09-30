//===- Util.cpp -------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OperationSupport.h"

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
    } else if (mrt.hasRank()) {
      ret << "D" << mrt.getRank();
    }
    const Type elem = mrt.getElementType();
    ret << getMangledType(elem);
  } else if (FloatType ft = ty.dyn_cast<FloatType>()) {
    ret << "F" << ft.getWidth();
  } else if (const IntegerType it = ty.dyn_cast<const IntegerType>()) {
    ret << "I" << it.getWidth();
  } else if (const IndexType it = ty.dyn_cast<const IndexType>()) {
    ret << "I64";
  } else if (ty.dyn_cast<air::AsyncTokenType>()) {
    ret << "E";
  } else {
    Type t = ty;
    t.dump();
    assert(0 && "unhandled type in getMangledType");
  }
  return ret.str();
}

std::string getMangledFuncName(ModuleOp module, std::string prefix,
                               FunctionType fnTy) {
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
} // namespace

void coalesceLoops(AffineForOp outer, AffineForOp inner) {
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
  auto iv_new_inner = builder.create<AffineApplyOp>(
      loc, AffineMap::get(1, 0, iv_new_inner_expr), iv_new);
  auto iv_new_outer = builder.create<AffineApplyOp>(
      loc, AffineMap::get(1, 0, iv_new_outer_expr), iv_new);
  SmallPtrSet<Operation *, 2> keep{iv_new_inner, iv_new_outer};
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

void normalizeLoop(AffineForOp afo) {
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

func::FuncOp getMangledFunction(ModuleOp module, std::string prefix,
                                ArrayRef<Value> operands,
                                ArrayRef<Type> retTys) {
  Builder builder(module);

  SmallVector<Type, 16> tys;
  for (auto o : operands)
    tys.push_back(o.getType());

  auto fnTy = builder.getFunctionType(tys, retTys);

  std::string fnName = getMangledFuncName(module, prefix, fnTy);
  auto fn = module.lookupSymbol<func::FuncOp>(fnName);

  if (!fn) {
    fn = func::FuncOp::create(builder.getUnknownLoc(), fnName, fnTy);
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
  } else {
    return 1;
  }
}

// Get the parent scf.for op of an iter_arg
scf::ForOp getForRegionIterArgsOwner(Value val) {
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return scf::ForOp();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast<scf::ForOp>(containingOp);
}

// Get the parent air.launch_herd op of a tile id
air::HerdOp getHerdArgOwner(Value val) {
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return air::HerdOp();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast<air::HerdOp>(containingOp);
}

// Get the parent air.hierarchy op of a tile id
air::HierarchyInterface getHierarchyArgOwner(Value val) {
  auto ivArg = val.dyn_cast<BlockArgument>();
  if (!ivArg)
    return air::HierarchyInterface();
  assert(ivArg.getOwner() && "unlinked block argument");
  auto *containingOp = ivArg.getOwner()->getParentOp();
  return dyn_cast<air::HierarchyInterface>(containingOp);
}

// Get operation's "id" attribute
int getIdAttr(Operation *op) {
  auto idAttr = op->getAttrOfType<IntegerAttr>("id");
  assert(idAttr && "op has no attribute named 'id'");
  return idAttr.getInt();
}

// Renumber the DMA ops
void renumberDmaOps(func::FuncOp func, std::string mode = "herd") {
  unsigned id = 0;
  if (mode == "global") {
    // Renumber DMA ops per entire module
    func->walk([&](Operation *func_dma) {
      if (dyn_cast<xilinx::air::DmaMemcpyInterface>(func_dma)) {
        func_dma->setAttr(
            "id",
            mlir::IntegerAttr::get(
                mlir::IntegerType::get(func_dma->getContext(), 32), ++id));
      }
    });
  } else if (mode == "herd") {
    for (auto herd : func.getOps<xilinx::air::HerdOp>()) {
      id = 0;
      // Renumber DMA ops per air herd
      herd->walk([&](Operation *herd_dma) {
        if (dyn_cast<xilinx::air::DmaMemcpyInterface>(herd_dma)) {
          herd_dma->setAttr(
              "id",
              mlir::IntegerAttr::get(
                  mlir::IntegerType::get(herd_dma->getContext(), 32), ++id));
        }
      });
    }
  } else
    assert(false && "Unknown dma renumber mode. Supported modes: global, herd");
}

} // namespace air
} // namespace xilinx