//===- AIRVerifyHierarchyLocality.cpp ---------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRVerifyHierarchyLocality.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#include <functional>

#define DEBUG_TYPE "air-verify-hierarchy-locality"

using namespace mlir;

namespace {

// The memory level that "matches" each hierarchy op. An L1 alloc is the local
// scratchpad of an air.herd; an L2 alloc lives in segment-shared memory; an L3
// alloc is host/system memory.
static std::optional<xilinx::air::MemorySpace>
matchingMemoryLevel(xilinx::air::HierarchyInterface H) {
  Operation *op = H.getOperation();
  if (isa<xilinx::air::HerdOp>(op))
    return xilinx::air::MemorySpace::L1;
  if (isa<xilinx::air::SegmentOp>(op))
    return xilinx::air::MemorySpace::L2;
  if (isa<xilinx::air::LaunchOp>(op))
    return xilinx::air::MemorySpace::L3;
  return std::nullopt;
}

static bool memrefIsAtLevel(MemRefType ty, xilinx::air::MemorySpace level) {
  return ty.getMemorySpaceAsInt() == static_cast<unsigned>(level);
}

// True if `op` is lexically inside `region` (any depth).
static bool insideRegion(Region &region, Operation *op) {
  if (!op)
    return false;
  for (Operation *p = op; p; p = p->getParentOp())
    if (p->getParentRegion() == &region)
      return true;
  return false;
}

// Walk the use chain of a kernel block argument inside the hierarchy body,
// descending through pure, view-producing ops, and collect every "terminal
// access" — an op that consumes the memref as a memory operation
// (DMA, channel, load/store, subview, opaque call, etc.). The op is stored
// together with the value through which it consumes the memref, so the
// caller can ask "which side of a dma_memcpy_nd is this".
struct TerminalAccess {
  Operation *op;
  Value consumed; // The SSA value at which `op` consumes the memref.
};

static void collectTerminalAccesses(Value root, Region &body,
                                    SmallVectorImpl<TerminalAccess> &out) {
  llvm::SmallPtrSet<Operation *, 16> visited;
  SmallVector<Value> worklist;
  worklist.push_back(root);

  while (!worklist.empty()) {
    Value v = worklist.pop_back_val();
    for (OpOperand &use : v.getUses()) {
      Operation *user = use.getOwner();
      if (!insideRegion(body, user))
        continue;
      if (!visited.insert(user).second)
        continue;

      // memref.subview is a *region-narrowing* terminal: its offsets/sizes
      // encode exactly which slice of the memref this PE will touch. Record
      // it as the terminal access and stop — descendants operate on the
      // narrowed view, so re-checking them is redundant and would over-flag
      // benign callers (e.g., linalg.fill on the subview).
      if (isa<memref::SubViewOp>(user)) {
        out.push_back({user, v});
        continue;
      }

      // Type-changing view ops (cast / reinterpret_cast / expand_shape /
      // collapse_shape / transpose) do not narrow the access region — they
      // are pure passthroughs as far as disjointness is concerned. Descend
      // into uses; do not record as a terminal.
      if (isa<memref::CastOp, memref::ExpandShapeOp, memref::CollapseShapeOp,
              memref::TransposeOp, memref::ReinterpretCastOp>(user)) {
        for (Value r : user->getResults())
          worklist.push_back(r);
        continue;
      }

      // air.execute is transparent: the memref is yielded back unchanged, so
      // descend into its results so users of the yielded value are visited.
      if (isa<xilinx::air::ExecuteOp>(user)) {
        for (Value r : user->getResults())
          worklist.push_back(r);
        continue;
      }

      // Nested hierarchy ops (e.g., air.segment / air.herd inside an
      // air.launch): the kernel operand is forwarded through to a block
      // argument of the inner op's body. Descend by following the tied
      // block argument so accesses inside the nested body are visited.
      // This is what makes verification of outer-level operands work when
      // the actual access happens deeper in the nest.
      if (auto innerH = dyn_cast<xilinx::air::HierarchyInterface>(user)) {
        // Find the operand index of `v` among innerH's kernel operands.
        for (unsigned i = 0, e = innerH.getNumKernelOperands(); i < e; ++i) {
          if (innerH.getKernelOperand(i) == v) {
            worklist.push_back(innerH.getKernelArgument(i));
          }
        }
        continue;
      }

      // Everything else is a terminal access from this verifier's perspective.
      out.push_back({user, v});
    }
  }
}

// Resolve `v` to a constant index value if statically derivable, else nullopt.
static std::optional<int64_t> staticInt(Value v) {
  if (!v)
    return std::nullopt;
  return getConstantIntValue(v);
}

// True if `v` is provably zero. Recognizes:
//   - arith.constant 0
//   - block argument that is a hierarchy iteration variable whose
//     iteration-space size is statically 1 (only one value: 0)
//   - block argument that is a kernel arg of a hierarchy op whose tied
//     operand isProvablyZero (passthrough)
//   - affine.apply with constant map evaluating to 0 OR with all-zero inputs
//     and a map of the form (...) -> sum of c_i * x_i (no constant term)
//   - arith.addi/subi where both operands are provably zero
//   - arith.muli where at least one operand is provably zero
static bool isProvablyZero(Value v) {
  if (!v)
    return false;
  if (auto c = staticInt(v))
    return *c == 0;
  if (auto ba = dyn_cast<BlockArgument>(v)) {
    Operation *parent = ba.getOwner()->getParentOp();
    auto H = dyn_cast_if_present<xilinx::air::HierarchyInterface>(parent);
    if (!H)
      return false;
    ArrayRef<BlockArgument> ids = H.getIds();
    auto idIt = llvm::find(ids, ba);
    if (idIt != ids.end()) {
      unsigned i = std::distance(ids.begin(), idIt);
      auto sz = staticInt(H.getSizeOperands()[i]);
      return sz && *sz == 1;
    }
    ArrayRef<BlockArgument> kargs = H.getKernelArguments();
    auto kIt = llvm::find(kargs, ba);
    if (kIt != kargs.end()) {
      unsigned i = std::distance(kargs.begin(), kIt);
      return isProvablyZero(H.getKernelOperand(i));
    }
    return false;
  }
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (auto add = dyn_cast<arith::AddIOp>(def))
    return isProvablyZero(add.getLhs()) && isProvablyZero(add.getRhs());
  if (auto sub = dyn_cast<arith::SubIOp>(def))
    return isProvablyZero(sub.getLhs()) && isProvablyZero(sub.getRhs());
  if (auto mul = dyn_cast<arith::MulIOp>(def))
    return isProvablyZero(mul.getLhs()) || isProvablyZero(mul.getRhs());
  if (auto cast = dyn_cast<arith::IndexCastOp>(def))
    return isProvablyZero(cast.getIn());
  if (auto apply = dyn_cast<affine::AffineApplyOp>(def)) {
    for (Value in : apply.getOperands())
      if (!isProvablyZero(in))
        return false;
    // All inputs are zero → result is the affine map evaluated at zeros.
    AffineMap map = apply.getAffineMap();
    if (map.getNumResults() != 1)
      return false;
    SmallVector<int64_t> zeroDims(map.getNumDims(), 0);
    SmallVector<int64_t> zeroSyms(map.getNumSymbols(), 0);
    std::function<std::optional<int64_t>(AffineExpr)> eval =
        [&](AffineExpr e) -> std::optional<int64_t> {
      if (auto c = dyn_cast<AffineConstantExpr>(e))
        return c.getValue();
      if (auto d = dyn_cast<AffineDimExpr>(e))
        return zeroDims[d.getPosition()];
      if (auto s = dyn_cast<AffineSymbolExpr>(e))
        return zeroSyms[s.getPosition()];
      if (auto bin = dyn_cast<AffineBinaryOpExpr>(e)) {
        auto lhs = eval(bin.getLHS());
        auto rhs = eval(bin.getRHS());
        if (!lhs || !rhs)
          return std::nullopt;
        switch (bin.getKind()) {
        case AffineExprKind::Add:
          return *lhs + *rhs;
        case AffineExprKind::Mul:
          return *lhs * *rhs;
        case AffineExprKind::Mod:
          return *rhs == 0 ? std::nullopt : std::optional<int64_t>(*lhs % *rhs);
        case AffineExprKind::FloorDiv:
          return *rhs == 0 ? std::nullopt : std::optional<int64_t>(*lhs / *rhs);
        case AffineExprKind::CeilDiv:
          return *rhs == 0 ? std::nullopt
                           : std::optional<int64_t>((*lhs + *rhs - 1) / *rhs);
        default:
          return std::nullopt;
        }
      }
      return std::nullopt;
    };
    auto val = eval(map.getResult(0));
    return val && *val == 0;
  }
  return false;
}

// Resolve `v` to a (BlockArgument, multiplier) pair if `v` is an affine-style
// expression of exactly one block argument with a positive constant scaling
// factor, otherwise return nullopt. Walks through:
//   - the block argument itself        →  (BA, 1)
//   - arith.muli/addi/subi by constant
//   - affine.apply with a single-symbol/dim affine_map of the form (s) -> s*c
//
// Returns the BA and the absolute coefficient |c|; sign doesn't matter for
// disjointness because what counts is whether distinct BA values produce
// distinct memory points spaced by at least the access size.
static std::optional<std::pair<BlockArgument, int64_t>>
asLinearInOneBlockArg(Value v) {
  if (!v)
    return std::nullopt;
  if (auto ba = dyn_cast<BlockArgument>(v)) {
    // If this block argument is a kernel argument of an inner hierarchy op
    // (e.g., %arg10 in `air.segment ... args(%arg10 = %launch_iv)`), trace
    // back through the tied kernel operand. This lets us prove that an
    // offset like `affine.apply (%segment_arg * 64)` is actually a function
    // of the OUTER launch IV, when checking from the launch's perspective.
    Operation *parent = ba.getOwner()->getParentOp();
    if (auto innerH =
            dyn_cast_if_present<xilinx::air::HierarchyInterface>(parent)) {
      ArrayRef<BlockArgument> kernelArgs = innerH.getKernelArguments();
      auto it = llvm::find(kernelArgs, ba);
      if (it != kernelArgs.end()) {
        unsigned idx = std::distance(kernelArgs.begin(), it);
        Value tiedOperand = innerH.getKernelOperand(idx);
        // Recurse on the operand passed in from the outer scope.
        return asLinearInOneBlockArg(tiedOperand);
      }
    }
    return std::make_pair(ba, int64_t{1});
  }

  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;

  if (auto cst = staticInt(v))
    return std::nullopt; // pure constant: no IV dependence

  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    auto l = asLinearInOneBlockArg(add.getLhs());
    auto r = asLinearInOneBlockArg(add.getRhs());
    bool rConst =
        staticInt(add.getRhs()).has_value() || isProvablyZero(add.getRhs());
    bool lConst =
        staticInt(add.getLhs()).has_value() || isProvablyZero(add.getLhs());
    if (l && rConst)
      return l;
    if (r && lConst)
      return r;
    return std::nullopt;
  }
  if (auto sub = dyn_cast<arith::SubIOp>(def)) {
    auto l = asLinearInOneBlockArg(sub.getLhs());
    bool rConst =
        staticInt(sub.getRhs()).has_value() || isProvablyZero(sub.getRhs());
    if (l && rConst)
      return l;
    return std::nullopt;
  }
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    auto l = asLinearInOneBlockArg(mul.getLhs());
    auto r = asLinearInOneBlockArg(mul.getRhs());
    auto lc = staticInt(mul.getLhs());
    auto rc = staticInt(mul.getRhs());
    if (l && rc)
      return std::make_pair(l->first, l->second * std::abs(*rc));
    if (r && lc)
      return std::make_pair(r->first, r->second * std::abs(*lc));
    return std::nullopt;
  }
  if (auto cast = dyn_cast<arith::IndexCastOp>(def))
    return asLinearInOneBlockArg(cast.getIn());

  if (auto apply = dyn_cast<affine::AffineApplyOp>(def)) {
    AffineMap map = apply.getAffineMap();
    if (map.getNumResults() != 1)
      return std::nullopt;
    AffineExpr expr = map.getResult(0);
    // Find a single dim or symbol that is the only non-constant input,
    // and require the expression to be linear in it.
    SmallVector<Value> inputs(apply.getOperands().begin(),
                              apply.getOperands().end());
    BlockArgument theBA = nullptr;
    unsigned theIdx = 0;
    bool theIsSymbol = false;
    for (unsigned i = 0; i < inputs.size(); ++i) {
      if (staticInt(inputs[i]))
        continue;
      auto inner = asLinearInOneBlockArg(inputs[i]);
      if (!inner || inner->second != 1)
        return std::nullopt;
      if (theBA && theBA != inner->first)
        return std::nullopt; // multiple distinct block args
      theBA = inner->first;
      theIdx = i;
      theIsSymbol = i >= map.getNumDims();
    }
    if (!theBA)
      return std::nullopt;

    // Substitute constants for all other inputs and ask AffineExpr what the
    // coefficient on the surviving variable is.
    SmallVector<AffineExpr> dimSubs(map.getNumDims());
    SmallVector<AffineExpr> symSubs(map.getNumSymbols());
    auto *ctx = expr.getContext();
    for (unsigned i = 0; i < inputs.size(); ++i) {
      bool isSymbol = i >= map.getNumDims();
      unsigned localIdx = isSymbol ? i - map.getNumDims() : i;
      if (i == theIdx) {
        AffineExpr placeholder = isSymbol ? getAffineSymbolExpr(localIdx, ctx)
                                          : getAffineDimExpr(localIdx, ctx);
        if (isSymbol)
          symSubs[localIdx] = placeholder;
        else
          dimSubs[localIdx] = placeholder;
      } else {
        auto c = staticInt(inputs[i]);
        if (!c)
          return std::nullopt;
        AffineExpr cstExpr = getAffineConstantExpr(*c, ctx);
        if (isSymbol)
          symSubs[localIdx] = cstExpr;
        else
          dimSubs[localIdx] = cstExpr;
      }
    }
    AffineExpr substituted = expr.replaceDimsAndSymbols(dimSubs, symSubs);
    // Substituted expression should now be of the form `c0 * IV + c1`.
    // Probe for the coefficient by evaluating at IV=0 and IV=1.
    SmallVector<int64_t> probe0(map.getNumDims(), 0),
        probe0Sym(map.getNumSymbols(), 0);
    SmallVector<int64_t> probe1(map.getNumDims(), 0),
        probe1Sym(map.getNumSymbols(), 0);
    if (theIsSymbol) {
      probe1Sym[theIdx - map.getNumDims()] = 1;
    } else {
      probe1[theIdx] = 1;
    }
    std::function<std::optional<int64_t>(AffineExpr, ArrayRef<int64_t>,
                                         ArrayRef<int64_t>)>
        eval = [&](AffineExpr e, ArrayRef<int64_t> dims,
                   ArrayRef<int64_t> syms) -> std::optional<int64_t> {
      // Simple recursive evaluator for linear AffineExpr trees.
      if (auto cst = dyn_cast<AffineConstantExpr>(e))
        return cst.getValue();
      if (auto d = dyn_cast<AffineDimExpr>(e))
        return dims[d.getPosition()];
      if (auto s = dyn_cast<AffineSymbolExpr>(e))
        return syms[s.getPosition()];
      if (auto bin = dyn_cast<AffineBinaryOpExpr>(e)) {
        auto lhs = eval(bin.getLHS(), dims, syms);
        auto rhs = eval(bin.getRHS(), dims, syms);
        if (!lhs || !rhs)
          return std::nullopt;
        switch (bin.getKind()) {
        case AffineExprKind::Add:
          return *lhs + *rhs;
        case AffineExprKind::Mul:
          return *lhs * *rhs;
        case AffineExprKind::Mod:
          if (*rhs == 0)
            return std::nullopt;
          return *lhs % *rhs;
        case AffineExprKind::FloorDiv:
          if (*rhs == 0)
            return std::nullopt;
          return *lhs / *rhs;
        case AffineExprKind::CeilDiv:
          if (*rhs == 0)
            return std::nullopt;
          return (*lhs + *rhs - 1) / *rhs;
        default:
          return std::nullopt;
        }
      }
      return std::nullopt;
    };
    auto v0 = eval(substituted, probe0, probe0Sym);
    auto v1 = eval(substituted, probe1, probe1Sym);
    if (!v0 || !v1)
      return std::nullopt;
    int64_t coef = std::abs(*v1 - *v0);
    if (coef == 0)
      return std::nullopt;
    return std::make_pair(theBA, coef);
  }

  return std::nullopt;
}

// Resolve an OpFoldResult-like (Value or constant int) to an int64_t when
// possible. Returns nullopt if the size is dynamic-and-unresolvable.
static std::optional<int64_t> resolveSize(Value v,
                                          std::optional<int64_t> staticSz) {
  if (staticSz)
    return staticSz;
  if (v)
    return staticInt(v);
  return std::nullopt;
}

// Static iteration extent of an scf.for / scf.parallel IV, or nullopt if not
// statically known. Returns (upper - lower + step - 1) / step (ceil-div),
// counting the number of distinct iv values.
static std::optional<int64_t> scfIvExtent(BlockArgument iv) {
  Operation *parent = iv.getOwner()->getParentOp();
  Value lb, ub, step;
  unsigned ivIdx = ~0u;
  if (auto forOp = dyn_cast_if_present<scf::ForOp>(parent)) {
    if (iv != forOp.getInductionVar())
      return std::nullopt;
    lb = forOp.getLowerBound();
    ub = forOp.getUpperBound();
    step = forOp.getStep();
  } else if (auto parOp = dyn_cast_if_present<scf::ParallelOp>(parent)) {
    auto ivs = parOp.getInductionVars();
    auto it = llvm::find(ivs, iv);
    if (it == ivs.end())
      return std::nullopt;
    ivIdx = std::distance(ivs.begin(), it);
    lb = parOp.getLowerBound()[ivIdx];
    ub = parOp.getUpperBound()[ivIdx];
    step = parOp.getStep()[ivIdx];
  } else {
    return std::nullopt;
  }
  auto lbC = staticInt(lb);
  auto ubC = staticInt(ub);
  auto stC = staticInt(step);
  if (!lbC || !ubC || !stC || *stC <= 0 || *ubC <= *lbC)
    return std::nullopt;
  return (*ubC - *lbC + *stC - 1) / *stC;
}

// Return the static iteration extent of a hierarchy IV's iteration space, or
// nullopt if not statically known. Caller is responsible for ensuring `iv` is
// a hierarchy IV (BlockArgument owned by an air HierarchyInterface).
static std::optional<int64_t> hierarchyIvExtent(BlockArgument iv) {
  Operation *parent = iv.getOwner()->getParentOp();
  auto H = dyn_cast_if_present<xilinx::air::HierarchyInterface>(parent);
  if (!H)
    return std::nullopt;
  ArrayRef<BlockArgument> ids = H.getIds();
  auto it = llvm::find(ids, iv);
  if (it == ids.end())
    return std::nullopt;
  unsigned i = std::distance(ids.begin(), it);
  return staticInt(H.getSizeOperands()[i]);
}

// True if `v` transitively depends on any of `ivs` through a chain of pure
// index-arithmetic ops and hierarchy passthrough kernel args. Conservative:
// only descends through arith index ops and affine.apply (does NOT descend
// through memref.load or other side-effecting ops). Used both for the
// channel-index per-PE replication signal and for detecting func.call
// scalar operands that carry the iv as a partition hint.
static bool dependsOnAnyIv(Value v, ArrayRef<BlockArgument> ivs) {
  llvm::SmallPtrSet<Value, 8> visited;
  std::function<bool(Value)> rec = [&](Value v) -> bool {
    if (!v || !visited.insert(v).second)
      return false;
    if (auto ba = dyn_cast<BlockArgument>(v)) {
      if (llvm::is_contained(ivs, ba))
        return true;
      Operation *parent = ba.getOwner()->getParentOp();
      if (auto innerH =
              dyn_cast_if_present<xilinx::air::HierarchyInterface>(parent)) {
        ArrayRef<BlockArgument> kargs = innerH.getKernelArguments();
        auto it = llvm::find(kargs, ba);
        if (it != kargs.end()) {
          unsigned i = std::distance(kargs.begin(), it);
          return rec(innerH.getKernelOperand(i));
        }
      }
      return false;
    }
    Operation *def = v.getDefiningOp();
    if (!def)
      return false;
    if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
             arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::IndexCastOp,
             affine::AffineApplyOp>(def))
      return false;
    for (Value op : def->getOperands())
      if (rec(op))
        return true;
    return false;
  };
  return rec(v);
}

// True if `op` is a hierarchy op that lies strictly inside `outerH.getBody()`.
static bool isInnerHierarchyOf(Operation *op,
                               xilinx::air::HierarchyInterface outerH) {
  if (!op || op == outerH.getOperation())
    return false;
  return outerH.getBody().findAncestorOpInRegion(*op) != nullptr;
}

// Decomposition of an offset expression with respect to outerH's IVs.
struct OffsetDecomp {
  // Coefficient (signed) on each outerH IV that appears in the offset.
  // IVs not present have coefficient 0.
  llvm::SmallDenseMap<BlockArgument, int64_t, 4> ivCoeffs;
  // Per-outer-iteration access-span contribution from inner-hierarchy /
  // scf.for / scf.parallel IVs (always >= 0).
  int64_t innerSpan = 0;
};

// Decompose an offset expression into (per-outer-iv coefficients,
// inner-iteration extra span). outerH is the hierarchy we're checking
// partitioning for; `outerIvs` are its iteration variables. Returns nullopt
// if the expression contains an unrecognized term (e.g., a non-IV /
// non-passthrough BlockArgument, or a non-affine op).
static std::optional<OffsetDecomp>
decomposeOffset(Value v, ArrayRef<BlockArgument> outerIvs,
                xilinx::air::HierarchyInterface outerH);

// Linear-in-`iv` analysis with inner-hierarchy-IV span tracking. Used by the
// per-iv interface; internally calls decomposeOffset and extracts the single
// iv coefficient.
//
// When checking partitioning of `iv` (an IV of `outerH`), inner-hierarchy IVs
// (e.g., a herd's `tx` when `outerH` is the launch) iterate over the SAME
// range for every outer iteration. They don't break linearity in `iv`; they
// just expand the per-outer-iteration access span. We track that as
// `extraSpan` (the maximum offset contribution beyond the linear-in-iv term).
//
// Returns (coefficient on `iv`, extraSpan from inner IVs), or nullopt on
// non-affine / unrecognized operands. extraSpan is always >= 0.
static std::optional<std::pair<int64_t, int64_t>>
coeffOfIvWithInnerSpan(Value v, BlockArgument iv,
                       xilinx::air::HierarchyInterface outerH) {
  if (!v)
    return std::make_pair<int64_t, int64_t>(0, 0);

  if (auto c = staticInt(v))
    return std::make_pair<int64_t, int64_t>(0, 0);
  if (isProvablyZero(v))
    return std::make_pair<int64_t, int64_t>(0, 0);

  if (auto ba = dyn_cast<BlockArgument>(v)) {
    if (ba == iv)
      return std::make_pair<int64_t, int64_t>(1, 0);
    Operation *parent = ba.getOwner()->getParentOp();
    auto H = dyn_cast_if_present<xilinx::air::HierarchyInterface>(parent);
    if (H) {
      // Hierarchy passthrough kernel arg: trace through to the tied operand
      // in the outer scope.
      ArrayRef<BlockArgument> kargs = H.getKernelArguments();
      auto kIt = llvm::find(kargs, ba);
      if (kIt != kargs.end()) {
        unsigned idx = std::distance(kargs.begin(), kIt);
        return coeffOfIvWithInnerSpan(H.getKernelOperand(idx), iv, outerH);
      }
      // Inner-hierarchy IV: contributes (extent - 1) to the per-outer-iter
      // span and 0 to the iv coefficient. Only treat as inner if `H` is
      // strictly inside outerH (not outerH itself).
      ArrayRef<BlockArgument> ids = H.getIds();
      auto idIt = llvm::find(ids, ba);
      if (idIt != ids.end() && H.getOperation() != outerH.getOperation() &&
          isInnerHierarchyOf(H.getOperation(), outerH)) {
        auto e = hierarchyIvExtent(ba);
        if (!e)
          return std::nullopt;
        int64_t span = (*e > 0) ? (*e - 1) : 0;
        return std::make_pair<int64_t, int64_t>(0, std::move(span));
      }
    }
    // scf.for / scf.parallel IV inside outerH body: same treatment as inner
    // hierarchy IV — contributes (extent - 1) to span, 0 to coefficient on iv.
    Operation *parent2 = ba.getOwner()->getParentOp();
    if (parent2 &&
        (isa<scf::ForOp>(parent2) || isa<scf::ParallelOp>(parent2)) &&
        isInnerHierarchyOf(parent2, outerH)) {
      auto e = scfIvExtent(ba);
      if (!e)
        return std::nullopt;
      int64_t span = (*e > 0) ? (*e - 1) : 0;
      return std::make_pair<int64_t, int64_t>(0, std::move(span));
    }
    return std::nullopt; // unknown BlockArgument
  }

  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;

  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    auto l = coeffOfIvWithInnerSpan(add.getLhs(), iv, outerH);
    auto r = coeffOfIvWithInnerSpan(add.getRhs(), iv, outerH);
    if (!l || !r)
      return std::nullopt;
    return std::make_pair(l->first + r->first, l->second + r->second);
  }
  if (auto sub = dyn_cast<arith::SubIOp>(def)) {
    auto l = coeffOfIvWithInnerSpan(sub.getLhs(), iv, outerH);
    auto r = coeffOfIvWithInnerSpan(sub.getRhs(), iv, outerH);
    if (!l || !r)
      return std::nullopt;
    // Spans accumulate regardless of sign (max range only grows).
    return std::make_pair(l->first - r->first, l->second + r->second);
  }
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    auto lc = staticInt(mul.getLhs());
    auto rc = staticInt(mul.getRhs());
    if (lc) {
      auto r = coeffOfIvWithInnerSpan(mul.getRhs(), iv, outerH);
      if (!r)
        return std::nullopt;
      int64_t k = std::abs(*lc);
      return std::make_pair(*lc * r->first, k * r->second);
    }
    if (rc) {
      auto l = coeffOfIvWithInnerSpan(mul.getLhs(), iv, outerH);
      if (!l)
        return std::nullopt;
      int64_t k = std::abs(*rc);
      return std::make_pair(*rc * l->first, k * l->second);
    }
    return std::nullopt;
  }
  if (auto cast = dyn_cast<arith::IndexCastOp>(def))
    return coeffOfIvWithInnerSpan(cast.getIn(), iv, outerH);

  // affine.apply: evaluate the affine map symbolically. For each input,
  // compute its (coeff_on_iv, span). Then walk the AffineExpr and combine.
  // Add: components add. Mul (by constant): scale both coeff and span.
  // Mod/FloorDiv/CeilDiv: only safe when divisor is constant and the
  // dividend has no iv dependence (coeff == 0); span becomes <= |divisor|-1.
  if (auto apply = dyn_cast<affine::AffineApplyOp>(def)) {
    AffineMap map = apply.getAffineMap();
    if (map.getNumResults() != 1)
      return std::nullopt;
    SmallVector<Value> inputs(apply.getOperands().begin(),
                              apply.getOperands().end());
    SmallVector<std::pair<int64_t, int64_t>> dimVals(map.getNumDims(), {0, 0});
    SmallVector<std::pair<int64_t, int64_t>> symVals(map.getNumSymbols(),
                                                     {0, 0});
    for (unsigned i = 0; i < inputs.size(); ++i) {
      auto sub = coeffOfIvWithInnerSpan(inputs[i], iv, outerH);
      if (!sub)
        return std::nullopt;
      bool isSymbol = i >= map.getNumDims();
      unsigned localIdx = isSymbol ? i - map.getNumDims() : i;
      if (isSymbol)
        symVals[localIdx] = *sub;
      else
        dimVals[localIdx] = *sub;
    }
    std::function<std::optional<std::pair<int64_t, int64_t>>(AffineExpr)> eval =
        [&](AffineExpr e) -> std::optional<std::pair<int64_t, int64_t>> {
      if (auto c = dyn_cast<AffineConstantExpr>(e))
        return std::make_pair<int64_t, int64_t>(0, 0);
      if (auto d = dyn_cast<AffineDimExpr>(e))
        return dimVals[d.getPosition()];
      if (auto s = dyn_cast<AffineSymbolExpr>(e))
        return symVals[s.getPosition()];
      auto bin = dyn_cast<AffineBinaryOpExpr>(e);
      if (!bin)
        return std::nullopt;
      auto lhs = eval(bin.getLHS());
      auto rhs = eval(bin.getRHS());
      if (!lhs || !rhs)
        return std::nullopt;
      auto lConstE = dyn_cast<AffineConstantExpr>(bin.getLHS());
      auto rConstE = dyn_cast<AffineConstantExpr>(bin.getRHS());
      switch (bin.getKind()) {
      case AffineExprKind::Add:
        return std::make_pair(lhs->first + rhs->first,
                              lhs->second + rhs->second);
      case AffineExprKind::Mul: {
        // One side must be a constant for affine to be valid.
        if (rConstE) {
          int64_t k = rConstE.getValue();
          int64_t kAbs = std::abs(k);
          return std::make_pair(k * lhs->first, kAbs * lhs->second);
        }
        if (lConstE) {
          int64_t k = lConstE.getValue();
          int64_t kAbs = std::abs(k);
          return std::make_pair(k * rhs->first, kAbs * rhs->second);
        }
        return std::nullopt;
      }
      case AffineExprKind::Mod:
      case AffineExprKind::FloorDiv:
      case AffineExprKind::CeilDiv: {
        // Only safe when divisor is a positive constant AND the dividend
        // does not depend on iv (coeff == 0). The result then has coeff 0
        // and bounded span: at most divisor-1 for Mod, span/divisor for div.
        if (!rConstE)
          return std::nullopt;
        int64_t d = rConstE.getValue();
        if (d <= 0)
          return std::nullopt;
        if (lhs->first != 0)
          return std::nullopt;
        if (bin.getKind() == AffineExprKind::Mod) {
          int64_t span = std::min<int64_t>(lhs->second, d - 1);
          return std::make_pair<int64_t, int64_t>(0, std::move(span));
        }
        // FloorDiv / CeilDiv
        int64_t span = lhs->second / d;
        return std::make_pair<int64_t, int64_t>(0, std::move(span));
      }
      case AffineExprKind::Constant:
      case AffineExprKind::DimId:
      case AffineExprKind::SymbolId:
        // These are leaf expressions, not binary ops; should be unreachable
        // here but listed for switch completeness.
        return std::nullopt;
      }
      return std::nullopt;
    };
    return eval(map.getResult(0));
  }

  return std::nullopt;
}

// Recursive helper for decomposeOffset. Accumulates contributions of `v`
// (multiplied by `coeff`) into `d`. Returns false on bail.
static bool decomposeRecursive(Value v, int64_t coeff,
                               ArrayRef<BlockArgument> outerIvs,
                               xilinx::air::HierarchyInterface outerH,
                               OffsetDecomp &d);

// Decompose an offset expression into (per-outer-iv coefficients,
// inner-iteration extra span). See struct OffsetDecomp doc.
static std::optional<OffsetDecomp>
decomposeOffset(Value v, ArrayRef<BlockArgument> outerIvs,
                xilinx::air::HierarchyInterface outerH) {
  OffsetDecomp d;
  if (!decomposeRecursive(v, /*coeff=*/1, outerIvs, outerH, d))
    return std::nullopt;
  return d;
}

static bool decomposeRecursive(Value v, int64_t coeff,
                               ArrayRef<BlockArgument> outerIvs,
                               xilinx::air::HierarchyInterface outerH,
                               OffsetDecomp &d) {
  if (!v)
    return true;
  if (staticInt(v).has_value())
    return true; // constant: no iv contribution.
  if (isProvablyZero(v))
    return true;

  if (auto ba = dyn_cast<BlockArgument>(v)) {
    if (llvm::is_contained(outerIvs, ba)) {
      d.ivCoeffs[ba] += coeff;
      return true;
    }
    Operation *parent = ba.getOwner()->getParentOp();
    auto H = dyn_cast_if_present<xilinx::air::HierarchyInterface>(parent);
    if (H) {
      // Hierarchy passthrough kernel arg: trace through.
      ArrayRef<BlockArgument> kargs = H.getKernelArguments();
      auto kIt = llvm::find(kargs, ba);
      if (kIt != kargs.end()) {
        unsigned i = std::distance(kargs.begin(), kIt);
        return decomposeRecursive(H.getKernelOperand(i), coeff, outerIvs,
                                  outerH, d);
      }
      // Inner-hierarchy IV (not outerH itself): contributes (extent-1) to
      // per-outer-iteration span.
      ArrayRef<BlockArgument> ids = H.getIds();
      auto idIt = llvm::find(ids, ba);
      if (idIt != ids.end() && H.getOperation() != outerH.getOperation() &&
          isInnerHierarchyOf(H.getOperation(), outerH)) {
        auto e = hierarchyIvExtent(ba);
        if (!e)
          return false;
        d.innerSpan += std::abs(coeff) * (*e > 0 ? *e - 1 : 0);
        return true;
      }
    }
    // scf.for / scf.parallel IV inside outerH body.
    if (parent && (isa<scf::ForOp>(parent) || isa<scf::ParallelOp>(parent)) &&
        isInnerHierarchyOf(parent, outerH)) {
      auto e = scfIvExtent(ba);
      if (!e)
        return false;
      d.innerSpan += std::abs(coeff) * (*e > 0 ? *e - 1 : 0);
      return true;
    }
    return false; // unknown BA
  }

  Operation *def = v.getDefiningOp();
  if (!def)
    return false;

  if (auto add = dyn_cast<arith::AddIOp>(def))
    return decomposeRecursive(add.getLhs(), coeff, outerIvs, outerH, d) &&
           decomposeRecursive(add.getRhs(), coeff, outerIvs, outerH, d);
  if (auto sub = dyn_cast<arith::SubIOp>(def))
    return decomposeRecursive(sub.getLhs(), coeff, outerIvs, outerH, d) &&
           decomposeRecursive(sub.getRhs(), -coeff, outerIvs, outerH, d);
  if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    auto lc = staticInt(mul.getLhs());
    auto rc = staticInt(mul.getRhs());
    if (lc)
      return decomposeRecursive(mul.getRhs(), coeff * (*lc), outerIvs, outerH,
                                d);
    if (rc)
      return decomposeRecursive(mul.getLhs(), coeff * (*rc), outerIvs, outerH,
                                d);
    return false;
  }
  if (auto cast = dyn_cast<arith::IndexCastOp>(def))
    return decomposeRecursive(cast.getIn(), coeff, outerIvs, outerH, d);

  if (auto apply = dyn_cast<affine::AffineApplyOp>(def)) {
    AffineMap map = apply.getAffineMap();
    if (map.getNumResults() != 1)
      return false;
    SmallVector<Value> inputs(apply.getOperands().begin(),
                              apply.getOperands().end());
    SmallVector<OffsetDecomp> dimVals(map.getNumDims());
    SmallVector<OffsetDecomp> symVals(map.getNumSymbols());
    for (unsigned i = 0; i < inputs.size(); ++i) {
      OffsetDecomp inner;
      if (!decomposeRecursive(inputs[i], 1, outerIvs, outerH, inner))
        return false;
      bool isSymbol = i >= map.getNumDims();
      unsigned localIdx = isSymbol ? i - map.getNumDims() : i;
      if (isSymbol)
        symVals[localIdx] = std::move(inner);
      else
        dimVals[localIdx] = std::move(inner);
    }
    std::function<std::optional<OffsetDecomp>(AffineExpr)> eval =
        [&](AffineExpr e) -> std::optional<OffsetDecomp> {
      OffsetDecomp r;
      if (isa<AffineConstantExpr>(e))
        return r;
      if (auto dim = dyn_cast<AffineDimExpr>(e))
        return dimVals[dim.getPosition()];
      if (auto sym = dyn_cast<AffineSymbolExpr>(e))
        return symVals[sym.getPosition()];
      auto bin = dyn_cast<AffineBinaryOpExpr>(e);
      if (!bin)
        return std::nullopt;
      auto lhs = eval(bin.getLHS());
      auto rhs = eval(bin.getRHS());
      if (!lhs || !rhs)
        return std::nullopt;
      auto lConst = dyn_cast<AffineConstantExpr>(bin.getLHS());
      auto rConst = dyn_cast<AffineConstantExpr>(bin.getRHS());
      switch (bin.getKind()) {
      case AffineExprKind::Add: {
        OffsetDecomp out = *lhs;
        for (auto &kv : rhs->ivCoeffs)
          out.ivCoeffs[kv.first] += kv.second;
        out.innerSpan += rhs->innerSpan;
        return out;
      }
      case AffineExprKind::Mul: {
        if (rConst) {
          int64_t k = rConst.getValue();
          OffsetDecomp out;
          for (auto &kv : lhs->ivCoeffs)
            out.ivCoeffs[kv.first] = kv.second * k;
          out.innerSpan = lhs->innerSpan * std::abs(k);
          return out;
        }
        if (lConst) {
          int64_t k = lConst.getValue();
          OffsetDecomp out;
          for (auto &kv : rhs->ivCoeffs)
            out.ivCoeffs[kv.first] = kv.second * k;
          out.innerSpan = rhs->innerSpan * std::abs(k);
          return out;
        }
        return std::nullopt;
      }
      case AffineExprKind::Mod:
      case AffineExprKind::FloorDiv:
      case AffineExprKind::CeilDiv: {
        if (!rConst)
          return std::nullopt;
        int64_t dv = rConst.getValue();
        if (dv <= 0)
          return std::nullopt;
        if (!lhs->ivCoeffs.empty())
          return std::nullopt;
        OffsetDecomp out;
        if (bin.getKind() == AffineExprKind::Mod)
          out.innerSpan = std::min<int64_t>(lhs->innerSpan, dv - 1);
        else
          out.innerSpan = lhs->innerSpan / dv;
        return out;
      }
      case AffineExprKind::Constant:
      case AffineExprKind::DimId:
      case AffineExprKind::SymbolId:
        return std::nullopt;
      }
      return std::nullopt;
    };
    auto res = eval(map.getResult(0));
    if (!res)
      return false;
    for (auto &kv : res->ivCoeffs)
      d.ivCoeffs[kv.first] += kv.second * coeff;
    d.innerSpan += res->innerSpan * std::abs(coeff);
    return true;
  }

  return false;
}

// Lex-packing partition condition for a single dim.
//
// For an offset expression `Σ c_iv * iv` (over a subset of the outer IVs in
// `decomp.ivCoeffs`) with per-iteration access span `staticSize +
// decomp.innerSpan`, decide whether distinct iv tuples (varying *only the
// IVs that appear in this dim*) produce disjoint access regions.
//
// Sufficient condition (sort iv coefficients ascending by |c|):
//   |c_1| ≥ size + innerSpan
//   |c_2| ≥ size + innerSpan + |c_1| * (extent_1 - 1)
//   ...
//   |c_k| ≥ size + innerSpan + Σ_{j<k} |c_j| * (extent_j - 1)
//
// IVs not in `decomp.ivCoeffs` are *not* constrained by this dim — those
// must be covered by some other dim (or by a per-iv check elsewhere).
static bool dimLexPacks(const OffsetDecomp &decomp, int64_t staticSize) {
  llvm::SmallVector<std::pair<BlockArgument, int64_t>, 4> entries;
  for (auto &kv : decomp.ivCoeffs) {
    if (kv.second != 0)
      entries.push_back({kv.first, std::abs(kv.second)});
  }
  if (entries.empty())
    return false;
  llvm::sort(entries,
             [](const auto &a, const auto &b) { return a.second < b.second; });
  int64_t acc = staticSize + decomp.innerSpan;
  for (const auto &[iv, c] : entries) {
    if (c < acc)
      return false;
    auto e = hierarchyIvExtent(iv);
    if (!e)
      return false;
    int64_t mul = (*e > 0) ? (*e - 1) : 0;
    acc += c * mul;
  }
  return true;
}

// True if iterating `iv` produces disjoint per-instance ranges in this access
// dimension (legacy per-iv check). Kept for backward compat / used by the
// per-PE replication path.
static bool dimPartitionsBy(Value offset, Value sizeVal,
                            std::optional<int64_t> sizeStatic, BlockArgument iv,
                            xilinx::air::HierarchyInterface outerH) {
  auto lin = coeffOfIvWithInnerSpan(offset, iv, outerH);
  if (!lin)
    return false;
  int64_t coeff = std::abs(lin->first);
  int64_t extraSpan = lin->second;
  if (coeff == 0)
    return false;
  auto sz = resolveSize(sizeVal, sizeStatic);
  if (!sz)
    return false;
  return coeff >= (*sz + extraSpan);
}

// Result of analyzing a single terminal access.
enum class CheckResult {
  Disjoint,     // proven disjoint across all herd IVs
  NotDisjoint,  // proven NOT disjoint
  Inconclusive, // analysis bailed
};

struct CheckOutcome {
  CheckResult result;
  std::string reason;
  Operation *atOp;
};

// For a terminal access through `consumed`, determine offsets and sizes from
// the op kind and check whether each herd IV partitions some access dim.
// Sizes are stored as (Value, std::optional<int64_t>) so we can resolve
// static-attribute sizes from memref.subview alongside SSA-value sizes from
// channel/dma ops.
static CheckOutcome checkTerminal(const TerminalAccess &t,
                                  ArrayRef<BlockArgument> ivs,
                                  xilinx::air::HierarchyInterface outerH) {
  Operation *op = t.op;

  SmallVector<Value> offsets;
  SmallVector<Value> sizesVal;
  SmallVector<std::optional<int64_t>> sizesStatic;
  bool hasPattern = false;

  auto pushSize = [&](Value v) {
    sizesVal.push_back(v);
    sizesStatic.push_back(std::nullopt);
  };
  auto pushStaticSize = [&](int64_t s) {
    sizesVal.push_back(nullptr);
    sizesStatic.push_back(s);
  };

  // Read-only accesses don't need disjointness across iterations: multiple
  // readers of the same memory range is not a race. Only writes require
  // per-iteration partitioning. AIR-specific read-only terminals:
  //   - air.channel.put on its memref (reads src, sends to channel)
  //   - air.dma_memcpy_nd on its src side
  // For other ops, query MemoryEffectOpInterface and skip the access if
  // it's read-only.
  if (auto chan = dyn_cast<xilinx::air::ChannelPutOp>(op)) {
    if (chan.getMemref() == t.consumed)
      return {CheckResult::Disjoint, {}, op};
  }
  if (auto dma = dyn_cast<xilinx::air::DmaMemcpyNdOp>(op)) {
    if (dma.getSrc() == t.consumed)
      return {CheckResult::Disjoint, {}, op};
  }
  // Channel index can disambiguate per-PE memref accesses. For an
  // `air.channel.get @ch[%pid] (%dst[][][])` (or put), even though the
  // memref offsets don't reference the iteration variable, the channel
  // INDEX does — so each iteration instance reads/writes a logically
  // distinct stream of data into/from its own (LOCAL-cloned) memref. Skip
  // the partitioning check when the channel index transitively depends on
  // any iteration variable (through any chain of pure ops, not just the
  // affine subset asLinearInOneBlockArg handles — patterns like
  // `arith.divsi %iv, %c2` come up in practice).
  if (auto chan = dyn_cast<xilinx::air::ChannelInterface>(op)) {
    if (chan.getMemref() == t.consumed) {
      for (Value idx : chan.getIndices()) {
        if (dependsOnAnyIv(idx, ivs))
          return {CheckResult::Disjoint, {}, op};
      }
    }
  }
  if (auto memEffectOp = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<MemoryEffects::EffectInstance, 4> effects;
    memEffectOp.getEffectsOnValue(t.consumed, effects);
    if (!effects.empty()) {
      bool hasWrite = false;
      for (auto &eff : effects) {
        if (isa<MemoryEffects::Write>(eff.getEffect()) ||
            isa<MemoryEffects::Free>(eff.getEffect())) {
          hasWrite = true;
          break;
        }
      }
      if (!hasWrite)
        return {CheckResult::Disjoint, {}, op};
    }
  }

  if (auto chan = dyn_cast<xilinx::air::ChannelInterface>(op)) {
    if (chan.getMemref() == t.consumed) {
      for (Value v : chan.getOffsets())
        offsets.push_back(v);
      for (Value v : chan.getSizes())
        pushSize(v);
      hasPattern = true;
    }
  } else if (auto dma = dyn_cast<xilinx::air::DmaMemcpyNdOp>(op)) {
    if (dma.getDst() == t.consumed) {
      for (Value v : dma.getDstOffsets())
        offsets.push_back(v);
      for (Value v : dma.getDstSizes())
        pushSize(v);
      hasPattern = true;
    } else if (dma.getSrc() == t.consumed) {
      for (Value v : dma.getSrcOffsets())
        offsets.push_back(v);
      for (Value v : dma.getSrcSizes())
        pushSize(v);
      hasPattern = true;
    }
  } else if (auto sv = dyn_cast<memref::SubViewOp>(op)) {
    if (sv.getSource() == t.consumed) {
      for (auto ofr : sv.getMixedOffsets()) {
        if (auto v = dyn_cast<Value>(ofr))
          offsets.push_back(v);
        else
          offsets.push_back(nullptr); // static offset — no IV dependence
      }
      for (auto ofr : sv.getMixedSizes()) {
        if (auto v = dyn_cast<Value>(ofr)) {
          pushSize(v);
        } else {
          auto attr = cast<IntegerAttr>(cast<Attribute>(ofr));
          pushStaticSize(attr.getInt());
        }
      }
      hasPattern = true;
    }
  } else if (auto store = dyn_cast<memref::StoreOp>(op)) {
    if (store.getMemRef() == t.consumed) {
      for (Value idx : store.getIndices()) {
        offsets.push_back(idx);
        pushStaticSize(1);
      }
      hasPattern = true;
    }
  } else if (auto load = dyn_cast<memref::LoadOp>(op)) {
    if (load.getMemRef() == t.consumed) {
      for (Value idx : load.getIndices()) {
        offsets.push_back(idx);
        pushStaticSize(1);
      }
      hasPattern = true;
    }
  } else if (auto vstore = dyn_cast<vector::TransferWriteOp>(op)) {
    if (vstore.getBase() == t.consumed) {
      for (Value idx : vstore.getIndices())
        offsets.push_back(idx);
      // Vector transfer writes a slice of vector shape into the memref;
      // the access size in each dim is the vector dim size.
      auto vecTy = vstore.getVectorType();
      for (int64_t s : vecTy.getShape())
        pushStaticSize(s);
      hasPattern = true;
    }
  } else if (auto vload = dyn_cast<vector::TransferReadOp>(op)) {
    if (vload.getBase() == t.consumed) {
      for (Value idx : vload.getIndices())
        offsets.push_back(idx);
      auto vecTy = vload.getVectorType();
      for (int64_t s : vecTy.getShape())
        pushStaticSize(s);
      hasPattern = true;
    }
  }

  // No structured access pattern → treat as full-buffer access by every
  // iteration; disjointness fails unless there are no IVs.
  if (!hasPattern) {
    // For `func.call`, `memref.load/store`, or anything we don't recognize:
    // be conservative.
    if (ivs.empty())
      return {CheckResult::Disjoint, {}, op};
    return {CheckResult::Inconclusive,
            "unrecognized terminal access; cannot determine offsets/sizes", op};
  }

  // Empty offsets/sizes (`[][][]`) means whole-buffer access.
  if (offsets.empty()) {
    if (ivs.empty())
      return {CheckResult::Disjoint, {}, op};
    return {CheckResult::NotDisjoint,
            "access spans the entire memref ([][][]); every iteration "
            "instance touches the same locations",
            op};
  }

  // Per-dim joint partition check. For each dim, decompose the offset and
  // try the lex-packing condition over all outer IVs that appear in that
  // dim. If any dim's lex-packing covers a given iv, that iv is partitioned
  // by that dim. Across dims, every iv must be covered by at least one dim.
  llvm::SmallVector<std::optional<OffsetDecomp>, 4> dimDecomps;
  llvm::SmallVector<bool, 4> dimLex;
  llvm::SmallVector<int64_t, 4> dimStaticSize;
  for (unsigned d = 0; d < offsets.size() && d < sizesVal.size(); ++d) {
    Value off = offsets[d];
    if (!off) {
      dimDecomps.push_back(std::nullopt);
      dimLex.push_back(false);
      dimStaticSize.push_back(0);
      continue;
    }
    auto sz = resolveSize(sizesVal[d], sizesStatic[d]);
    if (!sz) {
      dimDecomps.push_back(std::nullopt);
      dimLex.push_back(false);
      dimStaticSize.push_back(0);
      continue;
    }
    auto decomp = decomposeOffset(off, ivs, outerH);
    dimStaticSize.push_back(*sz);
    if (!decomp) {
      dimDecomps.push_back(std::nullopt);
      dimLex.push_back(false);
      continue;
    }
    dimDecomps.push_back(decomp);
    dimLex.push_back(dimLexPacks(*decomp, *sz));
  }

  // For each IV, find a dim where the iv appears AND that dim lex-packs.
  for (BlockArgument iv : ivs) {
    bool found = false;
    for (unsigned d = 0; d < dimDecomps.size(); ++d) {
      if (!dimDecomps[d] || !dimLex[d])
        continue;
      auto it = dimDecomps[d]->ivCoeffs.find(iv);
      if (it != dimDecomps[d]->ivCoeffs.end() && it->second != 0) {
        found = true;
        break;
      }
    }
    // Fallback: legacy per-iv check (covers patterns the multi-IV decomp
    // doesn't reach yet, e.g., subview offsets with mixed Value/static).
    if (!found) {
      for (unsigned d = 0; d < offsets.size() && d < sizesVal.size(); ++d) {
        Value off = offsets[d];
        if (!off)
          continue;
        if (dimPartitionsBy(off, sizesVal[d], sizesStatic[d], iv, outerH)) {
          found = true;
          break;
        }
      }
    }
    if (!found) {
      // Either the IV genuinely does not partition (rejection) or the
      // analysis bailed on the relevant offsets (inconclusive).
      // Distinguish: if any offset for this access has the IV "somewhere"
      // but with a non-static size or a non-recognized expression → bail.
      bool sawIVSomewhere = false;
      for (unsigned d = 0; d < offsets.size(); ++d) {
        Value off = offsets[d];
        if (!off)
          continue;
        auto lin = asLinearInOneBlockArg(off);
        if (lin && lin->first == iv) {
          sawIVSomewhere = true;
          break;
        }
      }
      if (!sawIVSomewhere) {
        std::string r;
        llvm::raw_string_ostream os(r);
        os << "iteration variable does not appear in any offset of this "
              "access; iterations cannot be disjoint";
        return {CheckResult::NotDisjoint, os.str(), op};
      }
      return {CheckResult::Inconclusive,
              "could not prove that the access size in the iteration-indexed "
              "dimension is small enough to keep iterations disjoint",
              op};
    }
  }

  return {CheckResult::Disjoint, {}, op};
}

// Verify one hierarchy op.
static LogicalResult verifyOne(xilinx::air::HierarchyInterface H, bool strict) {
  auto level = matchingMemoryLevel(H);
  if (!level)
    return success();

  bool anyFailure = false;
  for (unsigned i = 0, e = H.getNumKernelOperands(); i < e; ++i) {
    Value operand = H.getKernelOperand(i);
    auto memTy = dyn_cast<MemRefType>(operand.getType());
    if (!memTy)
      continue;
    if (!memrefIsAtLevel(memTy, *level))
      continue;

    Operation *defOp = operand.getDefiningOp();
    if (defOp && insideRegion(H.getBody(), defOp))
      continue; // (R1) defined inside the body — implicit per-instance copy.

    // The air-shrink-memref-sizes-by-access pass annotates the alloc it
    // produces with `shrinkage = true`. By construction it has shrunk the
    // memref to the per-PE access size, and the air-to-aie LOCAL clone
    // path will materialize one private copy per core. Trust the
    // shrinkage marker as the shrink pass's promise of per-PE replication
    // — equivalent to having sunk the alloc into the herd body, just left
    // annotated rather than physically moved.
    if (defOp && defOp->hasAttr("air.shrinkage"))
      continue;
    // The alloc may live inside an air.execute wrapper for async tokens;
    // unwrap one level and check there too.
    if (defOp) {
      if (auto exec = dyn_cast<xilinx::air::ExecuteOp>(defOp)) {
        for (Operation &child : exec.getChildOps()) {
          if (child.hasAttr("air.shrinkage")) {
            defOp = &child;
            break;
          }
        }
        if (defOp->hasAttr("air.shrinkage"))
          continue;
      }
    }

    BlockArgument blockArg = H.getKernelArgument(i);
    SmallVector<TerminalAccess> terminals;
    collectTerminalAccesses(blockArg, H.getBody(), terminals);

    if (terminals.empty())
      continue; // operand passed in but not used: vacuously fine.

    // Filter out IVs whose iteration-space size is statically 1: with only
    // one iteration value, every access is trivially "disjoint" across
    // iterations and the partitioning check is vacuous. This matters for
    // single-iteration launches/segments/herds (e.g. an air.launch wrapped
    // around a unary kernel like relu), where the IV exists but isn't used
    // in any offset and the verifier would otherwise false-flag every
    // kernel operand.
    SmallVector<BlockArgument> ivs;
    {
      ArrayRef<BlockArgument> allIvs = H.getIds();
      OperandRange sizeOps = H.getSizeOperands();
      for (auto [iv, sz] : llvm::zip(allIvs, sizeOps)) {
        auto staticSz = getConstantIntValue(sz);
        if (staticSz && *staticSz == 1)
          continue; // trivial dim — skip
        ivs.push_back(iv);
      }
    }

    if (ivs.empty())
      continue; // every iteration dim is size 1 — nothing to check.

    // Two-pass check on terminals:
    //   Pass 1: look for any terminal that proves per-PE replication
    //           (e.g., air.channel.get/put with iv-dependent channel index
    //           — the alloc is then per-PE-cloned at lowering, so EVERY
    //           use inside the herd body operates on the PE-private copy).
    //           If found, accept the operand and skip pass 2.
    //   Pass 2: only if pass 1 found nothing, require every terminal
    //           individually to prove disjointness.
    bool perPeReplicated = false;
    for (const TerminalAccess &t : terminals) {
      // Channel ops with iv-dependent channel index disambiguate per-PE
      // memref accesses (each PE gets a private (LOCAL-cloned) memref
      // populated from a different channel slot).
      if (auto chan = dyn_cast<xilinx::air::ChannelInterface>(t.op)) {
        if (chan.getMemref() == t.consumed) {
          CheckOutcome outcome = checkTerminal(t, ivs, H);
          if (outcome.result == CheckResult::Disjoint) {
            perPeReplicated = true;
            break;
          }
        }
      }
      // External kernel func.call sinks: the call is opaque, but if any
      // non-memref operand of the call depends on a herd IV, that scalar
      // is presumably used by the kernel as a partition offset/index. AIE
      // external kernels follow this convention: the per-PE offset is
      // passed as an i32 RTP-style argument computed from the herd IV
      // (e.g., `arith.muli %iv, %size : i32`). Treat this as a per-PE
      // replication signal so the verifier doesn't reject a working
      // example just because the partition is encoded in a kernel arg
      // instead of a memref offset.
      if (auto call = dyn_cast<func::CallOp>(t.op)) {
        bool anyIvDep = false;
        for (Value arg : call.getOperands()) {
          if (isa<MemRefType>(arg.getType()))
            continue;
          if (dependsOnAnyIv(arg, ivs)) {
            anyIvDep = true;
            break;
          }
        }
        if (anyIvDep) {
          perPeReplicated = true;
          break;
        }
      }
    }
    if (perPeReplicated)
      continue; // operand is per-PE replicated; accept all its uses.

    for (const TerminalAccess &t : terminals) {
      CheckOutcome outcome = checkTerminal(t, ivs, H);
      if (outcome.result == CheckResult::Disjoint)
        continue;

      auto opName = H->getName().getStringRef();
      InFlightDiagnostic diag =
          (outcome.result == CheckResult::Inconclusive && !strict)
              ? H->emitWarning()
              : H->emitOpError();
      diag << "kernel operand #" << i << " of type " << memTy << ": "
           << outcome.reason << "; the alloc must be defined inside the "
           << opName
           << " body, or the access pattern must statically partition the "
              "memref over the "
           << opName << " iteration variables";
      diag.attachNote(outcome.atOp->getLoc()) << "access here";

      if (outcome.result == CheckResult::Inconclusive && !strict)
        continue;
      anyFailure = true;
      break; // one diagnostic per operand
    }
  }
  return failure(anyFailure);
}

class AIRVerifyHierarchyLocality
    : public xilinx::air::impl::AIRVerifyHierarchyLocalityBase<
          AIRVerifyHierarchyLocality> {
public:
  AIRVerifyHierarchyLocality() = default;
  AIRVerifyHierarchyLocality(const AIRVerifyHierarchyLocality &) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<xilinx::air::airDialect, memref::MemRefDialect,
                affine::AffineDialect, arith::ArithDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp m = getOperation();
    bool failed = false;
    m.walk([&](xilinx::air::HierarchyInterface H) {
      if (verifyOne(H, this->strict).failed())
        failed = true;
    });
    if (failed)
      signalPassFailure();
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRVerifyHierarchyLocalityPass() {
  return std::make_unique<AIRVerifyHierarchyLocality>();
}

} // namespace air
} // namespace xilinx
