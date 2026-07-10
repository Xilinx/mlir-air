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

// If `ba` is a hierarchy kernel-arg (i.e., `ba` is a body block argument of
// an air.launch / air.segment / air.herd that comes from `args(... = ...)`),
// return the tied operand from the outer scope. Otherwise return nullopt.
//
// Walks at most one hierarchy hop. Callers that need to trace through several
// nested hierarchies should re-invoke on the returned value (which is the
// natural recursion in every analysis below). Centralizes a passthrough
// pattern that was previously inlined identically in 4+ call sites.
static std::optional<Value> hierarchyKernelArgPassthrough(BlockArgument ba) {
  Operation *parent = ba.getOwner()->getParentOp();
  auto H = dyn_cast_if_present<xilinx::air::HierarchyInterface>(parent);
  if (!H)
    return std::nullopt;
  ArrayRef<BlockArgument> kargs = H.getKernelArguments();
  auto it = llvm::find(kargs, ba);
  if (it == kargs.end())
    return std::nullopt;
  return H.getKernelOperand(std::distance(kargs.begin(), it));
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
  SmallVector<Value> stack;
  stack.push_back(v);
  while (!stack.empty()) {
    Value cur = stack.pop_back_val();
    if (!cur || !visited.insert(cur).second)
      continue;
    if (auto ba = dyn_cast<BlockArgument>(cur)) {
      if (llvm::is_contained(ivs, ba))
        return true;
      if (auto upstream = hierarchyKernelArgPassthrough(ba))
        stack.push_back(*upstream);
      continue;
    }
    Operation *def = cur.getDefiningOp();
    if (!def)
      continue;
    if (!isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
             arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::IndexCastOp,
             affine::AffineApplyOp>(def))
      continue;
    for (Value op : def->getOperands())
      stack.push_back(op);
  }
  return false;
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

// Recursive helper for decomposeOffset. Accumulates contributions of `v`
// (multiplied by `coeff`) into `d`. Returns false on bail.
static bool decomposeRecursive(Value v, int64_t coeff,
                               ArrayRef<BlockArgument> outerIvs,
                               xilinx::air::HierarchyInterface outerH,
                               OffsetDecomp &d);

// Evaluate an AffineExpr over previously-decomposed input slots, producing
// an OffsetDecomp for the expression's value. Returns nullopt on patterns
// the analysis can't represent (e.g. iv * iv, mod/div with iv coeffs).
static std::optional<OffsetDecomp>
evalAffineExpr(AffineExpr e, ArrayRef<OffsetDecomp> dimVals,
               ArrayRef<OffsetDecomp> symVals) {
  if (isa<AffineConstantExpr>(e)) {
    OffsetDecomp r;
    return r;
  }
  if (auto dim = dyn_cast<AffineDimExpr>(e))
    return dimVals[dim.getPosition()];
  if (auto sym = dyn_cast<AffineSymbolExpr>(e))
    return symVals[sym.getPosition()];
  auto bin = dyn_cast<AffineBinaryOpExpr>(e);
  if (!bin)
    return std::nullopt;
  auto lhs = evalAffineExpr(bin.getLHS(), dimVals, symVals);
  auto rhs = evalAffineExpr(bin.getRHS(), dimVals, symVals);
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
}

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

  if (auto ba = dyn_cast<BlockArgument>(v)) {
    if (llvm::is_contained(outerIvs, ba)) {
      d.ivCoeffs[ba] += coeff;
      return true;
    }
    if (auto upstream = hierarchyKernelArgPassthrough(ba))
      return decomposeRecursive(*upstream, coeff, outerIvs, outerH, d);
    Operation *parent = ba.getOwner()->getParentOp();
    if (auto H = dyn_cast_if_present<xilinx::air::HierarchyInterface>(parent)) {
      ArrayRef<BlockArgument> ids = H.getIds();
      auto idIt = llvm::find(ids, ba);
      if (idIt != ids.end()) {
        // outerH IV that the caller filtered out of `outerIvs` because its
        // iteration size is statically 1: the only value is 0, contributes
        // 0 to the offset.
        if (H.getOperation() == outerH.getOperation()) {
          auto e = hierarchyIvExtent(ba);
          if (e && *e == 1)
            return true;
          return false;
        }
        // Inner-hierarchy IV (NOT outerH itself): iterates the same range
        // for every outer iteration → contributes (extent-1) to the
        // per-outer-iteration access span, not to iv linearity.
        if (isInnerHierarchyOf(H.getOperation(), outerH)) {
          auto e = hierarchyIvExtent(ba);
          if (!e)
            return false;
          d.innerSpan += std::abs(coeff) * (*e > 0 ? *e - 1 : 0);
          return true;
        }
      }
    }
    // scf.for / scf.parallel IV inside outerH body — same treatment as inner
    // hierarchy IV.
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
    auto res = evalAffineExpr(map.getResult(0), dimVals, symVals);
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
//
// TODO: this is sufficient but not necessary. A sound disjointness check
// would build the access region as a presburger::IntegerRelation
// (mlir::affine::MemRefRegion) and use IntegerRelation::isIntegerEmpty()
// on the intersection of access(iv_a) ∩ access(iv_b) restricted to
// iv_a ≠ iv_b. The lex-packing condition over-rejects programs where
// individual dims don't dominate but the joint access still partitions
// (e.g., interleaved-stride patterns). See follow-up PR.
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
  // distinct stream of data into/from its own (LOCAL-cloned) memref.
  // Skip the partitioning check when the channel index transitively
  // depends on any iteration variable (through any chain of pure index
  // ops, including non-affine forms like `arith.divsi %iv, %c2`).
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
  } else if (auto vt = dyn_cast<VectorTransferOpInterface>(op);
             vt && vt.getBase() == t.consumed) {
    for (Value idx : vt.getIndices())
      offsets.push_back(idx);
    // Per memref dim, find which vector dim (if any) it maps to via the
    // permutation map; non-vectorized memref dims access size 1.
    AffineMap pm = vt.getPermutationMap();
    auto vecShape = vt.getVectorType().getShape();
    for (unsigned d = 0; d < pm.getNumInputs(); ++d) {
      int64_t sz = 1;
      for (unsigned r = 0; r < pm.getNumResults(); ++r) {
        auto dim = dyn_cast<AffineDimExpr>(pm.getResult(r));
        if (dim && dim.getPosition() == d) {
          sz = vecShape[r];
          break;
        }
      }
      pushStaticSize(sz);
    }
    hasPattern = true;
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
    if (!found) {
      // Distinguish "iv genuinely doesn't appear anywhere" (definitive race
      // → NotDisjoint) from "iv appears but lex-packing didn't hold or the
      // decomposition bailed" (Inconclusive). Probe each offset
      // independently of size resolution: if iv has a nonzero coefficient
      // in any offset, the issue is partition arithmetic (or unknown size),
      // not absence.
      bool ivAppearsSomewhere = false;
      for (unsigned d = 0; d < offsets.size(); ++d) {
        Value off = offsets[d];
        if (!off)
          continue;
        auto probe = decomposeOffset(off, ivs, outerH);
        if (!probe)
          continue;
        auto it = probe->ivCoeffs.find(iv);
        if (it != probe->ivCoeffs.end() && it->second != 0) {
          ivAppearsSomewhere = true;
          break;
        }
      }
      if (!ivAppearsSomewhere) {
        return {CheckResult::NotDisjoint,
                "iteration variable does not appear in any offset of this "
                "access; iterations cannot be disjoint",
                op};
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

    // air-shrink-memref-sizes-by-access marks an alloc with
    // `air.shrinkage = true` once it has shrunk the memref to per-PE
    // size; the air-to-aie LOCAL clone path then materializes a private
    // copy per core. Trust ONLY the true value — the shrink pass also
    // sets the attribute to false on its failure paths.
    auto isShrunk = [](Operation *op) {
      auto attr = op->getAttrOfType<BoolAttr>("air.shrinkage");
      return attr && attr.getValue();
    };
    if (defOp && isShrunk(defOp))
      continue;
    // The alloc may be wrapped in air.execute for async tokens; the
    // marker may live on the execute or its inner alloc.
    if (auto exec = dyn_cast_if_present<xilinx::air::ExecuteOp>(defOp)) {
      if (isShrunk(exec))
        continue;
      bool ok = false;
      for (Operation &child : exec.getChildOps())
        if (isShrunk(&child)) {
          ok = true;
          break;
        }
      if (ok)
        continue;
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

    // A shared L1 buffer with a cross-core producer/consumer dependence (one
    // core reads what another writes) is a legitimate non-partitioned herd
    // operand -- neighbor tiles communicate through it. Accept it.
    if (auto herd = dyn_cast<xilinx::air::HerdOp>(H.getOperation()))
      if (xilinx::air::herdBufferHasCrossCoreDependence(herd, operand))
        continue;

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
      // iv-dependent channel index → each PE consumes a different slot,
      // alloc gets LOCAL-cloned per core. Inspect the index directly: a
      // checkTerminal=Disjoint result also fires for read-only paths
      // and would mis-trigger replication on an unrelated channel.put.
      if (auto chan = dyn_cast<xilinx::air::ChannelInterface>(t.op)) {
        if (chan.getMemref() == t.consumed) {
          for (Value idx : chan.getIndices()) {
            if (dependsOnAnyIv(idx, ivs)) {
              perPeReplicated = true;
              break;
            }
          }
          if (perPeReplicated)
            break;
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
      //
      // TODO: this is a heuristic — the kernel is opaque and may use the
      // iv-dependent scalar for something other than memref partition
      // (e.g., a loop bound). A function-declaration-level access summary
      // attribute would be more precise; pursue if false-positives surface.
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
    registry.insert<xilinx::air::airDialect, memref::MemRefDialect,
                    affine::AffineDialect, arith::ArithDialect, scf::SCFDialect,
                    vector::VectorDialect, func::FuncDialect>();
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
