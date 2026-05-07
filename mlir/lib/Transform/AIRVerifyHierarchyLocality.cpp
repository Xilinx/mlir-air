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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
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
  if (auto ba = dyn_cast<BlockArgument>(v))
    return std::make_pair(ba, int64_t{1});

  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;

  if (auto cst = staticInt(v))
    return std::nullopt; // pure constant: no IV dependence

  if (auto add = dyn_cast<arith::AddIOp>(def)) {
    auto l = asLinearInOneBlockArg(add.getLhs());
    auto r = asLinearInOneBlockArg(add.getRhs());
    auto lc = staticInt(add.getLhs());
    auto rc = staticInt(add.getRhs());
    if (l && rc)
      return l;
    if (r && lc)
      return r;
    return std::nullopt;
  }
  if (auto sub = dyn_cast<arith::SubIOp>(def)) {
    auto l = asLinearInOneBlockArg(sub.getLhs());
    auto rc = staticInt(sub.getRhs());
    if (l && rc)
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

// True if iterating `iv` produces disjoint per-instance ranges in this access
// dimension. The condition is: offset depends linearly on `iv` with absolute
// coefficient `c`, AND the access size in the same dim is ≤ c.
static bool dimPartitionsBy(Value offset, Value sizeVal,
                            std::optional<int64_t> sizeStatic,
                            BlockArgument iv) {
  auto lin = asLinearInOneBlockArg(offset);
  if (!lin)
    return false;
  if (lin->first != iv)
    return false;
  auto sz = resolveSize(sizeVal, sizeStatic);
  if (!sz)
    return false;
  return lin->second >= *sz;
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
                                  ArrayRef<BlockArgument> ivs) {
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

  // For each IV, find a dim that cleanly partitions.
  for (BlockArgument iv : ivs) {
    bool found = false;
    for (unsigned d = 0; d < offsets.size() && d < sizesVal.size(); ++d) {
      Value off = offsets[d];
      // A static offset cannot depend on an IV; skip it.
      if (!off)
        continue;
      if (dimPartitionsBy(off, sizesVal[d], sizesStatic[d], iv)) {
        found = true;
        break;
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

    for (const TerminalAccess &t : terminals) {
      CheckOutcome outcome = checkTerminal(t, ivs);
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
