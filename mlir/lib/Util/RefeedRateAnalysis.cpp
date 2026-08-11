//===- RefeedRateAnalysis.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Util/RefeedRateAnalysis.h"
#include "air/Util/Util.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

namespace {

// Enumerating a phase loop multiplies the traversal cost by its trip count.
// These bound the blow-up: a loop is only enumerated when it is short enough
// to be a mode selector, and the whole traversal gives up past the step
// budget rather than spending unbounded time on a pathological nest.
constexpr int64_t kMaxEnumeratedTrip = 256;
constexpr int64_t kMaxTraversalSteps = 1 << 20;
// One phase per dispatch-loop iteration. A superkernel runs a few hundred;
// past this the traversal cost outweighs the check and the analysis declines
// rather than silently summing distinct modes together.
constexpr int64_t kMaxPhases = 1024;
// An emission whose bundle subscripts do not resolve drives this many edges at
// most before the analysis declines.
constexpr int64_t kMaxBundleEdges = 1024;

// Follow `v` through hierarchy operand/argument ties and index casts to the
// outermost value denoting the same quantity. A mode selector read inside an
// air.herd and the same selector read in the enclosing air.segment must map
// to one root, or the two sides of a channel would be keyed on different
// phases and never compared.
Value traceToRoot(Value v) {
  bool changed = true;
  while (changed && v) {
    changed = false;
    if (auto cast = v.getDefiningOp<arith::IndexCastOp>()) {
      v = cast.getIn();
      changed = true;
      continue;
    }
    auto ba = dyn_cast<BlockArgument>(v);
    if (!ba)
      continue;
    Operation *owner = ba.getOwner()->getParentOp();
    Value tied = llvm::TypeSwitch<Operation *, Value>(owner)
                     .Case<air::LaunchOp, air::SegmentOp, air::HerdOp>(
                         [&](auto h) { return h.getTiedKernelOperand(ba); })
                     .Default([](Operation *) { return Value(); });
    if (tied) {
      v = tied;
      changed = true;
    }
  }
  return v;
}

// The refeed factor a put carries today. air::getRefeedCount resolves the
// per-emission override and the channel declaration; the third carrier is the
// L2 rendezvous buffer the put reads from, which air-to-aie propagates onto
// the aie.buffer separately. Consult it between the two so the analysis sees
// the same number the lowering will.
int64_t declaredRefeed(ChannelInterface c) {
  Operation *op = c.getOperation();
  if (op->getAttrOfType<IntegerAttr>(air::attrs::RefeedCount))
    return air::getRefeedCount(op);
  if (Value memref = c.getMemref()) {
    Operation *def = memref.getDefiningOp();
    // An allocation is wrapped in an air.execute; the carrier sits on the
    // memref.alloc inside, which is what air-to-aie reads.
    if (auto exec = dyn_cast_if_present<air::ExecuteOp>(def)) {
      Operation *term = exec.getRegion().front().getTerminator();
      unsigned idx = cast<OpResult>(memref).getResultNumber();
      // Results are (token, values...), so the terminator operand is one back.
      if (idx >= 1 && idx - 1 < term->getNumOperands())
        def = term->getOperand(idx - 1).getDefiningOp();
    }
    if (def && def->getAttrOfType<IntegerAttr>(air::attrs::RefeedCount))
      return air::getRefeedCount(def);
  }
  return air::getRefeedCount(c);
}

// True when `iv` can change a transfer rate: it steers a mode switch, bounds
// another loop, or sizes a transfer. Such a loop must be enumerated; any
// other loop only contributes its trip count as a repetition factor.
bool ivIsRateRelevant(Value iv) {
  llvm::SetVector<Value> worklist;
  worklist.insert(iv);
  for (unsigned i = 0; i < worklist.size(); ++i) {
    for (OpOperand &use : worklist[i].getUses()) {
      Operation *user = use.getOwner();
      if (auto sw = dyn_cast<scf::IndexSwitchOp>(user))
        if (use.get() == sw.getArg())
          return true;
      if (auto f = dyn_cast<scf::ForOp>(user))
        if (use.get() == f.getLowerBound() || use.get() == f.getUpperBound() ||
            use.get() == f.getStep())
          return true;
      // Only a *size* operand changes the token count. Offsets and strides
      // move the same transfer around; enumerating a loop for them would
      // multiply the traversal cost for nothing.
      if (auto c = dyn_cast<ChannelInterface>(user))
        for (OpFoldResult ofr : c.getMixedSizes())
          if (dyn_cast<Value>(ofr) == use.get())
            return true;
      if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
              arith::DivUIOp, arith::RemSIOp, arith::RemUIOp, arith::MinSIOp,
              arith::MaxSIOp, arith::IndexCastOp>(user))
        worklist.insert(user->getResult(0));
    }
  }
  return false;
}

} // namespace

// Declared in the air namespace (not anonymous) so the friend declaration in
// RefeedRateAnalysis names this class.
namespace xilinx {
namespace air {

class RefeedRateBuilder {
public:
  RefeedRateBuilder(RefeedRateAnalysis &result, Operation *scope)
      : result(result), scope(scope) {}

  void run();

private:
  using Env = DenseMap<Value, int64_t>;

  std::optional<int64_t> eval(Value v, const Env &env);
  std::optional<int64_t> tripCount(scf::ForOp f, const Env &env);
  void visitBlock(Block &b, Env &env, int64_t mult);
  void visitOp(mlir::Operation &op, Env &env, int64_t mult);
  void record(ChannelInterface c, const Env &env, int64_t mult);
  void poison(StringRef chan, Operation *culprit);

  RefeedRateAnalysis &result;
  Operation *scope;
  unsigned phaseIdx = 0;
  int64_t steps = 0;
  bool budgetBlown = false;
  DenseSet<StringRef> poisoned;
};

std::optional<int64_t> RefeedRateBuilder::eval(Value v, const Env &env) {
  if (!v)
    return std::nullopt;
  if (auto it = env.find(v); it != env.end())
    return it->second;
  if (auto c = getConstantIntValue(v))
    return c;
  // A value re-read inside a hierarchy is the operand it was tied to. Resolve
  // on the root so a mode discriminator computed once at launch level is
  // visible to every nested air.segment and air.herd.
  if (Value root = traceToRoot(v); root && root != v)
    return eval(root, env);
  Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;

  if (auto sw = dyn_cast<scf::IndexSwitchOp>(def)) {
    auto sel = eval(sw.getArg(), env);
    if (!sel)
      return std::nullopt;
    Region *region = &sw.getDefaultRegion();
    for (auto [i, c] : llvm::enumerate(sw.getCases()))
      if (c == *sel)
        region = &sw.getCaseRegions()[i];
    auto yield = cast<scf::YieldOp>(region->front().getTerminator());
    return eval(yield.getOperand(cast<OpResult>(v).getResultNumber()), env);
  }

  auto binary = [&](auto fn) -> std::optional<int64_t> {
    auto a = eval(def->getOperand(0), env);
    auto b = eval(def->getOperand(1), env);
    if (!a || !b)
      return std::nullopt;
    return fn(*a, *b);
  };
  // Same, for an operation that can itself fail to evaluate (division).
  auto binaryChecked = [&](auto fn) -> std::optional<int64_t> {
    auto a = eval(def->getOperand(0), env);
    auto b = eval(def->getOperand(1), env);
    if (!a || !b)
      return std::nullopt;
    return fn(*a, *b);
  };
  return llvm::TypeSwitch<Operation *, std::optional<int64_t>>(def)
      .Case<arith::IndexCastOp, arith::IndexCastUIOp>(
          [&](auto op) { return eval(op.getIn(), env); })
      .Case<arith::ExtUIOp, arith::ExtSIOp, arith::TruncIOp>(
          [&](auto op) { return eval(op.getIn(), env); })
      // The mode discriminator of a superkernel is a comparison against the
      // dispatch index (`arm = wave < N`), recomputed in every hierarchy.
      // Folding it is what lets one bound dispatch index resolve every arm.
      .Case<arith::CmpIOp>([&](arith::CmpIOp op) -> std::optional<int64_t> {
        auto a = eval(op.getLhs(), env);
        auto b = eval(op.getRhs(), env);
        if (!a || !b)
          return std::nullopt;
        switch (op.getPredicate()) {
        case arith::CmpIPredicate::eq:
          return *a == *b;
        case arith::CmpIPredicate::ne:
          return *a != *b;
        case arith::CmpIPredicate::slt:
        case arith::CmpIPredicate::ult:
          return *a < *b;
        case arith::CmpIPredicate::sle:
        case arith::CmpIPredicate::ule:
          return *a <= *b;
        case arith::CmpIPredicate::sgt:
        case arith::CmpIPredicate::ugt:
          return *a > *b;
        case arith::CmpIPredicate::sge:
        case arith::CmpIPredicate::uge:
          return *a >= *b;
        }
        return std::nullopt;
      })
      .Case<arith::SelectOp>([&](arith::SelectOp op) -> std::optional<int64_t> {
        auto c = eval(op.getCondition(), env);
        if (!c)
          return std::nullopt;
        return eval(*c ? op.getTrueValue() : op.getFalseValue(), env);
      })
      .Case<arith::AddIOp>([&](auto) { return binary(std::plus<int64_t>()); })
      .Case<arith::SubIOp>([&](auto) { return binary(std::minus<int64_t>()); })
      .Case<arith::MulIOp>(
          [&](auto) { return binary(std::multiplies<int64_t>()); })
      // A zero divisor is not evaluable. Folding it to 0 would turn an
      // unresolvable expression into a concrete trip count or size and skew
      // the rates; the channel must be reported unanalyzable instead.
      .Case<arith::DivSIOp, arith::DivUIOp>([&](auto) {
        return binaryChecked([](int64_t a, int64_t b) {
          return b ? std::optional<int64_t>(a / b) : std::nullopt;
        });
      })
      .Case<arith::RemSIOp, arith::RemUIOp>([&](auto) {
        return binaryChecked([](int64_t a, int64_t b) {
          return b ? std::optional<int64_t>(a % b) : std::nullopt;
        });
      })
      .Case<arith::MinSIOp>([&](auto) {
        return binary([](int64_t a, int64_t b) { return std::min(a, b); });
      })
      .Case<arith::MaxSIOp>([&](auto) {
        return binary([](int64_t a, int64_t b) { return std::max(a, b); });
      })
      .Default([](Operation *) { return std::nullopt; });
}

std::optional<int64_t> RefeedRateBuilder::tripCount(scf::ForOp f,
                                                    const Env &env) {
  auto lb = eval(f.getLowerBound(), env);
  auto ub = eval(f.getUpperBound(), env);
  auto st = eval(f.getStep(), env);
  if (!lb || !ub || !st || *st == 0)
    return std::nullopt;
  if (*ub <= *lb)
    return 0;
  return (*ub - *lb + *st - 1) / *st;
}

void RefeedRateBuilder::poison(StringRef chan, Operation *culprit) {
  if (poisoned.insert(chan).second) {
    result.valid = false;
    result.unanalyzable.emplace_back(chan, culprit);
  }
}

void RefeedRateBuilder::record(ChannelInterface c, const Env &env,
                               int64_t mult) {
  StringRef name = c.getChanName();
  // Explicit access patterns win; a bare emission moves the whole memref.
  int64_t size = 1;
  auto sizes = c.getMixedSizes();
  if (sizes.empty()) {
    auto ty = dyn_cast<ShapedType>(c.getMemref().getType());
    if (!ty || !ty.hasStaticShape())
      return poison(name, c.getOperation());
    size = static_cast<int64_t>(air::getTensorVolume(ty));
  } else {
    for (OpFoldResult ofr : sizes) {
      std::optional<int64_t> d;
      if (auto attr = dyn_cast<Attribute>(ofr))
        d = cast<IntegerAttr>(attr).getInt();
      else
        d = eval(cast<Value>(ofr), env);
      if (!d)
        return poison(name, c.getOperation());
      size *= *d;
    }
  }

  // Resolve which edge of the bundle this emission drives. An
  // air.channel @c [4, 4] is sixteen independent edges; aggregating them
  // double-counts whichever side spells its subscripts out. A subscript that
  // does not resolve -- typically a herd induction variable -- means the
  // emission drives every edge along that dimension.
  //
  // Spatial fan-out needs no special case here: a broadcast channel is
  // declared with a unit bundle (the fan-out lives in broadcast_shape), so a
  // put and its per-core gets all land on the same single edge.
  llvm::SmallVector<int64_t, 2> bundle;
  int64_t fanout = 1;
  if (auto decl = air::getChannelDeclarationThroughSymbol(c)) {
    if (auto bs = decl->getAttrOfType<ArrayAttr>("broadcast_shape"))
      for (auto a : bs)
        fanout *= cast<IntegerAttr>(a).getInt();
    for (auto a : decl.getSize())
      bundle.push_back(cast<IntegerAttr>(a).getInt());
  }
  if (bundle.empty())
    bundle.push_back(1);

  // On a broadcast channel the subscripts name the destination tile, not a
  // bundle edge: one put reaches all of them, and the per-tile gets are
  // scaled back down by the fan-out at the end of the traversal.
  if (fanout > 1) {
    bool isPutOp = isa<air::ChannelPutOp>(c.getOperation());
    auto &rate = result.rates[{name, 0u, phaseIdx}];
    // Only the destinations the traversal actually replicated need scaling
    // back: a get inside an air.herd was visited once per tile, while one
    // spelled out at memtile level was visited exactly once.
    if (!isPutOp && c->getParentOfType<air::HerdOp>())
      rate.fanout = fanout;
    if (isPutOp) {
      int64_t n = declaredRefeed(c);
      rate.rawSupply += mult * size;
      rate.supply += mult * size * n;
      rate.puts.push_back({c, mult * size, n});
    } else {
      rate.demand += mult * size;
      rate.gets.push_back({c, mult * size, 1});
    }
    return;
  }

  llvm::SmallVector<int64_t, 2> fixed(bundle.size(), -1);
  auto indices = c.getIndices();
  for (auto [dim, idx] : llvm::enumerate(indices)) {
    if (dim >= bundle.size())
      break;
    if (auto v = eval(idx, env))
      fixed[dim] = *v;
  }

  int64_t edges = 1;
  for (auto [dim, extent] : llvm::enumerate(bundle))
    edges *= fixed[dim] < 0 ? extent : 1;
  if (edges <= 0 || edges > kMaxBundleEdges)
    return poison(name, c.getOperation());

  int64_t tokens = mult * size;
  bool isPut = isa<air::ChannelPutOp>(c.getOperation());
  int64_t n = isPut ? declaredRefeed(c) : 1;

  // Walk the cartesian product of the unresolved dimensions.
  llvm::SmallVector<int64_t, 2> cur(bundle.size(), 0);
  for (int64_t e = 0; e < edges; ++e) {
    int64_t rem = e, linear = 0;
    for (auto [dim, extent] : llvm::enumerate(bundle)) {
      if (fixed[dim] >= 0) {
        cur[dim] = fixed[dim];
      } else {
        cur[dim] = rem % extent;
        rem /= extent;
      }
      linear = linear * extent + cur[dim];
    }
    auto &rate = result.rates[{name, static_cast<unsigned>(linear), phaseIdx}];
    if (isPut) {
      rate.rawSupply += tokens;
      rate.supply += tokens * n;
      rate.puts.push_back({c, tokens, n});
    } else {
      rate.demand += tokens;
      rate.gets.push_back({c, tokens, 1});
    }
  }
}

void RefeedRateBuilder::visitBlock(Block &b, Env &env, int64_t mult) {
  for (Operation &op : b) {
    if (budgetBlown)
      return;
    visitOp(op, env, mult);
  }
}

void RefeedRateBuilder::visitOp(Operation &op, Env &env, int64_t mult) {
  if (budgetBlown)
    return;
  if (++steps > kMaxTraversalSteps) {
    budgetBlown = true;
    result.valid = false;
    return;
  }
  {
    if (auto c = dyn_cast<ChannelInterface>(&op)) {
      record(c, env, mult);
      return;
    }
    if (auto f = dyn_cast<scf::ForOp>(&op)) {
      auto trip = tripCount(f, env);
      if (!trip) {
        // The bound is not resolvable at this phase; every channel below is
        // counted at an unknown rate, so drop those channels rather than
        // guess.
        f.walk([&](ChannelInterface c) { poison(c.getChanName(), f); });
        return;
      }
      if (*trip == 0)
        return;
      Value iv = f.getInductionVar();
      auto lb = eval(f.getLowerBound(), env);
      auto st = eval(f.getStep(), env);
      if (*trip <= kMaxEnumeratedTrip && ivIsRateRelevant(iv)) {
        for (int64_t i = 0; i < *trip; ++i) {
          env[iv] = *lb + i * *st;
          visitBlock(*f.getBody(), env, mult);
        }
        env.erase(iv);
      } else {
        visitBlock(*f.getBody(), env, mult * *trip);
      }
      return;
    }
    // Per-tile specialization inside a herd is written as scf.if on the tile
    // ids. With the ids bound the condition folds, so only the taken branch
    // contributes -- visiting both would count every tile's transfer on every
    // tile.
    if (auto ifOp = dyn_cast<scf::IfOp>(&op)) {
      auto cond = eval(ifOp.getCondition(), env);
      if (!cond) {
        ifOp.walk([&](ChannelInterface c) { poison(c.getChanName(), ifOp); });
        return;
      }
      Region &taken = *cond ? ifOp.getThenRegion() : ifOp.getElseRegion();
      if (!taken.empty())
        visitBlock(taken.front(), env, mult);
      return;
    }
    if (isa<affine::AffineIfOp>(&op)) {
      op.walk([&](ChannelInterface c) { poison(c.getChanName(), &op); });
      return;
    }
    // scf.parallel iterations are independent instances, exactly like an
    // scf.for trip: commonly one per memtile column, with the induction
    // variable naming the bundle edge.
    if (auto par = dyn_cast<scf::ParallelOp>(&op)) {
      int64_t trip = 1;
      bool ok = true;
      for (auto [lb, ub, st] :
           llvm::zip(par.getLowerBound(), par.getUpperBound(), par.getStep())) {
        auto l = eval(lb, env), u = eval(ub, env), t = eval(st, env);
        if (!l || !u || !t || *t == 0) {
          ok = false;
          break;
        }
        trip *= *u > *l ? (*u - *l + *t - 1) / *t : 0;
      }
      if (!ok) {
        par.walk([&](ChannelInterface c) { poison(c.getChanName(), par); });
        return;
      }
      if (trip == 0)
        return;
      bool relevant = llvm::any_of(par.getInductionVars(), [](Value iv) {
        return ivIsRateRelevant(iv);
      });
      if (trip <= kMaxEnumeratedTrip && relevant &&
          par.getInductionVars().size() == 1) {
        Value iv = par.getInductionVars()[0];
        auto lb = eval(par.getLowerBound()[0], env);
        auto st = eval(par.getStep()[0], env);
        for (int64_t i = 0; i < trip; ++i) {
          env[iv] = *lb + i * *st;
          visitBlock(*par.getBody(), env, mult);
        }
        env.erase(iv);
      } else {
        visitBlock(*par.getBody(), env, mult * trip);
      }
      return;
    }
    if (auto sw = dyn_cast<scf::IndexSwitchOp>(&op)) {
      auto sel = eval(sw.getArg(), env);
      if (!sel) {
        sw.walk([&](ChannelInterface c) { poison(c.getChanName(), sw); });
        return;
      }
      Region *region = &sw.getDefaultRegion();
      for (auto [i, c] : llvm::enumerate(sw.getCases()))
        if (c == *sel)
          region = &sw.getCaseRegions()[i];
      visitBlock(region->front(), env, mult);
      return;
    }
    // air.herd iterates over space: its body is written once but each tile
    // drives its own bundle edge, and the subscripts that pick the edge are
    // functions of the tile ids. Enumerate the tiles so those subscripts
    // resolve -- without it every tile's transfer would be spread across the
    // whole bundle and counted many times over.
    if (auto herd = dyn_cast<air::HerdOp>(&op)) {
      llvm::SmallVector<int64_t, 2> extents;
      for (Value sz : herd.getSizeOperands()) {
        auto n = eval(sz, env);
        if (!n || *n <= 0 || *n > kMaxBundleEdges) {
          herd.walk([&](ChannelInterface c) { poison(c.getChanName(), herd); });
          return;
        }
        extents.push_back(*n);
      }
      auto ids = herd.getIds();
      int64_t tiles = 1;
      for (int64_t e : extents)
        tiles *= e;
      for (int64_t t = 0; t < tiles; ++t) {
        int64_t rem = t;
        for (auto [dim, extent] : llvm::enumerate(extents)) {
          int64_t v = rem % extent;
          rem /= extent;
          env[ids[dim]] = v;
        }
        visitBlock(herd.getBody().front(), env, mult);
      }
      for (BlockArgument id : ids)
        env.erase(id);
      return;
    }
    // air.launch / air.segment bodies are traversed once.
    for (Region &r : op.getRegions())
      for (Block &nested : r)
        visitBlock(nested, env, mult);
  }
}

void RefeedRateBuilder::run() {
  if (scope->getNumRegions() == 0 || scope->getRegion(0).empty())
    return;
  Block &entry = scope->getRegion(0).front();
  Env env;

  // A cyclo-static phase is one iteration of the dispatch loop: the single
  // top-level scf.for that encloses ALL of the function's dataflow. Every mode
  // switch in these programs is recomputed from the dispatch index in each
  // hierarchy (`arm = wave < N`), so binding that index resolves all of them at
  // once -- no cross-hierarchy matching of switch ops is needed, and arms never
  // get summed together.
  //
  // A function whose channel ops are not all inside one such loop has no
  // dispatch loop to speak of; it is analyzed as a single phase, with every
  // loop contributing a plain repetition factor.
  scf::ForOp dispatch;
  {
    unsigned loopsWithChannels = 0;
    bool channelsOutside = false;
    for (Operation &op : entry) {
      bool hasChan = false;
      op.walk([&](ChannelInterface) { hasChan = true; });
      if (!hasChan)
        continue;
      if (auto f = dyn_cast<scf::ForOp>(&op)) {
        ++loopsWithChannels;
        dispatch = f;
      } else {
        channelsOutside = true;
      }
    }
    if (loopsWithChannels != 1 || channelsOutside)
      dispatch = nullptr;
  }

  result.phases.push_back({});
  if (!dispatch) {
    for (Operation &op : entry)
      visitOp(op, env, 1);
  } else {
    for (Operation &op : entry)
      if (&op != dispatch.getOperation())
        visitOp(op, env, 1);
    auto trip = tripCount(dispatch, env);
    auto lb = eval(dispatch.getLowerBound(), env);
    auto st = eval(dispatch.getStep(), env);
    if (!trip || !lb || !st || *trip > kMaxPhases) {
      dispatch.walk(
          [&](ChannelInterface c) { poison(c.getChanName(), dispatch); });
    } else {
      Value iv = dispatch.getInductionVar();
      for (int64_t i = 0; i < *trip; ++i) {
        result.phases.push_back({{iv, *lb + i * *st}});
        phaseIdx = result.phases.size() - 1;
        Env phaseEnv;
        phaseEnv[iv] = *lb + i * *st;
        visitBlock(*dispatch.getBody(), phaseEnv, 1);
      }
      phaseIdx = 0;
    }
  }

  for (auto &[key, rate] : result.rates)
    if (rate.fanout > 1)
      rate.demand /= rate.fanout;

  // Group the edges so each (channel, bundle edge) can be judged as a whole.
  llvm::MapVector<std::pair<StringRef, unsigned>, SmallVector<unsigned, 8>>
      byEdge;
  for (auto &[key, rate] : result.rates)
    byEdge[{key.channel, key.bundleIndex}].push_back(key.phase);

  using EdgeRate = RefeedRateAnalysis::EdgeRate;
  auto siteSet = [](ArrayRef<EdgeRate::Site> v) {
    SmallVector<Operation *> ops;
    for (auto s : v)
      ops.push_back(s.op.getOperation());
    llvm::sort(ops);
    ops.erase(llvm::unique(ops), ops.end());
    return ops;
  };

  // Collapses the identical report a mode repeated over many dispatch
  // iterations would produce -- one bug, not a hundred. Keyed on the edge as
  // well as the numbers: two edges of one bundle can be unbalanced by the same
  // amount and are still two separate findings.
  llvm::DenseSet<std::tuple<const void *, unsigned, int64_t, int64_t>> seen;
  for (auto &[edge, phaseList] : byEdge) {
    if (poisoned.contains(edge.first))
      continue;

    // Does each side come from the same emissions in every phase? A side that
    // varies is gated by the mode switch; a side that does not is common code.
    bool putsVary = false, getsVary = false;
    SmallVector<Operation *> refPuts, refGets;
    bool first = true;
    for (unsigned ph : phaseList) {
      const EdgeRate &r = result.rates[{edge.first, edge.second, ph}];
      auto p = siteSet(r.puts), g = siteSet(r.gets);
      if (first) {
        refPuts = p;
        refGets = g;
        first = false;
        continue;
      }
      putsVary |= p != refPuts;
      getsVary |= g != refGets;
    }

    // Exactly one side gated by the arm means the per-phase equation is not
    // well posed: an arm-independent consumer loop is fed by arm-specific
    // producers (or the reverse), so the tokens balance only once the whole
    // dispatch is summed. Judge those on the total instead.
    bool crossArm = putsVary != getsVary;
    if (crossArm) {
      EdgeRate tot;
      for (unsigned ph : phaseList) {
        const EdgeRate &r = result.rates[{edge.first, edge.second, ph}];
        tot.supply += r.supply;
        tot.rawSupply += r.rawSupply;
        tot.demand += r.demand;
        llvm::append_range(tot.puts, r.puts);
        llvm::append_range(tot.gets, r.gets);
      }
      if (tot.puts.empty() || tot.gets.empty() || tot.supply == tot.demand)
        continue;
      result.imbalances.push_back({true, edge.first, edge.second, {}, tot});
      continue;
    }

    for (unsigned ph : phaseList) {
      const EdgeRate &rate = result.rates[{edge.first, edge.second, ph}];
      if (rate.puts.empty() || rate.gets.empty() || rate.supply == rate.demand)
        continue;
      if (!seen.insert(
                   {edge.first.data(), edge.second, rate.supply, rate.demand})
               .second)
        continue;
      result.imbalances.push_back(
          {false, edge.first, edge.second, result.phases[ph], rate});
    }
  }
}

} // namespace air
} // namespace xilinx

RefeedRateAnalysis::RefeedRateAnalysis(Operation *scope) {
  RefeedRateBuilder(*this, scope).run();
}

FailureOr<int64_t>
RefeedRateAnalysis::inferRefeedCount(StringRef chanName, unsigned bundleIndex,
                                     const PhaseKey &phase) const {
  auto phaseIt = llvm::find(phases, phase);
  if (phaseIt == phases.end())
    return failure();
  auto it = rates.find(
      {chanName, bundleIndex,
       static_cast<unsigned>(std::distance(phases.begin(), phaseIt))});
  if (it == rates.end())
    return failure();
  const EdgeRate &r = it->second;
  if (r.rawSupply <= 0 || r.demand <= 0)
    return failure();
  if (r.demand % r.rawSupply)
    return failure();
  return r.demand / r.rawSupply;
}

std::string RefeedRateAnalysis::phaseToString(const PhaseKey &phase) const {
  if (phase.empty())
    return "the single phase";
  std::string s;
  llvm::raw_string_ostream os(s);
  llvm::interleave(
      phase, os,
      [&](const std::pair<Value, int64_t> &sel) {
        os << "iteration " << sel.second << " of the dispatch loop on ";
        sel.first.printAsOperand(os, OpPrintingFlags());
      },
      ", ");
  return s;
}
