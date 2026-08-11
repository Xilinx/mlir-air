//===- RefeedRateAnalysis.h -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_UTIL_REFEED_RATE_ANALYSIS_H
#define AIR_UTIL_REFEED_RATE_ANALYSIS_H

#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace xilinx {
namespace air {

// Cyclo-static token-rate analysis for air.channel edges.
//
// An air.channel is balanced when, over one iteration of the enclosing
// dispatch, the tokens its producers put equal the tokens its consumers get:
//
//     sum_puts(reps * size * refeed)  ==  sum_gets(reps * size)
//
// The refeed factor is the count-free re-broadcast multiplier (a single
// resident buffer re-sent N times per production; see attrs::RefeedCount).
// Solving the equation for it is what lets the compiler derive the count
// instead of trusting a hand-written attribute.
//
// Two properties of real AIR programs make a plain SDF sum wrong:
//
//   1. Mutually exclusive modes. A superkernel selects an scf.index_switch
//      arm from a runtime value; each arm has its own rate vector, and only
//      one arm runs per dispatch. Summing over both arms hides a deficit in
//      one behind a surplus in the other. Rates are therefore computed per
//      arm ("phase" in cyclo-static dataflow terms).
//
//   2. Phase-dependent trip counts. Inner loop bounds are commonly an
//      scf.index_switch on an outer induction variable -- a constant table
//      indexed by the phase. Those bounds look dynamic to a syntactic trip
//      count query but are statically enumerable, so the analysis evaluates
//      index arithmetic symbolically with the enclosing IVs bound.
//
// Spatial fan-out is not a rate deficit: one put on a channel carrying a
// broadcast_shape feeds every destination. Both sides are therefore counted
// per destination -- a get inside an air.herd counts once, not once per core,
// and a broadcast put is likewise not multiplied by the fan-out.
class RefeedRateAnalysis {
public:
  // Identifies one mutually exclusive mode: the arms chosen at every
  // scf.index_switch whose selector is not statically known. Entries are
  // (selector root value, case value), ordered by selector discovery so the
  // key is comparable across hierarchy levels.
  using PhaseKey = llvm::SmallVector<std::pair<mlir::Value, int64_t>, 2>;

  struct EdgeRate {
    // Producer tokens with the declared refeed factor applied.
    int64_t supply = 0;
    // Producer tokens with every refeed factor forced to 1. The inferred
    // refeed count is demand / rawSupply when that divides exactly.
    int64_t rawSupply = 0;
    int64_t demand = 0;
    // Destinations a single broadcast payload reaches. Every tile's get is
    // accumulated into `demand` and divided by this once the traversal is
    // done, which keeps the arithmetic exact however the destinations are
    // split across air.herd ops.
    int64_t fanout = 1;
    // One entry per contributing emission, for diagnostics: the op, the tokens
    // it moved on this edge, and the refeed factor applied (1 for a get).
    struct Site {
      ChannelInterface op;
      int64_t tokens;
      int64_t refeed;
    };
    llvm::SmallVector<Site, 4> puts;
    llvm::SmallVector<Site, 4> gets;
  };

  struct Imbalance {
    // True when the two sides are gated by different arms, so the balance was
    // summed over the whole dispatch instead of checked per phase.
    bool wholeDispatch = false;
    llvm::StringRef channel;
    // Linearized index into the channel's bundle: an air.channel @c [4, 4] is
    // sixteen independent edges, each with its own balance equation.
    unsigned bundleIndex;
    PhaseKey phase;
    EdgeRate rate;
  };

  // Identifies one dataflow edge in one phase.
  struct RateKey {
    llvm::StringRef channel;
    unsigned bundleIndex;
    unsigned phase;
    bool operator==(const RateKey &o) const {
      return channel == o.channel && bundleIndex == o.bundleIndex &&
             phase == o.phase;
    }
  };

  explicit RefeedRateAnalysis(mlir::Operation *scope);

  // False when any channel op sits under control flow the analysis could not
  // resolve. Callers must fall back to the declared attribute; the rates of
  // the affected channels are not reported.
  bool isValid() const { return valid; }

  // Channels the analysis gave up on, with the op that defeated it.
  llvm::ArrayRef<std::pair<llvm::StringRef, mlir::Operation *>>
  getUnanalyzable() const {
    return unanalyzable;
  }

  // Phases whose balance equation does not close under the declared refeed
  // counts. A negative (supply - demand) is a deficit and deadlocks; a
  // positive one over-supplies and only wastes bandwidth.
  llvm::ArrayRef<Imbalance> getImbalances() const { return imbalances; }

  // Refeed count implied by the balance equation for `chanName` in `phase`.
  // Fails when the ratio is not a positive integer, or when the channel was
  // not analyzable.
  mlir::FailureOr<int64_t> inferRefeedCount(llvm::StringRef chanName,
                                            unsigned bundleIndex,
                                            const PhaseKey &phase) const;

  // Rates of every edge, in discovery order.
  const llvm::MapVector<RateKey, EdgeRate> &getRates() const { return rates; }
  const llvm::SmallVector<PhaseKey> &getPhases() const { return phases; }

  // Human-readable phase, e.g. "arm 0 of the switch on %arg22".
  std::string phaseToString(const PhaseKey &phase) const;

private:
  friend class RefeedRateBuilder;

  bool valid = true;
  llvm::SmallVector<PhaseKey> phases;
  llvm::MapVector<RateKey, EdgeRate> rates;
  llvm::SmallVector<Imbalance, 4> imbalances;
  llvm::SmallVector<std::pair<llvm::StringRef, mlir::Operation *>> unanalyzable;
};

} // namespace air
} // namespace xilinx

namespace llvm {
template <>
struct DenseMapInfo<xilinx::air::RefeedRateAnalysis::RateKey> {
  using Key = xilinx::air::RefeedRateAnalysis::RateKey;
  static Key getEmptyKey() { return {StringRef(), ~0u, ~0u}; }
  static Key getTombstoneKey() { return {StringRef(), ~0u, ~1u}; }
  static unsigned getHashValue(const Key &k) {
    return hash_combine(k.channel, k.bundleIndex, k.phase);
  }
  static bool isEqual(const Key &a, const Key &b) { return a == b; }
};
} // namespace llvm

#endif // AIR_UTIL_REFEED_RATE_ANALYSIS_H
