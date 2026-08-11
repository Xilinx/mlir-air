//===- AIRVerifyRefeedBalance.cpp -------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRVerifyRefeedBalance.h"
#include "air/Dialect/AIR/AIRDialect.h"
#include "air/Util/RefeedRateAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "air-verify-refeed-balance"

using namespace mlir;
using namespace xilinx;

namespace xilinx {
namespace air {
#define GEN_PASS_DEF_AIRVERIFYREFEEDBALANCE
#include "air/Transform/Passes.h.inc"
} // namespace air
} // namespace xilinx

namespace {

class AIRVerifyRefeedBalance
    : public xilinx::air::impl::AIRVerifyRefeedBalanceBase<
          AIRVerifyRefeedBalance> {
public:
  AIRVerifyRefeedBalance() = default;
  AIRVerifyRefeedBalance(const AIRVerifyRefeedBalance &) {}

  void runOnOperation() override {
    bool sawDeficit = false;
    bool sawSurplus = false;

    getOperation().walk([&](func::FuncOp f) {
      air::RefeedRateAnalysis rates(f);

      if (this->verbose) {
        for (auto &[chan, culprit] : rates.getUnanalyzable())
          culprit->emitRemark("air.channel @")
              << chan
              << " skipped: the transfer rate under this op is not "
                 "statically resolvable";
        llvm::errs() << "// ---- refeed rates for @" << f.getSymName()
                     << (rates.isValid() ? "" : " (incomplete)") << " ----\n";
        for (auto &p : rates.getPhases())
          llvm::errs() << "//   phase: " << rates.phaseToString(p) << "\n";
        for (auto &[key, rate] : rates.getRates())
          llvm::errs() << "//   @" << key.channel << "[" << key.bundleIndex
                       << "]  ["
                       << rates.phaseToString(rates.getPhases()[key.phase])
                       << "]  supply=" << rate.supply << " (raw "
                       << rate.rawSupply << ")  demand=" << rate.demand
                       << "  puts=" << rate.puts.size()
                       << " gets=" << rate.gets.size() << "\n";
      }

      for (const auto &imb : rates.getImbalances()) {
        int64_t delta = imb.rate.supply - imb.rate.demand;
        air::ChannelInterface producer = imb.rate.puts.front().op;
        Operation *anchor = producer.getOperation();
        // A whole-dispatch sum is a weaker statement than a per-phase one --
        // it cannot distinguish a real shortfall from producers and consumers
        // simply being gated differently -- so it never escalates to an error.
        bool fatal = delta < 0 && !imb.wholeDispatch;
        InFlightDiagnostic diag =
            fatal ? anchor->emitError() : anchor->emitWarning();
        diag << "air.channel @" << imb.channel << "[" << imb.bundleIndex
             << "] is unbalanced in "
             << (imb.wholeDispatch
                     ? std::string("the whole dispatch (its two sides are "
                                   "gated by different arms)")
                     : rates.phaseToString(imb.phase))
             << ": " << imb.rate.supply << " tokens supplied, "
             << imb.rate.demand << " consumed ("
             << (delta < 0 ? "deficit " : "surplus ") << std::abs(delta) << ")";
        if (imb.rate.rawSupply > 0) {
          if (imb.rate.demand % imb.rate.rawSupply == 0)
            diag.attachNote(anchor->getLoc())
                << "the balance closes at air.refeed_count = "
                << imb.rate.demand / imb.rate.rawSupply;
          else
            diag.attachNote(anchor->getLoc())
                << "no integer air.refeed_count closes the balance; "
                << imb.rate.demand << " consumed tokens are not a multiple of "
                << imb.rate.rawSupply << " produced";
        }
        for (auto s : imb.rate.puts)
          diag.attachNote(s.op.getLoc())
              << "producer: " << s.tokens << " tokens x refeed " << s.refeed;
        for (auto s : imb.rate.gets)
          diag.attachNote(s.op.getLoc())
              << "consumer: " << s.tokens << " tokens";
        (fatal ? sawDeficit : sawSurplus) = true;
      }
    });

    // A deficit starves the consumers and deadlocks the array, so it is always
    // fatal. A surplus only wastes bandwidth -- it is a warning unless the
    // caller asked for a strict check.
    if (sawDeficit || (sawSurplus && this->strict))
      signalPassFailure();
  }
};

} // namespace

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRVerifyRefeedBalancePass() {
  return std::make_unique<AIRVerifyRefeedBalance>();
}

} // namespace air
} // namespace xilinx
