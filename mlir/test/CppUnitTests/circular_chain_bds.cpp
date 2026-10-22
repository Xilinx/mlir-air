// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// getCircularChainBDs: the BDs of one circular chain, given each transfer's
// sends per fill. Each case parses channel puts, reads a transfer's sends per
// fill from a `sends` attribute (absent means unknown), and checks the BD
// sequence by the puts' `name` attributes.

#include "air/Conversion/AIRToAIESchedulingUtils.h"
#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <iostream>
#include <string>

using namespace mlir;
using namespace xilinx;

static int failures = 0;

static const char *kChannels = R"mlir(
  air.channel @a [1]
  air.channel @b [1]
  air.channel @c [1]
  air.channel @d [2]
)mlir";

// Wraps `puts` in a function over %m, with %c0 and %c1 in scope.
static std::string module(const std::string &puts) {
  return std::string(kChannels) + R"mlir(
  func.func @f(%m: memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
)mlir" + puts +
         R"mlir(
    return
  }
)mlir";
}

static void check(MLIRContext &ctx, const std::string &what,
                  const std::string &puts, const std::string &expected) {
  OwningOpRef<ModuleOp> mod =
      parseSourceString<ModuleOp>(module(puts), ParserConfig(&ctx));
  if (!mod) {
    std::cerr << "FAIL " << what << ": does not parse\n";
    failures++;
    return;
  }
  SmallVector<Operation *> unit;
  mod->walk([&](air::ChannelPutOp put) { unit.push_back(put); });

  auto sends = [](Operation *op) -> std::optional<int64_t> {
    if (auto n = op->getAttrOfType<IntegerAttr>("sends"))
      return n.getInt();
    return std::nullopt;
  };
  std::string actual;
  for (Operation *op : air::getCircularChainBDs(unit, sends))
    actual += op->getAttrOfType<StringAttr>("name").str();
  if (actual == expected)
    return;
  std::cerr << "FAIL " << what << ": got " << actual << ", expected "
            << expected << "\n";
  failures++;
}

static std::string put(const std::string &chan, const std::string &idx,
                       const std::string &name, int sends) {
  std::string attrs = "name = \"" + name + "\"";
  if (sends > 0)
    attrs += ", sends = " + std::to_string(sends);
  return "    air.channel.put @" + chan + "[" + idx + "] (%m[] [] []) {" +
         attrs + "} : (memref<4xi32>)\n";
}

int main() {
  DialectRegistry registry;
  registry.insert<air::airDialect, arith::ArithDialect, func::FuncDialect,
                  memref::MemRefDialect>();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  check(ctx, "nothing re-fed", put("a", "", "x", 1) + put("a", "", "z", 1),
        "xz");
  check(ctx, "one transfer", put("a", "", "x", 4), "x");

  // One flow: its transfers are sent back to back, in a single pass.
  check(ctx, "one flow, equal counts",
        put("a", "", "x", 4) + put("a", "", "z", 4), "xxxxzzzz");
  check(ctx, "one flow, unequal counts",
        put("a", "", "x", 2) + put("a", "", "z", 3), "xxzzz");
  check(ctx, "one slot spelled [] and [0]",
        put("a", "", "x", 2) + put("a", "%c0", "z", 2), "xxzz");

  // Separate flows: as many passes per fill as every flow allows.
  check(ctx, "two flows, equal counts",
        put("a", "", "x", 4) + put("b", "", "z", 4), "xz");
  check(ctx, "two flows, counts 4 and 2",
        put("a", "", "x", 4) + put("b", "", "z", 2), "xxz");
  check(ctx, "two flows, counts 6 and 4",
        put("a", "", "x", 6) + put("b", "", "z", 4), "xxxzz");
  check(ctx, "two slots of one channel",
        put("d", "%c0", "x", 4) + put("d", "%c1", "z", 2), "xxz");

  // A flow with two transfers needs a single pass, so the others are sent in
  // full within it.
  check(ctx, "shared flow next to a re-fed one",
        put("a", "", "x", 2) + put("a", "", "y", 1) + put("c", "", "w", 4),
        "xxywwww");

  check(ctx, "an unknown count leaves the chain alone",
        put("a", "", "x", 4) + put("a", "", "z", 0), "xz");

  if (failures)
    return 1;
  std::cout << "PASS\n";
  return 0;
}
