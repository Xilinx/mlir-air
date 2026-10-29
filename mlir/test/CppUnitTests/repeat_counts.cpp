// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// getRepeatCounts: the BD tasks of one DMA channel and the repeat count of
// each. Each case parses channel puts in loops, in program order, with %n as
// a run-time trip count. The result reads `count[names]` per task, then
// ` name=times` for each transfer reported as running several times in a row
// per trip. A count of `rt` is a circular chain; `bad` is one the chain
// cannot express.

#include "air/Conversion/AIRToAIESchedulingUtils.h"
#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <iostream>
#include <string>

using namespace mlir;
using namespace xilinx;

static int failures = 0;

// Wraps `body` in a function over buffers %x and %z, with %c0, %c1, %c2, %c3
// and %c5 in scope and %n as a run-time trip count.
static std::string module(const std::string &body) {
  return R"mlir(
  air.channel @a [1]
  func.func @f(%x: memref<4xi32>, %z: memref<4xi32>, %n: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c5 = arith.constant 5 : index
)mlir" + body +
         R"mlir(
    return
  }
)mlir";
}

static std::string count(int rep) {
  if (rep == air::kRunTimeRepeatCount)
    return "rt";
  if (rep == air::kUnsupportedRunTimeRepeatCount)
    return "bad";
  return std::to_string(rep);
}

static void check(MLIRContext &ctx, const std::string &what,
                  const std::string &body, const std::string &expected) {
  OwningOpRef<ModuleOp> mod =
      parseSourceString<ModuleOp>(module(body), ParserConfig(&ctx));
  if (!mod) {
    std::cerr << "FAIL " << what << ": does not parse\n";
    failures++;
    return;
  }
  std::vector<Operation *> ops;
  mod->walk([&](air::ChannelPutOp put) { ops.push_back(put); });
  auto name = [](Operation *op) {
    return op->getAttrOfType<StringAttr>("name").str();
  };

  llvm::DenseMap<Operation *, int64_t> timesPerTrip;
  std::string actual;
  for (auto &[rep, task] : air::getRepeatCounts(ops, &timesPerTrip)) {
    actual += count(rep) + "[";
    for (Operation *op : task)
      actual += name(op);
    actual += "]";
  }
  for (Operation *op : ops)
    if (timesPerTrip.count(op))
      actual += " " + name(op) + "=" + std::to_string(timesPerTrip.at(op));
  if (actual == expected)
    return;
  std::cerr << "FAIL " << what << ": got " << actual << ", expected "
            << expected << "\n";
  failures++;
}

// A put of buffer `buf`; `name` tells the transfers apart.
static std::string put(const std::string &buf, const std::string &name) {
  return "    air.channel.put @a[] (%" + buf + "[] [] []) {name = \"" + name +
         "\"} : (memref<4xi32>)\n";
}

// `body` in a loop of `trips` (an SSA value) trips.
static std::string loop(const std::string &trips, const std::string &body) {
  return "    scf.for %i_" + std::to_string(body.size()) + " = %c0 to " +
         trips + " step %c1 {\n" + body + "    }\n";
}

int main() {
  DialectRegistry registry;
  registry.insert<air::airDialect, arith::ArithDialect, func::FuncDialect,
                  memref::MemRefDialect, scf::SCFDialect>();
  MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  // Every transfer moves one buffer, so the chain is one BD that loops to
  // itself, sent once per transfer. How many times each transfer runs in a
  // row does not matter. The count is that of the loop around the kept
  // transfer; a channel of one task is a circular chain and does not use it.
  check(ctx, "one buffer in run-time and fixed loops",
        loop("%n", loop("%c5", put("x", "p"))) +
            loop("%c3", loop("%c5", put("x", "q"))),
        "4[p]");
  check(ctx, "one buffer, run-time loops of other bounds",
        loop("%n", loop("%c5", put("x", "p"))) +
            loop("%c3", loop("%n", put("x", "q"))),
        "4[p]");

  // A transfer alone in a fixed loop inside the run-time loop takes that many
  // BDs in a row.
  check(ctx, "transfer alone in a fixed loop",
        loop("%n", put("x", "p") + loop("%c2", put("z", "q"))), "rt[pq] q=2");
  // Two BDs: each transfer folded onto one must run as many times in a row as
  // the one kept.
  check(ctx, "folded transfers run unlike times in a row",
        loop("%n", put("x", "p") + loop("%c2", put("z", "q"))) +
            loop("%n", put("x", "r") + loop("%c3", put("z", "s"))),
        "rt[p]bad[q] q=2");
  check(ctx, "folded transfers run alike times in a row",
        loop("%n", put("x", "p") + loop("%c2", put("z", "q"))) +
            loop("%n", put("x", "r") + loop("%c2", put("z", "s"))),
        "rt[pq] q=2");

  if (failures)
    return 1;
  std::cout << "PASS\n";
  return 0;
}
