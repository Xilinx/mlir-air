// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

#include "air/Util/Util.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"

#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <iostream>
#include <string>

using namespace xilinx;

static int failures = 0;

static void expectEq(const std::string &what, int64_t actual,
                     int64_t expected) {
  if (actual == expected)
    return;
  std::cerr << "FAIL " << what << ": got " << actual << ", expected "
            << expected << "\n";
  failures++;
}

static void checkAIE2(AIE::AIEDevice device, const std::string &name) {
  const auto &tm = AIE::getTargetModel(device);

  air::DmaBdLimits shim = air::getDmaBdLimits(tm, air::MemorySpace::L3);
  expectEq(name + " shim dims", shim.dims, 3);
  expectEq(name + " shim maxWrap", shim.maxWrap, 1023);
  expectEq(name + " shim maxIterWrap", shim.maxIterWrap, 63);
  expectEq(name + " shim maxStride", shim.maxStride, 1 << 20);
  expectEq(name + " shim numBDs", shim.numBDs, 16);
  expectEq(name + " shim granuleBits", shim.granuleBits, 32);
  expectEq(name + " shim listDims", shim.listDims(), 4);
  expectEq(name + " shim maxWrapAt(0)", shim.maxWrapAt(0), 63);
  for (unsigned i = 1; i < 4; i++)
    expectEq(name + " shim maxWrapAt(" + std::to_string(i) + ")",
             shim.maxWrapAt(i), 1023);

  air::DmaBdLimits memtile = air::getDmaBdLimits(tm, air::MemorySpace::L2);
  expectEq(name + " memtile dims", memtile.dims, 4);
  expectEq(name + " memtile maxWrap", memtile.maxWrap, 1023);
  expectEq(name + " memtile maxStride", memtile.maxStride, 1 << 17);
  expectEq(name + " memtile numBDs", memtile.numBDs, 48);

  air::DmaBdLimits core = air::getDmaBdLimits(tm, air::MemorySpace::L1);
  expectEq(name + " core dims", core.dims, 3);
  expectEq(name + " core maxWrap", core.maxWrap, 255);
  expectEq(name + " core maxStride", core.maxStride, 1 << 13);
  expectEq(name + " core numBDs", core.numBDs, 16);
}

// AIE1 BDs move one contiguous run: one dim, no wrap limit, no iteration dim.
static void checkAIE1() {
  air::DmaBdLimits shim = air::getDmaBdLimits(
      AIE::getTargetModel(AIE::AIEDevice::xcvc1902), air::MemorySpace::L3);
  expectEq("xcvc1902 shim dims", shim.dims, 1);
  expectEq("xcvc1902 shim maxWrap", shim.maxWrap, -1);
  expectEq("xcvc1902 shim maxIterWrap", shim.maxIterWrap, 0);
  expectEq("xcvc1902 shim listDims", shim.listDims(), 1);
  expectEq("xcvc1902 shim maxWrapAt(0)", shim.maxWrapAt(0), -1);
  expectEq("xcvc1902 shim maxStride", shim.maxStride, 1 << 20);
}

static void checkHelpers() {
  air::DmaBdLimits none;
  expectEq("no limits listDims", none.listDims(), -1);
  expectEq("no limits strideFits", none.strideFits(int64_t(1) << 40), 1);

  air::DmaBdLimits l;
  l.maxStride = 100;
  expectEq("strideFits at limit", l.strideFits(100), 1);
  expectEq("strideFits over limit", l.strideFits(101), 0);

  l.granuleBits = 32;
  expectEq("bytesToGranules(0)", l.bytesToGranules(0), 0);
  expectEq("bytesToGranules(2)", l.bytesToGranules(2), 1);
  expectEq("bytesToGranules(4)", l.bytesToGranules(4), 1);
  expectEq("bytesToGranules(5)", l.bytesToGranules(5), 2);
}

// The op-based lookup reads the device that contains the op, else a device
// in its module, else npu1.
static void checkLookup() {
  mlir::MLIRContext ctx;
  ctx.loadDialect<AIE::AIEDialect>();

  auto aie1 = mlir::parseSourceString<mlir::ModuleOp>(
      "module { aie.device(xcvc1902) {} }", &ctx);
  if (!aie1) {
    std::cerr << "FAIL could not parse the xcvc1902 module\n";
    failures++;
    return;
  }
  expectEq("module with xcvc1902 device",
           air::getDmaBdLimits(*aie1, air::MemorySpace::L3).dims, 1);
  auto dev = *aie1->getOps<AIE::DeviceOp>().begin();
  expectEq("xcvc1902 device op",
           air::getDmaBdLimits(dev, air::MemorySpace::L3).dims, 1);

  auto empty = mlir::parseSourceString<mlir::ModuleOp>("module {}", &ctx);
  expectEq("module without a device",
           air::getDmaBdLimits(*empty, air::MemorySpace::L3).listDims(), 4);
}

int main() {
  checkAIE2(AIE::AIEDevice::npu1, "npu1");
  checkAIE2(AIE::AIEDevice::npu2, "npu2");
  checkAIE1();
  checkHelpers();
  checkLookup();
  if (failures) {
    std::cerr << failures << " check(s) failed\n";
    return 1;
  }
  return 0;
}
