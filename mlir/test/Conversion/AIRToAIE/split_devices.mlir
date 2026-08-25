//===- split_devices.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// %t.d replaces %T, which lit 23 removed. Unlike %T it is per-test rather than
// per-directory, and lit does not create it -- the pass opens files under the
// prefix and does not mkdir, so the directory has to exist first.
// RUN: rm -rf %t.d
// RUN: mkdir -p %t.d
// RUN: air-opt %s -air-split-devices='output-prefix=%t.d/' | FileCheck %s
// RUN: aie-opt %t.d/aie.TestSegment0.mlir | FileCheck -check-prefix=AIE %s

// CHECK-NOT: aie.device
// CHECK: func.func @main

// AIE: module @aie.TestSegment0
// AIE-NEXT: aie.device
// AIE-NOT: func.func @main

aie.device(xcvc1902) {
  %tile11 = aie.tile(1, 1)
} { sym_name = "TestSegment0" }

func.func @main(%a0: memref<1024xbf16>, %a1: memref<1024xbf16>, %a2: memref<1024xbf16>) {
  return
}
