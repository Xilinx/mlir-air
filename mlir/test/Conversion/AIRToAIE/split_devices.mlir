//===- split_devices.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-split-devices='output-prefix=%T/' | FileCheck %s
// RUN: aie-opt %T/aie.TestSegment0.mlir | FileCheck -check-prefix=AIE %s

// CHECK-NOT: AIE.device
// CHECK: func.func @main

// AIE: module @aie.TestSegment0
// AIE-NEXT: AIE.device
// AIE-NOT: func.func @main

AIE.device(xcvc1902) {
  %tile11 = AIE.tile(1, 1)
} { sym_name = "TestSegment0" }

func.func @main(%a0: memref<1024xbf16>, %a1: memref<1024xbf16>, %a2: memref<1024xbf16>) {
  return
}
