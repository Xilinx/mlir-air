//===- device_func_kept_trace.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu="trace-size=8192" %s | FileCheck %s

// The trace setup goes into the host's code, not into a function a core
// calls.

// CHECK-LABEL: func.func private @helper
// CHECK-NOT: aiex.npu.write32
// CHECK: return
// CHECK: aie.runtime_sequence @host
// CHECK: aiex.npu.write32

module {
  aie.device(npu2) @seg {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) {sym_name = "buf"} : memref<16xf32>
    func.func private @helper(%a: memref<16xf32>) {
      %c0 = arith.constant 0 : index
      %x = arith.constant 2.0 : f32
      memref.store %x, %a[%c0] : memref<16xf32>
      return
    }
    %core = aie.core(%t) {
      func.call @helper(%buf) : (memref<16xf32>) -> ()
      aie.end
    }
    func.func @host(%arg0: memref<16xf32>) {
      return
    }
  }
}
