//===- device_func_kept.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// The functions a core calls, directly or through one another, are core
// code: the cleanup of the host's functions leaves them alone, including a
// call with no results and no buffer operands (here one setting a core
// control register). A function in the device that no core calls is still
// cleaned up.

// CHECK-LABEL: func.func private @inner
// CHECK: call @llvm.aie2p.set.ctrl.reg
// CHECK: memref.store
// CHECK-LABEL: func.func private @helper
// CHECK: call @llvm.aie2p.set.ctrl.reg
// CHECK: call @inner
// CHECK-LABEL: func.func @ctrl
// CHECK-NOT: memref.store
// CHECK: return

module {
  aie.device(npu2) @seg {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) {sym_name = "buf"} : memref<16xf32>
    func.func private @llvm.aie2p.set.ctrl.reg(i32, i32)
    func.func private @inner(%a: memref<16xf32>) {
      %c1 = arith.constant 1 : i32
      %c12 = arith.constant 12 : i32
      func.call @llvm.aie2p.set.ctrl.reg(%c1, %c12) : (i32, i32) -> ()
      %c0 = arith.constant 0 : index
      %x = arith.constant 2.0 : f32
      memref.store %x, %a[%c0] : memref<16xf32>
      return
    }
    func.func private @helper(%a: memref<16xf32>) {
      %c1 = arith.constant 1 : i32
      %c12 = arith.constant 12 : i32
      func.call @llvm.aie2p.set.ctrl.reg(%c1, %c12) : (i32, i32) -> ()
      func.call @inner(%a) : (memref<16xf32>) -> ()
      return
    }
    %core = aie.core(%t) {
      func.call @helper(%buf) : (memref<16xf32>) -> ()
      aie.end
    }
    func.func @ctrl() {
      %l1 = memref.alloc() : memref<16xf32, 2>
      %c0 = arith.constant 0 : index
      %x = arith.constant 2.0 : f32
      memref.store %x, %l1[%c0] : memref<16xf32, 2>
      return
    }
  }
}
