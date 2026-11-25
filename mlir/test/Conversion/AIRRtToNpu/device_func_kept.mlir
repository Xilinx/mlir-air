//===- device_func_kept.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu --split-input-file %s | FileCheck %s

// The functions a core calls, directly or through one another, are core code:
// the rewrites of the host's code leave them as they are. Each case pairs one
// with a function in the device that no core calls, which is still rewritten.

// The cleanup of dead host code: a call with no results and no buffer
// operands (here one setting a core control register) stays.

// CHECK-LABEL: func.func private @inner
// CHECK: call @llvm.aie2p.set.ctrl.reg
// CHECK: memref.store
// CHECK-LABEL: func.func private @helper
// CHECK: call @llvm.aie2p.set.ctrl.reg
// CHECK: call @inner
// CHECK-LABEL: func.func @host
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
    func.func @host() {
      %l1 = memref.alloc() : memref<16xf32, 2>
      %c0 = arith.constant 0 : index
      %x = arith.constant 2.0 : f32
      memref.store %x, %l1[%c0] : memref<16xf32, 2>
      return
    }
  }
}

// -----

// Stores to L1, copies between arguments and affine loops stay in core code.
// In the host's code the stores and copies go, and the loops are unrolled.

// CHECK-LABEL: func.func private @core_l1_store
// CHECK: memref.store
// CHECK: affine.store
// CHECK-LABEL: func.func private @core_copy
// CHECK: memref.copy
// CHECK-LABEL: func.func private @core_loop
// CHECK: affine.for
// CHECK-LABEL: func.func @host_l1_store
// CHECK-NOT: store
// CHECK: return
// CHECK-LABEL: func.func @host_copy
// CHECK-NOT: memref.copy
// CHECK: return
// CHECK-LABEL: func.func @host_loop
// CHECK-NOT: affine.for
// CHECK: return

module {
  aie.device(npu2) @seg {
    %t = aie.tile(0, 2)
    %buf = aie.buffer(%t) {sym_name = "buf"} : memref<16xf32, 2>
    %src = aie.buffer(%t) {sym_name = "src"} : memref<16xf32, 2>
    %out = aie.buffer(%t) {sym_name = "out"} : memref<16xf32>
    func.func private @core_l1_store(%a: memref<16xf32, 2>) {
      %c0 = arith.constant 0 : index
      %x = arith.constant 2.0 : f32
      memref.store %x, %a[%c0] : memref<16xf32, 2>
      affine.store %x, %a[1] : memref<16xf32, 2>
      return
    }
    func.func private @core_copy(%a: memref<16xf32, 2>, %b: memref<16xf32, 2>) {
      memref.copy %a, %b : memref<16xf32, 2> to memref<16xf32, 2>
      return
    }
    func.func private @core_loop(%a: memref<16xf32>) {
      %x = arith.constant 2.0 : f32
      affine.for %i = 0 to 16 {
        affine.store %x, %a[%i] : memref<16xf32>
      }
      return
    }
    %core = aie.core(%t) {
      func.call @core_l1_store(%buf) : (memref<16xf32, 2>) -> ()
      func.call @core_copy(%src, %buf) : (memref<16xf32, 2>, memref<16xf32, 2>) -> ()
      func.call @core_loop(%out) : (memref<16xf32>) -> ()
      aie.end
    }
    func.func @host_l1_store(%a: memref<16xf32, 2>) {
      %c0 = arith.constant 0 : index
      %x = arith.constant 2.0 : f32
      memref.store %x, %a[%c0] : memref<16xf32, 2>
      affine.store %x, %a[1] : memref<16xf32, 2>
      return
    }
    func.func @host_copy(%a: memref<16xf32, 2>, %b: memref<16xf32, 2>) {
      memref.copy %a, %b : memref<16xf32, 2> to memref<16xf32, 2>
      return
    }
    func.func @host_loop(%a: memref<16xf32>) {
      %x = arith.constant 2.0 : f32
      affine.for %i = 0 to 16 {
        affine.store %x, %a[%i] : memref<16xf32>
      }
      return
    }
  }
}
