//===- device_func_kept.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s

// A function in the device is core code: the cleanup of the host's functions
// leaves it alone, including a call with no results and no buffer operands
// (here one setting a core control register).

// CHECK-LABEL: func.func private @callee
// CHECK: call @llvm.aie2p.set.ctrl.reg
// CHECK: memref.store
// CHECK: return

module {
  aie.device(npu2) @seg {
    %t = aie.tile(0, 2)
    func.func private @llvm.aie2p.set.ctrl.reg(i32, i32)
    func.func private @callee(%a: memref<16xf32>) {
      %c1 = arith.constant 1 : i32
      %c12 = arith.constant 12 : i32
      func.call @llvm.aie2p.set.ctrl.reg(%c1, %c12) : (i32, i32) -> ()
      %c0 = arith.constant 0 : index
      %x = arith.constant 2.0 : f32
      memref.store %x, %a[%c0] : memref<16xf32>
      return
    }
  }
}
