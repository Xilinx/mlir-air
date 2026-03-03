//===- scf_if_channel_in_loop.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// Test that air-dependency does not create cross-scope dependencies when
// async ops (air.channel.get) are inside scf.if inside scf.for, and a
// memref.dealloc follows the loop. The dealloc must NOT depend directly on
// the channel.get token (which is defined in a child region), but instead
// on the scf.for's loop-carried result token.

// CHECK-LABEL: func.func @channel_in_scf_if_loop
// CHECK: scf.for
// CHECK:   scf.if
// CHECK:     air.channel.get
// CHECK: air.execute
// CHECK:   memref.dealloc
module {
  air.channel @ch [1, 1]
  func.func @channel_in_scf_if_loop() {
    %c1 = arith.constant 1 : index
    air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1_0 = arith.constant 1 : index
      %alloc = memref.alloc() : memref<48xi32, 2 : i32>
      scf.for %i = %c0 to %c4 step %c1_0 {
        %cond = arith.cmpi eq, %i, %c0 : index
        scf.if %cond {
          air.channel.get @ch[] (%alloc[] [] []) : (memref<48xi32, 2 : i32>)
        } else {
        }
      }
      // dealloc after loop — must depend on loop result, not inner token
      memref.dealloc %alloc : memref<48xi32, 2 : i32>
    }
    return
  }
}
