//===- air_channel_lock_count_per_channel.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' | FileCheck %s

// A core buffer takes two transfers on chan_in and then sends itself out on
// chan_out. Each channel's DMA counts against the core on its own: the core
// releases the buffer to the MM2S DMA once, so the MM2S BD acquires 1, not
// the 2 that counting chan_in's writes against chan_out's read would give.

// CHECK-LABEL: aie.mem(%{{.*}}tile_0_2)
// CHECK: aie.dma_start(MM2S, 0, ^bb1
// CHECK-NEXT: ^bb1:
// CHECK-NEXT: aie.use_lock(%[[OUT_CONS:.*]], AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: aie.dma_bd
// CHECK-NEXT: aie.use_lock(%[[OUT_PROD:.*]], Release, %c1_i32)
// CHECK: aie.core(%{{.*}}tile_0_2)
// CHECK: aie.use_lock(%[[OUT_PROD]], AcquireGreaterEqual, %c1_i32)
// CHECK: aie.use_lock(%[[OUT_CONS]], Release, %c1_i32)

module {
  air.channel @chan_in [1, 1]
  air.channel @chan_out [1, 1]
  func.func @in_then_out(%arg0: memref<64xi32>, %arg1: memref<32xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%a=%arg0, %b=%arg1) : memref<64xi32>, memref<32xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      air.channel.put @chan_in[] (%a[%c0] [%c32] [%c1_l]) : (memref<64xi32>)
      air.channel.put @chan_in[] (%a[%c32] [%c32] [%c1_l]) : (memref<64xi32>)
      air.channel.get @chan_out[] (%b[] [] []) : (memref<32xi32>)
      air.segment @seg {
        %c1_s = arith.constant 1 : index
        air.herd @herd tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %buf = memref.alloc() : memref<32xi32, 2>
          air.channel.get @chan_in[%tx, %ty] (%buf[] [] []) : (memref<32xi32, 2>)
          air.channel.get @chan_in[%tx, %ty] (%buf[] [] []) : (memref<32xi32, 2>)
          air.channel.put @chan_out[%tx, %ty] (%buf[] [] []) : (memref<32xi32, 2>)
          memref.dealloc %buf : memref<32xi32, 2>
        }
      }
    }
    return
  }
}
