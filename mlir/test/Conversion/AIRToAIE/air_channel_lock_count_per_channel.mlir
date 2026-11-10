//===- air_channel_lock_count_per_channel.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='device=npu2 row-offset=2' --split-input-file | FileCheck %s

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

// -----

// A core buffer filled on chan_a and then on chan_b, the core reading it
// after each. Each S2MM BD acquires 1: a channel's count is its own
// transfers', not the core's reads of the buffer.

// CHECK-LABEL: aie.mem(%{{.*}}tile_0_2)
// CHECK: aie.dma_start(S2MM, 0, ^bb4
// CHECK-NEXT: ^bb4:
// CHECK-NEXT: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %c1_i32)
// CHECK: aie.dma_start(S2MM, 1, ^bb6
// CHECK-NEXT: ^bb6:
// CHECK-NEXT: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %c1_i32)

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  air.channel @chan_out [1, 1]
  func.func @two_inputs(%arg0: memref<32xi32>, %arg1: memref<32xi32>, %arg2: memref<32xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%a=%arg0, %b=%arg1, %o=%arg2) : memref<32xi32>, memref<32xi32>, memref<32xi32> {
      air.channel.put @chan_a[] (%a[] [] []) : (memref<32xi32>)
      air.channel.put @chan_b[] (%b[] [] []) : (memref<32xi32>)
      air.channel.get @chan_out[] (%o[] [] []) : (memref<32xi32>)
      air.segment @seg {
        %c1_s = arith.constant 1 : index
        air.herd @herd tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %buf = memref.alloc() : memref<32xi32, 2>
          %acc = memref.alloc() : memref<32xi32, 2>
          air.channel.get @chan_a[%tx, %ty] (%buf[] [] []) : (memref<32xi32, 2>)
          linalg.copy ins(%buf : memref<32xi32, 2>) outs(%acc : memref<32xi32, 2>)
          air.channel.get @chan_b[%tx, %ty] (%buf[] [] []) : (memref<32xi32, 2>)
          linalg.copy ins(%buf : memref<32xi32, 2>) outs(%acc : memref<32xi32, 2>)
          air.channel.put @chan_out[%tx, %ty] (%acc[] [] []) : (memref<32xi32, 2>)
          memref.dealloc %buf : memref<32xi32, 2>
          memref.dealloc %acc : memref<32xi32, 2>
        }
      }
    }
    return
  }
}
