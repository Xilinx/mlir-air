//===- air_channel_sibling_ring_no_rotation.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Two sibling get-loops on the SAME channel each drain the resident stream
// through their OWN 2-deep ping-pong ring (each loop body interleaves two
// buffers). This is the multi-phase resident-stream pattern: each phase (loop)
// consumes the whole stream once, one loop after the other.
//
// Unlike air_channel_peeled_rotation.mlir -- where a SINGLE steady-state loop
// holds the interleaved sites -- here TWO distinct loops each hold multiple
// sites. That is not one round-robin rotation; collapsing all four buffers into
// a single circular BD chain would make each loop see only every 4th block and
// mis-deliver. Two things must hold: (1) detectNBufferRotation rejects it (more
// than one enclosing loop holds multiple sites), and (2) getRepeatCounts keys
// tasks on loop identity, so the two equal-trip loops do NOT merge into one
// infinite chain. The result is one terminated counted task PER loop, each a
// 2-BD ring carrying its own repeat_count (trip - 1 = 7).

// CHECK: aie.device
// CHECK-DAG:   %[[TILE:.*]] = aie.tile(2, 3)
// CHECK:       aie.mem(%[[TILE]])
// Two separate counted tasks, each a 2-BD ring with repeat_count = 7 (asserted
// order-independently). Two distinct task_ids prove the two loops did not fuse.
// CHECK-DAG:     aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 7)
// CHECK-DAG:     aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 7)
// CHECK-DAG:     aie.dma_bd({{.*}}) {task_id = 0 : i32}
// CHECK-DAG:     aie.dma_bd({{.*}}) {task_id = 0 : i32}
// CHECK-DAG:     aie.dma_bd({{.*}}) {task_id = 1 : i32}
// CHECK-DAG:     aie.dma_bd({{.*}}) {task_id = 1 : i32}

air.channel @channel_0 [1, 1]
func.func @sibling_ring_phases() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c8_h = arith.constant 8 : index
        %t0, %buf0 = air.execute -> (memref<32x32xbf16, 2>) {
          %m = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %m : memref<32x32xbf16, 2>
        }
        %t1, %buf1 = air.execute -> (memref<32x32xbf16, 2>) {
          %m = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %m : memref<32x32xbf16, 2>
        }
        %t2, %buf2 = air.execute -> (memref<32x32xbf16, 2>) {
          %m = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %m : memref<32x32xbf16, 2>
        }
        %t3, %buf3 = air.execute -> (memref<32x32xbf16, 2>) {
          %m = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %m : memref<32x32xbf16, 2>
        }
        // phase 0: 2-deep ring over buf0/buf1, trip 8
        %p0 = scf.for %i = %c0_h to %c8_h step %c1_h iter_args(%d = %t0) -> (!air.async.token) {
          %g0 = air.channel.get async [%d] @channel_0[] (%buf0[] [] []) : (memref<32x32xbf16, 2>)
          %g1 = air.channel.get async [%g0] @channel_0[] (%buf1[] [] []) : (memref<32x32xbf16, 2>)
          scf.yield %g1 : !air.async.token
        }
        // phase 1: 2-deep ring over buf2/buf3, trip 8
        %p1 = scf.for %i = %c0_h to %c8_h step %c1_h iter_args(%d = %p0) -> (!air.async.token) {
          %g0 = air.channel.get async [%d] @channel_0[] (%buf2[] [] []) : (memref<32x32xbf16, 2>)
          %g1 = air.channel.get async [%g0] @channel_0[] (%buf3[] [] []) : (memref<32x32xbf16, 2>)
          scf.yield %g1 : !air.async.token
        }
        %d0 = air.execute [%p1] { memref.dealloc %buf0 : memref<32x32xbf16, 2> }
        %d1 = air.execute [%p1] { memref.dealloc %buf1 : memref<32x32xbf16, 2> }
        %d2 = air.execute [%p1] { memref.dealloc %buf2 : memref<32x32xbf16, 2> }
        %d3 = air.execute [%p1] { memref.dealloc %buf3 : memref<32x32xbf16, 2> }
      }
    }
  }
  return
}
