//===- air_channel_sibling_ring_nested_loop.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Production shape (as produced by the ping-pong transform for a persistent
// multi-phase decode loop): two sibling get-loops each drain the resident stream
// through their own 2-buffer ring, and the pair sits inside an OUTER repeating
// loop. The two inner loops must remain distinct BD tasks (keyed on the inner
// loop identity, not the equal trip count), and each task's repeat_count is the
// PRODUCT of the enclosing loop trip counts: outer 4 * inner 8 = 32, minus 1.

// CHECK: aie.device
// CHECK-DAG:   %[[TILE:.*]] = aie.tile(2, 3)
// CHECK:       aie.mem(%[[TILE]])
// CHECK-DAG:     aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 31)
// CHECK-DAG:     aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 31)
// CHECK-DAG:     aie.dma_bd({{.*}}) {task_id = 0 : i32}
// CHECK-DAG:     aie.dma_bd({{.*}}) {task_id = 0 : i32}
// CHECK-DAG:     aie.dma_bd({{.*}}) {task_id = 1 : i32}
// CHECK-DAG:     aie.dma_bd({{.*}}) {task_id = 1 : i32}

air.channel @channel_0 [1, 1]
func.func @nested_sibling_ring() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%a, %b) in (%c=%c1, %dd=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
        %c0 = arith.constant 0 : index
        %c1h = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
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
        %outer = scf.for %v = %c0 to %c4 step %c1h iter_args(%odep = %t0) -> (!air.async.token) {
          // phase 0: 2-buffer ring over buf0/buf1
          %p0 = scf.for %i = %c0 to %c8 step %c1h iter_args(%dep = %odep) -> (!air.async.token) {
            %g0 = air.channel.get async [%dep] @channel_0[] (%buf0[] [] []) : (memref<32x32xbf16, 2>)
            %g1 = air.channel.get async [%g0] @channel_0[] (%buf1[] [] []) : (memref<32x32xbf16, 2>)
            scf.yield %g1 : !air.async.token
          }
          // phase 1: 2-buffer ring over buf2/buf3
          %p1 = scf.for %i = %c0 to %c8 step %c1h iter_args(%dep = %p0) -> (!air.async.token) {
            %g0 = air.channel.get async [%dep] @channel_0[] (%buf2[] [] []) : (memref<32x32xbf16, 2>)
            %g1 = air.channel.get async [%g0] @channel_0[] (%buf3[] [] []) : (memref<32x32xbf16, 2>)
            scf.yield %g1 : !air.async.token
          }
          scf.yield %p1 : !air.async.token
        }
        %d0 = air.execute [%outer] { memref.dealloc %buf0 : memref<32x32xbf16, 2> }
        %d1 = air.execute [%outer] { memref.dealloc %buf1 : memref<32x32xbf16, 2> }
        %d2 = air.execute [%outer] { memref.dealloc %buf2 : memref<32x32xbf16, 2> }
        %d3 = air.execute [%outer] { memref.dealloc %buf3 : memref<32x32xbf16, 2> }
      }
    }
  }
  return
}
