//===- air_channel_three_phase_no_rotation.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Generalization of the sibling-ring case to N=3 phases: three sibling get-loops
// on the same channel, each with the SAME trip count, each draining its own
// buffer. They must become THREE distinct counted BD tasks (repeat_count = 7),
// not one infinite chain cycling all three buffers -- task partitioning keys on
// loop identity, so equal trip counts do not merge sibling loops.

// CHECK: aie.device
// CHECK-DAG:   %[[TILE:.*]] = aie.tile(2, 3)
// CHECK:       aie.mem(%[[TILE]])
// CHECK-DAG:     aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 7)
// CHECK-DAG:     aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 7)
// CHECK-DAG:     aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 7)
// CHECK-DAG:     aie.dma_bd({{.*}}) {task_id = 0 : i32}
// CHECK-DAG:     aie.dma_bd({{.*}}) {task_id = 1 : i32}
// CHECK-DAG:     aie.dma_bd({{.*}}) {task_id = 2 : i32}

air.channel @channel_0 [1, 1]
func.func @three_phase() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%a, %b) in (%c=%c1, %e=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
        %c0 = arith.constant 0 : index
        %c1h = arith.constant 1 : index
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
        %ph0 = scf.for %i = %c0 to %c8 step %c1h iter_args(%dep = %t0) -> (!air.async.token) {
          %g = air.channel.get async [%dep] @channel_0[] (%buf0[] [] []) : (memref<32x32xbf16, 2>)
          scf.yield %g : !air.async.token
        }
        %ph1 = scf.for %i = %c0 to %c8 step %c1h iter_args(%dep = %ph0) -> (!air.async.token) {
          %g = air.channel.get async [%dep] @channel_0[] (%buf1[] [] []) : (memref<32x32xbf16, 2>)
          scf.yield %g : !air.async.token
        }
        %ph2 = scf.for %i = %c0 to %c8 step %c1h iter_args(%dep = %ph1) -> (!air.async.token) {
          %g = air.channel.get async [%dep] @channel_0[] (%buf2[] [] []) : (memref<32x32xbf16, 2>)
          scf.yield %g : !air.async.token
        }
        %d0 = air.execute [%ph2] { memref.dealloc %buf0 : memref<32x32xbf16, 2> }
        %d1 = air.execute [%ph2] { memref.dealloc %buf1 : memref<32x32xbf16, 2> }
        %d2 = air.execute [%ph2] { memref.dealloc %buf2 : memref<32x32xbf16, 2> }
      }
    }
  }
  return
}
