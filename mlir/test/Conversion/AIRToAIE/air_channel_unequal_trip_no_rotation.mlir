//===- air_channel_unequal_trip_no_rotation.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Two channel.get sites on the SAME channel target DISTINCT buffers, each
// drained by its OWN separate scf.for loop (here 6 and 4 iterations). This is
// the unrolled multi-phase projection pattern: each phase consumes its buffer
// as a contiguous block, one loop after the other.
//
// This is NOT an N-buffer round-robin rotation. A rotation interleaves its
// buffers inside ONE shared loop (successive iterations hand off to the next
// buffer); here the two sites share no enclosing loop, so
// detectNBufferRotation must reject them. Collapsing block consumers into a
// single circular BD chain would make each phase see only every Nth block and
// mis-deliver data. They must fall through to per-op counted BDs: one
// terminated dma_start per buffer, each carrying its own repeat_count
// (trip - 1: 5 and 3), NOT a single dma_start whose next_bd cycles between the
// two buffers.
//
// The two counted tasks are asserted order-independently and without pinning
// their terminator to a single shared end block: generateDmaBdProgram may give
// each non-infinite task its own end block. The presence of two dma_start ops
// each with a distinct repeat_count (rather than one repeat_count-less
// dma_start whose chain cycles) is what proves this is not a rotation.

// CHECK: aie.device
// CHECK-DAG:   %[[TILE:.*]] = aie.tile(2, 3)
// CHECK-DAG:   aie.buffer(%[[TILE]]) {{.*}} : memref<32x32xbf16, 2>
// CHECK-DAG:   aie.buffer(%[[TILE]]) {{.*}} : memref<32x32xbf16, 2>
// CHECK:       aie.mem(%[[TILE]])
// CHECK-DAG:     aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 5)
// CHECK-DAG:     aie.dma_start(S2MM, 0, {{.*}}, {{.*}}, repeat_count = 3)
// CHECK-DAG:     aie.dma_bd(%{{.*}} : memref<32x32xbf16, 2> offset = 0 len = 1024) {task_id = 0
// CHECK-DAG:     aie.dma_bd(%{{.*}} : memref<32x32xbf16, 2> offset = 0 len = 1024) {task_id = 1
// CHECK-DAG:     aie.end

air.channel @channel_0 [1, 1]
func.func @unequal_trip_phases() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c4_h = arith.constant 4 : index
        %c6_h = arith.constant 6 : index
        %async_token_0, %buf0 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %async_token_1, %buf1 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        // phase 0: get buf0, trip count 6
        %3 = scf.for %i = %c0_h to %c6_h step %c1_h iter_args(%d = %async_token_0) -> (!air.async.token) {
          %g = air.channel.get async [%d] @channel_0[] (%buf0[] [] []) : (memref<32x32xbf16, 2>)
          scf.yield %g : !air.async.token
        }
        // phase 1: get buf1, trip count 4
        %4 = scf.for %i = %c0_h to %c4_h step %c1_h iter_args(%d = %3) -> (!air.async.token) {
          %g = air.channel.get async [%d] @channel_0[] (%buf1[] [] []) : (memref<32x32xbf16, 2>)
          scf.yield %g : !air.async.token
        }
        %async_token_d0 = air.execute [%4] {
          memref.dealloc %buf0 : memref<32x32xbf16, 2>
        }
        %async_token_d1 = air.execute [%4] {
          memref.dealloc %buf1 : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}
