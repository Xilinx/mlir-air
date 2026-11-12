//===- air_channel_rerun_circular_chain.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// The herd of air_channel_unequal_trip_no_rotation.mlir in a loop of its
// segment: it runs its body twice in one dispatch. Counted tasks (repeat
// counts 5 and 3) would run once and the second run's transfers would never
// start. The core is marked air.reruns and its channel's tasks are unrolled
// into one circular chain, in program order: phase 0's buffer six times,
// then phase 1's four times, back to the start.

// CHECK: aie.device
// CHECK:       aie.mem(%{{.*}})
// CHECK:         aie.dma_start(S2MM, 0, ^[[FIRST:bb[0-9]+]], ^{{bb[0-9]+}})
// CHECK-NOT:     repeat_count
// CHECK:       ^[[FIRST]]:
// CHECK-NEXT:    aie.use_lock
// CHECK-NEXT:    aie.dma_bd(%[[P0:.*]] : memref<32x32xbf16, 2> offset = 0 len = 1024)
// CHECK-COUNT-5: aie.dma_bd(%[[P0]] : memref<32x32xbf16, 2> offset = 0 len = 1024)
// CHECK-COUNT-3: aie.dma_bd(%[[P1:.*]] : memref<32x32xbf16, 2> offset = 0 len = 1024)
// CHECK:         aie.dma_bd(%[[P1]] : memref<32x32xbf16, 2> offset = 0 len = 1024)
// CHECK-NEXT:    aie.use_lock
// CHECK-NEXT:    aie.next_bd ^[[FIRST]]
// CHECK:       aie.core(%{{.*}})
// CHECK:       } {{{.*}}air.reruns

air.channel @channel_0 [1, 1]
func.func @rerun_phases() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %c0_s = arith.constant 0 : index
      %c2_s = arith.constant 2 : index
      %t0 = air.wait_all async
      %r = scf.for %w = %c0_s to %c2_s step %c1_0 iter_args(%t = %t0) -> (!air.async.token) {
      %2 = air.herd @herd_0 async [%t] tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) {
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
      scf.yield %2 : !air.async.token
      }
    }
  }
  return
}
