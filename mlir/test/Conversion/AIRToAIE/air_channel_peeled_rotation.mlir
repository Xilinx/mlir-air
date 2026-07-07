//===- air_channel_peeled_rotation.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Peeled / software-pipelined N-buffer rotation: two steady-state channel.get
// sites share ONE loop (interleaving buf0/buf1 per iteration) while a peel get
// to buf2 sits OUTSIDE the loop. The peel therefore has a different static trip
// count than the steady sites -- but this is still one genuine round-robin
// rotation, not a set of time-multiplexed block consumers.
//
// detectNBufferRotation must key off the shared enclosing loop (>=2 sites share
// it), NOT per-op trip-count equality, and fold all three buffers into a SINGLE
// circular BD chain (no repeat_count; the last BD's next_bd cycles back to the
// first). This is the pattern non-tile-aligned matmul (xrt/53,54,55
// matmul_padding) produces; a trip-count-equality gate would wrongly split it
// into terminated per-trip tasks and break round-robin delivery.

// CHECK: aie.device
// CHECK-DAG:   %[[TILE:.*]] = aie.tile(2, 3)
// CHECK:       aie.mem(%[[TILE]])
// Single circular chain: three BDs, none carrying a repeat_count, cycling back.
// CHECK:         aie.dma_start(S2MM, 0, ^[[BB1:.*]], ^{{.*}})
// CHECK-NOT:     repeat_count
// CHECK:       ^[[BB1]]:
// CHECK:         aie.dma_bd(%{{.*}} : memref<32x32xbf16, 2>
// CHECK:         aie.next_bd ^[[BB2:.*]]
// CHECK:       ^[[BB2]]:
// CHECK:         aie.dma_bd(%{{.*}} : memref<32x32xbf16, 2>
// CHECK:         aie.next_bd ^[[BB3:.*]]
// CHECK:       ^[[BB3]]:
// CHECK:         aie.dma_bd(%{{.*}} : memref<32x32xbf16, 2>
// CHECK:         aie.next_bd ^[[BB1]]

air.channel @channel_0 [1, 1]
func.func @peeled_rotation() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%a, %b) in (%c=%c1, %d=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
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
        // peel: buf2 once (trip 1), outside the steady loop
        %p = air.channel.get async [%t2] @channel_0[] (%buf2[] [] []) : (memref<32x32xbf16, 2>)
        // steady: buf0, buf1 interleaved inside one shared loop (trip 8)
        %s = scf.for %i = %c0 to %c8 step %c1_h iter_args(%dep = %p) -> (!air.async.token) {
          %g0 = air.channel.get async [%dep, %t0] @channel_0[] (%buf0[] [] []) : (memref<32x32xbf16, 2>)
          %g1 = air.channel.get async [%g0, %t1] @channel_0[] (%buf1[] [] []) : (memref<32x32xbf16, 2>)
          scf.yield %g1 : !air.async.token
        }
        %d0 = air.execute [%s] { memref.dealloc %buf0 : memref<32x32xbf16, 2> }
        %d1 = air.execute [%s] { memref.dealloc %buf1 : memref<32x32xbf16, 2> }
        %d2 = air.execute [%s] { memref.dealloc %buf2 : memref<32x32xbf16, 2> }
      }
    }
  }
  return
}
