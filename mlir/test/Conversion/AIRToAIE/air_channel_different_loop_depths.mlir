//===- air_channel_different_loop_depths.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// When channel.get operations on the same channel use the SAME buffer (shared
// Q/K pattern) at different loop depths, getUniqueBDPattern deduplicates them
// to a single op. This produces a single infinitely-cycling BD — the core
// loops via while(true) and the BD keeps accepting data from the same buffer.

// CHECK: aie.device
// CHECK:         %[[TILE:.*]] = aie.tile(2, 3)
// CHECK:         %[[BUF:.*]] = aie.buffer(%[[TILE]]) {{{.*}}} : memref<32x32xbf16, 2>

// Verify single cycling BD (NOT sequential tasks):
// CHECK:    aie.mem(%[[TILE]])  {
// CHECK:           aie.dma_start(S2MM, 0, ^[[BD:.*]], ^[[END:.*]])
// CHECK:         ^[[BD]]:
// CHECK:           aie.dma_bd(%[[BUF]] : memref<32x32xbf16, 2>
// CHECK:           aie.next_bd ^[[BD]]
// CHECK:         ^[[END]]:
// CHECK:           aie.end
// CHECK:         }

air.channel @channel_0 [1, 1]
func.func @different_loop_depths() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2_h = arith.constant 2 : index
        // Single shared buffer for both Q and K (QK_shared pattern)
        %async_token_0, %buf = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        // Q: channel.get OUTSIDE the loop (once)
        %3 = air.channel.get async [%async_token_0] @channel_0[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
        // K: channel.get INSIDE scf.for loop (runs 2 times), SAME buffer
        %4 = scf.for %arg12 = %c0_h to %c2_h step %c1_h iter_args(%dep = %3) -> (!air.async.token) {
          %5 = air.channel.get async [%dep] @channel_0[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
          scf.yield %5 : !air.async.token
        }
        // Deallocation
        %async_token_d0 = air.execute [%4] {
          memref.dealloc %buf : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}
