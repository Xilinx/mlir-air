//===- air_channel_prefix_suffix_bd.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Prefix + repeating suffix pattern [Q, K, K, K, K] should collapse to a 2-BD
// circular chain [Q, K], not generate 5 separate BDs.
// This tests the prefix+suffix detection in getRepeatCounts().

// CHECK: aie.device
// CHECK:         %[[TILE:.*]] = aie.tile(2, 3)

// Verify 2-BD circular chain: bb1 -> bb2 -> bb1 (loops back)
// Without the prefix+suffix collapse, this would generate 5 BDs.
// CHECK:    aie.mem(%[[TILE]])  {
// CHECK:           aie.dma_start(S2MM, 0, ^[[BB1:.*]], ^[[END:.*]])
// CHECK:         ^[[BB1]]:
// CHECK:           aie.dma_bd({{.*}} : memref<32x32xbf16, 2>
// CHECK:           aie.next_bd ^[[BB2:.*]]
// CHECK:         ^[[BB2]]:
// CHECK:           aie.dma_bd({{.*}} : memref<32x32xbf16, 2>
// CHECK:           aie.next_bd ^[[BB1]]
// CHECK:         ^[[END]]:
// CHECK:           aie.end
// CHECK:         }

air.channel @channel_0 [1, 1]
func.func @prefix_suffix_collapse() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) {
        // Allocate Q and K buffers
        %async_token_q, %buf_q = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %async_token_k, %buf_k = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        // 1 channel.get to buf_q (prefix), then 4 channel.gets to buf_k (repeating suffix)
        // Pattern: [Q, K, K, K, K] -> should collapse to [Q, K] circular chain
        %3 = air.channel.get async [%async_token_q] @channel_0[] (%buf_q[] [] []) : (memref<32x32xbf16, 2>)
        %4 = air.channel.get async [%async_token_k, %3] @channel_0[] (%buf_k[] [] []) : (memref<32x32xbf16, 2>)
        %5 = air.channel.get async [%4] @channel_0[] (%buf_k[] [] []) : (memref<32x32xbf16, 2>)
        %6 = air.channel.get async [%5] @channel_0[] (%buf_k[] [] []) : (memref<32x32xbf16, 2>)
        %7 = air.channel.get async [%6] @channel_0[] (%buf_k[] [] []) : (memref<32x32xbf16, 2>)
        // Deallocations
        %async_token_dq = air.execute [%7] {
          memref.dealloc %buf_q : memref<32x32xbf16, 2>
        }
        %async_token_dk = air.execute [%7] {
          memref.dealloc %buf_k : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}
