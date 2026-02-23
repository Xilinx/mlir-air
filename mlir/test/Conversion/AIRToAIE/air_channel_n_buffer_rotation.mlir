//===- air_channel_n_buffer_rotation.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// 4-buffer rotation should generate single circular BD chain, not terminated sequences.
// This tests the N-buffer rotation detection in getRepeatCounts().

// CHECK: aie.device
// CHECK:         %[[TILE:.*]] = aie.tile(2, 3)
// CHECK:         %[[BUF3:.*]] = aie.buffer(%[[TILE]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[BUF2:.*]] = aie.buffer(%[[TILE]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[BUF1:.*]] = aie.buffer(%[[TILE]]) {{{.*}}} : memref<32x32xbf16, 2>
// CHECK:         %[[BUF0:.*]] = aie.buffer(%[[TILE]]) {{{.*}}} : memref<32x32xbf16, 2>

// Verify circular BD chain: bb1 -> bb2 -> bb3 -> bb4 -> bb1 (loops back)
// CHECK:    aie.mem(%[[TILE]])  {
// CHECK:           aie.dma_start(S2MM, 0, ^[[BB1:.*]], ^[[END:.*]])
// CHECK:         ^[[BB1]]:
// CHECK:           aie.dma_bd(%[[BUF3]] : memref<32x32xbf16, 2>
// CHECK:           aie.next_bd ^[[BB2:.*]]
// CHECK:         ^[[BB2]]:
// CHECK:           aie.dma_bd(%[[BUF2]] : memref<32x32xbf16, 2>
// CHECK:           aie.next_bd ^[[BB3:.*]]
// CHECK:         ^[[BB3]]:
// CHECK:           aie.dma_bd(%[[BUF1]] : memref<32x32xbf16, 2>
// CHECK:           aie.next_bd ^[[BB4:.*]]
// CHECK:         ^[[BB4]]:
// CHECK:           aie.dma_bd(%[[BUF0]] : memref<32x32xbf16, 2>
// CHECK:           aie.next_bd ^[[BB1]]
// CHECK:         ^[[END]]:
// CHECK:           aie.end
// CHECK:         }

air.channel @channel_0 [1, 1]
func.func @four_buffer_rotation() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) {
        // Allocate 4 buffers for rotation
        %async_token_0, %buf0 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %async_token_1, %buf1 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %async_token_2, %buf2 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %async_token_3, %buf3 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        // 4 channel.get operations using different buffers - forms rotation pattern
        %3 = air.channel.get async [%async_token_0] @channel_0[] (%buf0[] [] []) : (memref<32x32xbf16, 2>)
        %4 = air.channel.get async [%async_token_1, %3] @channel_0[] (%buf1[] [] []) : (memref<32x32xbf16, 2>)
        %5 = air.channel.get async [%async_token_2, %4] @channel_0[] (%buf2[] [] []) : (memref<32x32xbf16, 2>)
        %6 = air.channel.get async [%async_token_3, %5] @channel_0[] (%buf3[] [] []) : (memref<32x32xbf16, 2>)
        // Deallocations
        %async_token_d0 = air.execute [%6] {
          memref.dealloc %buf0 : memref<32x32xbf16, 2>
        }
        %async_token_d1 = air.execute [%6] {
          memref.dealloc %buf1 : memref<32x32xbf16, 2>
        }
        %async_token_d2 = air.execute [%6] {
          memref.dealloc %buf2 : memref<32x32xbf16, 2>
        }
        %async_token_d3 = air.execute [%6] {
          memref.dealloc %buf3 : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}
