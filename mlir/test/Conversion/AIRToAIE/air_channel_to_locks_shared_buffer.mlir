//===- air_channel_to_locks_shared_buffer.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" --split-input-file | FileCheck %s

// Two outbound channel.put ops sharing the same L1 staging buffer on the same
// DMA channel. Unlike ping-pong (where different buffers alternate), here the
// SAME buffer is reused for consecutive sends. Verifies that acquire/release
// are interleaved per-put (not batched at block start/end) to prevent the
// second put from overwriting the buffer before the DMA reads the first.

// CHECK: aie.device
// CHECK:         %[[TILE_MT:.*]] = aie.tile(2, 1)
// CHECK:         %[[TILE:.*]] = aie.tile(2, 3)

// One lock pair for the compute tile's MM2S channel (wlock init=1, rlock init=0)
// CHECK:         %[[WLOCK:.*]] = aie.lock(%[[TILE]], {{[0-9]+}}) {init = 1 : i32}
// CHECK:         %[[RLOCK:.*]] = aie.lock(%[[TILE]], {{[0-9]+}}) {init = 0 : i32}

// One shared buffer
// CHECK:         %[[BUF:.*]] = aie.buffer(%[[TILE]]) {{{.*}}} : memref<32x32xbf16, 2>

// DMA program: single BD using the shared buffer and lock pair
// CHECK:    aie.mem(%[[TILE]])  {
// CHECK:           aie.dma_start(MM2S, 0, ^bb1, ^bb2)
// CHECK:         ^bb1:
// CHECK:           aie.use_lock(%[[RLOCK]], AcquireGreaterEqual, 1)
// CHECK:           aie.dma_bd(%[[BUF]] : memref<32x32xbf16, 2>, 0, 1024)
// CHECK:           aie.use_lock(%[[WLOCK]], Release, 1)
// CHECK:           aie.next_bd ^bb1
// CHECK:         ^bb2:
// CHECK:           aie.end
// CHECK:         }

// Core body: interleaved acquire(wlock)/release(rlock) per put, NOT batched
// CHECK:    aie.core(%[[TILE]])  {
// CHECK:           aie.use_lock(%[[WLOCK]], AcquireGreaterEqual, 1)
// CHECK-NEXT:      aie.use_lock(%[[RLOCK]], Release, 1)
// CHECK-NEXT:      aie.use_lock(%[[WLOCK]], AcquireGreaterEqual, 1)
// CHECK-NEXT:      aie.use_lock(%[[RLOCK]], Release, 1)
// CHECK:           aie.end
// CHECK:         }

air.channel @channel_0 [1, 1]
func.func @shared_buffer_sequential_puts() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      // L2 destination buffer
      %async_token_0, %l2_buf = air.execute -> (memref<32x32xbf16, 1>) {
        %alloc = memref.alloc() : memref<32x32xbf16, 1>
        air.execute_terminator %alloc : memref<32x32xbf16, 1>
      }
      // Segment-level consumer
      %3 = air.channel.get async @channel_0[] (%l2_buf[] [] []) : (memref<32x32xbf16, 1>)
      // Herd with shared L1 buffer
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) {
        %c0 = arith.constant 0 : index
        %async_token_2, %buf = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        // Two consecutive puts from the same buffer on the same channel.
        // Without the fix, both acquires would be placed at block start,
        // causing the second acquire to deadlock (wlock consumed by first).
        %tok_1 = air.channel.put async [%async_token_2] @channel_0[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
        %tok_2 = air.channel.put async [%tok_1] @channel_0[] (%buf[] [] []) : (memref<32x32xbf16, 2>)
        %async_token_3 = air.execute [%tok_2] {
          memref.dealloc %buf : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}
