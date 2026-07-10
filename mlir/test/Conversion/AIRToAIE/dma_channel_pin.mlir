//===- dma_channel_pin.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// Compute-tile DMA channel pin. A channel decl carrying `air.tile_dma_channel =
// N` forces its compute-tile flow onto physical DMA channel N. A single S2MM
// flow would otherwise take channel 0; the pin moves it to channel 1. Used to
// keep two flows on the same tile on fixed, distinct physical channels when
// their routes would collide.

// CHECK: aie.device
// CHECK: %[[TILE:.*]] = aie.tile(2, 3)
// CHECK: aie.mem(%[[TILE]])
// CHECK: aie.dma_start(S2MM, 1
air.channel @pinnedIn [1, 1] {air.tile_dma_channel = 1 : i32}
func.func @tile_dma_channel_pin() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) {
        %async_token_0, %buf0 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        %g = air.channel.get async [%async_token_0] @pinnedIn[] (%buf0[] [] []) : (memref<32x32xbf16, 2>)
        %async_token_d0 = air.execute [%g] {
          memref.dealloc %buf0 : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}
