//===- dma_channel_pin_invalid.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" -verify-diagnostics -split-input-file

// An `air.tile_dma_channel` pin outside the tile's available DMA-channel range
// is a hard error, not a silent invalid allocation. A compute tile has 2 S2MM
// DMA channels, so pinning channel 9 must be rejected.

air.channel @pinnedIn [1, 1] {air.tile_dma_channel = 9 : i32}
func.func @tile_dma_channel_out_of_range() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) {
        %async_token_0, %buf0 = air.execute -> (memref<32x32xbf16, 2>) {
          %alloc = memref.alloc() : memref<32x32xbf16, 2>
          air.execute_terminator %alloc : memref<32x32xbf16, 2>
        }
        // expected-error @+1 {{air.tile_dma_channel = 9 is out of range [0, 2) for the S2MM DMA channels of tile}}
        %g = air.channel.get async [%async_token_0] @pinnedIn[] (%buf0[] [] []) : (memref<32x32xbf16, 2>)
        %async_token_d0 = air.execute [%g] {
          memref.dealloc %buf0 : memref<32x32xbf16, 2>
        }
      }
    }
  }
  return
}
