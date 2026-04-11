//===- dma_to_channel_auto_packet_l3_l2.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that auto-packet detection counts L3↔L2 channels (segment-level
// endpoints on L2 memrefs) toward shim DMA pressure. Three L3→L2 input
// channels exceed the per-column limit of 2.

// RUN: air-opt %s -air-dma-to-channel 2>&1 | FileCheck %s

// CHECK-COUNT-3: air.channel {{.*}} {channel_type = "dma_packet"}
// CHECK-NOT: channel_type = "dma_packet"

module {
  func.func @l3_to_l2_overflow(%arg0: memref<1024xbf16>,
      %arg1: memref<1024xbf16>, %arg2: memref<1024xbf16>,
      %arg3: memref<1024xbf16>) {
    air.launch () in () args(%a=%arg0, %b=%arg1, %d=%arg2, %co=%arg3)
        : memref<1024xbf16>, memref<1024xbf16>,
          memref<1024xbf16>, memref<1024xbf16> {
      air.segment @seg args(%sa=%a, %sb=%b, %sd=%d, %sco=%co)
          : memref<1024xbf16>, memref<1024xbf16>,
            memref<1024xbf16>, memref<1024xbf16> {
        // L2 buffers at segment level
        %l2_a = memref.alloc() : memref<1024xbf16, 1>
        %l2_b = memref.alloc() : memref<1024xbf16, 1>
        %l2_d = memref.alloc() : memref<1024xbf16, 1>
        %l2_c = memref.alloc() : memref<1024xbf16, 1>
        // L3→L2 DMAs (3 inputs, 1 output)
        air.dma_memcpy_nd (%l2_a[] [] [], %sa[] [] []) :
            (memref<1024xbf16, 1>, memref<1024xbf16>)
        air.dma_memcpy_nd (%l2_b[] [] [], %sb[] [] []) :
            (memref<1024xbf16, 1>, memref<1024xbf16>)
        air.dma_memcpy_nd (%l2_d[] [] [], %sd[] [] []) :
            (memref<1024xbf16, 1>, memref<1024xbf16>)
        air.dma_memcpy_nd (%sco[] [] [], %l2_c[] [] []) :
            (memref<1024xbf16>, memref<1024xbf16, 1>)
        memref.dealloc %l2_a : memref<1024xbf16, 1>
        memref.dealloc %l2_b : memref<1024xbf16, 1>
        memref.dealloc %l2_d : memref<1024xbf16, 1>
        memref.dealloc %l2_c : memref<1024xbf16, 1>
      }
    }
    return
  }
}
