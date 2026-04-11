//===- dma_to_channel_auto_packet_single_herd.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that auto-packet detection works for single-herd segments.
// One 1x1 herd with 3 inputs = 3 input channels > per-column limit of 2.

// RUN: air-opt %s -air-dma-to-channel 2>&1 | FileCheck %s

// CHECK-COUNT-3: air.channel {{.*}} {channel_type = "dma_packet"}
// CHECK-NOT: channel_type = "dma_packet"

module {
  func.func @single_herd_overflow(%arg0: memref<1024xbf16>,
      %arg1: memref<1024xbf16>, %arg2: memref<1024xbf16>,
      %arg3: memref<1024xbf16>) {
    air.launch () in () args(%a=%arg0, %b=%arg1, %d=%arg2, %co=%arg3)
        : memref<1024xbf16>, memref<1024xbf16>,
          memref<1024xbf16>, memref<1024xbf16> {
      air.segment @seg args(%sa=%a, %sb=%b, %sd=%d, %sco=%co)
          : memref<1024xbf16>, memref<1024xbf16>,
            memref<1024xbf16>, memref<1024xbf16> {
        %c1 = arith.constant 1 : index
        air.herd @herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1)
            args(%ha=%sa, %hb=%sb, %hd=%sd, %hc=%sco)
            : memref<1024xbf16>, memref<1024xbf16>,
              memref<1024xbf16>, memref<1024xbf16> {
          %buf_a = memref.alloc() : memref<1024xbf16, 2>
          %buf_b = memref.alloc() : memref<1024xbf16, 2>
          %buf_d = memref.alloc() : memref<1024xbf16, 2>
          %buf_c = memref.alloc() : memref<1024xbf16, 2>
          air.dma_memcpy_nd (%buf_a[] [] [], %ha[] [] []) :
              (memref<1024xbf16, 2>, memref<1024xbf16>)
          air.dma_memcpy_nd (%buf_b[] [] [], %hb[] [] []) :
              (memref<1024xbf16, 2>, memref<1024xbf16>)
          air.dma_memcpy_nd (%buf_d[] [] [], %hd[] [] []) :
              (memref<1024xbf16, 2>, memref<1024xbf16>)
          air.dma_memcpy_nd (%hc[] [] [], %buf_c[] [] []) :
              (memref<1024xbf16>, memref<1024xbf16, 2>)
          memref.dealloc %buf_a : memref<1024xbf16, 2>
          memref.dealloc %buf_b : memref<1024xbf16, 2>
          memref.dealloc %buf_d : memref<1024xbf16, 2>
          memref.dealloc %buf_c : memref<1024xbf16, 2>
        }
      }
    }
    return
  }
}
