//===- dma_to_channel_auto_packet.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that air-dma-to-channel auto-detects when packet switching is needed
// for multi-herd designs. When a segment has multiple herds and the total
// L3-to-L1 input channels exceed 2 per column, the pass should mark them
// as dma_packet.

// RUN: air-opt %s -air-dma-to-channel | FileCheck %s

// Two herds with 2 inputs each = 4 input channels > 2 per-column limit.
// CHECK-COUNT-4: air.channel {{.*}} {channel_type = "dma_packet"}

// Two output channels stay as default (2 outputs <= 2 per-column limit).
// The default channel_type is "dma_stream" which is not printed when it
// matches the default.

module {
  func.func @dual_herd(%arg0: memref<1024xbf16>, %arg1: memref<1024xbf16>,
                        %arg2: memref<1024xbf16>, %arg3: memref<1024xbf16>,
                        %arg4: memref<1024xbf16>, %arg5: memref<1024xbf16>) {
    air.launch () in () args(%a0=%arg0, %b0=%arg1, %a1=%arg2, %b1=%arg3,
                              %co=%arg4, %cm=%arg5)
        : memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
          memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16> {
      air.segment @seg args(%sa0=%a0, %sb0=%b0, %sa1=%a1, %sb1=%b1,
                             %sco=%co, %scm=%cm)
          : memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16>,
            memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16> {
        %c1 = arith.constant 1 : index
        air.herd @add_herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1)
            args(%ha=%sa0, %hb=%sb0, %hc=%sco)
            : memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16> {
          %buf_a = memref.alloc() : memref<1024xbf16, 2>
          %buf_b = memref.alloc() : memref<1024xbf16, 2>
          %buf_c = memref.alloc() : memref<1024xbf16, 2>
          air.dma_memcpy_nd (%buf_a[] [] [], %ha[] [] []) :
              (memref<1024xbf16, 2>, memref<1024xbf16>)
          air.dma_memcpy_nd (%buf_b[] [] [], %hb[] [] []) :
              (memref<1024xbf16, 2>, memref<1024xbf16>)
          air.dma_memcpy_nd (%hc[] [] [], %buf_c[] [] []) :
              (memref<1024xbf16>, memref<1024xbf16, 2>)
          memref.dealloc %buf_a : memref<1024xbf16, 2>
          memref.dealloc %buf_b : memref<1024xbf16, 2>
          memref.dealloc %buf_c : memref<1024xbf16, 2>
        }
        air.herd @mul_herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1)
            args(%ha=%sa1, %hb=%sb1, %hc=%scm)
            : memref<1024xbf16>, memref<1024xbf16>, memref<1024xbf16> {
          %buf_a = memref.alloc() : memref<1024xbf16, 2>
          %buf_b = memref.alloc() : memref<1024xbf16, 2>
          %buf_c = memref.alloc() : memref<1024xbf16, 2>
          air.dma_memcpy_nd (%buf_a[] [] [], %ha[] [] []) :
              (memref<1024xbf16, 2>, memref<1024xbf16>)
          air.dma_memcpy_nd (%buf_b[] [] [], %hb[] [] []) :
              (memref<1024xbf16, 2>, memref<1024xbf16>)
          air.dma_memcpy_nd (%hc[] [] [], %buf_c[] [] []) :
              (memref<1024xbf16>, memref<1024xbf16, 2>)
          memref.dealloc %buf_a : memref<1024xbf16, 2>
          memref.dealloc %buf_b : memref<1024xbf16, 2>
          memref.dealloc %buf_c : memref<1024xbf16, 2>
        }
      }
    }
    return
  }
}
