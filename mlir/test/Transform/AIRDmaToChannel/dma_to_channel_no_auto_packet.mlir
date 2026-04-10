//===- dma_to_channel_no_auto_packet.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Negative tests for auto-packet-switching detection in air-dma-to-channel.
// In all cases below, no channels should be upgraded to dma_packet.

// RUN: air-opt %s -air-dma-to-channel -split-input-file | FileCheck %s

// None of these cases should produce dma_packet channels.
// CHECK-NOT: channel_type = "dma_packet"

// Test 1: Two 1x1 herds with 1 input each = 2 input channels.
// Capacity = 2 channels/col * 1 col = 2. 2 <= 2 => no upgrade.
module {
  func.func @dual_herd_at_limit(%arg0: memref<1024xbf16>,
      %arg1: memref<1024xbf16>,
      %arg2: memref<1024xbf16>, %arg3: memref<1024xbf16>) {
    air.launch () in () args(%a0=%arg0, %a1=%arg1, %co=%arg2, %cm=%arg3)
        : memref<1024xbf16>, memref<1024xbf16>,
          memref<1024xbf16>, memref<1024xbf16> {
      air.segment @seg args(%sa0=%a0, %sa1=%a1, %sco=%co, %scm=%cm)
          : memref<1024xbf16>, memref<1024xbf16>,
            memref<1024xbf16>, memref<1024xbf16> {
        %c1 = arith.constant 1 : index
        air.herd @add_herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1)
            args(%ha=%sa0, %hc=%sco)
            : memref<1024xbf16>, memref<1024xbf16> {
          %buf_a = memref.alloc() : memref<1024xbf16, 2>
          %buf_c = memref.alloc() : memref<1024xbf16, 2>
          air.dma_memcpy_nd (%buf_a[] [] [], %ha[] [] []) :
              (memref<1024xbf16, 2>, memref<1024xbf16>)
          air.dma_memcpy_nd (%hc[] [] [], %buf_c[] [] []) :
              (memref<1024xbf16>, memref<1024xbf16, 2>)
          memref.dealloc %buf_a : memref<1024xbf16, 2>
          memref.dealloc %buf_c : memref<1024xbf16, 2>
        }
        air.herd @mul_herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1)
            args(%ha=%sa1, %hc=%scm)
            : memref<1024xbf16>, memref<1024xbf16> {
          %buf_a = memref.alloc() : memref<1024xbf16, 2>
          %buf_c = memref.alloc() : memref<1024xbf16, 2>
          air.dma_memcpy_nd (%buf_a[] [] [], %ha[] [] []) :
              (memref<1024xbf16, 2>, memref<1024xbf16>)
          air.dma_memcpy_nd (%hc[] [] [], %buf_c[] [] []) :
              (memref<1024xbf16>, memref<1024xbf16, 2>)
          memref.dealloc %buf_a : memref<1024xbf16, 2>
          memref.dealloc %buf_c : memref<1024xbf16, 2>
        }
      }
    }
    return
  }
}

// -----

// Test 2: Single-herd segment with 3 inputs. Single herd => early return,
// no auto-detection applied regardless of channel count.
module {
  func.func @single_herd(%arg0: memref<1024xbf16>,
      %arg1: memref<1024xbf16>, %arg2: memref<1024xbf16>,
      %arg3: memref<1024xbf16>) {
    air.launch () in () args(%a0=%arg0, %b0=%arg1, %d0=%arg2, %co=%arg3)
        : memref<1024xbf16>, memref<1024xbf16>,
          memref<1024xbf16>, memref<1024xbf16> {
      air.segment @seg args(%sa0=%a0, %sb0=%b0, %sd0=%d0, %sco=%co)
          : memref<1024xbf16>, memref<1024xbf16>,
            memref<1024xbf16>, memref<1024xbf16> {
        %c1 = arith.constant 1 : index
        air.herd @only_herd tile (%tx, %ty) in (%sx=%c1, %sy=%c1)
            args(%ha=%sa0, %hb=%sb0, %hd=%sd0, %hc=%sco)
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
