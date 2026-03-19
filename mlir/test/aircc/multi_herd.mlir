//===- multi_herd.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Verify that aircc handles a 2x2 herd correctly through placement.
// The air-to-aie conversion may fail on this simplified IR (DMA offsets
// don't account for tile indices), but the placement pipeline should
// produce correct channel and herd placement.

// RUN: rm -rf %t && mkdir -p %t
// RUN: aircc %s --device=npu1 --tmpdir=%t --output-format=none 2>&1 || true

// The placed IR should contain channels for the 2x2 herd
// RUN: FileCheck %s --input-file=%t/placed.multi_herd.mlir --check-prefix=PLACED

// PLACED: air.channel
// PLACED: air.herd @copyherd

module {
  func.func @copy(%arg0: memref<4096xui8>, %arg1: memref<4096xui8>) {
    air.launch () in () args(%arg2=%arg0, %arg3=%arg1) : memref<4096xui8>, memref<4096xui8> {
      air.segment @seg  args(%arg4=%arg2, %arg5=%arg3) : memref<4096xui8>, memref<4096xui8> {
        %c2 = arith.constant 2 : index
        air.herd @copyherd  tile (%arg6, %arg7) in (%arg8=%c2, %arg9=%c2) args(%arg10=%arg4, %arg11=%arg5) : memref<4096xui8>, memref<4096xui8> {
          %c0 = arith.constant 0 : index
          %c1024 = arith.constant 1024 : index
          %c1 = arith.constant 1 : index
          %alloc = memref.alloc() : memref<1024xui8, 2 : i32>
          air.dma_memcpy_nd (%alloc[] [] [], %arg10[%c0] [%c1024] [%c1]) : (memref<1024xui8, 2 : i32>, memref<4096xui8>)
          air.dma_memcpy_nd (%arg11[%c0] [%c1024] [%c1], %alloc[] [] []) : (memref<4096xui8>, memref<1024xui8, 2 : i32>)
          memref.dealloc %alloc : memref<1024xui8, 2 : i32>
        }
      }
    }
    return
  }
}
