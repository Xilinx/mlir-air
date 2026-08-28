//===- dma_preserves_attrs.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// air-dependency re-instantiates air.dma_memcpy_nd with the async interface.
// The builder it uses takes only the access pattern and the padding, so every
// other attribute has to be carried across explicitly. `channel` matters most:
// air-dependency runs BEFORE air-dma-to-channel (see aircc.cpp), so dropping it
// here means a named channel never reaches the pass that reads it, and the DMA
// silently lowers onto a fresh point-to-point channel instead.

// RUN: air-opt %s -air-dependency -split-input-file | FileCheck %s

// CHECK-LABEL: func.func @preserves
// CHECK: air.dma_memcpy_nd async
// CHECK-SAME: air.front_end_note = "keep me"
// CHECK-SAME: channel = @named
// CHECK-SAME: channel_indices = array<i64: 1>
air.channel @named [4]
func.func @preserves(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%a=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %cst1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<32x32xi32, 2>
    air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @named, channel_indices = array<i64: 1>, air.front_end_note = "keep me"} : (memref<32x32xi32, 2>, memref<64x64xi32>)
    memref.dealloc %alloc : memref<32x32xi32, 2>
  }
  return
}

// -----

// Same for the cross-rank attributes, which the builder also does not take.

// CHECK-LABEL: func.func @preserves_cross_rank
// CHECK: air.dma_memcpy_nd async
// CHECK-SAME: dst_rank = 1 : i64
func.func @preserves_cross_rank() {
  %c2 = arith.constant 2 : index
  air.rank (%rx) in (%sx = %c2) {
    %dst = memref.alloc() {air.symmetric} : memref<128xf32>
    %src = memref.alloc() : memref<128xf32, 2>
    air.dma_memcpy_nd (%dst[] [] [], %src[] [] []) {dst_rank = 1 : i64}
        : (memref<128xf32>, memref<128xf32, 2>)
  }
  return
}
