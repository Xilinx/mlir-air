//===- split_launch_for_padding_dma_memtile.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-split-launch-for-padding='pad-location=memtile' | FileCheck %s

// Tests memtile padding with DMA ops (pad-location=memtile + dma_memcpy_nd).
// M=300, M_TILE=64 -> launchM=5, last M-block has 300-4*64=44 rows, pad=20
// N=300, N_TILE=64 -> launchN=5, last N-block has 300-4*64=44 cols, pad=20
// Expected: 4 partitions, L2->L1 DMAs get pad_after, L3->L2 sizes reduced.

// CHECK-LABEL: func.func @dma_memtile_padding

// Interior: 4x4 launch, no padding on any DMA.
// CHECK: air.segment @dma_memtile_padding_0_interior
// CHECK: air.dma_memcpy_nd
// CHECK-NOT: pad_after

// M-boundary: L2->L1 A DMA (id=3) has pad_after, B DMA (id=4) unchanged.
// CHECK: air.segment @dma_memtile_padding_0_m_boundary
// CHECK: air.dma_memcpy_nd ({{.*}}) {id = 3 : i32, pad_after = array<i32: 20, 0>, pad_before = array<i32: 0, 0>}
// CHECK: air.dma_memcpy_nd ({{.*}}) {id = 4 : i32}
// CHECK-NOT: pad_after

// N-boundary: L2->L1 B DMA (id=4) has pad_after, A DMA (id=3) unchanged.
// CHECK: air.segment @dma_memtile_padding_0_n_boundary
// CHECK: air.dma_memcpy_nd ({{.*}}) {id = 3 : i32}
// CHECK-NOT: pad_after
// CHECK: air.dma_memcpy_nd ({{.*}}) {id = 4 : i32, pad_after = array<i32: 0, 20>, pad_before = array<i32: 0, 0>}

// Corner: both A and B DMA ops have padding.
// CHECK: air.segment @dma_memtile_padding_0_corner
// CHECK: air.dma_memcpy_nd ({{.*}}) {id = 3 : i32, pad_after = array<i32: 20, 0>, pad_before = array<i32: 0, 0>}
// CHECK: air.dma_memcpy_nd ({{.*}}) {id = 4 : i32, pad_after = array<i32: 0, 20>, pad_before = array<i32: 0, 0>}

module {
  func.func @dma_memtile_padding(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>) {
    %c1 = arith.constant 1 : index
    %c5 = arith.constant 5 : index
    air.launch (%arg9, %arg10, %arg11) in (%arg12=%c5, %arg13=%c5, %arg14=%c1) args(%arg15=%arg0, %arg16=%arg1, %arg17=%arg2) : memref<*xbf16>, memref<*xbf16>, memref<*xbf16> attributes {air.actual_sizes = array<i64: 300, 300, 1>} {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c300 = arith.constant 300 : index

      %offset_a = arith.muli %arg9, %c64 : index
      %offset_b = arith.muli %arg10, %c64 : index

      // L2 buffer allocs at launch level
      %alloc_a = memref.alloc() : memref<64x64xbf16, 1 : i32>
      %alloc_b = memref.alloc() : memref<64x64xbf16, 1 : i32>

      // L3->L2 DMA (launch level): src=L3 (arg0/arg1), dst=L2 (alloc)
      air.dma_memcpy_nd (%alloc_a[] [] [], %arg15[%offset_a, %c0] [%c64, %c64] [%c300, %c1_0]) {id = 1 : i32} : (memref<64x64xbf16, 1 : i32>, memref<*xbf16>)
      air.dma_memcpy_nd (%alloc_b[] [] [], %arg16[%c0, %offset_b] [%c64, %c64] [%c300, %c1_0]) {id = 2 : i32} : (memref<64x64xbf16, 1 : i32>, memref<*xbf16>)

      air.segment @dma_memtile_padding_0  args(%seg_a=%alloc_a, %seg_b=%alloc_b) : memref<64x64xbf16, 1 : i32>, memref<64x64xbf16, 1 : i32> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c64_s = arith.constant 64 : index

        // L1 buffer allocs
        %l1_a = memref.alloc() : memref<64x64xbf16, 2 : i32>
        %l1_b = memref.alloc() : memref<64x64xbf16, 2 : i32>

        // L2->L1 DMA (segment level): src=L2 (seg arg), dst=L1 (alloc)
        air.dma_memcpy_nd (%l1_a[] [] [], %seg_a[%c0_s, %c0_s] [%c64_s, %c64_s] [%c64_s, %c1_s]) {id = 3 : i32} : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 1 : i32>)
        air.dma_memcpy_nd (%l1_b[] [] [], %seg_b[%c0_s, %c0_s] [%c64_s, %c64_s] [%c64_s, %c1_s]) {id = 4 : i32} : (memref<64x64xbf16, 2 : i32>, memref<64x64xbf16, 1 : i32>)

        memref.dealloc %l1_a : memref<64x64xbf16, 2 : i32>
        memref.dealloc %l1_b : memref<64x64xbf16, 2 : i32>
      }

      memref.dealloc %alloc_a : memref<64x64xbf16, 1 : i32>
      memref.dealloc %alloc_b : memref<64x64xbf16, 1 : i32>
    }
    return
  }
}
