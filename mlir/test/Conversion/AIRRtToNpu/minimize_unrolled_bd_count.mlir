//===- minimize_unrolled_bd_count.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file %s | FileCheck %s

// tileIllegalWrapDim splits ONE oversized dim into two factors and hands
// everything past the 4-dim shim limit to the affine.for unroller, which costs
// one BD per iteration. Because the split never reconsiders the dims NEXT to
// the one it cut, it can leave a partition that is legal but needlessly fine.
//
// This is Gemma-3-4B's GeGLU at padded prefill 4096 (4096x10240 bf16, 8-wide
// herd). The airrt descriptor is [512, 2, 8, 640] x [81920, 40960, 640, 1].
// Splitting dim 0 alone gives [16, 32, 2, 8, 640], i.e. SIXTEEN BDs each
// covering only 32*2 = 64 of the 1024-step outer walk -- a shim tile's entire
// 16-BD budget on one channel, so the design failed to compile.
//
// Dims 0 and 1 are adjacent in the iteration space (81920 == 40960 * 2), so
// they describe one 1024-step extent walked at stride 40960 and can be
// re-partitioned freely. The pair that maximizes per-BD volume is 32 x 16:
// 16 is the largest inner factor keeping the outer stride under the 1 MiB
// ceiling (40960 * 16 = 655360, where 40960 * 32 would overflow it), and 32 is
// the largest outer factor inside the 6-bit position-0 wrap bound. That covers
// 512 of the 1024 steps, so TWO BDs replace sixteen.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.dma_bd(%arg0 : memref<4096x10240xbf16> offset = 0 len = 81920 sizes = [32, 16, 8, 640] strides = [655360, 40960, 640, 1])
// CHECK: aie.dma_bd(%arg0 : memref<4096x10240xbf16> offset = 20971520 len = 81920 sizes = [32, 16, 8, 640] strides = [655360, 40960, 640, 1])
// CHECK-NOT: aie.dma_bd

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "forward_0"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<4096x10240xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c8_i64 = arith.constant 8 : i64
    %c512_i64 = arith.constant 512 : i64
    %c640_i64 = arith.constant 640 : i64
    %c40960_i64 = arith.constant 40960 : i64
    %c81920_i64 = arith.constant 81920 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    affine.for %arg1 = 0 to 1 {
      %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c512_i64, %c2_i64, %c8_i64, %c640_i64], [%c81920_i64, %c40960_i64, %c640_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<4096x10240xbf16>) : !airrt.event
    }
    return
  }
}

// -----

// Non-mergeable neighbour: dim 0 stride 4096 != dim 1 stride 64 * wrap 8, so
// there is no run to re-partition and the plain one-dim split stands. 128
// splits into 4 x 32 and the four BDs are emitted unchanged. This is the
// dma_memcpy_split.mlir shape, pinned here so the re-partition cannot start
// firing on it.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.dma_bd(%arg0 : memref<128x8x8x64xbf16> offset = 0 len = 1024 sizes = [32, 8, 8, 16] strides = [4096, 64, 512, 1])
// CHECK: aie.dma_bd(%arg0 : memref<128x8x8x64xbf16> offset = 131072
// CHECK: aie.dma_bd(%arg0 : memref<128x8x8x64xbf16> offset = 262144
// CHECK: aie.dma_bd(%arg0 : memref<128x8x8x64xbf16> offset = 393216

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "forward_0"}
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<128x8x8x64xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c64_i64 = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %c512_i64 = arith.constant 512 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    affine.for %arg1 = 0 to 1 {
      %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<128x8x8x64xbf16>) : !airrt.event
    }
    return
  }
}
