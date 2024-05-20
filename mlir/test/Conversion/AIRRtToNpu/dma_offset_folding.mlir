//===- dma_offset_folding.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-----------------------------------------------------------------------------===//


// RUN: air-opt -airrt-to-npu --split-input-file %s | FileCheck %s

// 
//Test correctness of generated offsets, wraps and strides
//
//
// CHECK-LABEL: aie.device(npu1_4col)
// CHECK: aie.shim_dma_allocation @airMemcpyId19(S2MM, 0, 0)
// CHECK: memref.global "public" @airMemcpyId19 : memref<128x128xf32, 1>
// CHECK: aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId4 : memref<128x128xbf16, 1>
// CHECK: aie.shim_dma_allocation @airMemcpyId5(MM2S, 1, 0)
// CHECK: memref.global "public" @airMemcpyId5 : memref<16x8x8x16xbf16, 1>


// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 0][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 8][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 128][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 16][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 256][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 24][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 0, 384][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 8192][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 128, 0][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 8192][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 8][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 128, 128][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 8192][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 16][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 128, 256][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 8192][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 24][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 128, 384][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 16384][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 256, 0][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 16384][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 8][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 256, 128][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 16384][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 16][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 256, 256][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 16384][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 24][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 256, 384][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 24576][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 0][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 384, 0][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 24576][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 8][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 384, 128][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 24576][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 16][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 384, 256][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 24576][1, 4, 128, 16][0, 16, 64, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg1[0, 0, 0, 24][16, 8, 8, 8][2048, 32, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<32768xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg2[0, 0, 384, 384][1, 1, 128, 128][0, 0, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId19} : memref<512x512xf32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}

module {
  aie.device(npu1_4col) {
    aie.shim_dma_allocation @airMemcpyId19(S2MM, 0, 0)
    memref.global "public" @airMemcpyId19 : memref<128x128xf32, 1>
    aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
    memref.global "public" @airMemcpyId4 : memref<128x128xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId5(MM2S, 1, 0)
    memref.global "public" @airMemcpyId5 : memref<16x8x8x16xbf16, 1>
  } {sym_name = "forward_0"}
  airrt.module_metadata{
    airrt.segment_metadata attributes {sym_name = "forward_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd_0"}
    }
  }
  func.func @forward(%arg0: memref<512x128xbf16>, %arg1: memref<16x8x8x64xbf16>, %arg2: memref<512x512xf32>) -> memref<512x512xf32> {
    %c384_i64 = arith.constant 384 : i64
    %c48_i64 = arith.constant 48 : i64
    %c3_i64 = arith.constant 3 : i64
    %c256_i64 = arith.constant 256 : i64
    %c2_i64 = arith.constant 2 : i64
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c512_i64 = arith.constant 512 : i64
    %c64_i64 = arith.constant 64 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c4_i64 = arith.constant 4 : i64
    %c1_i64 = arith.constant 1 : i64
    %c128_i64 = arith.constant 128 : i64
    %c32_i64 = arith.constant 32 : i64
    %c0_i64 = arith.constant 0 : i64
    %c19_i32 = arith.constant 19 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "forward_0" : i64
    %0 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c19_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    %p_0 = airrt.segment_load "forward_0" : i64
    %3 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c1_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c1_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c16_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %5 = airrt.dma_memcpy_nd(%c19_i32, %c0_i64, %c1_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c128_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    %p_1 = airrt.segment_load "forward_0" : i64
    %6 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c2_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %7 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c2_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c32_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %8 = airrt.dma_memcpy_nd(%c19_i32, %c0_i64, %c2_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c256_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    %p_2 = airrt.segment_load "forward_0" : i64
    %9 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c3_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %10 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c3_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c48_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %11 = airrt.dma_memcpy_nd(%c19_i32, %c0_i64, %c3_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c384_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    %p_3 = airrt.segment_load "forward_0" : i64
    %12 = airrt.dma_memcpy_nd(%c4_i32, %c1_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c128_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %13 = airrt.dma_memcpy_nd(%c5_i32, %c1_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %14 = airrt.dma_memcpy_nd(%c19_i32, %c1_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c128_i64, %c0_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    %p_4 = airrt.segment_load "forward_0" : i64
    %15 = airrt.dma_memcpy_nd(%c4_i32, %c1_i64, %c1_i64, %arg0[%c0_i64, %c0_i64, %c128_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %16 = airrt.dma_memcpy_nd(%c5_i32, %c1_i64, %c1_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c16_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %17 = airrt.dma_memcpy_nd(%c19_i32, %c1_i64, %c1_i64, %arg2[%c0_i64, %c0_i64, %c128_i64, %c128_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    %p_5 = airrt.segment_load "forward_0" : i64
    %18 = airrt.dma_memcpy_nd(%c4_i32, %c1_i64, %c2_i64, %arg0[%c0_i64, %c0_i64, %c128_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %19 = airrt.dma_memcpy_nd(%c5_i32, %c1_i64, %c2_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c32_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %20 = airrt.dma_memcpy_nd(%c19_i32, %c1_i64, %c2_i64, %arg2[%c0_i64, %c0_i64, %c128_i64, %c256_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    %p_6 = airrt.segment_load "forward_0" : i64
    %21 = airrt.dma_memcpy_nd(%c4_i32, %c1_i64, %c3_i64, %arg0[%c0_i64, %c0_i64, %c128_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %22 = airrt.dma_memcpy_nd(%c5_i32, %c1_i64, %c3_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c48_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %23 = airrt.dma_memcpy_nd(%c19_i32, %c1_i64, %c3_i64, %arg2[%c0_i64, %c0_i64, %c128_i64, %c384_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    %p_7 = airrt.segment_load "forward_0" : i64
    %24 = airrt.dma_memcpy_nd(%c4_i32, %c2_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %25 = airrt.dma_memcpy_nd(%c5_i32, %c2_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %26 = airrt.dma_memcpy_nd(%c19_i32, %c2_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    %p_8 = airrt.segment_load "forward_0" : i64
    %27 = airrt.dma_memcpy_nd(%c4_i32, %c2_i64, %c1_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %28 = airrt.dma_memcpy_nd(%c5_i32, %c2_i64, %c1_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c16_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %29 = airrt.dma_memcpy_nd(%c19_i32, %c2_i64, %c1_i64, %arg2[%c0_i64, %c0_i64, %c256_i64, %c128_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    %p_9 = airrt.segment_load "forward_0" : i64
    %30 = airrt.dma_memcpy_nd(%c4_i32, %c2_i64, %c2_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %31 = airrt.dma_memcpy_nd(%c5_i32, %c2_i64, %c2_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c32_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %32 = airrt.dma_memcpy_nd(%c19_i32, %c2_i64, %c2_i64, %arg2[%c0_i64, %c0_i64, %c256_i64, %c256_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    %p_10 = airrt.segment_load "forward_0" : i64
    %33 = airrt.dma_memcpy_nd(%c4_i32, %c2_i64, %c3_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %34 = airrt.dma_memcpy_nd(%c5_i32, %c2_i64, %c3_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c48_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %35 = airrt.dma_memcpy_nd(%c19_i32, %c2_i64, %c3_i64, %arg2[%c0_i64, %c0_i64, %c256_i64, %c384_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    %p_11 = airrt.segment_load "forward_0" : i64
    %36 = airrt.dma_memcpy_nd(%c4_i32, %c3_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c384_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %37 = airrt.dma_memcpy_nd(%c5_i32, %c3_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %38 = airrt.dma_memcpy_nd(%c19_i32, %c3_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c384_i64, %c0_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    %p_12 = airrt.segment_load "forward_0" : i64
    %39 = airrt.dma_memcpy_nd(%c4_i32, %c3_i64, %c1_i64, %arg0[%c0_i64, %c0_i64, %c384_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %40 = airrt.dma_memcpy_nd(%c5_i32, %c3_i64, %c1_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c16_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %41 = airrt.dma_memcpy_nd(%c19_i32, %c3_i64, %c1_i64, %arg2[%c0_i64, %c0_i64, %c384_i64, %c128_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    %p_13 = airrt.segment_load "forward_0" : i64
    %42 = airrt.dma_memcpy_nd(%c4_i32, %c3_i64, %c2_i64, %arg0[%c0_i64, %c0_i64, %c384_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %43 = airrt.dma_memcpy_nd(%c5_i32, %c3_i64, %c2_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c32_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %44 = airrt.dma_memcpy_nd(%c19_i32, %c3_i64, %c2_i64, %arg2[%c0_i64, %c0_i64, %c384_i64, %c256_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    %p_14 = airrt.segment_load "forward_0" : i64
    %45 = airrt.dma_memcpy_nd(%c4_i32, %c3_i64, %c3_i64, %arg0[%c0_i64, %c0_i64, %c384_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %46 = airrt.dma_memcpy_nd(%c5_i32, %c3_i64, %c3_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c48_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %47 = airrt.dma_memcpy_nd(%c19_i32, %c3_i64, %c3_i64, %arg2[%c0_i64, %c0_i64, %c384_i64, %c384_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    return %arg2 : memref<512x512xf32>
  }
}
