//===- buffer_memref_to_args.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -canonicalize -cse --split-input-file %s | FileCheck %s

// Converts AIEX.npu.dma_memcpy_nd src/dst memref to function params.

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK: aie.runtime_sequence @func0(%[[VAL_0:.*]]: memref<8x16xi32>, %[[VAL_1:.*]]: memref<16x8xi32>, %[[VAL_2:.*]]: memref<8x8xi32>) {
// CHECK:   aiex.npu.dma_memcpy_nd(%[[VAL_0]][0, 0, 0, 0][1, 1, 8, 16][0, 0, 16, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<8x16xi32>
// CHECK:   aiex.npu.dma_memcpy_nd(%[[VAL_1]][0, 0, 0, 0][1, 1, 16, 8][0, 0, 8, 1]) {id = 1 : i64, metadata = @airMemcpyId4} : memref<16x8xi32>
// CHECK:   aiex.npu.dma_memcpy_nd(%[[VAL_2]][0, 0, 0, 0][1, 1, 8, 8][0, 0, 8, 1]) {id = 2 : i64, metadata = @airMemcpyId16} : memref<8x8xi32>

#map = affine_map<(d0)[] -> (d0 * 8)>
module {
  aie.device(npu1_1col) {
    aie.shim_dma_allocation @airMemcpyId16(S2MM, 0, 0)
    aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
    aie.shim_dma_allocation @airMemcpyId5(MM2S, 0, 0)
  } {sym_name = "segment_0"}
  func.func @func0(%arg0: memref<8x16xi32>, %arg1: memref<16x8xi32>, %arg2: memref<8x8xi32>) {
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c16_i32 = arith.constant 16 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() : memref<8x8xi32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 1 {
        %0 = affine.apply #map(%arg4)[]
        %1 = affine.apply #map(%arg3)[]
        %2 = arith.index_cast %arg3 : index to i64
        %3 = arith.index_cast %arg4 : index to i64
        %4 = arith.index_cast %1 : index to i64
        %5 = airrt.dma_memcpy_nd(%c4_i32, %2, %3, %arg0[%c0_i64, %c0_i64, %4, %c0_i64], [%c1_i64, %c1_i64, %c8_i64, %c16_i64], [%c0_i64, %c0_i64, %c16_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<8x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %6 = airrt.wait_all %5 : !airrt.event
        %7 = arith.index_cast %0 : index to i64
        %8 = airrt.dma_memcpy_nd(%c4_i32, %2, %3, %arg1[%c0_i64, %c0_i64, %c0_i64, %7], [%c1_i64, %c1_i64, %c16_i64, %c8_i64], [%c0_i64, %c0_i64, %c8_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %9 = airrt.dma_memcpy_nd(%c16_i32, %2, %3, %alloc[%c0_i64, %c0_i64, %4, %7], [%c1_i64, %c1_i64, %c8_i64, %c8_i64], [%c0_i64, %c0_i64, %c8_i64]) {metadata = @airMemcpyId16} : (i32, i64, i64, memref<8x8xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %p = airrt.segment_load "segment_0" : i64
      }
    }
    memref.copy %alloc, %arg2 : memref<8x8xi32> to memref<8x8xi32>
    return
  }
}

// -----

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK: aie.runtime_sequence @func1(%[[VAL_0:.*]]: memref<8x16xi32>, %[[VAL_1:.*]]: memref<16x8xi32>, %[[VAL_2:.*]]: memref<8x8xi32>) {
// CHECK:   aiex.npu.dma_memcpy_nd(%[[VAL_0]][0, 0, 0, 0][1, 1, 8, 16][0, 0, 16, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<8x16xi32>
// CHECK:   aiex.npu.dma_memcpy_nd(%[[VAL_1]][0, 0, 0, 0][1, 1, 16, 8][0, 0, 8, 1]) {id = 1 : i64, metadata = @airMemcpyId4} : memref<16x8xi32>
// CHECK:   aiex.npu.dma_memcpy_nd(%[[VAL_2]][0, 0, 0, 0][1, 1, 8, 8][0, 0, 8, 1]) {id = 2 : i64, metadata = @airMemcpyId16} : memref<8x8xi32>

#map = affine_map<(d0)[] -> (d0 * 8)>
module {
  aie.device(npu1_1col) {
    aie.shim_dma_allocation @airMemcpyId16(S2MM, 0, 0)
    aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
    aie.shim_dma_allocation @airMemcpyId5(MM2S, 0, 0)
  } {sym_name = "segment_0"}
  func.func @func1() {
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c16_i32 = arith.constant 16 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<8x16xi32>
    memref.assume_alignment %0, 64 : memref<8x16xi32>
    %1 = memref.alloc() : memref<16x8xi32>
    memref.assume_alignment %1, 64 : memref<16x8xi32>
    %2 = memref.alloc() : memref<8x8xi32>
    memref.assume_alignment %2, 64 : memref<8x8xi32>
    affine.for %arg0 = 0 to 1 {
      affine.for %arg1 = 0 to 1 {
        %3 = affine.apply #map(%arg1)[]
        %4 = affine.apply #map(%arg0)[]
        %5 = arith.index_cast %arg0 : index to i64
        %6 = arith.index_cast %arg1 : index to i64
        %7 = arith.index_cast %4 : index to i64
        %8 = airrt.dma_memcpy_nd(%c4_i32, %5, %6, %0[%c0_i64, %c0_i64, %7, %c0_i64], [%c1_i64, %c1_i64, %c8_i64, %c16_i64], [%c0_i64, %c0_i64, %c16_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<8x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %9 = airrt.wait_all %8 : !airrt.event
        %10 = arith.index_cast %arg0 : index to i64
        %11 = arith.index_cast %arg1 : index to i64
        %12 = arith.index_cast %3 : index to i64
        %13 = airrt.dma_memcpy_nd(%c4_i32, %10, %11, %1[%c0_i64, %c0_i64, %c0_i64, %12], [%c1_i64, %c1_i64, %c16_i64, %c8_i64], [%c0_i64, %c0_i64, %c8_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %14 = affine.apply #map(%arg0)[]
        %15 = affine.apply #map(%arg1)[]
        %16 = arith.index_cast %arg0 : index to i64
        %17 = arith.index_cast %arg1 : index to i64
        %18 = arith.index_cast %14 : index to i64
        %19 = arith.index_cast %15 : index to i64
        %20 = airrt.dma_memcpy_nd(%c16_i32, %16, %17, %2[%c0_i64, %c0_i64, %18, %19], [%c1_i64, %c1_i64, %c8_i64, %c8_i64], [%c0_i64, %c0_i64, %c8_i64]) {metadata = @airMemcpyId16} : (i32, i64, i64, memref<8x8xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %p = airrt.segment_load "segment_0" : i64
      }
    }
    return
  }
}

// -----

// Bf16 datatype support.

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK: aie.runtime_sequence @func2(%[[VAL_0:.*]]: memref<2048x2048xbf16>, %[[VAL_1:.*]]: memref<2048x2048xbf16>, %[[VAL_2:.*]]: memref<2048x2048xbf16>) {
// CHECK:   aiex.npu.dma_memcpy_nd(%[[VAL_0]][0, 0, 0, 0][1, 8, 128, 256][0, 256, 2048, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<2048x2048xbf16>
// CHECK:   aiex.npu.dma_memcpy_nd(%[[VAL_1]][0, 0, 0, 0][1, 4, 512, 128][0, 1048576, 2048, 1]) {id = 1 : i64, metadata = @airMemcpyId7} : memref<2048x2048xbf16>
// CHECK:   aiex.npu.dma_memcpy_nd(%[[VAL_2]][0, 0, 0, 0][1, 1, 128, 128][0, 0, 2048, 1]) {id = 2 : i64, metadata = @airMemcpyId26} : memref<2048x2048xbf16>

module {
  aie.device(npu1_1col) {
    aie.shim_dma_allocation @airMemcpyId26(S2MM, 0, 0)
    aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
    aie.shim_dma_allocation @airMemcpyId10(MM2S, 0, 0)
    aie.shim_dma_allocation @airMemcpyId7(MM2S, 1, 0)
    aie.shim_dma_allocation @airMemcpyId13(MM2S, 1, 0)
  } {sym_name = "segment_0"}
  func.func @func2() {
    %c128_i64 = arith.constant 128 : i64
    %c8_i64 = arith.constant 8 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c256_i64 = arith.constant 256 : i64
    %c26_i32 = arith.constant 26 : i32
    %c7_i32 = arith.constant 7 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c0 = arith.constant 0 : index
    %0 = memref.alloc() : memref<2048x2048xbf16>
    %1 = airrt.wait_all : !airrt.event
    airrt.wait_all %1
    memref.assume_alignment %0, 64 : memref<2048x2048xbf16>
    %2 = airrt.wait_all : !airrt.event
    %3 = memref.alloc() : memref<2048x2048xbf16>
    %4 = airrt.wait_all : !airrt.event
    airrt.wait_all %4
    memref.assume_alignment %3, 64 : memref<2048x2048xbf16>
    %5 = airrt.wait_all : !airrt.event
    %6 = memref.alloc() : memref<2048x2048xbf16>
    %7 = airrt.wait_all : !airrt.event
    airrt.wait_all %7
    memref.assume_alignment %6, 64 : memref<2048x2048xbf16>
    %8 = airrt.wait_all : !airrt.event
    %9 = airrt.wait_all %8, %5, %2 : !airrt.event
    affine.for %arg0 = 0 to 1 {
      affine.for %arg1 = 0 to 1 {
        %10 = affine.apply affine_map<(d0)[] -> (d0 * 128)>(%arg0)[]
        %11 = airrt.wait_all : !airrt.event
        %12 = airrt.wait_all %11 : !airrt.event
        %13 = arith.index_cast %arg0 : index to i64
        %14 = arith.index_cast %arg1 : index to i64
        %15 = arith.index_cast %10 : index to i64
        %16 = airrt.dma_memcpy_nd(%c4_i32, %13, %14, %0[%c0_i64, %c0_i64, %15, %c0_i64], [%c1_i64, %c8_i64, %c128_i64, %c256_i64], [%c0_i64, %c256_i64, %c2048_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<2048x2048xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %17 = affine.apply affine_map<(d0)[] -> (d0 * 128)>(%arg1)[]
        %18 = airrt.wait_all : !airrt.event
        %19 = airrt.wait_all %18 : !airrt.event
        %20 = arith.index_cast %arg0 : index to i64
        %21 = arith.index_cast %arg1 : index to i64
        %22 = arith.index_cast %17 : index to i64
        %23 = airrt.dma_memcpy_nd(%c7_i32, %20, %21, %3[%c0_i64, %c0_i64, %c0_i64, %22], [%c1_i64, %c1_i64, %c2048_i64, %c128_i64], [%c0_i64, %c0_i64, %c2048_i64]) {metadata = @airMemcpyId13} : (i32, i64, i64, memref<2048x2048xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %24 = affine.apply affine_map<(d0)[] -> (d0 * 128)>(%arg0)[]
        %25 = airrt.wait_all : !airrt.event
        %26 = affine.apply affine_map<(d0)[] -> (d0 * 128)>(%arg1)[]
        %27 = airrt.wait_all : !airrt.event
        %28 = airrt.wait_all %27, %25 : !airrt.event
        %29 = arith.index_cast %arg0 : index to i64
        %30 = arith.index_cast %arg1 : index to i64
        %31 = arith.index_cast %24 : index to i64
        %32 = arith.index_cast %26 : index to i64
        %33 = airrt.dma_memcpy_nd(%c26_i32, %29, %30, %6[%c0_i64, %c0_i64, %31, %32], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c2048_i64]) {metadata = @airMemcpyId26} : (i32, i64, i64, memref<2048x2048xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %p = airrt.segment_load "segment_0" : i64
        %34 = airrt.wait_all : !airrt.event
      }
    }
    return
  }
}
