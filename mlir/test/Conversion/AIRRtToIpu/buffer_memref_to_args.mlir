//===- buffer_memref_to_args.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-ipu -canonicalize -cse --split-input-file %s | FileCheck %s

// Converts AIEX.ipu.dma_memcpy_nd src/dst memref to function params.

// CHECK-LABEL: aie.device(ipu)
// CHECK: func.func @func0(%[[VAL_0:.*]]: memref<8x16xi32>, %[[VAL_1:.*]]: memref<16x8xi32>, %[[VAL_2:.*]]: memref<8x8xi32>) {
// CHECK:   aiex.ipu.dma_memcpy_nd(0, 0, %[[VAL_0]][0, 0, 0, 0][1, 1, 8, 16][0, 0, 16]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<8x16xi32>
// CHECK:   aiex.ipu.dma_memcpy_nd(0, 0, %[[VAL_1]][0, 0, 0, 0][1, 1, 16, 8][0, 0, 8]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<16x8xi32>
// CHECK:   aiex.ipu.dma_memcpy_nd(0, 0, %[[VAL_2]][0, 0, 0, 0][1, 1, 8, 8][0, 0, 8]) {id = 2 : i64, metadata = @airMemcpyId16} : memref<8x8xi32>

#map = affine_map<()[s0] -> (s0 * 8)>
module {
  aie.device(ipu) {
    aie.shim_dma_allocation @airMemcpyId16(S2MM, 0, 0)
    memref.global "public" @airMemcpyId16 : memref<8x8xi32, 1>
    aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
    memref.global "public" @airMemcpyId4 : memref<8x16xi32, 1>
    aie.shim_dma_allocation @airMemcpyId5(MM2S, 0, 0)
    memref.global "public" @airMemcpyId5 : memref<16x8xi32, 1>
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
        %0 = affine.apply #map()[%arg4]
        %1 = affine.apply #map()[%arg3]
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

// CHECK-LABEL: aie.device(ipu)
// CHECK: func.func @func1(%[[VAL_0:.*]]: memref<8x16xi32>, %[[VAL_1:.*]]: memref<16x8xi32>, %[[VAL_2:.*]]: memref<8x8xi32>) {
// CHECK:   aiex.ipu.dma_memcpy_nd(0, 0, %[[VAL_0]][0, 0, 0, 0][1, 1, 8, 16][0, 0, 16]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<8x16xi32>
// CHECK:   aiex.ipu.dma_memcpy_nd(0, 0, %[[VAL_1]][0, 0, 0, 0][1, 1, 16, 8][0, 0, 8]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<16x8xi32>
// CHECK:   aiex.ipu.dma_memcpy_nd(0, 0, %[[VAL_2]][0, 0, 0, 0][1, 1, 8, 8][0, 0, 8]) {id = 2 : i64, metadata = @airMemcpyId16} : memref<8x8xi32>

#map = affine_map<()[s0] -> (s0 * 8)>
module {
  aie.device(ipu) {
    aie.shim_dma_allocation @airMemcpyId16(S2MM, 0, 0)
    memref.global "public" @airMemcpyId16 : memref<8x8xi32, 1>
    aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
    memref.global "public" @airMemcpyId4 : memref<8x16xi32, 1>
    aie.shim_dma_allocation @airMemcpyId5(MM2S, 0, 0)
    memref.global "public" @airMemcpyId5 : memref<16x8xi32, 1>
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
        %3 = affine.apply #map()[%arg1]
        %4 = affine.apply #map()[%arg0]
        %5 = arith.index_cast %arg0 : index to i64
        %6 = arith.index_cast %arg1 : index to i64
        %7 = arith.index_cast %4 : index to i64
        %8 = airrt.dma_memcpy_nd(%c4_i32, %5, %6, %0[%c0_i64, %c0_i64, %7, %c0_i64], [%c1_i64, %c1_i64, %c8_i64, %c16_i64], [%c0_i64, %c0_i64, %c16_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<8x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %9 = airrt.wait_all %8 : !airrt.event
        %10 = arith.index_cast %arg0 : index to i64
        %11 = arith.index_cast %arg1 : index to i64
        %12 = arith.index_cast %3 : index to i64
        %13 = airrt.dma_memcpy_nd(%c4_i32, %10, %11, %1[%c0_i64, %c0_i64, %c0_i64, %12], [%c1_i64, %c1_i64, %c16_i64, %c8_i64], [%c0_i64, %c0_i64, %c8_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %14 = affine.apply #map()[%arg0]
        %15 = affine.apply #map()[%arg1]
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
