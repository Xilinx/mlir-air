//===- airrt_to_ipu.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-ipu -canonicalize -cse --split-input-file %s | FileCheck %s

// Synchronous airrt.dma_memcpy_nd

// CHECK-LABEL: AIE.device(ipu)
// CHECK: AIE.shimDMAAllocation @airMemcpyId7(S2MM, 0, 0)
// CHECK: memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
// CHECK: AIE.shimDMAAllocation @airMemcpyId2(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
// CHECK: func.func @func0(%[[VAL_0:.*]]: memref<64xi32>, %[[VAL_1:.*]]: memref<64xi32>) {
// CHECK:   %[[CST_0:.*]] = arith.constant 0 : i32
// CHECK:   %[[CST_1:.*]] = arith.constant 1 : i32
// CHECK:   %[[CST_64:.*]] = arith.constant 64 : i32
// CHECK:   AIEX.ipu.dma_memcpy_nd(%[[CST_0]], %[[CST_0]], %[[VAL_0]][%[[CST_0]], %[[CST_0]], %[[CST_0]], %[[CST_0]]] [%[[CST_1]], %[[CST_1]], %[[CST_1]], %[[CST_64]]] [%[[CST_0]], %[[CST_0]], %[[CST_0]]]) {id = 1 : i32, metadata = @airMemcpyId2} : (i32, i32, memref<64xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
// CHECK:   AIEX.ipu.dma_memcpy_nd(%[[CST_0]], %[[CST_0]], %[[VAL_1]][%[[CST_0]], %[[CST_0]], %[[CST_0]], %[[CST_0]]] [%[[CST_1]], %[[CST_1]], %[[CST_1]], %[[CST_64]]] [%[[CST_0]], %[[CST_0]], %[[CST_0]]]) {id = 2 : i32, metadata = @airMemcpyId7} : (i32, i32, memref<64xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
// CHECK:   AIEX.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK:   return
// CHECK: }
// CHECK: {sym_name = "segment0"}

module {
  AIE.device(ipu) {
    AIE.shimDMAAllocation @airMemcpyId7(S2MM, 0, 0)
    memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
    AIE.shimDMAAllocation @airMemcpyId2(MM2S, 0, 0)
    memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
  } {sym_name = "segment0"}
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_3 [1, 1]
  func.func @func0(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c64_i64 = arith.constant 64 : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
    %p = airrt.segment_load "segment0" : i64
    %c7_i32 = arith.constant 7 : i32
    airrt.dma_memcpy_nd(%c7_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
    return
  }
}

// -----

// Asynchronous airrt.dma_memcpy_nd

// CHECK-LABEL: AIE.device(ipu) {
// CHECK: AIE.shimDMAAllocation @airMemcpyId7(S2MM, 0, 0)
// CHECK: memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
// CHECK: AIE.shimDMAAllocation @airMemcpyId2(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
// CHECK: func.func @func1(%[[VAL_0:.*]]: memref<64xi32>, %[[VAL_1:.*]]: memref<64xi32>) {
// CHECK:   %[[CST_0:.*]] = arith.constant 0 : i32
// CHECK:   %[[CST_1:.*]] = arith.constant 1 : i32
// CHECK:   %[[CST_64:.*]] = arith.constant 64 : i32
// CHECK:   AIEX.ipu.dma_memcpy_nd(%[[CST_0]], %[[CST_0]], %[[VAL_0]][%[[CST_0]], %[[CST_0]], %[[CST_0]], %[[CST_0]]] [%[[CST_1]], %[[CST_1]], %[[CST_1]], %[[CST_64]]] [%[[CST_0]], %[[CST_0]], %[[CST_0]]]) {id = 1 : i32, metadata = @airMemcpyId2} : (i32, i32, memref<64xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
// CHECK:   AIEX.ipu.dma_memcpy_nd(%[[CST_0]], %[[CST_0]], %[[VAL_1]][%[[CST_0]], %[[CST_0]], %[[CST_0]], %[[CST_0]]] [%[[CST_1]], %[[CST_1]], %[[CST_1]], %[[CST_64]]] [%[[CST_0]], %[[CST_0]], %[[CST_0]]]) {id = 2 : i32, metadata = @airMemcpyId7} : (i32, i32, memref<64xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
// CHECK:   AIEX.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK:   return
// CHECK: }
// CHECK: } {sym_name = "segment0"}

module {
  AIE.device(ipu) {
    AIE.shimDMAAllocation @airMemcpyId7(S2MM, 0, 0)
    memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
    AIE.shimDMAAllocation @airMemcpyId2(MM2S, 0, 0)
    memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
  } {sym_name = "segment0"}
  airrt.module_metadata{
  }
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_3 [1, 1]
  func.func @func1(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c64_i64 = arith.constant 64 : i64
    %0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %p = airrt.segment_load "segment0" : i64
    %1 = airrt.wait_all : !airrt.event
    %c7_i32 = arith.constant 7 : i32
    %2 = airrt.dma_memcpy_nd(%c7_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    return
  }
}

// -----

// air.launch iteration space unrolling

// CHECK-LABEL: AIE.device(ipu) {
// CHECK: AIE.shimDMAAllocation @airMemcpyId16(S2MM, 0, 0)
// CHECK: memref.global "public" @airMemcpyId16 : memref<32x32xi32, 1>
// CHECK: AIE.shimDMAAllocation @airMemcpyId5(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId5 : memref<32x32xi32, 1>
// CHECK: AIE.shimDMAAllocation @airMemcpyId6(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId6 : memref<32x32xi32, 1>
// CHECK: AIE.shimDMAAllocation @airMemcpyId7(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId7 : memref<32x32xi32, 1>
// CHECK: func.func @func2(%[[VAL_0:.*]]: memref<32x32xi32>, %[[VAL_1:.*]]: memref<32x32xi32>, %[[VAL_2:.*]]: memref<32x32xi32>) {
// CHECK:   %[[CST_0:.*]] = arith.constant 0 : i32
// CHECK:   %[[CST_1:.*]] = arith.constant 1 : i32
// CHECK:   %[[CST_32:.*]] = arith.constant 32 : i32
// CHECK:   AIEX.ipu.dma_memcpy_nd(%[[CST_0]], %[[CST_0]], %[[VAL_2]][%[[CST_0]], %[[CST_0]], %[[CST_0]], %[[CST_0]]] [%[[CST_1]], %[[CST_1]], %[[CST_32]], %[[CST_32]]] [%[[CST_0]], %[[CST_0]], %[[CST_32]]]) {id = 1 : i32, metadata = @airMemcpyId5} : (i32, i32, memref<32x32xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
// CHECK:   AIEX.ipu.dma_memcpy_nd(%[[CST_0]], %[[CST_0]], %[[VAL_0]][%[[CST_0]], %[[CST_0]], %[[CST_0]], %[[CST_0]]] [%[[CST_1]], %[[CST_1]], %[[CST_32]], %[[CST_32]]] [%[[CST_0]], %[[CST_0]], %[[CST_32]]]) {id = 2 : i32, metadata = @airMemcpyId6} : (i32, i32, memref<32x32xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
// CHECK:   AIEX.ipu.dma_memcpy_nd(%[[CST_0]], %[[CST_0]], %[[VAL_1]][%[[CST_0]], %[[CST_0]], %[[CST_0]], %[[CST_0]]] [%[[CST_1]], %[[CST_1]], %[[CST_32]], %[[CST_32]]] [%[[CST_0]], %[[CST_0]], %[[CST_32]]]) {id = 3 : i32, metadata = @airMemcpyId7} : (i32, i32, memref<32x32xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
// CHECK:   AIEX.ipu.dma_memcpy_nd(%[[CST_0]], %[[CST_0]], %[[VAL_2]][%[[CST_0]], %[[CST_0]], %[[CST_0]], %[[CST_0]]] [%[[CST_1]], %[[CST_1]], %[[CST_32]], %[[CST_32]]] [%[[CST_0]], %[[CST_0]], %[[CST_32]]]) {id = 4 : i32, metadata = @airMemcpyId16} : (i32, i32, memref<32x32xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
// CHECK:   AIEX.ipu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK:   return
// CHECK: }
// CHECK: } {sym_name = "segment_0"}

#map = affine_map<()[s0] -> (s0 * 32)>
module {
  AIE.device(ipu) {
    AIE.shimDMAAllocation @airMemcpyId16(S2MM, 0, 0)
    memref.global "public" @airMemcpyId16 : memref<32x32xi32, 1>
    AIE.shimDMAAllocation @airMemcpyId5(MM2S, 0, 0)
    memref.global "public" @airMemcpyId5 : memref<32x32xi32, 1>
    AIE.shimDMAAllocation @airMemcpyId6(MM2S, 0, 0)
    memref.global "public" @airMemcpyId6 : memref<32x32xi32, 1>
    AIE.shimDMAAllocation @airMemcpyId7(MM2S, 0, 0)
    memref.global "public" @airMemcpyId7 : memref<32x32xi32, 1>
  } {sym_name = "segment_0"}
  airrt.module_metadata{
  }
  func.func @func2(%arg0: memref<32x32xi32>, %arg1: memref<32x32xi32>, %arg2: memref<32x32xi32>) {
    %c32_i64 = arith.constant 32 : i64
    %c16_i32 = arith.constant 16 : i32
    %c7_i32 = arith.constant 7 : i32
    %c6_i32 = arith.constant 6 : i32
    %c5_i32 = arith.constant 5 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 1 {
        %0 = affine.apply #map()[%arg3]
        %1 = affine.apply #map()[%arg4]
        %2 = arith.index_cast %arg3 : index to i64
        %3 = arith.index_cast %arg4 : index to i64
        %4 = arith.index_cast %0 : index to i64
        %5 = arith.index_cast %1 : index to i64
        %6 = airrt.dma_memcpy_nd(%c5_i32, %2, %3, %arg2[%c0_i64, %c0_i64, %4, %5], [%c1_i64, %c1_i64, %c32_i64, %c32_i64], [%c0_i64, %c0_i64, %c32_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<32x32xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %7 = airrt.dma_memcpy_nd(%c6_i32, %2, %3, %arg0[%c0_i64, %c0_i64, %4, %c0_i64], [%c1_i64, %c1_i64, %c32_i64, %c32_i64], [%c0_i64, %c0_i64, %c32_i64]) {metadata = @airMemcpyId6} : (i32, i64, i64, memref<32x32xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %8 = airrt.dma_memcpy_nd(%c7_i32, %2, %3, %arg1[%c0_i64, %c0_i64, %c0_i64, %5], [%c1_i64, %c1_i64, %c32_i64, %c32_i64], [%c0_i64, %c0_i64, %c32_i64]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<32x32xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %9 = airrt.dma_memcpy_nd(%c16_i32, %2, %3, %arg2[%c0_i64, %c0_i64, %4, %5], [%c1_i64, %c1_i64, %c32_i64, %c32_i64], [%c0_i64, %c0_i64, %c32_i64]) {metadata = @airMemcpyId16} : (i32, i64, i64, memref<32x32xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %p = airrt.segment_load "segment_0" : i64
      }
    }
    return
  }
}

// -----

// air.launch iteration space unrolling 2

// CHECK-LABEL: AIE.device(ipu) {
// CHECK:  func.func @func3(%[[ARG0:.*]]: memref<8x8xi32>)
// CHECK:  %[[CST0:.*]] = arith.constant 0 : i32
// CHECK:  %[[CST1:.*]] = arith.constant 1 : i32
// CHECK:  %[[CST4:.*]] = arith.constant 4 : i32
// CHECK:  %[[CST8:.*]] = arith.constant 8 : i32
// CHECK:  AIEX.ipu.dma_memcpy_nd(%[[CST0]], %[[CST0]], %[[ARG0]][%[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]]] [%[[CST1]], %[[CST1]], %[[CST4]], %[[CST4]]] [%[[CST0]], %[[CST0]], %[[CST8]]]) {id = 1 : i32, metadata = @airMemcpyId14} : (i32, i32, memref<8x8xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
// CHECK:  AIEX.ipu.dma_memcpy_nd(%[[CST0]], %[[CST1]], %[[ARG0]][%[[CST0]], %[[CST0]], %[[CST0]], %[[CST4]]] [%[[CST1]], %[[CST1]], %[[CST4]], %[[CST4]]] [%[[CST0]], %[[CST0]], %[[CST8]]]) {id = 2 : i32, metadata = @airMemcpyId14_1} : (i32, i32, memref<8x8xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
// CHECK:  AIEX.ipu.dma_memcpy_nd(%[[CST1]], %[[CST0]], %[[ARG0]][%[[CST0]], %[[CST0]], %[[CST4]], %[[CST0]]] [%[[CST1]], %[[CST1]], %[[CST4]], %[[CST4]]] [%[[CST0]], %[[CST0]], %[[CST8]]]) {id = 3 : i32, metadata = @airMemcpyId14_2} : (i32, i32, memref<8x8xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
// CHECK:  AIEX.ipu.dma_memcpy_nd(%[[CST1]], %[[CST1]], %[[ARG0]][%[[CST0]], %[[CST0]], %[[CST4]], %[[CST4]]] [%[[CST1]], %[[CST1]], %[[CST4]], %[[CST4]]] [%[[CST0]], %[[CST0]], %[[CST8]]]) {id = 4 : i32, metadata = @airMemcpyId14_3} : (i32, i32, memref<8x8xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])

#map = affine_map<()[s0] -> (s0 * 4)>
module {
  AIE.device(ipu) {
    AIE.shimDMAAllocation @airMemcpyId14(S2MM, 0, 0)
    memref.global "public" @airMemcpyId14 : memref<4x4xi32, 2>
    AIE.shimDMAAllocation @airMemcpyId14_1(S2MM, 1, 0)
    memref.global "public" @airMemcpyId14_1 : memref<4x4xi32, 2>
    AIE.shimDMAAllocation @airMemcpyId14_2(S2MM, 0, 1)
    memref.global "public" @airMemcpyId14_2 : memref<4x4xi32, 2>
    AIE.shimDMAAllocation @airMemcpyId14_3(S2MM, 1, 1)
    memref.global "public" @airMemcpyId14_3 : memref<4x4xi32, 2>
  } {sym_name = "segment_0"}
  airrt.module_metadata{
  }
  func.func @func3(%arg0: memref<8x8xi32>) {
    %c4_i64 = arith.constant 4 : i64
    %c8_i64 = arith.constant 8 : i64
    %c14_i32 = arith.constant 14 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 2 {
          affine.for %arg4 = 0 to 2 {
            %0 = affine.apply #map()[%arg3]
            %1 = affine.apply #map()[%arg4]
            %2 = arith.index_cast %arg3 : index to i64
            %3 = arith.index_cast %arg4 : index to i64
            %4 = arith.index_cast %0 : index to i64
            %5 = arith.index_cast %1 : index to i64
            %6 = airrt.dma_memcpy_nd(%c14_i32, %2, %3, %arg0[%c0_i64, %c0_i64, %4, %5], [%c1_i64, %c1_i64, %c4_i64, %c4_i64], [%c0_i64, %c0_i64, %c8_i64]) {metadata = @airMemcpyId14} : (i32, i64, i64, memref<8x8xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
          }
        }
        %p = airrt.segment_load "segment_0" : i64
      }
    }
    return
  }
}

// -----

// ObjectFifo lowering

// CHECK-LABEL: AIE.device(ipu) {
// CHECK:  func.func @func4(%[[ARG0:.*]]: memref<8x8xi32>)
// CHECK:  %[[CST0]] = arith.constant 0 : i32
// CHECK:  %[[CST1]] = arith.constant 1 : i32
// CHECK:  %[[CST4]] = arith.constant 4 : i32
// CHECK:  %[[CST8]] = arith.constant 8 : i32
// CHECK:  AIEX.ipu.dma_memcpy_nd(%[[CST0]], %[[CST0]], %[[ARG0]][%[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]]] [%[[CST1]], %[[CST1]], %[[CST4]], %[[CST4]]] [%[[CST0]], %[[CST0]], %[[CST8]]]) {id = 1 : i32, metadata = @air_channel_1} : (i32, i32, memref<8x8xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
// CHECK:  AIEX.ipu.dma_memcpy_nd(%[[CST0]], %[[CST1]], %[[ARG0]][%[[CST0]], %[[CST0]], %[[CST0]], %[[CST4]]] [%[[CST1]], %[[CST1]], %[[CST4]], %[[CST4]]] [%[[CST0]], %[[CST0]], %[[CST8]]]) {id = 2 : i32, metadata = @air_channel_3} : (i32, i32, memref<8x8xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
// CHECK:  AIEX.ipu.dma_memcpy_nd(%[[CST1]], %[[CST0]], %[[ARG0]][%[[CST0]], %[[CST0]], %[[CST4]], %[[CST0]]] [%[[CST1]], %[[CST1]], %[[CST4]], %[[CST4]]] [%[[CST0]], %[[CST0]], %[[CST8]]]) {id = 3 : i32, metadata = @air_channel_2} : (i32, i32, memref<8x8xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])
// CHECK:  AIEX.ipu.dma_memcpy_nd(%[[CST1]], %[[CST1]], %[[ARG0]][%[[CST0]], %[[CST0]], %[[CST4]], %[[CST4]]] [%[[CST1]], %[[CST1]], %[[CST4]], %[[CST4]]] [%[[CST0]], %[[CST0]], %[[CST8]]]) {id = 4 : i32, metadata = @air_channel_4} : (i32, i32, memref<8x8xi32>, [i32, i32, i32, i32], [i32, i32, i32, i32], [i32, i32, i32])

#map = affine_map<()[s0] -> (s0 * 4)>
module {
  AIE.device(ipu) {
    %tile_0_3 = AIE.tile(0, 3)
    %tile_1_3 = AIE.tile(1, 3)
    %tile_0_4 = AIE.tile(0, 4)
    %tile_1_4 = AIE.tile(1, 4)
    %tile_0_0 = AIE.tile(0, 0)
    %tile_1_0 = AIE.tile(1, 0)
    AIE.objectFifo @air_channel_4(%tile_1_4, {%tile_0_0}, 1 : i32) : !AIE.objectFifo<memref<4x4xi32>>
    AIE.objectFifo @air_channel_3(%tile_0_4, {%tile_0_0}, 1 : i32) : !AIE.objectFifo<memref<4x4xi32>>
    AIE.objectFifo @air_channel_2(%tile_1_3, {%tile_1_0}, 1 : i32) : !AIE.objectFifo<memref<4x4xi32>>
    AIE.objectFifo @air_channel_1(%tile_0_3, {%tile_1_0}, 1 : i32) : !AIE.objectFifo<memref<4x4xi32>>
  } {sym_name = "segment_0"}
  airrt.module_metadata{
  }
  func.func @func4(%arg0: memref<8x8xi32>) {
    %c4_i64 = arith.constant 4 : i64
    %c8_i64 = arith.constant 8 : i64
    %c14_i32 = arith.constant 14 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg1 = 0 to 1 {
      affine.for %arg2 = 0 to 1 {
        affine.for %arg3 = 0 to 2 {
          affine.for %arg4 = 0 to 2 {
            %0 = affine.apply #map()[%arg3]
            %1 = affine.apply #map()[%arg4]
            %2 = arith.index_cast %arg3 : index to i64
            %3 = arith.index_cast %arg4 : index to i64
            %4 = arith.index_cast %0 : index to i64
            %5 = arith.index_cast %1 : index to i64
            %6 = airrt.dma_memcpy_nd(%c14_i32, %2, %3, %arg0[%c0_i64, %c0_i64, %4, %5], [%c1_i64, %c1_i64, %c4_i64, %c4_i64], [%c0_i64, %c0_i64, %c8_i64]) {metadata = @air_channel_1} : (i32, i64, i64, memref<8x8xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
          }
        }
        %p = airrt.segment_load "segment_0" : i64
      }
    }
    return
  }
}
