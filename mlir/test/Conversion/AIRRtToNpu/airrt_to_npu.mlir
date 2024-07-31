//===- airrt_to_npu.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -canonicalize -cse --split-input-file %s | FileCheck %s

// Synchronous airrt.dma_memcpy_nd

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK: aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 0)
// CHECK: memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
// CHECK: aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
// CHECK: aiex.runtime_sequence @func0(%[[VAL_0:.*]]: memref<64xi32>, %[[VAL_1:.*]]: memref<64xi32>) {
// CHECK:   aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_0]][0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @airMemcpyId2} : memref<64xi32>
// CHECK:   aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_1]][0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 1 : i64, metadata = @airMemcpyId7} : memref<64xi32>
// CHECK:   aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: }
// CHECK: {sym_name = "segment0"}

module {
  aie.device(npu1_1col) {
    aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 0)
    memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
    aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 0)
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

// CHECK-LABEL: aie.device(npu1_1col) {
// CHECK: aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 0)
// CHECK: memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
// CHECK: aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
// CHECK: aiex.runtime_sequence @func1(%[[VAL_0:.*]]: memref<64xi32>, %[[VAL_1:.*]]: memref<64xi32>) {
// CHECK:   aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_0]][0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 0 : i64, metadata = @airMemcpyId2} : memref<64xi32>
// CHECK:   aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_1]][0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1]) {id = 1 : i64, metadata = @airMemcpyId7} : memref<64xi32>
// CHECK:   aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: }
// CHECK: } {sym_name = "segment0"}

module {
  aie.device(npu1_1col) {
    aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 0)
    memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
    aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 0)
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

// CHECK-LABEL: aie.device(npu1_1col) {
// CHECK: aie.shim_dma_allocation @airMemcpyId16(S2MM, 0, 0)
// CHECK: memref.global "public" @airMemcpyId16 : memref<32x32xi32, 1>
// CHECK: aie.shim_dma_allocation @airMemcpyId5(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId5 : memref<32x32xi32, 1>
// CHECK: aie.shim_dma_allocation @airMemcpyId6(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId6 : memref<32x32xi32, 1>
// CHECK: aie.shim_dma_allocation @airMemcpyId7(MM2S, 0, 0)
// CHECK: memref.global "public" @airMemcpyId7 : memref<32x32xi32, 1>
// CHECK: aiex.runtime_sequence @func2(%[[VAL_0:.*]]: memref<32x32xi32>, %[[VAL_1:.*]]: memref<32x32xi32>, %[[VAL_2:.*]]: memref<32x32xi32>) {
// CHECK:   aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_2]][0, 0, 0, 0][1, 1, 32, 32][0, 0, 32, 1]) {id = 0 : i64, metadata = @airMemcpyId5} : memref<32x32xi32>
// CHECK:   aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_0]][0, 0, 0, 0][1, 1, 32, 32][0, 0, 32, 1]) {id = 1 : i64, metadata = @airMemcpyId6} : memref<32x32xi32>
// CHECK:   aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_1]][0, 0, 0, 0][1, 1, 32, 32][0, 0, 32, 1]) {id = 2 : i64, metadata = @airMemcpyId7} : memref<32x32xi32>
// CHECK:   aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_2]][0, 0, 0, 0][1, 1, 32, 32][0, 0, 32, 1]) {id = 3 : i64, metadata = @airMemcpyId16} : memref<32x32xi32>
// CHECK:   aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: }
// CHECK: } {sym_name = "segment_0"}

#map = affine_map<()[s0] -> (s0 * 32)>
module {
  aie.device(npu1_1col) {
    aie.shim_dma_allocation @airMemcpyId16(S2MM, 0, 0)
    memref.global "public" @airMemcpyId16 : memref<32x32xi32, 1>
    aie.shim_dma_allocation @airMemcpyId5(MM2S, 0, 0)
    memref.global "public" @airMemcpyId5 : memref<32x32xi32, 1>
    aie.shim_dma_allocation @airMemcpyId6(MM2S, 0, 0)
    memref.global "public" @airMemcpyId6 : memref<32x32xi32, 1>
    aie.shim_dma_allocation @airMemcpyId7(MM2S, 0, 0)
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

// CHECK-LABEL: aie.device(npu1_2col) {
// CHECK:  aiex.runtime_sequence @func3(%[[ARG0:.*]]: memref<8x8xi32>)
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 0, 0][2, 2, 4, 4][32, 4, 8, 1]) {id = 0 : i64, metadata = @airMemcpyId14} : memref<8x8xi32>

#map = affine_map<()[s0] -> (s0 * 4)>
module {
  aie.device(npu1_2col) {
    aie.shim_dma_allocation @airMemcpyId14(S2MM, 0, 0)
    memref.global "public" @airMemcpyId14 : memref<4x4xi32, 2>
    aie.shim_dma_allocation @airMemcpyId14_1(S2MM, 1, 0)
    memref.global "public" @airMemcpyId14_1 : memref<4x4xi32, 2>
    aie.shim_dma_allocation @airMemcpyId14_2(S2MM, 0, 1)
    memref.global "public" @airMemcpyId14_2 : memref<4x4xi32, 2>
    aie.shim_dma_allocation @airMemcpyId14_3(S2MM, 1, 1)
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

// objectfifo lowering

// CHECK-LABEL: aie.device(npu1_2col)
// CHECK:  aiex.runtime_sequence @func4(%[[ARG0:.*]]: memref<8x8xi32>)
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 0, 0][2, 2, 4, 4][32, 4, 8, 1]) {id = 0 : i64, metadata = @air_channel_1} : memref<8x8xi32>

#map = affine_map<()[s0] -> (s0 * 4)>
module {
  aie.device(npu1_2col) {
    %tile_0_3 = aie.tile(0, 3)
    %tile_1_3 = aie.tile(1, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_1_4 = aie.tile(1, 4)
    %tile_0_0 = aie.tile(0, 0)
    %tile_1_0 = aie.tile(1, 0)
    aie.objectfifo @air_channel_4(%tile_1_4, {%tile_0_0}, 1 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @air_channel_3(%tile_0_4, {%tile_0_0}, 1 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @air_channel_2(%tile_1_3, {%tile_1_0}, 1 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @air_channel_1(%tile_0_3, {%tile_1_0}, 1 : i32) : !aie.objectfifo<memref<4x4xi32>>
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

// -----

// Unroll repeat pattern

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK:  aiex.runtime_sequence @func5(%[[ARG0:.*]]: memref<8x8xi32>, %[[ARG1:.*]]: memref<8x8xi32>, %[[ARG2:.*]]: memref<8x8xi32>)
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 0, 0][2, 1, 4, 8][0, 0, 8, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<8x8xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 4, 0][2, 1, 4, 8][0, 0, 8, 1]) {id = 1 : i64, metadata = @airMemcpyId4} : memref<8x8xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG1]][0, 0, 0, 0][2, 2, 8, 4][0, 4, 8, 1]) {id = 2 : i64, metadata = @airMemcpyId5} : memref<8x8xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG2]][0, 0, 0, 0][2, 2, 4, 4][32, 4, 8, 1]) {id = 3 : i64, metadata = @airMemcpyId16} : memref<8x8xi32>

#map = affine_map<()[s0] -> (s0 * 4)>
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    aie.shim_dma_allocation @airMemcpyId16(S2MM, 0, 0)
    memref.global "public" @airMemcpyId16 : memref<4x4xi32, 1>
    aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
    memref.global "public" @airMemcpyId4 : memref<4x8xi32, 1>
    aie.shim_dma_allocation @airMemcpyId5(MM2S, 1, 0)
    memref.global "public" @airMemcpyId5 : memref<8x4xi32, 1>
  } {sym_name = "segment_0"}
  airrt.module_metadata{
  }
  func.func @func5(%arg0: memref<8x8xi32>, %arg1: memref<8x8xi32>, %arg2: memref<8x8xi32>) {
    %c4_i64 = arith.constant 4 : i64
    %c8_i64 = arith.constant 8 : i64
    %c16_i32 = arith.constant 16 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        %0 = affine.apply #map()[%arg3]
        %1 = arith.index_cast %arg3 : index to i64
        %2 = arith.index_cast %arg4 : index to i64
        %3 = arith.index_cast %0 : index to i64
        %4 = airrt.dma_memcpy_nd(%c4_i32, %1, %2, %arg0[%c0_i64, %c0_i64, %3, %c0_i64], [%c1_i64, %c1_i64, %c4_i64, %c8_i64], [%c0_i64, %c0_i64, %c8_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<8x8xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %5 = affine.apply #map()[%arg4]
        %6 = arith.index_cast %5 : index to i64
        %7 = airrt.dma_memcpy_nd(%c5_i32, %1, %2, %arg1[%c0_i64, %c0_i64, %c0_i64, %6], [%c1_i64, %c1_i64, %c8_i64, %c4_i64], [%c0_i64, %c0_i64, %c8_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<8x8xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %8 = airrt.dma_memcpy_nd(%c16_i32, %1, %2, %arg2[%c0_i64, %c0_i64, %3, %6], [%c1_i64, %c1_i64, %c4_i64, %c4_i64], [%c0_i64, %c0_i64, %c8_i64]) {metadata = @airMemcpyId16} : (i32, i64, i64, memref<8x8xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %p = airrt.segment_load "segment_0" : i64
      }
    }
    return
  }
}

// -----

// Populate repeat dimension (highest dimension)

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK:  aiex.runtime_sequence @func6(%[[ARG0:.*]]: memref<8x16xi32>, %[[ARG1:.*]]: memref<16x32xi32>, %[[ARG2:.*]]: memref<8x32xi32>)
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 0, 0][2, 1, 8, 16][0, 0, 16, 1]) {id = 0 : i64, metadata = @airMemcpyId4} : memref<8x16xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG1]][0, 0, 0, 0][1, 2, 16, 16][0, 16, 32, 1]) {id = 1 : i64, metadata = @airMemcpyId5} : memref<16x32xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG2]][0, 0, 0, 0][1, 2, 8, 16][0, 16, 32, 1]) {id = 2 : i64, metadata = @airMemcpyId12} : memref<8x32xi32>

#map = affine_map<()[s0] -> (s0 * 8)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId12(S2MM, 0, 0)
    memref.global "public" @airMemcpyId12 : memref<1x1x8x16xi32, 1>
    aie.shim_dma_allocation @airMemcpyId4(MM2S, 0, 0)
    memref.global "public" @airMemcpyId4 : memref<1x1x8x16xi32, 1>
    aie.shim_dma_allocation @airMemcpyId5(MM2S, 1, 0)
    memref.global "public" @airMemcpyId5 : memref<1x1x16x16xi32, 1>
  } {sym_name = "segment_0"}
  airrt.module_metadata{
  }
  func.func @func6(%arg0: memref<8x16xi32>, %arg1: memref<16x32xi32>, %arg2: memref<8x32xi32>) {
    %c32_i64 = arith.constant 32 : i64
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c12_i32 = arith.constant 12 : i32
    %c5_i32 = arith.constant 5 : i32
    %c4_i32 = arith.constant 4 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 2 {
        %0 = affine.apply #map()[%arg3]
        %1 = arith.index_cast %arg3 : index to i64
        %2 = arith.index_cast %arg4 : index to i64
        %3 = arith.index_cast %0 : index to i64
        %4 = airrt.dma_memcpy_nd(%c4_i32, %1, %2, %arg0[%c0_i64, %c0_i64, %3, %c0_i64], [%c1_i64, %c1_i64, %c8_i64, %c16_i64], [%c0_i64, %c0_i64, %c16_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<8x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %5 = affine.apply #map1()[%arg4]
        %6 = arith.index_cast %5 : index to i64
        %7 = airrt.dma_memcpy_nd(%c5_i32, %1, %2, %arg1[%c0_i64, %c0_i64, %c0_i64, %6], [%c1_i64, %c1_i64, %c16_i64, %c16_i64], [%c0_i64, %c0_i64, %c32_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x32xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %8 = airrt.dma_memcpy_nd(%c12_i32, %1, %2, %arg2[%c0_i64, %c0_i64, %3, %6], [%c1_i64, %c1_i64, %c8_i64, %c16_i64], [%c0_i64, %c0_i64, %c32_i64]) {metadata = @airMemcpyId12} : (i32, i64, i64, memref<8x32xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %p = airrt.segment_load "segment_0" : i64
      }
    }
    return
  }
}

// -----

// Unroll repeat pattern + populate repeat dimension

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK:  aiex.runtime_sequence @func7(%[[ARG0:.*]]: memref<2048x512xi32>, %[[ARG1:.*]]: memref<512x2048xi32>, %[[ARG2:.*]]: memref<2048x2048xi32>)
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 0, 0][4, 8, 64, 64][0, 64, 512, 1]) {id = 0 : i64, metadata = @airMemcpyId20} : memref<2048x512xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 64, 0][4, 8, 64, 64][0, 64, 512, 1]) {id = 1 : i64, metadata = @airMemcpyId20} : memref<2048x512xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 128, 0][4, 8, 64, 64][0, 64, 512, 1]) {id = 2 : i64, metadata = @airMemcpyId20} : memref<2048x512xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 192, 0][4, 8, 64, 64][0, 64, 512, 1]) {id = 3 : i64, metadata = @airMemcpyId20} : memref<2048x512xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG1]][0, 0, 0, 0][4, 4, 512, 64][0, 64, 2048, 1]) {id = 4 : i64, metadata = @airMemcpyId21} : memref<512x2048xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG2]][0, 0, 0, 0][4, 4, 64, 64][131072, 64, 2048, 1]) {id = 5 : i64, metadata = @airMemcpyId26} : memref<2048x2048xi32>

#map = affine_map<()[s0] -> (s0 * 64)>
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId26(S2MM, 0, 0)
    memref.global "public" @airMemcpyId26 : memref<64x64xi32, 1>
    aie.shim_dma_allocation @airMemcpyId20(MM2S, 0, 0)
    memref.global "public" @airMemcpyId20 : memref<64x64xi32, 1>
    aie.shim_dma_allocation @airMemcpyId21(MM2S, 1, 0)
    memref.global "public" @airMemcpyId21 : memref<64x64xi32, 1>
  } {sym_name = "segment_0"}
  airrt.module_metadata{
  }
  func.func @func7(%arg0: memref<2048x512xi32>, %arg1: memref<512x2048xi32>) {
    %c2048_i64 = arith.constant 2048 : i64
    %c8_i64 = arith.constant 8 : i64
    %c512_i64 = arith.constant 512 : i64
    %c64_i64 = arith.constant 64 : i64
    %c26_i32 = arith.constant 26 : i32
    %c15_i32 = arith.constant 15 : i32
    %c14_i32 = arith.constant 14 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() : memref<2048x2048xi32>
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 4 {
        %0 = affine.apply #map()[%arg3]
        %1 = arith.index_cast %arg3 : index to i64
        %2 = arith.index_cast %arg4 : index to i64
        %3 = arith.index_cast %0 : index to i64
        %4 = airrt.dma_memcpy_nd(%c14_i32, %1, %2, %arg0[%c0_i64, %c0_i64, %3, %c0_i64], [%c1_i64, %c8_i64, %c64_i64, %c64_i64], [%c0_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId20} : (i32, i64, i64, memref<2048x512xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %5 = affine.apply #map()[%arg4]
        %6 = arith.index_cast %arg3 : index to i64
        %7 = arith.index_cast %arg4 : index to i64
        %8 = arith.index_cast %5 : index to i64
        %9 = airrt.dma_memcpy_nd(%c15_i32, %6, %7, %arg1[%c0_i64, %c0_i64, %c0_i64, %8], [%c1_i64, %c1_i64, %c512_i64, %c64_i64], [%c0_i64, %c0_i64, %c2048_i64]) {metadata = @airMemcpyId21} : (i32, i64, i64, memref<512x2048xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %10 = affine.apply #map()[%arg3]
        %11 = affine.apply #map()[%arg4]
        %12 = arith.index_cast %arg3 : index to i64
        %13 = arith.index_cast %arg4 : index to i64
        %14 = arith.index_cast %10 : index to i64
        %15 = arith.index_cast %11 : index to i64
        %16 = airrt.dma_memcpy_nd(%c26_i32, %12, %13, %alloc[%c0_i64, %c0_i64, %14, %15], [%c1_i64, %c1_i64, %c64_i64, %c64_i64], [%c0_i64, %c0_i64, %c2048_i64]) {metadata = @airMemcpyId26} : (i32, i64, i64, memref<2048x2048xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %p = airrt.segment_load "segment_0" : i64
      }
    }
    return
  }
}

// -----

// check that lowering works for the herd_load case

// CHECK-LABEL: @func8
// CHECK: aiex.npu.dma_memcpy_nd
// CHECK: aiex.npu.sync
module {
  aie.device(npu1_1col) {
    aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 0)
    memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
  } {sym_name = "herd"}
  func.func @func8(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c7_i32 = arith.constant 7 : i32
    %c64_i64 = arith.constant 64 : i64
    %p = airrt.herd_load "herd" : i64
    airrt.dma_memcpy_nd(%c7_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
    return
  }
}

// -----

// Dealing with scenarios where wrap dimension in airrt.dma_memcpy_nd goes beyond the [0, 1023] hardware limit.

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK:  aiex.runtime_sequence @func9(%[[ARG0:.*]]: memref<2048x2048xi32>, %[[ARG1:.*]]: memref<2048x2048xi32>, %[[ARG2:.*]]: memref<2048x2048xi32>)
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 0, 0][4, 8, 64, 256][0, 256, 2048, 1]) {id = 0 : i64, metadata = @airMemcpyId20} : memref<2048x2048xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 64, 0][4, 8, 64, 256][0, 256, 2048, 1]) {id = 1 : i64, metadata = @airMemcpyId20} : memref<2048x2048xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 128, 0][4, 8, 64, 256][0, 256, 2048, 1]) {id = 2 : i64, metadata = @airMemcpyId20} : memref<2048x2048xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 192, 0][4, 8, 64, 256][0, 256, 2048, 1]) {id = 3 : i64, metadata = @airMemcpyId20} : memref<2048x2048xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG1]][0, 0, 0, 0][4, 4, 512, 64][64, 1048576, 2048, 1]) {id = 4 : i64, metadata = @airMemcpyId21} : memref<2048x2048xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG1]][0, 0, 0, 0][4, 4, 512, 64][64, 1048576, 2048, 1]) {id = 5 : i64, metadata = @airMemcpyId21} : memref<2048x2048xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG1]][0, 0, 0, 0][4, 4, 512, 64][64, 1048576, 2048, 1]) {id = 6 : i64, metadata = @airMemcpyId21} : memref<2048x2048xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG1]][0, 0, 0, 0][4, 4, 512, 64][64, 1048576, 2048, 1]) {id = 7 : i64, metadata = @airMemcpyId21} : memref<2048x2048xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG2]][0, 0, 0, 0][4, 4, 64, 64][131072, 64, 2048, 1]) {id = 8 : i64, metadata = @airMemcpyId26} : memref<2048x2048xi32>

#map = affine_map<()[s0] -> (s0 * 64)>
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId26(S2MM, 0, 0)
    memref.global "public" @airMemcpyId26 : memref<64x64xi32, 1>
    aie.shim_dma_allocation @airMemcpyId14(MM2S, 0, 0)
    memref.global "public" @airMemcpyId14 : memref<64x256xi32, 1>
    aie.shim_dma_allocation @airMemcpyId20(MM2S, 0, 0)
    memref.global "public" @airMemcpyId20 : memref<64x256xi32, 1>
    aie.shim_dma_allocation @airMemcpyId15(MM2S, 1, 0)
    memref.global "public" @airMemcpyId15 : memref<256x64xi32, 1>
    aie.shim_dma_allocation @airMemcpyId21(MM2S, 1, 0)
    memref.global "public" @airMemcpyId21 : memref<256x64xi32, 1>
  } {sym_name = "segment_0"}
  airrt.module_metadata{
  }
  func.func @func9(%arg0: memref<2048x2048xi32>, %arg1: memref<2048x2048xi32>) {
    %c64_i64 = arith.constant 64 : i64
    %c8_i64 = arith.constant 8 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c256_i64 = arith.constant 256 : i64
    %c26_i32 = arith.constant 26 : i32
    %c15_i32 = arith.constant 15 : i32
    %c14_i32 = arith.constant 14 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() : memref<2048x2048xi32>
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 4 {
        %0 = affine.apply #map()[%arg3]
        %1 = arith.index_cast %arg3 : index to i64
        %2 = arith.index_cast %arg4 : index to i64
        %3 = arith.index_cast %0 : index to i64
        %4 = airrt.dma_memcpy_nd(%c14_i32, %1, %2, %arg0[%c0_i64, %c0_i64, %3, %c0_i64], [%c1_i64, %c8_i64, %c64_i64, %c256_i64], [%c0_i64, %c256_i64, %c2048_i64]) {metadata = @airMemcpyId20} : (i32, i64, i64, memref<2048x2048xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %5 = affine.apply #map()[%arg4]
        %6 = arith.index_cast %arg3 : index to i64
        %7 = arith.index_cast %arg4 : index to i64
        %8 = arith.index_cast %5 : index to i64
        %9 = airrt.dma_memcpy_nd(%c15_i32, %6, %7, %arg1[%c0_i64, %c0_i64, %c0_i64, %8], [%c1_i64, %c1_i64, %c2048_i64, %c64_i64], [%c0_i64, %c0_i64, %c2048_i64]) {metadata = @airMemcpyId21} : (i32, i64, i64, memref<2048x2048xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %10 = affine.apply #map()[%arg3]
        %11 = affine.apply #map()[%arg4]
        %12 = arith.index_cast %arg3 : index to i64
        %13 = arith.index_cast %arg4 : index to i64
        %14 = arith.index_cast %10 : index to i64
        %15 = arith.index_cast %11 : index to i64
        %16 = airrt.dma_memcpy_nd(%c26_i32, %12, %13, %alloc[%c0_i64, %c0_i64, %14, %15], [%c1_i64, %c1_i64, %c64_i64, %c64_i64], [%c0_i64, %c0_i64, %c2048_i64]) {metadata = @airMemcpyId26} : (i32, i64, i64, memref<2048x2048xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %p = airrt.segment_load "segment_0" : i64
      }
    }
    return
  }
}

// -----

// Dealing with scenarios where wrap dimension in airrt.dma_memcpy_nd goes beyond the [0, 1023] hardware limit (test case 2).

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK:  aiex.runtime_sequence @func10(%[[ARG0:.*]]: memref<2654208xi32>)
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 0, 0][3, 768, 3, 32][128, 3456, 1152, 1]) {id = 0 : i64, metadata = @airMemcpyId21} : memref<2654208xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 0, 0][3, 768, 3, 32][128, 3456, 1152, 1]) {id = 1 : i64, metadata = @airMemcpyId21} : memref<2654208xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 0, 0][3, 768, 3, 32][128, 3456, 1152, 1]) {id = 2 : i64, metadata = @airMemcpyId21} : memref<2654208xi32>

#map = affine_map<()[s0] -> (s0 * 64)>
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId21(MM2S, 0, 2)
    memref.global "public" @airMemcpyId21 : memref<256x64xbf16, 1>
  } {sym_name = "segment_0"}
  airrt.module_metadata{
  }
  func.func @func10(%arg2: memref<2304x2304xbf16>) {
    %c64_i64 = arith.constant 64 : i64
    %c8_i64 = arith.constant 8 : i64
    %c2304_i64 = arith.constant 2304 : i64
    %c256_i64 = arith.constant 256 : i64
    %c26_i32 = arith.constant 26 : i32
    %c15_i32 = arith.constant 15 : i32
    %c14_i32 = arith.constant 14 : i32
    %c21_i32 = arith.constant 21 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg0 = 0 to 3 {
      affine.for %arg1 = 0 to 3 {
        %p = airrt.segment_load "segment_0" : i64
        %34 = affine.apply affine_map<()[s0] -> (s0 * 256)>()[%arg1]
        %39 = arith.index_cast %arg0 : index to i64
        %40 = arith.index_cast %arg1 : index to i64
        %41 = arith.index_cast %34 : index to i64
        %42 = airrt.dma_memcpy_nd(%c21_i32, %39, %40, %arg2[%c0_i64, %c0_i64, %c0_i64, %41], [%c1_i64, %c1_i64, %c2304_i64, %c64_i64], [%c0_i64, %c0_i64, %c2304_i64]) {metadata = @airMemcpyId21} : (i32, i64, i64, memref<2304x2304xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      }
    }
    return
  }
}

// -----

// 16-bit type conversion

// CHECK-LABEL: aiex.runtime_sequence @func11
// CHECK-SAME: %arg0: memref<8192xi32>
// CHECK-NEXT: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][4, 4, 32, 16][2048, 16, 64, 1]){{.*}}: memref<8192xi32>
module {
  aie.device(npu1_1col) {
    func.func @func11(%arg0: memref<128x128xbf16>, %arg1: memref<128x128xbf16>) {
      %c0_i32 = arith.constant 0 : i32
      %c0_i64 = arith.constant 0 : i64
      %c4_i64 = arith.constant 4 : i64
      %c32_i64 = arith.constant 32 : i64
      %c128_i64 = arith.constant 128 : i64
      %c4096_i64 = arith.constant 4096 : i64
      airrt.dma_memcpy_nd(%c0_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c4_i64, %c4_i64, %c32_i64, %c32_i64], [%c4096_i64, %c32_i64, %c128_i64]) {metadata = @md0} : (i32, i64, i64, memref<128x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
      return
    }
  }
}

// -----

// 16-bit conversion with dma operands that aren't function arguments

// CHECK-LABEL: func.func @func12
// CHECK-SAME: %arg0: memref<16xi32>
// CHECK-NEXT: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][1, 1, 1, 16][0, 0, 0, 1]) {{.*}} : memref<16xi32>
module {
 func.func @func12() {
    %c1_i32 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %alloc = memref.alloc() : memref<32xbf16>
    airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %alloc[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @md0} : (i32, i64, i64, memref<32xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
    return
  }
}

// -----

// Before PR https://github.com/Xilinx/mlir-air/pull/447 running
// `air-opt --cse  --canonicalize -airrt-to-npu`
// on the function in the test produced:
//
//  func.func @func12(%arg0: memref<16xi32>) {
//    %alloc = memref.alloc() : memref<32xbf16>
//    memref.assume_alignment %alloc, 64 : memref<32xbf16>
//    aiex.npu.dma_memcpy_nd(0, 0, %arg0 ...
//    return
//  }
//
// PR 447 relocates the memref.assume_alignment op so that calling
// `air-opt -airrt-to-npu` on the function in the test now produces:
//
//  func.func @func12(%arg0: memref<16xi32>) {
//    memref.assume_alignment %arg0, 64 : memref<16xi32>
//    aiex.npu.dma_memcpy_nd(0, 0, %arg0 ...
//    return
//  }
//
// The key difference is that memref.alloc is removed.

// CHECK-LABEL: func13
// CHECK-NOT: memref.alloc
// CHECK: memref.assume_alignment
// CHECK-SAME: memref<16xi32>
// CHECK-NOT: memref.alloc
// CHECK: return
module {
  func.func @func13() {

    %c1_i32 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %alloc = memref.alloc() : memref<32xbf16>

    // assert that the alignment of %alloc is 64 bits:
    memref.assume_alignment %alloc, 64 : memref<32xbf16>
    airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %alloc[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @md0} : (i32, i64, i64, memref<32xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
    return
  }
}

// -----

// Multi-dimensional offset collapsing

// CHECK-LABEL: func.func @func14
// CHECK-SAME: %arg0: memref<512xi32>
// CHECK-NEXT: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 264][1, 1, 16, 8][0, 0, 16, 1]) {id = 0 : i64, metadata = @md0} : memref<512xi32>
module {
 func.func @func14(%arg0 : memref<32x32xbf16>) {
    %c1_i32 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c16_i64 = arith.constant 16 : i64
    %c32_i64 = arith.constant 32 : i64
    airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c16_i64, %c16_i64], [%c1_i64, %c1_i64, %c16_i64, %c16_i64], [%c0_i64, %c0_i64, %c32_i64]) {metadata = @md0} : (i32, i64, i64, memref<32x32xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
    return
  }
}

// -----

// Loop carried event

// CHECK-LABEL: func.func @func15
// CHECK-NEXT: return
module {
  func.func @func15() {
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c2048 = arith.constant 2048 : index
    %9 = airrt.wait_all : !airrt.event
    %11:1 = scf.for %arg6 = %c0 to %c2048 step %c512 iter_args(%arg7 = %9) -> (!airrt.event) {
      %12 = airrt.wait_all : !airrt.event
      scf.yield %12 : !airrt.event
    }
    return
  }
}

// -----

// Multiple Shim DMAs.

// CHECK: aie.shim_dma_allocation @airMemcpyId45(S2MM, 0, 0)
// CHECK: aie.shim_dma_allocation @airMemcpyId46(S2MM, 1, 0)
// CHECK: aie.shim_dma_allocation @airMemcpyId47(S2MM, 0, 1)
// CHECK: aie.shim_dma_allocation @airMemcpyId48(S2MM, 1, 1)
// CHECK: aie.shim_dma_allocation @airMemcpyId7(MM2S, 0, 0)
// CHECK: aie.shim_dma_allocation @airMemcpyId17(MM2S, 0, 0)
// CHECK: aie.shim_dma_allocation @airMemcpyId12(MM2S, 1, 0)
// CHECK: aie.shim_dma_allocation @airMemcpyId22(MM2S, 1, 0)
// CHECK-LABEL: aiex.runtime_sequence @func16
// CHECK-SAME: %[[VAL_0:.*]]: memref<262144xi32>, %[[VAL_1:.*]]: memref<262144xi32>, %[[VAL_2:.*]]: memref<131072xi32>) {
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_0]][0, 0, 0, 0][2, 4, 256, 128][0, 128, 512, 1]) {id = 0 : i64, metadata = @airMemcpyId7} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_0]][0, 0, 0, 131072][2, 4, 256, 128][0, 128, 512, 1]) {id = 1 : i64, metadata = @airMemcpyId7} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_1]][0, 0, 0, 0][2, 2, 512, 128][128, 131072, 256, 1]) {id = 2 : i64, metadata = @airMemcpyId12} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_1]][0, 0, 0, 0][2, 2, 512, 128][128, 131072, 256, 1]) {id = 3 : i64, metadata = @airMemcpyId12} : memref<262144xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_2]][0, 0, 0, 0][2, 2, 64, 128][65536, 128, 256, 1]) {id = 4 : i64, metadata = @airMemcpyId45} : memref<131072xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_2]][0, 0, 0, 16384][2, 2, 64, 128][65536, 128, 256, 1]) {id = 5 : i64, metadata = @airMemcpyId46} : memref<131072xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_2]][0, 0, 0, 32768][2, 2, 64, 128][65536, 128, 256, 1]) {id = 0 : i64, metadata = @airMemcpyId47} : memref<131072xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %[[VAL_2]][0, 0, 0, 49152][2, 2, 64, 128][65536, 128, 256, 1]) {id = 1 : i64, metadata = @airMemcpyId48} : memref<131072xi32>
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.sync {channel = 1 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}
// CHECK: aiex.npu.sync {channel = 1 : i32, column = 1 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}

#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 256 + 64)>
#map2 = affine_map<()[s0] -> (s0 * 256 + 128)>
#map3 = affine_map<()[s0] -> (s0 * 256 + 192)>
module {
  aie.device(npu1_1col) {
    aie.shim_dma_allocation @airMemcpyId45(S2MM, 0, 0)
    memref.global "public" @airMemcpyId45 : memref<256x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId46(S2MM, 1, 0)
    memref.global "public" @airMemcpyId46 : memref<256x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId47(S2MM, 0, 1)
    memref.global "public" @airMemcpyId47 : memref<256x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId48(S2MM, 1, 1)
    memref.global "public" @airMemcpyId48 : memref<256x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId7(MM2S, 0, 0)
    memref.global "public" @airMemcpyId7 : memref<256x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId17(MM2S, 0, 0)
    memref.global "public" @airMemcpyId17 : memref<256x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId12(MM2S, 1, 0)
    memref.global "public" @airMemcpyId12 : memref<256x256xbf16, 1>
    aie.shim_dma_allocation @airMemcpyId22(MM2S, 1, 0)
    memref.global "public" @airMemcpyId22 : memref<256x256xbf16, 1>
  } {sym_name = "segment_0"}
  func.func @func16(%arg0: memref<512x1024xbf16>, %arg1: memref<1024x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c64_i64 = arith.constant 64 : i64
    %c512_i64 = arith.constant 512 : i64
    %c4_i64 = arith.constant 4 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c256_i64 = arith.constant 256 : i64
    %c45_i32 = arith.constant 45 : i32
    %c12_i32 = arith.constant 12 : i32
    %c7_i32 = arith.constant 7 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg3 = 0 to 2 {
      affine.for %arg4 = 0 to 2 {
        %0 = affine.apply #map()[%arg3]
        %1 = arith.index_cast %arg3 : index to i64
        %2 = arith.index_cast %arg4 : index to i64
        %3 = arith.index_cast %0 : index to i64
        %4 = airrt.dma_memcpy_nd(%c7_i32, %1, %2, %arg0[%c0_i64, %c0_i64, %3, %c0_i64], [%c1_i64, %c4_i64, %c256_i64, %c256_i64], [%c0_i64, %c256_i64, %c1024_i64]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<512x1024xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %5 = affine.apply #map()[%arg4]
        %6 = arith.index_cast %arg3 : index to i64
        %7 = arith.index_cast %arg4 : index to i64
        %8 = arith.index_cast %5 : index to i64
        %9 = airrt.dma_memcpy_nd(%c12_i32, %6, %7, %arg1[%c0_i64, %c0_i64, %c0_i64, %8], [%c1_i64, %c1_i64, %c1024_i64, %c256_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId12} : (i32, i64, i64, memref<1024x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %10 = affine.apply #map()[%arg4]
        %11 = affine.apply #map()[%arg3]
        %12 = arith.index_cast %arg3 : index to i64
        %13 = arith.index_cast %arg4 : index to i64
        %14 = arith.index_cast %11 : index to i64
        %15 = arith.index_cast %10 : index to i64
        %16 = airrt.dma_memcpy_nd(%c45_i32, %12, %13, %arg2[%c0_i64, %c0_i64, %14, %15], [%c1_i64, %c1_i64, %c64_i64, %c256_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId45} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %17 = affine.apply #map1()[%arg3]
        %18 = arith.index_cast %arg3 : index to i64
        %19 = arith.index_cast %arg4 : index to i64
        %20 = arith.index_cast %17 : index to i64
        %21 = arith.index_cast %10 : index to i64
        %22 = airrt.dma_memcpy_nd(%c45_i32, %18, %19, %arg2[%c0_i64, %c0_i64, %20, %21], [%c1_i64, %c1_i64, %c64_i64, %c256_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId46} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %23 = affine.apply #map2()[%arg3]
        %24 = arith.index_cast %arg3 : index to i64
        %25 = arith.index_cast %arg4 : index to i64
        %26 = arith.index_cast %23 : index to i64
        %27 = arith.index_cast %10 : index to i64
        %28 = airrt.dma_memcpy_nd(%c45_i32, %24, %25, %arg2[%c0_i64, %c0_i64, %26, %27], [%c1_i64, %c1_i64, %c64_i64, %c256_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId47} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %29 = affine.apply #map3()[%arg3]
        %30 = arith.index_cast %arg3 : index to i64
        %31 = arith.index_cast %arg4 : index to i64
        %32 = arith.index_cast %29 : index to i64
        %33 = arith.index_cast %10 : index to i64
        %34 = airrt.dma_memcpy_nd(%c45_i32, %30, %31, %arg2[%c0_i64, %c0_i64, %32, %33], [%c1_i64, %c1_i64, %c64_i64, %c256_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId48} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %35 = airrt.wait_all %16, %22, %28, %34 : !airrt.event
        %p = airrt.segment_load "segment_0" : i64
      }
    }
    return
  }
}

// -----

// AIRRt alloc / dealloc.

// CHECK-LABEL: func.func @func17
// CHECK-NEXT: return
module {
  func.func @func17() {
    %0 = airrt.alloc : memref<8x16xi32, 1 : i32>
    %1 = airrt.alloc : memref<32x16xi32, 1 : i32>
    %2 = airrt.alloc : memref<8x32xi32, 1 : i32>
    airrt.dealloc %0 : memref<8x16xi32, 1 : i32>
    airrt.dealloc %1 : memref<32x16xi32, 1 : i32>
    airrt.dealloc %2 : memref<8x32xi32, 1 : i32>
    return
  }
}

// -----

// Avoid folding for loop into wrap-and-stride, if the outcome is stride > 1M; unroll BDs instead.

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK:  aiex.runtime_sequence @func18(%[[ARG0:.*]]: memref<8192x32768xi32>)
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 0, 0][1, 4, 64, 64][0, 64, 32768, 1]) {id = 0 : i64, metadata = @airMemcpyId26} : memref<8192x32768xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 64, 0][1, 4, 64, 64][0, 64, 32768, 1]) {id = 1 : i64, metadata = @airMemcpyId26} : memref<8192x32768xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 128, 0][1, 4, 64, 64][0, 64, 32768, 1]) {id = 2 : i64, metadata = @airMemcpyId26} : memref<8192x32768xi32>
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 192, 0][1, 4, 64, 64][0, 64, 32768, 1]) {id = 3 : i64, metadata = @airMemcpyId26} : memref<8192x32768xi32>

#map = affine_map<()[s0] -> (s0 * 64)>
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId26(S2MM, 0, 0)
    memref.global "public" @airMemcpyId26 : memref<64x64xi32, 1>
  } {sym_name = "segment_0"}
  func.func @func18() {
    %c32768_i64 = arith.constant 32768 : i64
    %c8_i64 = arith.constant 8 : i64
    %c512_i64 = arith.constant 512 : i64
    %c64_i64 = arith.constant 64 : i64
    %c26_i32 = arith.constant 26 : i32
    %c15_i32 = arith.constant 15 : i32
    %c14_i32 = arith.constant 14 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() : memref<8192x32768xi32>
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 4 {
        %10 = affine.apply #map()[%arg3]
        %11 = affine.apply #map()[%arg4]
        %12 = arith.index_cast %arg3 : index to i64
        %13 = arith.index_cast %arg4 : index to i64
        %14 = arith.index_cast %10 : index to i64
        %15 = arith.index_cast %11 : index to i64
        %16 = airrt.dma_memcpy_nd(%c26_i32, %12, %13, %alloc[%c0_i64, %c0_i64, %14, %15], [%c1_i64, %c1_i64, %c64_i64, %c64_i64], [%c0_i64, %c0_i64, %c32768_i64]) {metadata = @airMemcpyId26} : (i32, i64, i64, memref<8192x32768xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %p = airrt.segment_load "segment_0" : i64
      }
    }
    return
  }
}

// -----

// Big memref.

// CHECK-LABEL: aie.device(npu1_1col)
// CHECK:  aiex.runtime_sequence @func19(%[[ARG0:.*]]: memref<308x2432xi32>)
// CHECK:  aiex.npu.dma_memcpy_nd(0, 0, %[[ARG0]][0, 0, 0, 0][4, 19, 28, 128][0, 128, 2432, 1]) {id = 0 : i64, metadata = @airMemcpyId26} : memref<308x2432xi32>

#map = affine_map<()[s0] -> (s0 * 64)>
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId26(S2MM, 0, 0)
    memref.global "public" @airMemcpyId26 : memref<64x64xi32, 1>
  } {sym_name = "segment_0"}
  func.func @func19() {
    %c2432_i64 = arith.constant 2432 : i64
    %c8_i64 = arith.constant 8 : i64
    %c512_i64 = arith.constant 512 : i64
    %c128_i64 = arith.constant 128 : i64
    %c64_i64 = arith.constant 64 : i64
    %c28_i64 = arith.constant 28 : i64
    %c26_i32 = arith.constant 26 : i32
    %c19_i64 = arith.constant 19 : i64
    %c14_i32 = arith.constant 14 : i32
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %alloc = memref.alloc() : memref<308x2432xi32>
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 1 {
        %10 = affine.apply #map()[%arg3]
        %11 = affine.apply #map()[%arg4]
        %12 = arith.index_cast %arg3 : index to i64
        %13 = arith.index_cast %arg4 : index to i64
        %14 = arith.index_cast %10 : index to i64
        %15 = arith.index_cast %11 : index to i64
        %16 = airrt.dma_memcpy_nd(%c26_i32, %12, %13, %alloc[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c19_i64, %c28_i64, %c128_i64], [%c0_i64, %c128_i64, %c2432_i64]) {metadata = @airMemcpyId26} : (i32, i64, i64, memref<308x2432xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
        %p = airrt.segment_load "segment_0" : i64
      }
    }
    return
  }
}

// -----

// Purge scf.parallel op which contains only no-ops.

// CHECK-LABEL: func20
// CHECK: return
module {
  func.func @func20() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c152 = arith.constant 152 : index
    %51 = airrt.wait_all : !airrt.event
    %52 = scf.for %arg3 = %c0 to %c152 step %c1 iter_args(%arg4 = %51) -> (!airrt.event) {
      %61 = airrt.wait_all : !airrt.event
      %62 = airrt.wait_all %arg4, %61 : !airrt.event
      %63 = airrt.wait_all : !airrt.event
      %64 = airrt.wait_all %arg4, %63 : !airrt.event
      %65 = scf.parallel (%arg5) = (%c0) to (%c2) step (%c1) init (%arg4) -> !airrt.event {
        %66 = airrt.wait_all : !airrt.event
        %67 = airrt.wait_all %arg4, %66 : !airrt.event
        scf.reduce(%67 : !airrt.event) {
        ^bb0(%arg6: !airrt.event, %arg7: !airrt.event):
          %68 = airrt.wait_all %arg6, %arg7 : !airrt.event
          scf.reduce.return %68 : !airrt.event
        }
      }
      scf.yield %65 : !airrt.event
    }
    return
  }
}

// -----

// Outermost wrap must be in range [1:64] for AIE2.

// CHECK-LABEL: func21
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 0][38, 2, 64, 32][77824, 32, 1216, 1]) {id = 0 : i64, metadata = @airMemcpyId10} : memref<11829248xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 2957312][38, 2, 64, 32][77824, 32, 1216, 1]) {id = 1 : i64, metadata = @airMemcpyId10} : memref<11829248xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 5914624][38, 2, 64, 32][77824, 32, 1216, 1]) {id = 2 : i64, metadata = @airMemcpyId10} : memref<11829248xi32>
// CHECK: aiex.npu.dma_memcpy_nd(0, 0, %arg0[0, 0, 0, 8871936][38, 2, 64, 32][77824, 32, 1216, 1]) {id = 3 : i64, metadata = @airMemcpyId10} : memref<11829248xi32>
// CHECK: return

#map = affine_map<()[s0] -> (s0 * 128)>
module {
  aie.device(npu1_4col) {
    aie.shim_dma_allocation @airMemcpyId10(MM2S, 1, 0)
    memref.global "public" @airMemcpyId10 : memref<1x2x64x64xbf16, 1 : i32>
  } {sym_name = "matmul_bf16_large_dispatch_0_matmul_308x2432x9728_bf16_0"}
  airrt.module_metadata{
  }
  func.func @func21(%arg0: memref<9728x2432xbf16>) {
    %c2_i64 = arith.constant 2 : i64
    %c2432_i64 = arith.constant 2432 : i64
    %c155648_i64 = arith.constant 155648 : i64
    %c152_i64 = arith.constant 152 : i64
    %c64_i64 = arith.constant 64 : i64
    %c10_i32 = arith.constant 10 : i32
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 1 {
        %0 = affine.apply #map()[%arg4]
        %1 = arith.index_cast %arg3 : index to i64
        %2 = arith.index_cast %arg4 : index to i64
        %3 = arith.index_cast %0 : index to i64
        %4 = airrt.dma_memcpy_nd(%c10_i32, %1, %2, %arg0[%c0_i64, %c0_i64, %c0_i64, %3], [%c152_i64, %c2_i64, %c64_i64, %c64_i64], [%c155648_i64, %c64_i64, %c2432_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<9728x2432xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
      }
    }
    return
  }
}
