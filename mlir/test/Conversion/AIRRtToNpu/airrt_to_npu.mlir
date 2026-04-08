//===- airrt_to_npu.mlir ---------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -canonicalize -cse --split-input-file %s | FileCheck %s

// Synchronous airrt.dma_memcpy_nd.

// CHECK-LABEL: aie.device(npu1_1col) @segment0
// CHECK: aie.shim_dma_allocation @airMemcpyId7(%shim_noc_tile_0_0, S2MM, 0)
// CHECK: aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
// CHECK: aie.runtime_sequence @func0(%[[VAL_0:.*]]: memref<64xi32>, %[[VAL_1:.*]]: memref<64xi32>) {
// CHECK:   aiex.dma_configure_task_for @airMemcpyId2 {
// CHECK:     aie.dma_bd(%[[VAL_0]] : memref<64xi32>, 0, 64
// CHECK:   aiex.dma_start_task
// CHECK:   %[[T1:.*]] = aiex.dma_configure_task_for @airMemcpyId7 {
// CHECK:     aie.dma_bd(%[[VAL_1]] : memref<64xi32>, 0, 64
// CHECK:   } {issue_token = true}
// CHECK:   aiex.dma_start_task(%[[T1]])
// CHECK: }

module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId7(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
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
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
    %p = airrt.segment_load "segment0" : i64
    %c7_i32 = arith.constant 7 : i32
    airrt.dma_memcpy_nd(%c7_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
    return
  }
}

// -----

// Asynchronous airrt.dma_memcpy_nd

// CHECK-LABEL: aie.device(npu1_1col) @segment0 {
// CHECK: aie.shim_dma_allocation @airMemcpyId7(%shim_noc_tile_0_0, S2MM, 0)
// CHECK: aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
// CHECK: aie.runtime_sequence @func1(%[[VAL_0:.*]]: memref<64xi32>, %[[VAL_1:.*]]: memref<64xi32>) {
// CHECK:   aiex.dma_configure_task_for @airMemcpyId2 {
// CHECK:     aie.dma_bd(%[[VAL_0]] : memref<64xi32>, 0, 64
// CHECK:   aiex.dma_start_task
// CHECK:   %[[T1:.*]] = aiex.dma_configure_task_for @airMemcpyId7 {
// CHECK:     aie.dma_bd(%[[VAL_1]] : memref<64xi32>, 0, 64
// CHECK:   } {issue_token = true}
// CHECK:   aiex.dma_start_task(%[[T1]])
// CHECK:   aiex.dma_await_task(%[[T1]])
// CHECK: }
// CHECK: }

module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId7(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
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
    %0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %p = airrt.segment_load "segment0" : i64
    %1 = airrt.wait_all : !airrt.event
    %c7_i32 = arith.constant 7 : i32
    %2 = airrt.dma_memcpy_nd(%c7_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %0, %2
    return
  }
}

// -----

// air.launch iteration space unrolling

// CHECK-LABEL: aie.device(npu1_1col) @segment_0 {
// CHECK: aie.shim_dma_allocation @airMemcpyId16(%shim_noc_tile_0_0, S2MM, 0)
// CHECK: aie.shim_dma_allocation @airMemcpyId5(%shim_noc_tile_0_0, MM2S, 0)
// CHECK: aie.runtime_sequence @func2(%[[VAL_0:.*]]: memref<32x32xi32>, %[[VAL_1:.*]]: memref<32x32xi32>, %[[VAL_2:.*]]: memref<32x32xi32>) {
// CHECK:   aiex.dma_configure_task_for @airMemcpyId5 {
// CHECK:     aie.dma_bd(%[[VAL_2]] : memref<32x32xi32>, 0, 1024
// CHECK:   aiex.dma_start_task
// CHECK:   aiex.dma_configure_task_for @airMemcpyId5 {
// CHECK:     aie.dma_bd(%[[VAL_0]] : memref<32x32xi32>, 0, 1024
// CHECK:   aiex.dma_start_task
// CHECK:   aiex.dma_configure_task_for @airMemcpyId5 {
// CHECK:     aie.dma_bd(%[[VAL_1]] : memref<32x32xi32>, 0, 1024
// CHECK:   aiex.dma_start_task
// CHECK:   %[[T3:.*]] = aiex.dma_configure_task_for @airMemcpyId16 {
// CHECK:     aie.dma_bd(%[[VAL_2]] : memref<32x32xi32>, 0, 1024
// CHECK:   } {issue_token = true}
// CHECK:   aiex.dma_start_task(%[[T3]])
// CHECK:   aiex.dma_await_task(%[[T3]])
// CHECK: }
// CHECK: }

#map = affine_map<(d0)[] -> (d0 * 32)>
module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId16(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @airMemcpyId5(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @airMemcpyId6(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @airMemcpyId7(%shim_noc_tile_0_0, MM2S, 0)
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
        %0 = affine.apply #map(%arg3)[]
        %1 = affine.apply #map(%arg4)[]
        %2 = arith.index_cast %arg3 : index to i64
        %3 = arith.index_cast %arg4 : index to i64
        %4 = arith.index_cast %0 : index to i64
        %5 = arith.index_cast %1 : index to i64
        %6 = airrt.dma_memcpy_nd(%c5_i32, %2, %3, %arg2[%c0_i64, %c0_i64, %4, %5], [%c1_i64, %c1_i64, %c32_i64, %c32_i64], [%c0_i64, %c0_i64, %c32_i64, %c1_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<32x32xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
        %7 = airrt.dma_memcpy_nd(%c6_i32, %2, %3, %arg0[%c0_i64, %c0_i64, %4, %c0_i64], [%c1_i64, %c1_i64, %c32_i64, %c32_i64], [%c0_i64, %c0_i64, %c32_i64, %c1_i64]) {metadata = @airMemcpyId6} : (i32, i64, i64, memref<32x32xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
        %8 = airrt.dma_memcpy_nd(%c7_i32, %2, %3, %arg1[%c0_i64, %c0_i64, %c0_i64, %5], [%c1_i64, %c1_i64, %c32_i64, %c32_i64], [%c0_i64, %c0_i64, %c32_i64, %c1_i64]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<32x32xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
        %9 = airrt.dma_memcpy_nd(%c16_i32, %2, %3, %arg2[%c0_i64, %c0_i64, %4, %5], [%c1_i64, %c1_i64, %c32_i64, %c32_i64], [%c0_i64, %c0_i64, %c32_i64, %c1_i64]) {metadata = @airMemcpyId16} : (i32, i64, i64, memref<32x32xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
        airrt.wait_all %6, %7, %8, %9
        %p = airrt.segment_load "segment_0" : i64
      }
    }
    return
  }
}

// -----

// air.launch iteration space unrolling 2

// CHECK-LABEL: aie.device(npu1_2col) @segment_0 {
// CHECK:  aie.runtime_sequence @func3(%[[ARG0:.*]]: memref<8x8xi32>)
// CHECK:  %[[T0:.*]] = aiex.dma_configure_task_for @airMemcpyId14 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8x8xi32>, 0, 16
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task(%[[T0]])
// CHECK:  aiex.dma_await_task(%[[T0]])
// CHECK:  %[[T1:.*]] = aiex.dma_configure_task_for @airMemcpyId14 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8x8xi32>, 4, 16
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task(%[[T1]])
// CHECK:  aiex.dma_await_task(%[[T1]])
// CHECK:  %[[T2:.*]] = aiex.dma_configure_task_for @airMemcpyId14 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8x8xi32>, 32, 16
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task(%[[T2]])
// CHECK:  aiex.dma_await_task(%[[T2]])
// CHECK:  %[[T3:.*]] = aiex.dma_configure_task_for @airMemcpyId14 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8x8xi32>, 36, 16
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task(%[[T3]])
// CHECK:  aiex.dma_await_task(%[[T3]])

#map = affine_map<(d0)[] -> (d0 * 4)>
module {
  aie.device(npu1_2col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId14(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @airMemcpyId14_1(%shim_noc_tile_0_0, S2MM, 1)
    aie.shim_dma_allocation @airMemcpyId14_2(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @airMemcpyId14_3(%shim_noc_tile_0_0, S2MM, 1)
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
            %0 = affine.apply #map(%arg3)[]
            %1 = affine.apply #map(%arg4)[]
            %2 = arith.index_cast %arg3 : index to i64
            %3 = arith.index_cast %arg4 : index to i64
            %4 = arith.index_cast %0 : index to i64
            %5 = arith.index_cast %1 : index to i64
            %6 = airrt.dma_memcpy_nd(%c14_i32, %2, %3, %arg0[%c0_i64, %c0_i64, %4, %5], [%c1_i64, %c1_i64, %c4_i64, %c4_i64], [%c0_i64, %c0_i64, %c8_i64, %c1_i64]) {metadata = @airMemcpyId14} : (i32, i64, i64, memref<8x8xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
            airrt.wait_all %6
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

// CHECK-LABEL: aie.device(npu1_2col) @segment_0
// CHECK:  aie.runtime_sequence @func4(%[[ARG0:.*]]: memref<8x8xi32>)
// CHECK:  %[[T0:.*]] = aiex.dma_configure_task_for @air_channel_1 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8x8xi32>, 0, 16
// CHECK:  aiex.dma_start_task(%[[T0]])
// CHECK:  aiex.dma_await_task(%[[T0]])
// CHECK:  %[[T1:.*]] = aiex.dma_configure_task_for @air_channel_1 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8x8xi32>, 4, 16
// CHECK:  aiex.dma_start_task(%[[T1]])
// CHECK:  aiex.dma_await_task(%[[T1]])
// CHECK:  %[[T2:.*]] = aiex.dma_configure_task_for @air_channel_1 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8x8xi32>, 32, 16
// CHECK:  aiex.dma_start_task(%[[T2]])
// CHECK:  aiex.dma_await_task(%[[T2]])
// CHECK:  %[[T3:.*]] = aiex.dma_configure_task_for @air_channel_1 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8x8xi32>, 36, 16
// CHECK:  aiex.dma_start_task(%[[T3]])
// CHECK:  aiex.dma_await_task(%[[T3]])

#map = affine_map<(d0)[] -> (d0 * 4)>
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
            %0 = affine.apply #map(%arg3)[]
            %1 = affine.apply #map(%arg4)[]
            %2 = arith.index_cast %arg3 : index to i64
            %3 = arith.index_cast %arg4 : index to i64
            %4 = arith.index_cast %0 : index to i64
            %5 = arith.index_cast %1 : index to i64
            %6 = airrt.dma_memcpy_nd(%c14_i32, %2, %3, %arg0[%c0_i64, %c0_i64, %4, %5], [%c1_i64, %c1_i64, %c4_i64, %c4_i64], [%c0_i64, %c0_i64, %c8_i64, %c1_i64]) {metadata = @air_channel_1} : (i32, i64, i64, memref<8x8xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
            airrt.wait_all %6
          }
        }
        %p = airrt.segment_load "segment_0" : i64
      }
    }
    return
  }
}

// -----

// check that lowering works for the herd_load case

// CHECK-LABEL: @func8
// CHECK: aiex.dma_configure_task_for @airMemcpyId7
// CHECK: aiex.dma_start_task
module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId7(%shim_noc_tile_0_0, S2MM, 0)
  } {sym_name = "herd"}
  airrt.module_metadata{
    airrt.segment_metadata attributes {sym_name = ""} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 0 : i64, sym_name = "herd"}
    }
  }
  func.func @func8(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c7_i32 = arith.constant 7 : i32
    %c64_i64 = arith.constant 64 : i64
    %p = airrt.herd_load "herd" () : () -> i64
    airrt.dma_memcpy_nd(%c7_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
    return
  }
}

// -----

// 16-bit type conversion

// CHECK-LABEL: aie.runtime_sequence @func11
// CHECK-SAME: %arg0: memref<128x128xbf16>
// CHECK-NEXT: aiex.dma_configure_task_for @md0 {
// CHECK:        aie.dma_bd(%arg0 : memref<128x128xbf16>, 0, 4096
// CHECK: } {repeat_count = 3 : i32}
// CHECK: aiex.dma_start_task
module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @md0(%shim_noc_tile_0_0, MM2S, 0)
    func.func @func11(%arg0: memref<128x128xbf16>, %arg1: memref<128x128xbf16>) {
      %c0_i32 = arith.constant 0 : i32
      %c0_i64 = arith.constant 0 : i64
      %c4_i64 = arith.constant 4 : i64
      %c32_i64 = arith.constant 32 : i64
      %c128_i64 = arith.constant 128 : i64
      %c4096_i64 = arith.constant 4096 : i64
      %c1_i64 = arith.constant 1 : i64
      airrt.dma_memcpy_nd(%c0_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c4_i64, %c4_i64, %c32_i64, %c32_i64], [%c4096_i64, %c32_i64, %c128_i64, %c1_i64]) {metadata = @md0} : (i32, i64, i64, memref<128x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
      return
    }
  }
}

// -----

// 16-bit conversion with dma operands that aren't function arguments

// CHECK-LABEL: aie.runtime_sequence @func12
// CHECK-SAME: %arg0: memref<32xbf16>
// CHECK-NEXT: aiex.dma_configure_task_for @md0 {
// CHECK:        aie.dma_bd(%arg0 : memref<32xbf16>, 0, 32
// CHECK:      aiex.dma_start_task
module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @md0(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "segment_0"}
  func.func @func12() {
    %c1_i32 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %alloc = memref.alloc() : memref<32xbf16>
    airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %alloc[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @md0} : (i32, i64, i64, memref<32xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
    %p = airrt.segment_load "segment_0" : i64
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
//    aiex.npu.dma_memcpy_nd(%arg0 ...
//    return
//  }
//
// PR 447 relocates the memref.assume_alignment op so that calling
// `air-opt -airrt-to-npu` on the function in the test now produces:
//
//  func.func @func12(%arg0: memref<16xi32>) {
//    memref.assume_alignment %arg0, 64 : memref<16xi32>
//    aiex.npu.dma_memcpy_nd(%arg0 ...
//    return
//  }
//
// The key difference is that memref.alloc is removed.

// CHECK-LABEL: aie.runtime_sequence @func13
// CHECK-SAME: memref<32xbf16>
// CHECK-NOT: memref.alloc
// CHECK: aiex.dma_configure_task_for @md0
// CHECK: aie.dma_bd
// CHECK: aiex.dma_start_task
module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @md0(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "segment_0"}
  func.func @func13() {
    %c1_i32 = arith.constant 1 : i32
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c32_i64 = arith.constant 32 : i64
    %alloc = memref.alloc() : memref<32xbf16>
    // assert that the alignment of %alloc is 64 bits:
    memref.assume_alignment %alloc, 64 : memref<32xbf16>
    airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %alloc[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c32_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @md0} : (i32, i64, i64, memref<32xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
    %p = airrt.segment_load "segment_0" : i64
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
// CHECK:  aie.runtime_sequence @func18(%[[ARG0:.*]]: memref<8192x32768xi32>)
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 0, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 64, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 128, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 192, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 2097152, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 2097216, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 2097280, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 2097344, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 4194304, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 4194368, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 4194432, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 4194496, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 6291456, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 6291520, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 6291584, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task
// CHECK:  aiex.dma_configure_task_for @airMemcpyId26 {
// CHECK:    aie.dma_bd(%[[ARG0]] : memref<8192x32768xi32>, 6291648, 4096, [<size = 64, stride = 32768>, <size = 64, stride = 1>])
// CHECK:  } {issue_token = true}
// CHECK:  aiex.dma_start_task

#map = affine_map<(d0)[] -> (d0 * 64)>
module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId26(%tile_0_0, S2MM, 0)
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
        %10 = affine.apply #map(%arg3)[]
        %11 = affine.apply #map(%arg4)[]
        %12 = arith.index_cast %arg3 : index to i64
        %13 = arith.index_cast %arg4 : index to i64
        %14 = arith.index_cast %10 : index to i64
        %15 = arith.index_cast %11 : index to i64
        %16 = airrt.dma_memcpy_nd(%c26_i32, %12, %13, %alloc[%c0_i64, %c0_i64, %14, %15], [%c1_i64, %c1_i64, %c64_i64, %c64_i64], [%c0_i64, %c0_i64, %c32768_i64, %c1_i64]) {metadata = @airMemcpyId26} : (i32, i64, i64, memref<8192x32768xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
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

// CHECK-LABEL: aie.runtime_sequence @func21
// CHECK: aiex.dma_configure_task_for @airMemcpyId10 {
// CHECK:   aie.dma_bd(%arg0 : memref<9728x2432xbf16>, 0, 8192, [<size = 38, stride = 155648>, <size = 2, stride = 64>, <size = 64, stride = 2432>, <size = 64, stride = 1>])
// CHECK: } {repeat_count = 37 : i32}
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId10 {
// CHECK:   aie.dma_bd(%arg0 : memref<9728x2432xbf16>, 5914624, 8192, [<size = 38, stride = 155648>, <size = 2, stride = 64>, <size = 64, stride = 2432>, <size = 64, stride = 1>])
// CHECK: } {repeat_count = 37 : i32}
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId10 {
// CHECK:   aie.dma_bd(%arg0 : memref<9728x2432xbf16>, 11829248, 8192, [<size = 38, stride = 155648>, <size = 2, stride = 64>, <size = 64, stride = 2432>, <size = 64, stride = 1>])
// CHECK: } {repeat_count = 37 : i32}
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId10 {
// CHECK:   aie.dma_bd(%arg0 : memref<9728x2432xbf16>, 17743872, 8192, [<size = 38, stride = 155648>, <size = 2, stride = 64>, <size = 64, stride = 2432>, <size = 64, stride = 1>])
// CHECK: } {repeat_count = 37 : i32}
// CHECK: aiex.dma_start_task

#map = affine_map<(d0)[] -> (d0 * 128)>
module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId10(%shim_noc_tile_0_0, MM2S, 1)
  } {sym_name = "segment_0"}
  airrt.module_metadata{
  }
  func.func @func21(%arg0: memref<9728x2432xbf16>) {
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c2432_i64 = arith.constant 2432 : i64
    %c155648_i64 = arith.constant 155648 : i64
    %c152_i64 = arith.constant 152 : i64
    %c64_i64 = arith.constant 64 : i64
    %c10_i32 = arith.constant 10 : i32
    %c0_i64 = arith.constant 0 : i64
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 1 {
        %0 = affine.apply #map(%arg4)[]
        %1 = arith.index_cast %arg3 : index to i64
        %2 = arith.index_cast %arg4 : index to i64
        %3 = arith.index_cast %0 : index to i64
        %4 = airrt.dma_memcpy_nd(%c10_i32, %1, %2, %arg0[%c0_i64, %c0_i64, %c0_i64, %3], [%c152_i64, %c2_i64, %c64_i64, %c64_i64], [%c155648_i64, %c64_i64, %c2432_i64, %c1_i64]) {metadata = @airMemcpyId10} : (i32, i64, i64, memref<9728x2432xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
        %p = airrt.segment_load "segment_0" : i64
      }
    }
    return
  }
}

// -----

// Test: Await ops are placed at WaitAllOp locations (clustered), not all at end.
// This test verifies that await ops appear after their corresponding start ops
// at the synchronization point (wait_all), not accumulated at the function end.

// CHECK-LABEL: aie.runtime_sequence @await_clustering
// CHECK: %[[T0:.*]] = aiex.dma_configure_task_for @input_channel
// CHECK: aiex.dma_start_task(%[[T0]])
// CHECK: %[[T1:.*]] = aiex.dma_configure_task_for @output_channel {
// CHECK: } {issue_token = true}
// CHECK: aiex.dma_start_task(%[[T1]])
// First synchronization point: free for input, await for output (order follows WaitAllOp operand order)
// CHECK: aiex.dma_free_task(%[[T0]])
// CHECK: aiex.dma_await_task(%[[T1]])
// Second batch of DMAs
// CHECK: %[[T2:.*]] = aiex.dma_configure_task_for @input_channel
// CHECK: aiex.dma_start_task(%[[T2]])
// CHECK: %[[T3:.*]] = aiex.dma_configure_task_for @output_channel {
// CHECK: } {issue_token = true}
// CHECK: aiex.dma_start_task(%[[T3]])
// Second synchronization point: free for input, await for output
// CHECK: aiex.dma_free_task(%[[T2]])
// CHECK: aiex.dma_await_task(%[[T3]])

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @input_channel(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @output_channel(%tile_0_0, S2MM, 0)
  } {sym_name = "segment_0"}
  airrt.module_metadata{}
  func.func @await_clustering(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    // First batch: input DMA (MM2S) and output DMA (S2MM)
    %0 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @input_channel} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @output_channel} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    // First synchronization point
    airrt.wait_all %0, %1
    // Second batch: another pair of DMAs
    %2 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @input_channel} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @output_channel} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    // Second synchronization point
    airrt.wait_all %2, %3
    %p = airrt.segment_load "segment_0" : i64
    return
  }
}

// -----

// Test: MM2S (input) channels get dma_free_task (no wait), S2MM (output) channels get dma_await_task.
// This verifies BD ID management - MM2S tasks need their BDs freed for reuse without waiting.

// CHECK-LABEL: aie.runtime_sequence @free_task_for_mm2s
// Input channel (MM2S) - no issue_token, gets free_task
// CHECK: %[[INPUT:.*]] = aiex.dma_configure_task_for @mm2s_input {
// CHECK-NOT: issue_token
// CHECK: aiex.dma_start_task(%[[INPUT]])
// Output channel (S2MM) - has issue_token, gets await_task
// CHECK: %[[OUTPUT:.*]] = aiex.dma_configure_task_for @s2mm_output {
// CHECK: } {issue_token = true}
// CHECK: aiex.dma_start_task(%[[OUTPUT]])
// At sync point: free for MM2S (input), await for S2MM (output) - order follows WaitAllOp operand order
// CHECK: aiex.dma_free_task(%[[INPUT]])
// CHECK: aiex.dma_await_task(%[[OUTPUT]])

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @mm2s_input(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @s2mm_output(%tile_0_0, S2MM, 0)
  } {sym_name = "segment_0"}
  airrt.module_metadata{}
  func.func @free_task_for_mm2s(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    // MM2S (host to device, input) - should get dma_free_task
    %0 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @mm2s_input} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    // S2MM (device to host, output) - should get dma_await_task
    %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @s2mm_output} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %0, %1
    %p = airrt.segment_load "segment_0" : i64
    return
  }
}

// -----

// Test: Multiple MM2S channels all get dma_free_task at sync point.
// Verifies that all input DMAs get their BDs freed for reuse.

// CHECK-LABEL: aie.runtime_sequence @multiple_mm2s_free
// CHECK: %[[A:.*]] = aiex.dma_configure_task_for @input_a
// CHECK: aiex.dma_start_task(%[[A]])
// CHECK: %[[B:.*]] = aiex.dma_configure_task_for @input_b
// CHECK: aiex.dma_start_task(%[[B]])
// CHECK: %[[C:.*]] = aiex.dma_configure_task_for @output_c {
// CHECK: } {issue_token = true}
// CHECK: aiex.dma_start_task(%[[C]])
// At sync point: free for inputs (in order), await for output (order follows WaitAllOp operand order)
// CHECK: aiex.dma_free_task(%[[A]])
// CHECK: aiex.dma_free_task(%[[B]])
// CHECK: aiex.dma_await_task(%[[C]])

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @input_a(%tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @input_b(%tile_0_0, MM2S, 1)
    aie.shim_dma_allocation @output_c(%tile_0_0, S2MM, 0)
  } {sym_name = "segment_0"}
  airrt.module_metadata{}
  func.func @multiple_mm2s_free(%arg0: memref<64xi32>, %arg1: memref<64xi32>, %arg2: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    // Two MM2S inputs
    %0 = airrt.dma_memcpy_nd(%c1_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @input_a} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @input_b} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    // One S2MM output
    %2 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @output_c} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64]) : !airrt.event
    airrt.wait_all %0, %1, %2
    %p = airrt.segment_load "segment_0" : i64
    return
  }
}

// -----

// Packet attribute transfer from airrt.dma_memcpy_nd to aie.dma_bd.
// The MM2S DMA with a packet attribute should produce a dma_bd with the packet
// field set. The S2MM DMA without a packet attribute should not.

// CHECK-LABEL: aie.device(npu1_1col) @segment0
// CHECK: aie.runtime_sequence @packet_attr_transfer(%[[VAL_0:.*]]: memref<64xi32>, %[[VAL_1:.*]]: memref<64xi32>) {
// CHECK:   aiex.dma_configure_task_for @airMemcpyId2 {
// CHECK:     aie.dma_bd(%[[VAL_0]] : memref<64xi32>, 0, 64,{{.*}}) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>}
// CHECK:   }
// CHECK:   aiex.dma_start_task
// CHECK:   %[[T1:.*]] = aiex.dma_configure_task_for @airMemcpyId7 {
// CHECK-NOT: packet
// CHECK:     aie.dma_bd(%[[VAL_1]] : memref<64xi32>, 0, 64
// CHECK:   } {issue_token = true}
// CHECK:   aiex.dma_start_task(%[[T1]])
// CHECK: }

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId7(%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @airMemcpyId2(%tile_0_0, MM2S, 0)
  } {sym_name = "segment0"}
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  func.func @packet_attr_transfer(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c7_i32 = arith.constant 7 : i32
    // MM2S DMA with packet attribute (direct L3->L1 packet-switched flow)
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId2, packet = #aie.packet_info<pkt_type = 0, pkt_id = 3>} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
    %p = airrt.segment_load "segment0" : i64
    // S2MM DMA without packet attribute
    airrt.dma_memcpy_nd(%c7_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
    return
  }
}

// -----

// Packet attribute transfer with multi-dimensional DMA (dims + packet).

// CHECK-LABEL: aie.device(npu1_1col) @segment0
// CHECK: aie.runtime_sequence @packet_attr_with_dims(%[[VAL_0:.*]]: memref<32x64xi32>, %[[VAL_1:.*]]: memref<64xi32>) {
// CHECK:   aiex.dma_configure_task_for @airMemcpyId2 {
// CHECK:     aie.dma_bd(%[[VAL_0]] : memref<32x64xi32>, 0, 128, [<size = 2, stride = 64>{{.*}}]) {packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>}
// CHECK:   }
// CHECK:   aiex.dma_start_task

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId7(%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @airMemcpyId2(%tile_0_0, MM2S, 0)
  } {sym_name = "segment0"}
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  func.func @packet_attr_with_dims(%arg0: memref<32x64xi32>, %arg1: memref<64xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c7_i32 = arith.constant 7 : i32
    // MM2S DMA with packet attribute and non-trivial dimensions
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c2_i64, %c64_i64], [%c0_i64, %c0_i64, %c64_i64, %c0_i64]) {metadata = @airMemcpyId2, packet = #aie.packet_info<pkt_type = 0, pkt_id = 5>} : (i32, i64, i64, memref<32x64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
    %p = airrt.segment_load "segment0" : i64
    airrt.dma_memcpy_nd(%c7_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
    return
  }
}

// -----

// stride=0 at dim 1 (broadcast pattern) should be folded into repeat_count
// and NOT passed to aie.dma_bd dimensions. Regression test for issue #1484.

// CHECK-LABEL: aie.runtime_sequence @broadcast_stride_zero
// CHECK-SAME: %[[ARG0:.*]]: memref<64xbf16>
// CHECK-NEXT: aiex.dma_configure_task_for @airMemcpyId2 {
// CHECK:        aie.dma_bd(%[[ARG0]] : memref<64xbf16>, 0, 64, [<size = 64, stride = 1>])
// CHECK: } {repeat_count = 3 : i32}
// CHECK: aiex.dma_start_task
module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
    func.func @broadcast_stride_zero(%arg0: memref<64xbf16>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %c64_i64 = arith.constant 64 : i64
      %c2_i32 = arith.constant 2 : i32
      // sizes=[2, 2, 1, 64] strides=[0, 0, 0, 1]
      // dim 0: stride=0 size=2 -> repeat_count=1
      // dim 1: stride=0 size=2 -> should fold into repeat_count, NOT pass to BD
      // Expected: repeat_count = (1+1)*2 - 1 = 3, BD dims = [<size=64, stride=1>]
      airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c2_i64, %c2_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
      return
    }
  }
}

// -----

// stride=0 at dim 2 only (broadcast pattern with non-zero stride at dim 1).
// Regression test for issue #1484.

// CHECK-LABEL: aie.runtime_sequence @broadcast_stride_zero_dim2
// CHECK-SAME: %[[ARG0:.*]]: memref<256xbf16>
// CHECK-NEXT: aiex.dma_configure_task_for @airMemcpyId3 {
// CHECK:        aie.dma_bd(%[[ARG0]] : memref<256xbf16>, 0, 128, [<size = 2, stride = 128>, <size = 64, stride = 1>])
// CHECK: } {repeat_count = 2 : i32}
// CHECK: aiex.dma_start_task
module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId3(%shim_noc_tile_0_0, MM2S, 0)
    func.func @broadcast_stride_zero_dim2(%arg0: memref<256xbf16>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %c3_i64 = arith.constant 3 : i64
      %c64_i64 = arith.constant 64 : i64
      %c128_i64 = arith.constant 128 : i64
      %c3_i32 = arith.constant 3 : i32
      // sizes=[1, 2, 3, 64] strides=[0, 128, 0, 1]
      // dim 0: size=1 -> repeat_count=0 (no repeat)
      // dim 1: stride=128 size=2 -> BD dim (non-zero stride)
      // dim 2: stride=0 size=3 -> fold into repeat_count: (0+1)*3-1=2
      // dim 3: stride=1 size=64 -> BD dim
      // Expected: repeat_count=2, BD dims=[<size=2, stride=128>, <size=64, stride=1>]
      // transferLen = 2*3*64=384 -> 384/3=128 per BD execution (after folding dim 2)
      airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c2_i64, %c3_i64, %c64_i64], [%c0_i64, %c128_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId3} : (i32, i64, i64, memref<256xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
      return
    }
  }
}

// -----

// stride=0 at both dim 1 and dim 2 (nested broadcast). Tests that nested
// repeat folding produces correct repeat_count. Regression test for issue #1484.

// CHECK-LABEL: aie.runtime_sequence @broadcast_stride_zero_nested
// CHECK-SAME: %[[ARG0:.*]]: memref<64xbf16>
// CHECK-NEXT: aiex.dma_configure_task_for @airMemcpyId4 {
// CHECK:        aie.dma_bd(%[[ARG0]] : memref<64xbf16>, 0, 64, [<size = 64, stride = 1>])
// CHECK: } {repeat_count = 11 : i32}
// CHECK: aiex.dma_start_task
module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, MM2S, 0)
    func.func @broadcast_stride_zero_nested(%arg0: memref<64xbf16>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c2_i64 = arith.constant 2 : i64
      %c3_i64 = arith.constant 3 : i64
      %c4_i64 = arith.constant 4 : i64
      %c64_i64 = arith.constant 64 : i64
      %c4_i32 = arith.constant 4 : i32
      // sizes=[2, 3, 2, 64] strides=[0, 0, 0, 1]
      // dim 0: stride=0 size=2 -> repeat_count=1
      // dim 1: stride=0 size=3 -> fold: repeat_count=(1+1)*3-1=5
      // dim 2: stride=0 size=2 -> fold: repeat_count=(5+1)*2-1=11
      // dim 3: stride=1 size=64 -> BD dim
      // Expected: repeat_count=11, BD dims=[<size=64, stride=1>]
      // Total transfers: 64 * 12 = 768 = 2*3*2*64. Correct.
      airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c2_i64, %c3_i64, %c2_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
      return
    }
  }
}

// -----

// 4D BD pattern with size-1 gap: sizes=[4, 1, 128, 64] strides=[65536, 64, 512, 1].
// The size-1 dim must be retained when use4thDimInBd is true, so that the 4th BD
// dimension provides iteration_stride to the hardware. Without the fix, the size-1
// dim is skipped, collapsing to 3 entries where the repeat dim appears in both
// repeat_count AND dimLayouts, causing validator rejection (4*128*64 != 8192).
// The fix retains size-1 middle dims when use4thDimInBd is true.

// CHECK-LABEL: aie.runtime_sequence @bd_repeat_dim_size1_gap
// CHECK-SAME: %[[ARG0:.*]]: memref<512x512xbf16>
// CHECK-NEXT: aiex.dma_configure_task_for @airMemcpyId5 {
// CHECK:        aie.dma_bd(%[[ARG0]] : memref<512x512xbf16>, 64, 8192, [<size = 4, stride = 65536>, <size = 1, stride = 64>, <size = 128, stride = 512>, <size = 64, stride = 1>])
// CHECK: } {repeat_count = 3 : i32}
// CHECK: aiex.dma_start_task
module {
  aie.device(npu1_1col) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId5(%shim_noc_tile_0_0, MM2S, 0)
    func.func @bd_repeat_dim_size1_gap(%arg0: memref<512x512xbf16>) {
      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c4_i64 = arith.constant 4 : i64
      %c64_i64 = arith.constant 64 : i64
      %c128_i64 = arith.constant 128 : i64
      %c512_i64 = arith.constant 512 : i64
      %c65536_i64 = arith.constant 65536 : i64
      %c5_i32 = arith.constant 5 : i32
      // sizes=[4, 1, 128, 64] strides=[65536, 64, 512, 1]
      // use4thDimInBd=true (strides[0]=65536 != 0), repeat_count=3
      // dim 0: size=4 stride=65536 -> iteration dim (4th BD dim)
      // dim 1: size=1 stride=64 -> retained (degenerate, preserves 4D layout)
      // dim 2: size=128 stride=512 -> BD dim
      // dim 3: size=64 stride=1 -> BD dim (innermost)
      // transferLen = 1*128*64 = 8192
      // Without fix: dim 1 skipped -> 3 entries -> validator: 4*128*64=32768 != 8192 FAIL
      // With fix: dim 1 retained -> 4 entries -> validator checks lowest 3: 1*128*64=8192 PASS
      airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c4_i64, %c1_i64, %c128_i64, %c64_i64], [%c65536_i64, %c64_i64, %c512_i64, %c1_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<512x512xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
      return
    }
  }
}
