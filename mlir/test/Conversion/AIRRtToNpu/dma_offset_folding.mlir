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
// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.shim_dma_allocation @airMemcpyId19(%shim_noc_tile_0_0, S2MM, 0)
// CHECK: aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, MM2S, 0)
// CHECK: aie.shim_dma_allocation @airMemcpyId5(%shim_noc_tile_0_0, MM2S, 1)

// Block 1
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 0, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 0, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_0:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_0]])
// CHECK: aiex.dma_await_task(%[[TASK_0]])

// Block 2
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 0, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 16, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_1:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_1]])
// CHECK: aiex.dma_await_task(%[[TASK_1]])

// Block 3
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 0, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 32, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_2:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_2]])
// CHECK: aiex.dma_await_task(%[[TASK_2]])

// Block 4
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 0, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 48, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_3:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_3]])
// CHECK: aiex.dma_await_task(%[[TASK_3]])

// Block 5
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 16384, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 0, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_4:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_4]])
// CHECK: aiex.dma_await_task(%[[TASK_4]])

// Block 6
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 16384, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 16, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_5:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_5]])
// CHECK: aiex.dma_await_task(%[[TASK_5]])

// Block 7
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 16384, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 32, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_6:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_6]])
// CHECK: aiex.dma_await_task(%[[TASK_6]])

// Block 8
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 16384, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 48, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_7:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_7]])
// CHECK: aiex.dma_await_task(%[[TASK_7]])

// Block 9
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 32768, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 0, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_8:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_8]])
// CHECK: aiex.dma_await_task(%[[TASK_8]])

// Block 10
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 32768, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 16, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_9:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_9]])
// CHECK: aiex.dma_await_task(%[[TASK_9]])

// Block 11
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 32768, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 32, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_10:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_10]])
// CHECK: aiex.dma_await_task(%[[TASK_10]])

// Block 12
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 32768, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 48, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_11:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_11]])
// CHECK: aiex.dma_await_task(%[[TASK_11]])

// Block 13
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 49152, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 0, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_12:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_12]])
// CHECK: aiex.dma_await_task(%[[TASK_12]])

// Block 14
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 49152, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 16, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_13:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_13]])
// CHECK: aiex.dma_await_task(%[[TASK_13]])

// Block 15
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 49152, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 32, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_14:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_14]])
// CHECK: aiex.dma_await_task(%[[TASK_14]])

// Block 16
// CHECK: aiex.dma_configure_task_for @airMemcpyId4
// CHECK: aie.dma_bd(%arg0 : memref<512x128xbf16>, 49152, 16384
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_configure_task_for @airMemcpyId5
// CHECK: aie.dma_bd(%arg1 : memref<16x8x8x64xbf16>, 48, 1024
// CHECK: aiex.dma_start_task
// CHECK: %[[TASK_15:.*]] = aiex.dma_configure_task_for @airMemcpyId19
// CHECK: {issue_token = true}
// CHECK: aiex.dma_start_task(%[[TASK_15]])
// CHECK: aiex.dma_await_task(%[[TASK_15]])

module {
  aie.device(npu1) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId19(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @airMemcpyId4(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @airMemcpyId5(%shim_noc_tile_0_0, MM2S, 1)
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
    airrt.wait_all %0, %1, %2
    %p_0 = airrt.segment_load "forward_0" : i64
    %3 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c1_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c1_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c16_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %5 = airrt.dma_memcpy_nd(%c19_i32, %c0_i64, %c1_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c128_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    airrt.wait_all %3, %4, %5
    %p_1 = airrt.segment_load "forward_0" : i64
    %6 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c2_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %7 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c2_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c32_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %8 = airrt.dma_memcpy_nd(%c19_i32, %c0_i64, %c2_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c256_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    airrt.wait_all %6, %7, %8
    %p_2 = airrt.segment_load "forward_0" : i64
    %9 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c3_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %10 = airrt.dma_memcpy_nd(%c5_i32, %c0_i64, %c3_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c48_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %11 = airrt.dma_memcpy_nd(%c19_i32, %c0_i64, %c3_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c384_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    airrt.wait_all %9, %10, %11
    %p_3 = airrt.segment_load "forward_0" : i64
    %12 = airrt.dma_memcpy_nd(%c4_i32, %c1_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c128_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %13 = airrt.dma_memcpy_nd(%c5_i32, %c1_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %14 = airrt.dma_memcpy_nd(%c19_i32, %c1_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c128_i64, %c0_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    airrt.wait_all %12, %13, %14
    %p_4 = airrt.segment_load "forward_0" : i64
    %15 = airrt.dma_memcpy_nd(%c4_i32, %c1_i64, %c1_i64, %arg0[%c0_i64, %c0_i64, %c128_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %16 = airrt.dma_memcpy_nd(%c5_i32, %c1_i64, %c1_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c16_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %17 = airrt.dma_memcpy_nd(%c19_i32, %c1_i64, %c1_i64, %arg2[%c0_i64, %c0_i64, %c128_i64, %c128_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    airrt.wait_all %15, %16, %17
    %p_5 = airrt.segment_load "forward_0" : i64
    %18 = airrt.dma_memcpy_nd(%c4_i32, %c1_i64, %c2_i64, %arg0[%c0_i64, %c0_i64, %c128_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %19 = airrt.dma_memcpy_nd(%c5_i32, %c1_i64, %c2_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c32_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %20 = airrt.dma_memcpy_nd(%c19_i32, %c1_i64, %c2_i64, %arg2[%c0_i64, %c0_i64, %c128_i64, %c256_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    airrt.wait_all %18, %19, %20
    %p_6 = airrt.segment_load "forward_0" : i64
    %21 = airrt.dma_memcpy_nd(%c4_i32, %c1_i64, %c3_i64, %arg0[%c0_i64, %c0_i64, %c128_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %22 = airrt.dma_memcpy_nd(%c5_i32, %c1_i64, %c3_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c48_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %23 = airrt.dma_memcpy_nd(%c19_i32, %c1_i64, %c3_i64, %arg2[%c0_i64, %c0_i64, %c128_i64, %c384_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    airrt.wait_all %21, %22, %23
    %p_7 = airrt.segment_load "forward_0" : i64
    %24 = airrt.dma_memcpy_nd(%c4_i32, %c2_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %25 = airrt.dma_memcpy_nd(%c5_i32, %c2_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %26 = airrt.dma_memcpy_nd(%c19_i32, %c2_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    airrt.wait_all %24, %25, %26
    %p_8 = airrt.segment_load "forward_0" : i64
    %27 = airrt.dma_memcpy_nd(%c4_i32, %c2_i64, %c1_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %28 = airrt.dma_memcpy_nd(%c5_i32, %c2_i64, %c1_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c16_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %29 = airrt.dma_memcpy_nd(%c19_i32, %c2_i64, %c1_i64, %arg2[%c0_i64, %c0_i64, %c256_i64, %c128_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    airrt.wait_all %27, %28, %29
    %p_9 = airrt.segment_load "forward_0" : i64
    %30 = airrt.dma_memcpy_nd(%c4_i32, %c2_i64, %c2_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %31 = airrt.dma_memcpy_nd(%c5_i32, %c2_i64, %c2_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c32_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %32 = airrt.dma_memcpy_nd(%c19_i32, %c2_i64, %c2_i64, %arg2[%c0_i64, %c0_i64, %c256_i64, %c256_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    airrt.wait_all %30, %31, %32
    %p_10 = airrt.segment_load "forward_0" : i64
    %33 = airrt.dma_memcpy_nd(%c4_i32, %c2_i64, %c3_i64, %arg0[%c0_i64, %c0_i64, %c256_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %34 = airrt.dma_memcpy_nd(%c5_i32, %c2_i64, %c3_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c48_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %35 = airrt.dma_memcpy_nd(%c19_i32, %c2_i64, %c3_i64, %arg2[%c0_i64, %c0_i64, %c256_i64, %c384_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    airrt.wait_all %33, %34, %35
    %p_11 = airrt.segment_load "forward_0" : i64
    %36 = airrt.dma_memcpy_nd(%c4_i32, %c3_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c384_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %37 = airrt.dma_memcpy_nd(%c5_i32, %c3_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %38 = airrt.dma_memcpy_nd(%c19_i32, %c3_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c384_i64, %c0_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    airrt.wait_all %36, %37, %38
    %p_12 = airrt.segment_load "forward_0" : i64
    %39 = airrt.dma_memcpy_nd(%c4_i32, %c3_i64, %c1_i64, %arg0[%c0_i64, %c0_i64, %c384_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %40 = airrt.dma_memcpy_nd(%c5_i32, %c3_i64, %c1_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c16_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %41 = airrt.dma_memcpy_nd(%c19_i32, %c3_i64, %c1_i64, %arg2[%c0_i64, %c0_i64, %c384_i64, %c128_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    airrt.wait_all %39, %40, %41
    %p_13 = airrt.segment_load "forward_0" : i64
    %42 = airrt.dma_memcpy_nd(%c4_i32, %c3_i64, %c2_i64, %arg0[%c0_i64, %c0_i64, %c384_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %43 = airrt.dma_memcpy_nd(%c5_i32, %c3_i64, %c2_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c32_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %44 = airrt.dma_memcpy_nd(%c19_i32, %c3_i64, %c2_i64, %arg2[%c0_i64, %c0_i64, %c384_i64, %c256_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    airrt.wait_all %42, %43, %44
    %p_14 = airrt.segment_load "forward_0" : i64
    %45 = airrt.dma_memcpy_nd(%c4_i32, %c3_i64, %c3_i64, %arg0[%c0_i64, %c0_i64, %c384_i64, %c0_i64], [%c1_i64, %c4_i64, %c128_i64, %c32_i64], [%c0_i64, %c32_i64, %c128_i64]) {metadata = @airMemcpyId4} : (i32, i64, i64, memref<512x128xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %46 = airrt.dma_memcpy_nd(%c5_i32, %c3_i64, %c3_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c48_i64], [%c16_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64]) {metadata = @airMemcpyId5} : (i32, i64, i64, memref<16x8x8x64xbf16>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    %47 = airrt.dma_memcpy_nd(%c19_i32, %c3_i64, %c3_i64, %arg2[%c0_i64, %c0_i64, %c384_i64, %c384_i64], [%c1_i64, %c1_i64, %c128_i64, %c128_i64], [%c0_i64, %c0_i64, %c512_i64]) {metadata = @airMemcpyId19} : (i32, i64, i64, memref<512x512xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
    affine.for %arg3 = 0 to 1 {
      %h = airrt.herd_load "herd_0" () : () -> i64
    }
    airrt.wait_all %45, %46, %47
    return %arg2 : memref<512x512xf32>
  }
}
