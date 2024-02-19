//===- air_channel_get_put.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-std | FileCheck %s

// CHECK-LABEL:   func.func @single_put_get
// CHECK: airrt.dma_memcpy_nd(%c3_i32, %{{.*}}, %{{.*}}, %arg0[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK: airrt.dma_memcpy_nd(%c4_i32, %{{.*}}, %{{.*}}, %arg1[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
air.channel @channel_1 [1, 1]
air.channel @channel_0 [1, 1]
func.func @single_put_get(%arg0: memref<32x16xi32>, %arg1: memref<32x16xi32>) {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %async_token = air.channel.put async  @channel_0[%c0, %c0] (%arg0[%c8, %c0] [%c8, %c16] [%c32, %c0]) {id = 1 : i32} : (memref<32x16xi32>)
  %async_token_0 = air.channel.get async @channel_1[%c0, %c0] (%arg1[%c8, %c0] [%c8, %c16] [%c32, %c0]) {id = 2 : i32} : (memref<32x16xi32>)
  air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) attributes {x_loc = 7 : i64, y_loc = 2 : i64} {
    %c0_1 = arith.constant 0 : index
    %c32_2 = arith.constant 32 : index
    %c16_3 = arith.constant 16 : index
    %c8_4 = arith.constant 8 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<16x8xi32, 2>
    %alloc_5 = memref.alloc() {sym_name = "scratch_copy"} : memref<16x8xi32, 2>
    air.channel.get  @channel_0[%arg2, %arg3] (%alloc[%c0_1, %c0_1] [%c8_4, %c16_3] [%c32_2, %c0_1]) {id = 3 : i32} : (memref<16x8xi32, 2>)
    affine.for %arg6 = 0 to 8 {
      affine.for %arg7 = 0 to 16 {
        %0 = affine.load %alloc[%arg7, %arg6] : memref<16x8xi32, 2>
        affine.store %0, %alloc_5[%arg7, %arg6] : memref<16x8xi32, 2>
      }
    }
    air.channel.put  @channel_1[%arg2, %arg3] (%alloc_5[%c0_1, %c0_1] [%c8_4, %c16_3] [%c32_2, %c0_1]) {id = 4 : i32} : (memref<16x8xi32, 2>)
    memref.dealloc %alloc_5 : memref<16x8xi32, 2>
    memref.dealloc %alloc : memref<16x8xi32, 2>
    air.herd_terminator
  }
  return
}

// CHECK-LABEL:   func.func @par_put_get
// CHECK: airrt.herd_load "herd_0" : i64
// CHECK: affine.for
// CHECK:   affine.for
// CHECK:     airrt.dma_memcpy_nd(%c3_i32, %{{.*}}, %{{.*}}, %arg0[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK:     airrt.dma_memcpy_nd(%c4_i32, %{{.*}}, %{{.*}}, %arg1[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK:   } {air.herd = "inner"}
// CHECK: } {affine_opt_label = "tiling", air.herd = "outer"}

air.channel @channel_3 [2, 2]
air.channel @channel_2 [2, 2]
func.func @par_put_get(%arg0: memref<32x16xi32>, %arg1: memref<32x16xi32>) {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %async_token_2 = air.wait_all async
  %5 = scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%async_token_2) -> !air.async.token {
    %async_token = air.channel.put async  @channel_2[%arg8, %arg9] (%arg0[%c8, %c0] [%c8, %c16] [%c32, %c0]) {id = 1 : i32} : (memref<32x16xi32>)
    scf.reduce(%async_token : !air.async.token) {
    ^bb0(%arg10: !air.async.token, %arg11: !air.async.token):
      %9 = air.wait_all async [%arg10, %arg11] 
      scf.reduce.return %9 : !air.async.token
    }
  }
  %6 = scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%async_token_2) -> !air.async.token {
    %async_token_0 = air.channel.get async @channel_3[%arg8, %arg9] (%arg1[%c8, %c0] [%c8, %c16] [%c32, %c0]) {id = 2 : i32} : (memref<32x16xi32>)
    scf.reduce(%async_token_0 : !air.async.token) {
    ^bb0(%arg10: !air.async.token, %arg11: !air.async.token):
      %9 = air.wait_all async [%arg10, %arg11] 
      scf.reduce.return %9 : !air.async.token
    }
  }
  air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) attributes {x_loc = 7 : i64, y_loc = 2 : i64} {
    %c0_1 = arith.constant 0 : index
    %c32_2 = arith.constant 32 : index
    %c16_3 = arith.constant 16 : index
    %c8_4 = arith.constant 8 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<16x8xi32, 2>
    %alloc_5 = memref.alloc() {sym_name = "scratch_copy"} : memref<16x8xi32, 2>
    air.channel.get  @channel_2[%arg2, %arg3] (%alloc[%c0_1, %c0_1] [%c8_4, %c16_3] [%c32_2, %c0_1]) {id = 3 : i32} : (memref<16x8xi32, 2>)
    affine.for %arg6 = 0 to 8 {
      affine.for %arg7 = 0 to 16 {
        %0 = affine.load %alloc[%arg7, %arg6] : memref<16x8xi32, 2>
        affine.store %0, %alloc_5[%arg7, %arg6] : memref<16x8xi32, 2>
      }
    }
    air.channel.put  @channel_3[%arg2, %arg3] (%alloc_5[%c0_1, %c0_1] [%c8_4, %c16_3] [%c32_2, %c0_1]) {id = 4 : i32} : (memref<16x8xi32, 2>)
    memref.dealloc %alloc_5 : memref<16x8xi32, 2>
    memref.dealloc %alloc : memref<16x8xi32, 2>
    air.herd_terminator
  }
  return
}

// CHECK-LABEL:   func.func @par_with_for_put_get
// CHECK: airrt.herd_load "herd_0" : i64
// CHECK: affine.for
// CHECK:   affine.for
// CHECK:     airrt.dma_memcpy_nd(%{{.*}}, %{{.*}}, %{{.*}}, %arg0[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK:     scf.for
// CHECK:       airrt.dma_memcpy_nd(%{{.*}}, %{{.*}}, %{{.*}}, %arg1[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK:       scf.yield
// CHECK:   } {air.herd = "inner"}
// CHECK: } {affine_opt_label = "tiling", air.herd = "outer"}
air.channel @channel_5 [2, 2]
air.channel @channel_4 [2, 2]
func.func @par_with_for_put_get(%arg0: memref<32x16xi32>, %arg1: memref<32x16xi32>) {
  %c8 = arith.constant 8 : index
  %c16 = arith.constant 16 : index
  %c32 = arith.constant 32 : index
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %async_token_2 = air.wait_all async
  %5 = scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%async_token_2) -> !air.async.token {
    %async_token = air.channel.put async  @channel_4[%arg8, %arg9] (%arg0[%c8, %c0] [%c8, %c16] [%c32, %c0]) {id = 1 : i32} : (memref<32x16xi32>)
    scf.reduce(%async_token : !air.async.token) {
    ^bb0(%arg10: !air.async.token, %arg11: !air.async.token):
      %9 = air.wait_all async [%arg10, %arg11] 
      scf.reduce.return %9 : !air.async.token
    }
  }
  %6 = scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%async_token_2) -> !air.async.token {
    %7 = scf.for %arg12 = %c0 to %c2 step %c1 iter_args(%arg13 = %async_token_2) -> (!air.async.token) {
      %async_token_0 = air.channel.get async [%arg13] @channel_5[%arg8, %arg9] (%arg1[%c8, %c0] [%c8, %c16] [%c32, %c0]) {id = 2 : i32} : (memref<32x16xi32>)
      scf.yield %async_token_0 : !air.async.token
    }
    scf.reduce(%7 : !air.async.token) {
    ^bb0(%arg10: !air.async.token, %arg11: !air.async.token):
      %9 = air.wait_all async [%arg10, %arg11] 
      scf.reduce.return %9 : !air.async.token
    }
  }
  air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) attributes {x_loc = 7 : i64, y_loc = 2 : i64} {
    %c0_1 = arith.constant 0 : index
    %c2_1 = arith.constant 2 : index
    %c1_1 = arith.constant 1 : index
    %c32_2 = arith.constant 32 : index
    %c16_3 = arith.constant 16 : index
    %c8_4 = arith.constant 8 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<16x8xi32, 2>
    %alloc_5 = memref.alloc() {sym_name = "scratch_copy"} : memref<16x8xi32, 2>
    air.channel.get  @channel_4[%arg2, %arg3] (%alloc[%c0_1, %c0_1] [%c8_4, %c16_3] [%c32_2, %c0_1]) {id = 3 : i32} : (memref<16x8xi32, 2>)
    affine.for %arg6 = 0 to 8 {
      affine.for %arg7 = 0 to 16 {
        %0 = affine.load %alloc[%arg7, %arg6] : memref<16x8xi32, 2>
        affine.store %0, %alloc_5[%arg7, %arg6] : memref<16x8xi32, 2>
      }
    }
    scf.for %arg12 = %c0_1 to %c2_1 step %c1_1 {
      air.channel.put  @channel_5[%arg2, %arg3] (%alloc_5[%c0_1, %c0_1] [%c8_4, %c16_3] [%c32_2, %c0_1]) {id = 4 : i32} : (memref<16x8xi32, 2>)
    }
    memref.dealloc %alloc_5 : memref<16x8xi32, 2>
    memref.dealloc %alloc : memref<16x8xi32, 2>
    air.herd_terminator
  }
  return
}

// CHECK-LABEL:   func.func @one_d_scf_parallel
// CHECK: affine.for
// CHECK:   airrt.dma_memcpy_nd(%{{.*}}, %{{.*}}, %{{.*}}, %arg0[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<128xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
// CHECK: } {affine_opt_label = "tiling"}

#map = affine_map<()[s0] -> (s0 * 64)>
air.channel @channel_6 [1, 1]
func.func @one_d_scf_parallel(%arg0: memref<128xf32>, %arg1: memref<128xf32>) {
  %c2 = arith.constant 2 : index
  %0 = air.launch async (%arg2) in (%arg3=%c2) args(%arg4=%arg0) : memref<128xf32> attributes {id = 1 : i32} {
    %c64 = arith.constant 64 : index
    %c1 = arith.constant 1 : index
    %async_token, %results = air.execute -> (index) {
      %3 = affine.apply #map()[%arg2]
      air.execute_terminator %3 : index
    }
    %1 = air.channel.put async [%async_token]  @channel_6[] (%arg4[%results] [%c64] [%c1]) {id = 1 : i32} : (memref<128xf32>)
    %2 = air.segment @segment_0 async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 2 : i64, y_size = 4 : i64} {
      %c1_0 = arith.constant 1 : index
      %c2_1 = arith.constant 2 : index
      %3 = air.wait_all async 
      %async_token_2, %results_3 = air.execute -> (memref<64xf32, 1>) {
        %alloc = memref.alloc() : memref<64xf32, 1>
        air.execute_terminator %alloc : memref<64xf32, 1>
      }
      %4 = air.channel.get async [%3, %async_token_2]  @channel_6[] (%results_3[] [] []) {id = 3 : i32} : (memref<64xf32, 1>)
      %5 = air.herd @herd_0 async [%4]  tile (%arg5, %arg6) in (%arg7=%c1_0, %arg8=%c2_1) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %async_token_5, %results_6 = air.execute -> (memref<32xf32, 2>) {
          %alloc = memref.alloc() : memref<32xf32, 2>
          air.execute_terminator %alloc : memref<32xf32, 2>
        }
        %async_token_7 = air.execute [%async_token_5] {
          memref.dealloc %results_6 : memref<32xf32, 2>
        }
        air.herd_terminator
      }
      %async_token_4 = air.execute [%4] {
        memref.dealloc %results_3 : memref<64xf32, 1>
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
