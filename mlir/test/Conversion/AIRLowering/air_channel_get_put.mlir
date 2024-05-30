//===- air_channel_get_put.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --split-input-file -air-to-std | FileCheck %s

// CHECK-LABEL:   func.func @single_put_get
// CHECK: affine.for %{{.*}} 0 to 2
// CHECK: affine.for %{{.*}} 0 to 2
// CHECK: airrt.segment_load
// CHECK: airrt.dma_memcpy_nd(%c3_i32, %{{.*}}, %{{.*}}, %arg0[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK: airrt.dma_memcpy_nd(%c4_i32, %{{.*}}, %{{.*}}, %arg1[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
module {
  air.channel @channel_1 [1, 1]
  air.channel @channel_0 [1, 1]
  func.func @single_put_get(%a0: memref<32x16xi32>, %a1: memref<32x16xi32>) {
    %c2_0 = arith.constant 2 : index
    air.launch (%arg2, %arg3) in (%arg4=%c2_0, %arg5=%c2_0) args(%arg0=%a0, %arg1=%a1) : memref<32x16xi32>, memref<32x16xi32> {
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %0 = air.channel.put async  @channel_0[%c0, %c0] (%arg0[%c8, %c0] [%c8, %c16] [%c32, %c0]) {id = 1 : i32} : (memref<32x16xi32>)
      %1 = air.channel.get async  @channel_1[%c0, %c0] (%arg1[%c8, %c0] [%c8, %c16] [%c32, %c0]) {id = 2 : i32} : (memref<32x16xi32>)
      air.segment @segment_0 {
        %c1_0 = arith.constant 1 : index
        air.herd @herd_0  tile (%arg10, %arg11) in (%arg12=%c1_0, %arg13=%c1_0) {
          %c0_4 = arith.constant 0 : index
          %c32_5 = arith.constant 32 : index
          %c16_6 = arith.constant 16 : index
          %c8_7 = arith.constant 8 : index
          %alloc = memref.alloc() {sym_name = "scratch"} : memref<16x8xi32, 2>
          %alloc_8 = memref.alloc() {sym_name = "scratch_copy"} : memref<16x8xi32, 2>
          air.channel.get  @channel_0[%arg10, %arg11] (%alloc[%c0_4, %c0_4] [%c8_7, %c16_6] [%c32_5, %c0_4]) {id = 3 : i32} : (memref<16x8xi32, 2>)
          affine.for %arg18 = 0 to 8 {
            affine.for %arg19 = 0 to 16 {
              %2 = affine.load %alloc[%arg19, %arg18] : memref<16x8xi32, 2>
              affine.store %2, %alloc_8[%arg19, %arg18] : memref<16x8xi32, 2>
            }
          }
          air.channel.put  @channel_1[%arg10, %arg11] (%alloc_8[%c0_4, %c0_4] [%c8_7, %c16_6] [%c32_5, %c0_4]) {id = 4 : i32} : (memref<16x8xi32, 2>)
          memref.dealloc %alloc_8 : memref<16x8xi32, 2>
          memref.dealloc %alloc : memref<16x8xi32, 2>
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// CHECK-LABEL:   func.func @par_put_get
// CHECK: affine.for %{{.*}} 0 to 1
// CHECK: affine.for %{{.*}} 0 to 1
// CHECK: airrt.segment_load "segment_0" : i64
// CHECK: airrt.dma_memcpy_nd(%c3_i32, %{{.*}}, %{{.*}}, %arg0[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK: airrt.dma_memcpy_nd(%c4_i32, %{{.*}}, %{{.*}}, %arg1[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK: airrt.herd_load "herd_0" : i64

module {
  air.channel @channel_3 [2, 2]
  air.channel @channel_2 [2, 2]
  func.func @par_put_get(%a0: memref<32x16xi32>, %a1: memref<32x16xi32>) {
    %c1_0 = arith.constant 1 : index
    air.launch (%arg2, %arg3) in (%arg4=%c1_0, %arg5=%c1_0) args(%arg0=%a0, %arg1=%a1) : memref<32x16xi32>, memref<32x16xi32> {
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %0 = air.wait_all async 
      %1 = scf.parallel (%a2, %a3) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%0) -> !air.async.token {
        %3 = air.channel.put async  @channel_2[%a2, %a3] (%arg0[%c8, %c0] [%c8, %c16] [%c32, %c0]) {id = 1 : i32} : (memref<32x16xi32>)
        scf.reduce(%3 : !air.async.token) {
        ^bb0(%a4: !air.async.token, %a5: !air.async.token):
          %4 = air.wait_all async [%a4, %a5] 
          scf.reduce.return %4 : !air.async.token
        }
      }
      %2 = scf.parallel (%a2, %a3) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%0) -> !air.async.token {
        %3 = air.channel.get async  @channel_3[%a2, %a3] (%arg1[%c8, %c0] [%c8, %c16] [%c32, %c0]) {id = 2 : i32} : (memref<32x16xi32>)
        scf.reduce(%3 : !air.async.token) {
        ^bb0(%a4: !air.async.token, %a5: !air.async.token):
          %4 = air.wait_all async [%a4, %a5] 
          scf.reduce.return %4 : !air.async.token
        }
      }
      air.segment @segment_0  args(%arg6=%arg2, %arg7=%arg3, %arg8=%arg4, %arg9=%arg5) : index, index, index, index {
        %c2_2 = arith.constant 2 : index
        %c2_3 = arith.constant 2 : index
        air.herd @herd_0  tile (%arg10, %arg11) in (%arg12=%c2_2, %arg13=%c2_3) args(%arg14=%arg6, %arg15=%arg7, %arg16=%arg8, %arg17=%arg9) : index, index, index, index {
          %c0_4 = arith.constant 0 : index
          %c32_5 = arith.constant 32 : index
          %c16_6 = arith.constant 16 : index
          %c8_7 = arith.constant 8 : index
          %alloc = memref.alloc() {sym_name = "scratch"} : memref<16x8xi32, 2>
          %alloc_8 = memref.alloc() {sym_name = "scratch_copy"} : memref<16x8xi32, 2>
          air.channel.get  @channel_2[%arg10, %arg11] (%alloc[%c0_4, %c0_4] [%c8_7, %c16_6] [%c32_5, %c0_4]) {id = 3 : i32} : (memref<16x8xi32, 2>)
          affine.for %arg18 = 0 to 8 {
            affine.for %arg19 = 0 to 16 {
              %3 = affine.load %alloc[%arg19, %arg18] : memref<16x8xi32, 2>
              affine.store %3, %alloc_8[%arg19, %arg18] : memref<16x8xi32, 2>
            }
          }
          air.channel.put  @channel_3[%arg10, %arg11] (%alloc_8[%c0_4, %c0_4] [%c8_7, %c16_6] [%c32_5, %c0_4]) {id = 4 : i32} : (memref<16x8xi32, 2>)
          memref.dealloc %alloc_8 : memref<16x8xi32, 2>
          memref.dealloc %alloc : memref<16x8xi32, 2>
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// CHECK-LABEL:   func.func @par_with_for_put_get
// CHECK: airrt.segment_load "segment_0" : i64
// CHECK: affine.for %{{.*}} 0 to 2
// CHECK:   affine.for %{{.*}} 0 to 2
// CHECK:     airrt.dma_memcpy_nd(%{{.*}}, %{{.*}}, %{{.*}}, %arg0[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK:     scf.for
// CHECK:       airrt.dma_memcpy_nd(%{{.*}}, %{{.*}}, %{{.*}}, %arg1[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<32x16xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
// CHECK:       scf.yield
// CHECK: airrt.herd_load "herd_0" : i64
module {
  air.channel @channel_5 [2, 2]
  air.channel @channel_4 [2, 2]
  func.func @par_with_for_put_get(%a0: memref<32x16xi32>, %a1: memref<32x16xi32>) {
    %c1_0 = arith.constant 1 : index
    %c1_1 = arith.constant 1 : index
    air.launch (%arg2, %arg3) in (%arg4=%c1_0, %arg5=%c1_1) args (%arg0=%a0, %arg1=%a1) : memref<32x16xi32>, memref<32x16xi32> {
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %0 = air.wait_all async 
      %1 = scf.parallel (%a2, %a3) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%0) -> !air.async.token {
        %3 = air.channel.put async  @channel_4[%a2, %a3] (%arg0[%c8, %c0] [%c8, %c16] [%c32, %c0]) {id = 1 : i32} : (memref<32x16xi32>)
        scf.reduce(%3 : !air.async.token) {
        ^bb0(%a4: !air.async.token, %a5: !air.async.token):
          %4 = air.wait_all async [%a4, %a5] 
          scf.reduce.return %4 : !air.async.token
        }
      }
      %2 = scf.parallel (%a2, %a3) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) init (%0) -> !air.async.token {
        %3 = scf.for %a4 = %c0 to %c2 step %c1 iter_args(%a5 = %0) -> (!air.async.token) {
          %4 = air.channel.get async [%a5]  @channel_5[%a2, %a3] (%arg1[%c8, %c0] [%c8, %c16] [%c32, %c0]) {id = 2 : i32} : (memref<32x16xi32>)
          scf.yield %4 : !air.async.token
        }
        scf.reduce(%3 : !air.async.token) {
        ^bb0(%a4: !air.async.token, %a5: !air.async.token):
          %4 = air.wait_all async [%a4, %a5] 
          scf.reduce.return %4 : !air.async.token
        }
      }
      air.segment @segment_0  args(%arg6=%arg2, %arg7=%arg3, %arg8=%arg4, %arg9=%arg5) : index, index, index, index {
        %c2_2 = arith.constant 2 : index
        %c2_3 = arith.constant 2 : index
        air.herd @herd_0  tile (%arg10, %arg11) in (%arg12=%c2_2, %arg13=%c2_3) args(%arg14=%arg6, %arg15=%arg7, %arg16=%arg8, %arg17=%arg9) : index, index, index, index {
          %c0_4 = arith.constant 0 : index
          %c2_5 = arith.constant 2 : index
          %c1_6 = arith.constant 1 : index
          %c32_7 = arith.constant 32 : index
          %c16_8 = arith.constant 16 : index
          %c8_9 = arith.constant 8 : index
          %alloc = memref.alloc() {sym_name = "scratch"} : memref<16x8xi32, 2>
          %alloc_10 = memref.alloc() {sym_name = "scratch_copy"} : memref<16x8xi32, 2>
          air.channel.get  @channel_4[%arg10, %arg11] (%alloc[%c0_4, %c0_4] [%c8_9, %c16_8] [%c32_7, %c0_4]) {id = 3 : i32} : (memref<16x8xi32, 2>)
          affine.for %arg18 = 0 to 8 {
            affine.for %arg19 = 0 to 16 {
              %3 = affine.load %alloc[%arg19, %arg18] : memref<16x8xi32, 2>
              affine.store %3, %alloc_10[%arg19, %arg18] : memref<16x8xi32, 2>
            }
          }
          scf.for %arg18 = %c0_4 to %c2_5 step %c1_6 {
            air.channel.put  @channel_5[%arg10, %arg11] (%alloc_10[%c0_4, %c0_4] [%c8_9, %c16_8] [%c32_7, %c0_4]) {id = 4 : i32} : (memref<16x8xi32, 2>)
          }
          memref.dealloc %alloc_10 : memref<16x8xi32, 2>
          memref.dealloc %alloc : memref<16x8xi32, 2>
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
// -----

// CHECK-LABEL:   func.func @one_d_scf_parallel
// CHECK: affine.for
// CHECK: airrt.segment_load "segment_0" : i64
// CHECK: airrt.dma_memcpy_nd(%{{.*}}, %{{.*}}, %{{.*}}, %arg0[%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], [%{{.*}}, %{{.*}}, %{{.*}}]) : (i32, i64, i64, memref<128xf32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64]) : !airrt.event
// CHECK: airrt.herd_load "herd_0" : i64

#map = affine_map<()[s0] -> (s0 * 64)>
module {
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
}
