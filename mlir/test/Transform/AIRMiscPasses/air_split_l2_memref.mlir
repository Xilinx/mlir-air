//===- air_split_l2_memref.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="mm2s-channel-count=6 s2mm-channel-count=6" | FileCheck %s

// CHECK: [[$MAP0:#map[0-9]*]] = affine_map<()[s0] -> (s0 * 256)>
// CHECK: [[$MAP1:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 256 + 64)>
// CHECK: [[$MAP2:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 256 + 128)>
// CHECK: [[$MAP3:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 256 + 192)>
// CHECK: [[$MAP4:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 64)>
// CHECK: air.channel @channel_2 [4, 1]
// CHECK: air.channel @channel_1 [1, 1]
// CHECK: air.channel @channel_0 [4, 4]
// CHECK-LABEL: func.func @test0
// CHECK: air.execute
// CHECK: air.execute
// CHECK: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: %[[CST0_0:.*]] = arith.constant 0 : index
// CHECK: %[[VAL0:.*]] = affine.apply [[$MAP0]]()
// CHECK: air.channel.get {{.*}} @channel_2[%[[CST0]], %[[CST0_0]]] (%{{.*}}[%[[VAL0]], 
// CHECK: %[[CST1:.*]] = arith.constant 1 : index
// CHECK: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: %[[VAL1:.*]] = affine.apply [[$MAP1]]()
// CHECK: air.channel.get {{.*}} @channel_2[%[[CST1]], %[[CST0]]] (%{{.*}}[%[[VAL1]]
// CHECK: %[[CST2:.*]] = arith.constant 2 : index
// CHECK: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: %[[VAL2:.*]] = affine.apply [[$MAP2]]()
// CHECK: air.channel.get {{.*}} @channel_2[%[[CST2]], %[[CST0]]] (%{{.*}}[%[[VAL2]]
// CHECK: %[[CST3:.*]] = arith.constant 3 : index
// CHECK: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: %[[VAL3:.*]] = affine.apply [[$MAP3]]()
// CHECK: air.channel.get {{.*}} @channel_2[%[[CST3]], %[[CST0]]] (%{{.*}}[%[[VAL3]]
// CHECK: air.herd
// CHECK: air.herd_terminator
// CHECK: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: %[[CST0_0:.*]] = arith.constant 0 : index
// CHECK: %[[CST0_1:.*]] = arith.constant 0 : index
// CHECK: %[[VAL4:.*]] = affine.apply [[$MAP0]]()
// CHECK: air.channel.put {{.*}} @channel_2[%[[CST0_0]], %[[CST0_1]]] (%{{.*}}[%[[VAL4]]
// CHECK: %[[CST1:.*]] = arith.constant 1 : index
// CHECK: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: %[[VAL5:.*]] = affine.apply [[$MAP1]]()
// CHECK: air.channel.put {{.*}} @channel_2[%[[CST1]], %[[CST0]]] (%{{.*}}[%[[VAL5]]
// CHECK: %[[CST2:.*]] = arith.constant 2 : index
// CHECK: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: %[[VAL6:.*]] = affine.apply [[$MAP2]]()
// CHECK: air.channel.put {{.*}} @channel_2[%[[CST2]], %[[CST0]]] (%{{.*}}[%[[VAL6]]
// CHECK: %[[CST3:.*]] = arith.constant 3 : index
// CHECK: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: %[[VAL7:.*]] = affine.apply [[$MAP3]]()
// CHECK: air.channel.put {{.*}} @channel_2[%[[CST3]], %[[CST0]]] (%{{.*}}[%[[VAL7]]


#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
air.channel @channel_1 [1, 1]
air.channel @channel_0 [4, 4]
func.func @test0(%arg0: memref<512x1024xbf16>, %arg1: memref<1024x512xbf16>, %arg2: memref<512x512xbf16>) {
  %c2 = arith.constant 2 : index
  %0 = air.launch async (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg2) : memref<512x512xbf16> attributes {id = 1 : i32} {
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %async_token, %results = air.execute -> (index) {
      %3 = affine.apply #map()[%arg3]
      air.execute_terminator %3 : index
    }
    %async_token_0, %results_1 = air.execute -> (index) {
      %3 = affine.apply #map()[%arg4]
      air.execute_terminator %3 : index
    }
    %1 = air.channel.get async [%async_token, %async_token_0]  @channel_1[] (%arg7[%results, %results_1] [%c256, %c256] [%c512, %c1]) {id = 3 : i32} : (memref<512x512xbf16>)
    %2 = air.segment @segment_0 async  {
      %c64 = arith.constant 64 : index
      %c1_2 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c0 = arith.constant 0 : index
      %c256_3 = arith.constant 256 : index
      %3 = air.wait_all async 
      %4 = air.wait_all async 
      %async_token_4, %results_5 = air.execute -> (memref<256x256xbf16, 1>) {
        %alloc = memref.alloc() : memref<256x256xbf16, 1>
        air.execute_terminator %alloc : memref<256x256xbf16, 1>
      }
      %5 = scf.parallel (%arg8, %arg9) = (%c0, %c0) to (%c4, %c4) step (%c1_2, %c1_2) init (%async_token_4) -> !air.async.token {
        %async_token_7, %results_8 = air.execute -> (index) {
          %9 = affine.apply #map1()[%arg8]
          air.execute_terminator %9 : index
        }
        %async_token_9, %results_10 = air.execute -> (index) {
          %9 = affine.apply #map1()[%arg9]
          air.execute_terminator %9 : index
        }
        %8 = air.channel.get async [%async_token_4, %async_token_9, %async_token_7]  @channel_0[%arg8, %arg9] (%results_5[%results_8, %results_10] [%c64, %c64] [%c256_3, %c1_2]) {id = 24 : i32} : (memref<256x256xbf16, 1>)
        scf.reduce(%8 : !air.async.token) {
        ^bb0(%arg10: !air.async.token, %arg11: !air.async.token):
          %9 = air.wait_all async [%arg10, %arg11] 
          scf.reduce.return %9 : !air.async.token
        }
      }
      %6 = air.herd @herd_0 async [%async_token_4]  tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c4) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
        %c64_7 = arith.constant 64 : index
        %c256_8 = arith.constant 256 : index
        %c4_9 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c1_10 = arith.constant 1 : index
        %c0_11 = arith.constant 0 : index
        %async_token_12, %results_13 = air.execute -> (memref<16x16x4x4xbf16, 2>) {
          %alloc = memref.alloc() : memref<16x16x4x4xbf16, 2>
          air.execute_terminator %alloc : memref<16x16x4x4xbf16, 2>
        }
        %8 = air.channel.put async [%async_token_12]  @channel_0[%arg8, %arg9] (%results_13[%c0_11, %c0_11, %c0_11] [%c64_7, %c16, %c4_9] [%c4_9, %c256_8, %c1_10]) {id = 41 : i32} : (memref<16x16x4x4xbf16, 2>)
        %async_token_14 = air.execute [%8] {
          memref.dealloc %results_13 : memref<16x16x4x4xbf16, 2>
        }
        air.herd_terminator
      }
      %7 = air.channel.put async [%3, %4, %6]  @channel_1[] (%results_5[] [] []) {id = 42 : i32} : (memref<256x256xbf16, 1>)
      %async_token_6 = air.execute [%7] {
        memref.dealloc %results_5 : memref<256x256xbf16, 1>
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
