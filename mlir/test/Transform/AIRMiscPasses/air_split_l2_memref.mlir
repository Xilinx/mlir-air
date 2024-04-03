//===- air_split_l2_memref.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref --split-input-file | FileCheck %s

// Tiling up a single L2 memref.

// CHECK: [[$MAP0:#map[0-9]*]] = affine_map<()[s0] -> (s0 * 256)>
// CHECK: [[$MAP1:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 256 + 64)>
// CHECK: [[$MAP2:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 256 + 128)>
// CHECK: [[$MAP3:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 256 + 192)>
// CHECK: air.channel @channel_2 [4, 1]
// CHECK: air.channel @channel_1 [1, 1]
// CHECK: air.channel @channel_0 [4, 4]
// CHECK-LABEL: func.func @test0
// CHECK: air.launch
// CHECK-DAG: %[[CST3:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[CST2:.*]] = arith.constant 2 : index
// CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
// CHECK: %[[VAL0:.*]] = affine.apply [[$MAP0]]()
// CHECK: %[[VAL1:.*]] = affine.apply [[$MAP1]]()
// CHECK: %[[VAL2:.*]] = affine.apply [[$MAP2]]()
// CHECK: %[[VAL3:.*]] = affine.apply [[$MAP3]]()
// CHECK: air.channel.get {{.*}} @channel_2[%[[CST0]], %[[CST0]]] (%{{.*}}[%[[VAL0]]
// CHECK: air.channel.get {{.*}} @channel_2[%[[CST1]], %[[CST0]]] (%{{.*}}[%[[VAL1]]
// CHECK: air.channel.get {{.*}} @channel_2[%[[CST2]], %[[CST0]]] (%{{.*}}[%[[VAL2]]
// CHECK: air.channel.get {{.*}} @channel_2[%[[CST3]], %[[CST0]]] (%{{.*}}[%[[VAL3]]
// CHECK: air.segment
// CHECK-COUNT-4: memref.alloc() : memref<64x256xbf16, 1>
// CHECK-COUNT-16: air.channel.get async{{.*}}@channel_0
// CHECK: air.herd
// CHECK-COUNT-4: air.channel.put async{{.*}}@channel_2


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

// -----

// Specializing L2 memrefs based on air.channels.

// CHECK: [[$MAP0:#map[0-9]*]] = affine_map<()[s0] -> (s0 * 256)>
// CHECK: [[$MAP1:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 256 + 64)>
// CHECK: [[$MAP2:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 256 + 128)>
// CHECK: [[$MAP3:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 256 + 192)>
// CHECK: [[$MAP4:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK-LABEL: func.func @test1
// CHECK: air.launch
// CHECK-COUNT-4: memref.alloc() : memref<64x2048xbf16, 1>
// CHECK-COUNT-4: air.channel.get {{.*}} @channel_4
// CHECK: scf.for
// CHECK-NEXT: air.execute
// CHECK-NEXT: affine.apply [[$MAP4]]()
// CHECK-NEXT: air.execute_terminator
// CHECK-NEXT: }
// CHECK-NEXT: air.channel.put {{.*}} @channel_0
// CHECK: scf.for
// CHECK-NEXT: air.execute
// CHECK-NEXT: affine.apply [[$MAP4]]()
// CHECK-NEXT: air.execute_terminator
// CHECK-NEXT: }
// CHECK-NEXT: air.channel.put {{.*}} @channel_1
// CHECK: scf.for
// CHECK-NEXT: air.execute
// CHECK-NEXT: affine.apply [[$MAP4]]()
// CHECK-NEXT: air.execute_terminator
// CHECK-NEXT: }
// CHECK-NEXT: air.channel.put {{.*}} @channel_2
// CHECK: scf.for
// CHECK-NEXT: air.execute
// CHECK-NEXT: affine.apply [[$MAP4]]()
// CHECK-NEXT: air.execute_terminator
// CHECK-NEXT: }
// CHECK-NEXT: air.channel.put {{.*}} @channel_3
// CHECK: air.herd
// CHECK: air.channel.get {{.*}} @channel_0
// CHECK: air.channel.get {{.*}} @channel_1
// CHECK: air.channel.get {{.*}} @channel_2
// CHECK: air.channel.get {{.*}} @channel_3

#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 8)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>
module {
  air.channel @channel_8 [1, 1]
  air.channel @channel_3 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_2 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_1 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_0 [1, 1] {broadcast_shape = [1, 4]}
  func.func @test1(%arg0: memref<2048x2048xbf16>, %arg1: memref<2048x2048xbf16>, %arg2: memref<2048x2048xbf16>) {
    %c8 = arith.constant 8 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c8, %arg6=%c8) args(%arg7=%arg0) : memref<2048x2048xbf16> attributes {id = 1 : i32} {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %async_token, %results = air.execute -> (index) {
        %3 = affine.apply #map()[%arg3]
        air.execute_terminator %3 : index
      }
      %1 = scf.for %arg8 = %c0 to %c2048 step %c256 iter_args(%arg9 = %async_token) -> (!air.async.token) {
        %3 = air.channel.put async [%arg9]  @channel_8[] (%arg7[%results, %arg8] [%c256, %c256] [%c2048, %c1]) {id = 1 : i32} : (memref<2048x2048xbf16>)
        scf.yield %3 : !air.async.token
      }
      %2 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c192 = arith.constant 192 : index
        %c128 = arith.constant 128 : index
        %c64 = arith.constant 64 : index
        %c8_0 = arith.constant 8 : index
        %c8192 = arith.constant 8192 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c0_2 = arith.constant 0 : index
        %c2048_3 = arith.constant 2048 : index
        %c256_4 = arith.constant 256 : index
        %async_token_5, %results_6 = air.execute -> (memref<256x2048xbf16, 1>) {
          %alloc = memref.alloc() : memref<256x2048xbf16, 1>
          air.execute_terminator %alloc : memref<256x2048xbf16, 1>
        }
        %3 = scf.for %arg8 = %c0_2 to %c2048_3 step %c256_4 iter_args(%arg9 = %async_token_5) -> (!air.async.token) {
          %9 = air.channel.get async [%arg9]  @channel_8[] (%results_6[%c0_2, %arg8] [%c256_4, %c256_4] [%c2048_3, %c1_1]) {id = 4 : i32} : (memref<256x2048xbf16, 1>)
          scf.yield %9 : !air.async.token
        }
        %4 = scf.for %arg8 = %c0_2 to %c256_4 step %c8_0 iter_args(%arg9 = %async_token_5) -> (!air.async.token) {
          %async_token_8, %results_9 = air.execute [%arg9] -> (index) {
            %10 = affine.apply #map1()[%arg8]
            air.execute_terminator %10 : index
          }
          %9 = air.channel.put async [%async_token_8]  @channel_0[] (%results_6[%c0_2, %c0_2, %c0_2, %results_9] [%c8_0, %c16, %c4, %c8_0] [%c8_0, %c8192, %c2048_3, %c1_1]) {id = 6 : i32} : (memref<256x2048xbf16, 1>)
          scf.yield %9 : !air.async.token
        }
        %5 = scf.for %arg8 = %c0_2 to %c256_4 step %c8_0 iter_args(%arg9 = %async_token_5) -> (!air.async.token) {
          %async_token_8, %results_9 = air.execute [%arg9] -> (index) {
            %10 = affine.apply #map1()[%arg8]
            air.execute_terminator %10 : index
          }
          %9 = air.channel.put async [%async_token_8]  @channel_1[] (%results_6[%c0_2, %c0_2, %c64, %results_9] [%c8_0, %c16, %c4, %c8_0] [%c8_0, %c8192, %c2048_3, %c1_1]) {id = 7 : i32} : (memref<256x2048xbf16, 1>)
          scf.yield %9 : !air.async.token
        }
        %6 = scf.for %arg8 = %c0_2 to %c256_4 step %c8_0 iter_args(%arg9 = %async_token_5) -> (!air.async.token) {
          %async_token_8, %results_9 = air.execute [%arg9] -> (index) {
            %10 = affine.apply #map1()[%arg8]
            air.execute_terminator %10 : index
          }
          %9 = air.channel.put async [%async_token_8]  @channel_2[] (%results_6[%c0_2, %c0_2, %c128, %results_9] [%c8_0, %c16, %c4, %c8_0] [%c8_0, %c8192, %c2048_3, %c1_1]) {id = 8 : i32} : (memref<256x2048xbf16, 1>)
          scf.yield %9 : !air.async.token
        }
        %7 = scf.for %arg8 = %c0_2 to %c256_4 step %c8_0 iter_args(%arg9 = %async_token_5) -> (!air.async.token) {
          %async_token_8, %results_9 = air.execute [%arg9] -> (index) {
            %10 = affine.apply #map1()[%arg8]
            air.execute_terminator %10 : index
          }
          %9 = air.channel.put async [%async_token_8]  @channel_3[] (%results_6[%c0_2, %c0_2, %c192, %results_9] [%c8_0, %c16, %c4, %c8_0] [%c8_0, %c8192, %c2048_3, %c1_1]) {id = 9 : i32} : (memref<256x2048xbf16, 1>)
          scf.yield %9 : !air.async.token
        }
        %8 = air.herd @herd_0 async  tile (%arg8, %arg9) in (%arg10=%c4, %arg11=%c4) attributes {id = 3 : i32, link_with = "mm.o"} {
          %c0_8 = arith.constant 0 : index
          %c256_9 = arith.constant 256 : index
          %c8_10 = arith.constant 8 : index
          %9 = air.wait_all async 
          %10 = scf.for %arg12 = %c0_8 to %c256_9 step %c8_10 iter_args(%arg13 = %9) -> (!air.async.token) {
            %async_token_11, %results_12 = air.execute -> (memref<8x16x4x8xbf16, 2>) {
              %alloc = memref.alloc() : memref<8x16x4x8xbf16, 2>
              air.execute_terminator %alloc : memref<8x16x4x8xbf16, 2>
            }
            %11 = affine.if #set()[%arg8, %arg9] -> !air.async.token {
              %12 = air.channel.get async [%async_token_11, %arg13]  @channel_0[%arg8, %arg9] (%results_12[] [] []) {id = 15 : i32} : (memref<8x16x4x8xbf16, 2>)
              affine.yield %12 : !air.async.token
            } else {
              %12 = affine.if #set1()[%arg8, %arg9] -> !air.async.token {
                %13 = air.channel.get async [%async_token_11, %arg13]  @channel_1[%arg8, %arg9] (%results_12[] [] []) {id = 16 : i32} : (memref<8x16x4x8xbf16, 2>)
                affine.yield %13 : !air.async.token
              } else {
                %13 = affine.if #set2()[%arg8, %arg9] -> !air.async.token {
                  %14 = air.channel.get async [%async_token_11, %arg13]  @channel_2[%arg8, %arg9] (%results_12[] [] []) {id = 17 : i32} : (memref<8x16x4x8xbf16, 2>)
                  affine.yield %14 : !air.async.token
                } else {
                  %14 = air.channel.get async [%async_token_11, %arg13]  @channel_3[%arg8, %arg9] (%results_12[] [] []) {id = 18 : i32} : (memref<8x16x4x8xbf16, 2>)
                  affine.yield %14 : !air.async.token
                }
                affine.yield %13 : !air.async.token
              }
              affine.yield %12 : !air.async.token
            }
            %async_token_13 = air.execute [%11] {
              memref.dealloc %results_12 : memref<8x16x4x8xbf16, 2>
            }
            scf.yield %async_token_13 : !air.async.token
          }
          air.herd_terminator
        }
        %async_token_7 = air.execute [%8] {
          memref.dealloc %results_6 : memref<256x2048xbf16, 1>
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// One-dimensional scf.parallel unrolling.

// CHECK: [[$MAP0:#map[0-9]*]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK: [[$MAP1:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 64)>
// CHECK: [[$MAP2:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 64 + 32)>
// CHECK-LABEL: func.func @test2
// CHECK: air.launch
// CHECK-COUNT-2: air.channel.get {{.*}} @channel_0
// CHECK: air.segment
// CHECK-COUNT-2: memref.alloc() : memref<8x32xi32, 1>
// CHECK-COUNT-2: air.channel.get {{.*}} @channel_4
// CHECK: air.herd
// CHECK: air.herd_terminator
// CHECK-COUNT-2: air.channel.put {{.*}} @channel_0

#map = affine_map<()[s0] -> (s0 * 8)>
#map1 = affine_map<()[s0] -> (s0 * 64)>
#map2 = affine_map<()[s0] -> (s0 * 32)>
module {
  air.channel @channel_5 [1, 1]
  air.channel @channel_4 [1, 2]
  func.func @test2(%arg0: memref<8x2048xi32>, %arg1: memref<2048x2048xi32>, %arg2: memref<8x2048xi32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c32) args(%arg7=%arg2) : memref<8x2048xi32> attributes {id = 1 : i32} {
      %c64 = arith.constant 64 : index
      %c8 = arith.constant 8 : index
      %c1_0 = arith.constant 1 : index
      %c2048 = arith.constant 2048 : index
      %async_token, %results = air.execute -> (index) {
        %3 = affine.apply #map()[%arg3]
        air.execute_terminator %3 : index
      }
      %async_token_1, %results_2 = air.execute -> (index) {
        %3 = affine.apply #map1()[%arg4]
        air.execute_terminator %3 : index
      }
      %1 = air.channel.get async [%async_token, %async_token_1]  @channel_5[] (%arg7[%results, %results_2] [%c8, %c64] [%c2048, %c1_0]) {id = 3 : i32} : (memref<8x2048xi32>)
      %2 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c32_3 = arith.constant 32 : index
        %c64_4 = arith.constant 64 : index
        %c8_5 = arith.constant 8 : index
        %c2 = arith.constant 2 : index
        %c1_6 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %async_token_7, %results_8 = air.execute -> (memref<8x64xi32, 1 : i32>) {
          %alloc = memref.alloc() : memref<8x64xi32, 1 : i32>
          air.execute_terminator %alloc : memref<8x64xi32, 1 : i32>
        }
        %3 = scf.parallel (%arg8) = (%c0) to (%c2) step (%c1_6) init (%async_token_7) -> !air.async.token {
          %6 = air.wait_all async 
          %async_token_10, %results_11 = air.execute -> (index) {
            %8 = affine.apply #map2()[%arg8]
            air.execute_terminator %8 : index
          }
          %7 = air.channel.get async [%async_token_7, %async_token_10, %6]  @channel_4[%c0, %arg8] (%results_8[%c0, %results_11] [%c8_5, %c32_3] [%c64_4, %c1_6]) {id = 8 : i32} : (memref<8x64xi32, 1 : i32>)
          scf.reduce(%7 : !air.async.token) {
          ^bb0(%arg9: !air.async.token, %arg10: !air.async.token):
            %8 = air.wait_all async [%arg9, %arg10] 
            scf.reduce.return %8 : !air.async.token
          }
        }
        %4 = air.herd @herd_0 async [%async_token_7]  tile (%arg8, %arg9) in (%arg10=%c1_6, %arg11=%c2) attributes {id = 3 : i32, link_with = "mm.o"} {
          %c0_10 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c2_11 = arith.constant 2 : index
          %c8_12 = arith.constant 8 : index
          %c32_13 = arith.constant 32 : index
          %c1_14 = arith.constant 1 : index
          %c16 = arith.constant 16 : index
          %async_token_15, %results_16 = air.execute -> (memref<8x2x4x4xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<8x2x4x4xi32, 2 : i32>
            air.execute_terminator %alloc : memref<8x2x4x4xi32, 2 : i32>
          }
          %6 = air.channel.put async [%async_token_15]  @channel_4[%arg8, %arg9] (%results_16[%c0_10, %c0_10, %c0_10, %c0_10] [%c2_11, %c4, %c8_12, %c4] [%c16, %c4, %c32_13, %c1_14]) {id = 11 : i32} : (memref<8x2x4x4xi32, 2 : i32>)
          %async_token_17 = air.execute [%6] {
            memref.dealloc %results_16 : memref<8x2x4x4xi32, 2 : i32>
          }
          air.herd_terminator
        }
        %5 = air.channel.put async [%4]  @channel_5[] (%results_8[] [] []) {id = 12 : i32} : (memref<8x64xi32, 1 : i32>)
        %async_token_9 = air.execute [%5] {
          memref.dealloc %results_8 : memref<8x64xi32, 1 : i32>
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Parallel-for-channel loop nest.

// CHECK: [[$MAP0:#map[0-9]*]] = affine_map<()[s0] -> (s0 * 64)>
// CHECK: [[$MAP1:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 64 + 32)>
// CHECK: [[$MAP2:#map[0-9]+]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK-LABEL: func.func @test3
// CHECK: air.launch
// CHECK: scf.for
// CHECK-COUNT-2: air.channel.put {{.*}} @channel_0
// CHECK: air.segment
// CHECK-COUNT-2: memref.alloc() : memref<2048x32xi32, 1>
// CHECK: scf.for
// CHECK-COUNT-2: air.channel.get {{.*}} @channel_0
// CHECK: scf.for
// CHECK: air.channel.put {{.*}} @channel_3
// CHECK: scf.for
// CHECK: air.channel.put {{.*}} @channel_3
// CHECK: air.herd
// CHECK: air.channel.get {{.*}} @channel_3
// CHECK: air.herd_terminator

#map = affine_map<()[s0] -> (s0 * 64)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#map2 = affine_map<()[s0] -> (s0 * 8)>
module {
  air.channel @channel_3 [1, 2]
  air.channel @channel_2 [1, 1]
  func.func @test3(%arg0: memref<2048x2048xi32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1, %arg4=%c32) args(%arg5=%arg0) : memref<2048x2048xi32> attributes {id = 1 : i32} {
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2048 = arith.constant 2048 : index
      %c256 = arith.constant 256 : index
      %async_token, %results = air.execute -> (index) {
        %3 = affine.apply #map()[%arg2]
        air.execute_terminator %3 : index
      }
      %1 = scf.for %arg6 = %c0 to %c2048 step %c256 iter_args(%arg7 = %async_token) -> (!air.async.token) {
        %3 = air.channel.put async [%arg7]  @channel_2[] (%arg5[%arg6, %results] [%c256, %c64] [%c2048, %c1_0]) {id = 2 : i32} : (memref<2048x2048xi32>)
        scf.yield %3 : !air.async.token
      }
      %2 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c512 = arith.constant 512 : index
        %c4 = arith.constant 4 : index
        %c64_1 = arith.constant 64 : index
        %c8 = arith.constant 8 : index
        %c2 = arith.constant 2 : index
        %c1_2 = arith.constant 1 : index
        %c0_3 = arith.constant 0 : index
        %c2048_4 = arith.constant 2048 : index
        %c256_5 = arith.constant 256 : index
        %async_token_6, %results_7 = air.execute -> (memref<2048x64xi32, 1 : i32>) {
          %alloc = memref.alloc() : memref<2048x64xi32, 1 : i32>
          air.execute_terminator %alloc : memref<2048x64xi32, 1 : i32>
        }
        %3 = scf.for %arg6 = %c0_3 to %c2048_4 step %c256_5 iter_args(%arg7 = %async_token_6) -> (!air.async.token) {
          %6 = air.channel.get async [%arg7]  @channel_2[] (%results_7[%arg6, %c0_3] [%c256_5, %c64_1] [%c64_1, %c1_2]) {id = 5 : i32} : (memref<2048x64xi32, 1 : i32>)
          scf.yield %6 : !air.async.token
        }
        %4 = scf.parallel (%arg6) = (%c0_3) to (%c2) step (%c1_2) init (%async_token_6) -> !air.async.token {
          %async_token_9, %results_10 = air.execute -> (index) {
            %8 = affine.apply #map1()[%arg6]
            air.execute_terminator %8 : index
          }
          %6 = air.wait_all async [%async_token_6, %async_token_9] 
          %7 = scf.for %arg7 = %c0_3 to %c256_5 step %c4 iter_args(%arg8 = %6) -> (!air.async.token) {
            %async_token_11, %results_12 = air.execute [%arg8] -> (index) {
              %9 = affine.apply #map2()[%arg7]
              air.execute_terminator %9 : index
            }
            %8 = air.channel.put async [%async_token_11]  @channel_3[%c0_3, %arg6] (%results_7[%c0_3, %c0_3, %results_12, %results_10] [%c8, %c4, %c8, %c4] [%c4, %c512, %c64_1, %c1_2]) {id = 7 : i32} : (memref<2048x64xi32, 1 : i32>)
            scf.yield %8 : !air.async.token
          }
          scf.reduce(%7 : !air.async.token) {
          ^bb0(%arg7: !air.async.token, %arg8: !air.async.token):
            %8 = air.wait_all async [%arg7, %arg8] 
            scf.reduce.return %8 : !air.async.token
          }
        }
        %5 = air.herd @herd_0 async [%async_token_6]  tile (%arg6, %arg7) in (%arg8=%c1_2, %arg9=%c2) attributes {id = 3 : i32, link_with = "mm.o"} {
          %c0_9 = arith.constant 0 : index
          %c4_10 = arith.constant 4 : index
          %c8_11 = arith.constant 8 : index
          %c32_12 = arith.constant 32 : index
          %c1_13 = arith.constant 1 : index
          %c128 = arith.constant 128 : index
          %c256_14 = arith.constant 256 : index
          %6 = air.wait_all async 
          %7 = scf.for %arg10 = %c0_9 to %c256_14 step %c4_10 iter_args(%arg11 = %6) -> (!air.async.token) {
            %async_token_15, %results_16 = air.execute -> (memref<8x4x8x4xi32, 2 : i32>) {
              %alloc = memref.alloc() : memref<8x4x8x4xi32, 2 : i32>
              air.execute_terminator %alloc : memref<8x4x8x4xi32, 2 : i32>
            }
            %8 = air.channel.get async [%async_token_15, %arg11]  @channel_3[%arg6, %arg7] (%results_16[%c0_9, %c0_9, %c0_9, %c0_9] [%c8_11, %c4_10, %c8_11, %c4_10] [%c128, %c32_12, %c4_10, %c1_13]) {id = 10 : i32} : (memref<8x4x8x4xi32, 2 : i32>)
            %async_token_17 = air.execute [%8] {
              memref.dealloc %results_16 : memref<8x4x8x4xi32, 2 : i32>
            }
            scf.yield %async_token_17 : !air.async.token
          }
          air.herd_terminator
        }
        %async_token_8 = air.execute [%5] {
          memref.dealloc %results_7 : memref<2048x64xi32, 1 : i32>
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Split by air.channels join/distribute pattern.

// CHECK-LABEL: func.func @test4
// CHECK: air.launch
// CHECK-COUNT-4: affine.apply
// CHECK-COUNT-4: air.channel.put {{.*}} @channel_0
// CHECK: air.segment
// CHECK-COUNT-4: memref.alloc() : memref<256x32xi32, 1>
// CHECK-COUNT-4: air.channel.get {{.*}} @channel_0
// CHECK: scf.for
// CHECK: air.channel.put {{.*}} @channel_4
// CHECK: scf.for
// CHECK: air.channel.put {{.*}} @channel_5
// CHECK: scf.for
// CHECK: air.channel.put {{.*}} @channel_6
// CHECK: scf.for
// CHECK: air.channel.put {{.*}} @channel_7
// CHECK: air.herd
// CHECK: affine.if
// CHECK: air.channel.get {{.*}} @channel_4
// CHECK: affine.if
// CHECK: air.channel.get {{.*}} @channel_5
// CHECK: affine.if
// CHECK: air.channel.get {{.*}} @channel_6
// CHECK: else
// CHECK: air.channel.get {{.*}} @channel_7
// CHECK: air.herd_terminator

#map = affine_map<()[s0] -> (s0 * 128)>
#map1 = affine_map<()[s0] -> (s0 * 8)>
#set = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 1 == 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 2 == 0)>
module {
  air.channel @channel_9 [1, 1]
  air.channel @channel_7 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_6 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_5 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_4 [1, 1] {broadcast_shape = [4, 1]}
  func.func @test4(%arg0: memref<512x256xi32>, %arg1: memref<256x512xi32>, %arg2: memref<512x512xi32>) {
    %c4 = arith.constant 4 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c4, %arg6=%c4) args(%arg7=%arg1) : memref<256x512xi32> attributes {id = 1 : i32} {
      %c512 = arith.constant 512 : index
      %c1 = arith.constant 1 : index
      %c256 = arith.constant 256 : index
      %c128 = arith.constant 128 : index
      %c0 = arith.constant 0 : index
      %async_token, %results = air.execute -> (index) {
        %3 = affine.apply #map()[%arg4]
        air.execute_terminator %3 : index
      }
      %1 = air.channel.put async [%async_token]  @channel_9[] (%arg7[%c0, %results] [%c256, %c128] [%c512, %c1]) {id = 2 : i32} : (memref<256x512xi32>)
      %2 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c96 = arith.constant 96 : index
        %c64 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %c8 = arith.constant 8 : index
        %c1_0 = arith.constant 1 : index
        %c128_1 = arith.constant 128 : index
        %c0_2 = arith.constant 0 : index
        %c4_3 = arith.constant 4 : index
        %3 = air.wait_all async 
        %async_token_4, %results_5 = air.execute -> (memref<256x128xi32, 1 : i32>) {
          %alloc = memref.alloc() : memref<256x128xi32, 1 : i32>
          air.execute_terminator %alloc : memref<256x128xi32, 1 : i32>
        }
        %4 = air.channel.get async [%3, %async_token_4]  @channel_9[] (%results_5[] [] []) {id = 5 : i32} : (memref<256x128xi32, 1 : i32>)
        %5 = scf.for %arg8 = %c0_2 to %c32 step %c4_3 iter_args(%arg9 = %async_token_4) -> (!air.async.token) {
          %async_token_7, %results_8 = air.execute [%arg9] -> (index) {
            %11 = affine.apply #map1()[%arg8]
            air.execute_terminator %11 : index
          }
          %10 = air.channel.put async [%async_token_7]  @channel_4[] (%results_5[%c0_2, %results_8, %c0_2] [%c8, %c32, %c4_3] [%c4_3, %c128_1, %c1_0]) {id = 10 : i32} : (memref<256x128xi32, 1 : i32>)
          scf.yield %10 : !air.async.token
        }
        %6 = scf.for %arg8 = %c0_2 to %c32 step %c4_3 iter_args(%arg9 = %async_token_4) -> (!air.async.token) {
          %async_token_7, %results_8 = air.execute [%arg9] -> (index) {
            %11 = affine.apply #map1()[%arg8]
            air.execute_terminator %11 : index
          }
          %10 = air.channel.put async [%async_token_7]  @channel_5[] (%results_5[%c0_2, %results_8, %c32] [%c8, %c32, %c4_3] [%c4_3, %c128_1, %c1_0]) {id = 11 : i32} : (memref<256x128xi32, 1 : i32>)
          scf.yield %10 : !air.async.token
        }
        %7 = scf.for %arg8 = %c0_2 to %c32 step %c4_3 iter_args(%arg9 = %async_token_4) -> (!air.async.token) {
          %async_token_7, %results_8 = air.execute [%arg9] -> (index) {
            %11 = affine.apply #map1()[%arg8]
            air.execute_terminator %11 : index
          }
          %10 = air.channel.put async [%async_token_7]  @channel_6[] (%results_5[%c0_2, %results_8, %c64] [%c8, %c32, %c4_3] [%c4_3, %c128_1, %c1_0]) {id = 12 : i32} : (memref<256x128xi32, 1 : i32>)
          scf.yield %10 : !air.async.token
        }
        %8 = scf.for %arg8 = %c0_2 to %c32 step %c4_3 iter_args(%arg9 = %async_token_4) -> (!air.async.token) {
          %async_token_7, %results_8 = air.execute [%arg9] -> (index) {
            %11 = affine.apply #map1()[%arg8]
            air.execute_terminator %11 : index
          }
          %10 = air.channel.put async [%async_token_7]  @channel_7[] (%results_5[%c0_2, %results_8, %c96] [%c8, %c32, %c4_3] [%c4_3, %c128_1, %c1_0]) {id = 13 : i32} : (memref<256x128xi32, 1 : i32>)
          scf.yield %10 : !air.async.token
        }
        %9 = air.herd @herd_0 async  tile (%arg8, %arg9) in (%arg10=%c4_3, %arg11=%c4_3) attributes {id = 3 : i32} {
          %c1024 = arith.constant 1024 : index
          %c0_7 = arith.constant 0 : index
          %c4_8 = arith.constant 4 : index
          %c32_9 = arith.constant 32 : index
          %c1_10 = arith.constant 1 : index
          %10 = air.wait_all async 
          %11 = scf.for %arg12 = %c0_7 to %c32_9 step %c4_8 iter_args(%arg13 = %10) -> (!air.async.token) {
            %async_token_11, %results_12 = air.execute -> (memref<8x4x8x4xi32, 2 : i32>) {
              %alloc = memref.alloc() : memref<8x4x8x4xi32, 2 : i32>
              air.execute_terminator %alloc : memref<8x4x8x4xi32, 2 : i32>
            }
            %12 = affine.if #set()[%arg8, %arg9] -> !air.async.token {
              %13 = air.channel.get async [%async_token_11, %arg13]  @channel_4[%arg8, %arg9] (%results_12[%c0_7] [%c1024] [%c1_10]) {id = 19 : i32} : (memref<8x4x8x4xi32, 2 : i32>)
              affine.yield %13 : !air.async.token
            } else {
              %13 = affine.if #set1()[%arg8, %arg9] -> !air.async.token {
                %14 = air.channel.get async [%async_token_11, %arg13]  @channel_5[%arg8, %arg9] (%results_12[%c0_7] [%c1024] [%c1_10]) {id = 20 : i32} : (memref<8x4x8x4xi32, 2 : i32>)
                affine.yield %14 : !air.async.token
              } else {
                %14 = affine.if #set2()[%arg8, %arg9] -> !air.async.token {
                  %15 = air.channel.get async [%async_token_11, %arg13]  @channel_6[%arg8, %arg9] (%results_12[%c0_7] [%c1024] [%c1_10]) {id = 21 : i32} : (memref<8x4x8x4xi32, 2 : i32>)
                  affine.yield %15 : !air.async.token
                } else {
                  %15 = air.channel.get async [%async_token_11, %arg13]  @channel_7[%arg8, %arg9] (%results_12[%c0_7] [%c1024] [%c1_10]) {id = 22 : i32} : (memref<8x4x8x4xi32, 2 : i32>)
                  affine.yield %15 : !air.async.token
                }
                affine.yield %14 : !air.async.token
              }
              affine.yield %13 : !air.async.token
            }
            %async_token_13 = air.execute [%async_token_11] {
              memref.dealloc %results_12 : memref<8x4x8x4xi32, 2 : i32>
            }
            scf.yield %12 : !air.async.token
          }
          air.herd_terminator
        }
        %async_token_6 = air.execute [%9] {
          memref.dealloc %results_5 : memref<256x128xi32, 1 : i32>
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
