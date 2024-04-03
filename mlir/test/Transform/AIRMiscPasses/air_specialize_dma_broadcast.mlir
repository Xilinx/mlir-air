//===- air_specialize_dma_broadcast.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-specialize-dma-broadcast --split-input-file | FileCheck %s

// Lowers broadcastable DMAs using affine.if
// CHECK: [[$SET0:#set[0-9]*]] = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
// CHECK: [[$SET1:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 1 >= 0)>
// CHECK: [[$SET2:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
// CHECK: [[$SET3:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 1 == 0)>
// CHECK-LABEL: @func0
// CHECK: air.herd @herd_0
// CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[CST64:.*]] = arith.constant 64 : index
// CHECK-DAG: %[[CST32:.*]] = arith.constant 32 : index
// CHECK: %[[EVENT0:.*]] = affine.if [[$SET0]]
// CHECK: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: %[[EVENT1:.*]] = air.dma_memcpy_nd {{.*}} ({{.*}}[] [] [], {{.*}}[%[[CST0]], {{.*}}] [%[[CST32]], %[[CST32]]] [%[[CST64]], %[[CST1]]]) {broadcast_set = [[$SET0]]{{.*}}id = [[#ID0:]]
// CHECK: affine.yield %[[EVENT1]]
// CHECK: %[[CST32_1:.*]] = arith.constant 32 : index
// CHECK: %[[EVENT2:.*]] = air.dma_memcpy_nd {{.*}} ({{.*}}[] [] [], {{.*}}[%[[CST32_1]], {{.*}}] [%[[CST32]], %[[CST32]]] [%[[CST64]], %[[CST1]]]) {broadcast_set = [[$SET1]]{{.*}}id = [[#ID0+1]]
// CHECK: affine.yield %[[EVENT2]]
// CHECK: %[[EVENT3:.*]] = affine.if [[$SET2]]
// CHECK: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: %[[EVENT4:.*]] = air.dma_memcpy_nd {{.*}} ({{.*}}[] [] [], {{.*}}[{{.*}}, %[[CST0]]] [%[[CST32]], %[[CST32]]] [%[[CST64]], %[[CST1]]]) {broadcast_set = [[$SET2]]{{.*}}id = [[#ID1:]]
// CHECK: affine.yield %[[EVENT4]]
// CHECK: %[[CST32_1:.*]] = arith.constant 32 : index
// CHECK: %[[EVENT5:.*]] = air.dma_memcpy_nd {{.*}} ({{.*}}[] [] [], {{.*}}[{{.*}}, %[[CST32_1]]] [%[[CST32]], %[[CST32]]] [%[[CST64]], %[[CST1]]]) {broadcast_set = [[$SET3]]{{.*}}id = [[#ID1+1]]
// CHECK: affine.yield %[[EVENT5]]

#map = affine_map<()[s0] -> (s0 * 32)>
#set = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 1 >= 0, s0 >= 0, -s0 + 1 >= 0)>
#set1 = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 1 >= 0, d1 - s0 == 0, s0 >= 0, -s0 + 1 >= 0)>
func.func @func0() {
  %c2 = arith.constant 2 : index
  %c0 = arith.constant 0 : index
  %c512 = arith.constant 512 : index
  %c64 = arith.constant 64 : index
  %0 = air.wait_all async 
  %1 = scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c512, %c512) step (%c64, %c64) init (%0) -> !air.async.token {
    %2 = scf.for %arg2 = %c0 to %c512 step %c64 iter_args(%arg3 = %0) -> (!air.async.token) {
      %async_token, %results = air.execute [%arg3] -> (memref<64x64xbf16, 1>) {
        %alloc = memref.alloc() : memref<64x64xbf16, 1>
        air.execute_terminator %alloc : memref<64x64xbf16, 1>
      } {id = 3 : i32}
      %async_token_0, %results_1 = air.execute [%arg3] -> (memref<64x64xbf16, 1>) {
        %alloc = memref.alloc() : memref<64x64xbf16, 1>
        air.execute_terminator %alloc : memref<64x64xbf16, 1>
      } {id = 4 : i32}
      %async_token_2, %results_3 = air.execute [%arg3] -> (memref<64x64xbf16, 1>) {
        %alloc = memref.alloc() : memref<64x64xbf16, 1>
        air.execute_terminator %alloc : memref<64x64xbf16, 1>
      } {id = 5 : i32}
      %3 = air.herd @herd_0 async  tile (%arg4, %arg5) in (%arg6=%c2, %arg7=%c2) args(%arg8=%results, %arg9=%results_1) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {id = 1 : i32} {
        %c1 = arith.constant 1 : index
        %c64_4 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %c2_5 = arith.constant 2 : index
        %c0_5 = arith.constant 0 : index
        %async_token_6, %results_7 = air.execute -> (index) {
          %6 = affine.apply #map()[%arg4]
          air.execute_terminator %6 : index
        } {id = 6 : i32}
        %async_token_8, %results_9 = air.execute -> (index) {
          %6 = affine.apply #map()[%arg5]
          air.execute_terminator %6 : index
        } {id = 7 : i32}
        %4 = air.wait_all async [%async_token_6, %async_token_8]
        %5 = scf.for %arg10 = %c0_5 to %c2_5 step %c1 iter_args(%arg11 = %4) -> (!air.async.token) {
          %async_token_36, %results_37 = air.execute [%arg11] -> (index) {
            %12 = affine.apply #map()[%arg10]
            air.execute_terminator %12 : index
          } {id = 16 : i32}
          %async_token_10, %results_11 = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          } {id = 8 : i32}
          %async_token_12, %results_13 = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          } {id = 9 : i32}
          %6 = air.dma_memcpy_nd async [%async_token_10, %arg11] (%results_11[] [] [], %arg8[%results_7, %results_37] [%c32, %c32] [%c64_4, %c1]) {broadcast_pattern = #set, id = 4 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          %7 = air.dma_memcpy_nd async [%async_token_12, %arg11] (%results_13[] [] [], %arg9[%results_37, %results_9] [%c32, %c32] [%c64_4, %c1]) {broadcast_pattern = #set1, id = 5 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          %8 = air.wait_all async [%6, %7] 
          scf.yield %8 : !air.async.token
        }
        air.herd_terminator
      }
      scf.yield %3 : !air.async.token
    }
    scf.reduce(%2 : !air.async.token) {
    ^bb0(%arg2: !air.async.token, %arg3: !air.async.token):
      %3 = air.wait_all async [%arg2, %arg3] 
      scf.reduce.return %3 : !air.async.token
    }
  }
  return
}

// -----

// Multi-dimensional DMA support.
// CHECK: [[$SET0:#set[0-9]*]] = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
// CHECK: [[$SET1:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 1 >= 0)>
// CHECK: [[$SET2:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
// CHECK: [[$SET3:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 1 == 0)>
// CHECK-LABEL: @func1
// CHECK: %[[EVENT0:.*]] = affine.if [[$SET0]]
// CHECK: %[[EVENT1:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET0]]{{.*}}id = [[#ID0:]]
// CHECK: affine.yield %[[EVENT1]]
// CHECK: %[[EVENT2:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET1]]{{.*}}id = [[#ID0+1]]
// CHECK: affine.yield %[[EVENT2]]
// CHECK: %[[EVENT3:.*]] = affine.if [[$SET2]]
// CHECK: %[[EVENT4:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET2]]{{.*}}id = [[#ID1:]]
// CHECK: affine.yield %[[EVENT4]]
// CHECK: %[[EVENT5:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET3]]{{.*}}id = [[#ID1+1]]
// CHECK: affine.yield %[[EVENT5]]

#map = affine_map<()[s0] -> (s0 * 32)>
#set = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 1 >= 0, s0 >= 0, -s0 + 1 >= 0)>
#set1 = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 1 >= 0, d1 - s0 == 0, s0 >= 0, -s0 + 1 >= 0)>
func.func @func1() {
  %c32 = arith.constant 32 : index
  %0 = air.launch async (%arg0, %arg1) in (%arg2=%c32, %arg3=%c32) attributes {id = 3 : i32} {
    %1 = air.segment @segment_0 async  attributes {id = 2 : i32} {
      %c2 = arith.constant 2 : index
      %async_token, %results = air.execute -> (memref<1x1x64x64xi32, 1>) {
        %alloc = memref.alloc() : memref<1x1x64x64xi32, 1>
        air.execute_terminator %alloc : memref<1x1x64x64xi32, 1>
      } {id = 9 : i32}
      %async_token_0, %results_1 = air.execute -> (memref<1x1x64x512xi32, 1>) {
        %alloc = memref.alloc() : memref<1x1x64x512xi32, 1>
        air.execute_terminator %alloc : memref<1x1x64x512xi32, 1>
      } {id = 10 : i32}
      %async_token_2, %results_3 = air.execute -> (memref<1x1x512x64xi32, 1>) {
        %alloc = memref.alloc() : memref<1x1x512x64xi32, 1>
        air.execute_terminator %alloc : memref<1x1x512x64xi32, 1>
      } {id = 11 : i32}
      %2 = air.herd @herd_0 async  tile (%arg4, %arg5) in (%arg6=%c2, %arg7=%c2) args(%arg8=%results_1, %arg9=%results_3) : memref<1x1x64x512xi32, 1>, memref<1x1x512x64xi32, 1> attributes {id = 1 : i32} {
        %c64 = arith.constant 64 : index
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c2048 = arith.constant 2048 : index
        %c8 = arith.constant 8 : index
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c32_4 = arith.constant 32 : index
        %async_token_5, %results_6 = air.execute -> (index) {
          %5 = affine.apply #map()[%arg4]
          air.execute_terminator %5 : index
        } {id = 12 : i32}
        %async_token_7, %results_8 = air.execute -> (index) {
          %5 = affine.apply #map()[%arg5]
          air.execute_terminator %5 : index
        } {id = 13 : i32}
        %async_token_9, %results_10 = air.execute -> (memref<1x1x4x8x4x8xi32, 2>) {
          %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2>
          air.execute_terminator %alloc : memref<1x1x4x8x4x8xi32, 2>
        } {id = 14 : i32}
        %3 = air.wait_all async [%async_token_5, %async_token_7]  {id = 2 : i32}
        %4 = scf.for %arg10 = %c0 to %c512 step %c32_4 iter_args(%arg11 = %3) -> (!air.async.token) {
          %async_token_11, %results_12 = air.execute -> (memref<1x1x4x8x4x8xi32, 2>) {
            %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2>
            air.execute_terminator %alloc : memref<1x1x4x8x4x8xi32, 2>
          } {id = 16 : i32}
          %async_token_13, %results_14 = air.execute -> (memref<1x1x4x4x8x8xi32, 2>) {
            %alloc = memref.alloc() : memref<1x1x4x4x8x8xi32, 2>
            air.execute_terminator %alloc : memref<1x1x4x4x8x8xi32, 2>
          } {id = 17 : i32}
          %5 = air.dma_memcpy_nd async [%async_token_11, %arg11] (%results_12[] [] [], %arg8[%c0, %c0, %results_6, %arg10] [%c4, %c8, %c4, %c8] [%c8, %c2048, %c512, %c1]) {broadcast_pattern = #set, id = 3 : i32} : (memref<1x1x4x8x4x8xi32, 2>, memref<1x1x64x512xi32, 1>)
          %6 = air.dma_memcpy_nd async [%async_token_13, %arg11] (%results_14[] [] [], %arg9[%c0, %c0, %arg10, %results_8] [%c4, %c4, %c8, %c8] [%c8, %c512, %c64, %c1]) {broadcast_pattern = #set1, id = 4 : i32} : (memref<1x1x4x4x8x8xi32, 2>, memref<1x1x512x64xi32, 1>)
          %7 = air.wait_all async [%5, %6]  {id = 1 : i32}
          scf.yield %7 : !air.async.token
        }
        air.herd_terminator
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// -----

// Broadcast into a single section in the affine set.

// CHECK: [[$SET0:#set[0-9]*]] = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
// CHECK-LABEL: @func2
// CHECK: %[[EVENT0:.*]] = affine.if [[$SET0]]
// CHECK: %[[EVENT1:.*]] = air.dma_memcpy_nd {{.*}}broadcast_set = [[$SET0]]{{.*}}
// CHECK: affine.yield %[[EVENT1]]
// CHECK-NEXT: } else {
// CHECK-NEXT: air.wait_all async
// CHECK-NEXT: affine.yield

#map = affine_map<()[s0] -> (s0 * 8)>
#map1 = affine_map<()[s0] -> (s0 * 32)>
#set = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 1 >= 0, s0 >= 0, -s0 >= 0)>
module {
  func.func @func2() {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c32) attributes {id = 3 : i32} {
      %1 = air.segment @segment_0 async  attributes {id = 2 : i32} {
        %c2 = arith.constant 2 : index
        %c1_0 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<8x2048xi32, 1 : i32>) {
          %alloc = memref.alloc() : memref<8x2048xi32, 1 : i32>
          air.execute_terminator %alloc : memref<8x2048xi32, 1 : i32>
        } {id = 1 : i32}
        %2 = air.herd @herd_0 async [%async_token]  tile (%arg4, %arg5) in (%arg6=%c1_0, %arg7=%c2) args(%arg8=%results) : memref<8x2048xi32, 1 : i32> attributes {id = 1 : i32} {
          %c64 = arith.constant 64 : index
          %c32_5 = arith.constant 32 : index
          %c1024 = arith.constant 1024 : index
          %c2048 = arith.constant 2048 : index
          %c8 = arith.constant 8 : index
          %c1_6 = arith.constant 1 : index
          %c0 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c256 = arith.constant 256 : index
          %async_token_7, %results_8 = air.execute -> (index) {
            %5 = affine.apply #map()[%arg4]
            air.execute_terminator %5 : index
          } {id = 3 : i32}
          %3 = air.wait_all async [%async_token_7]  {id = 2 : i32}
          %4 = scf.for %arg10 = %c0 to %c256 step %c4 iter_args(%arg11 = %3) -> (!air.async.token) {
            %async_token_11, %results_12 = air.execute [%arg11] -> (index) {
              %8 = affine.apply #map()[%arg10]
              air.execute_terminator %8 : index
            } {id = 5 : i32}
            %async_token_13, %results_14 = air.execute -> (memref<4x2x4x8xi32, 2 : i32>) {
              %alloc = memref.alloc() : memref<4x2x4x8xi32, 2 : i32>
              air.execute_terminator %alloc : memref<4x2x4x8xi32, 2 : i32>
            } {id = 6 : i32}
            %5 = air.dma_memcpy_nd async [%async_token_13, %async_token_11, %arg11] (%results_14[%c0] [%c256] [%c1_6], %arg8[%c0, %results_8, %results_12] [%c4, %c8, %c8] [%c8, %c2048, %c1_6]) {broadcast_pattern = #set, id = 1 : i32} : (memref<4x2x4x8xi32, 2 : i32>, memref<8x2048xi32, 1 : i32>)
            %async_token_17 = air.execute [%async_token_13] {
              memref.dealloc %results_14 : memref<4x2x4x8xi32, 2 : i32>
            } {id = 8 : i32}
            %7 = air.wait_all async [%arg11, %5]  {id = 1 : i32}
            scf.yield %7 : !air.async.token
          }
          air.herd_terminator
        }
        %async_token_3 = air.execute [%2] {
          memref.dealloc %results : memref<8x2048xi32, 1 : i32>
        } {id = 10 : i32}
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Multi-dimensional DMA broadcast specialization.

// CHECK: [[$SET0:#set[0-9]*]] = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
// CHECK: [[$SET1:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 1 == 0)>
// CHECK-LABEL: @func3
// CHECK: air.herd
// CHECK: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: %[[EVENT0:.*]] = affine.if [[$SET0]]
// CHECK: %[[CST0_0:.*]] = arith.constant 0 : index
// CHECK: %[[EVENT1:.*]] = air.dma_memcpy_nd {{.*}}(%{{.*}}[] [] [], %{{.*}}[%[[CST0]], %[[CST0]], %[[CST0]], %[[CST0_0]], %{{.*}}, %[[CST0]]]{{.*}}broadcast_set = [[$SET0]]{{.*}}
// CHECK: affine.yield %[[EVENT1]]
// CHECK-NEXT: } else {
// CHECK: %[[CST1:.*]] = arith.constant 1 : index
// CHECK: %[[EVENT1:.*]] = air.dma_memcpy_nd {{.*}}(%{{.*}}[] [] [], %{{.*}}[%[[CST0]], %[[CST0]], %[[CST0]], %[[CST1]], %{{.*}}, %[[CST0]]]{{.*}}broadcast_set = [[$SET1]]{{.*}}
// CHECK-NEXT: affine.yield

#map = affine_map<()[s0] -> (s0 * 8)>
#set = affine_set<(d0, d1)[s0] : (d0 >= 0, -d0 + 1 >= 0, d1 - s0 == 0, s0 >= 0, -s0 + 1 >= 0)>
module {
  func.func @func3() {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c16, %arg6=%c8) attributes {id = 4 : i32} {
      %1 = air.segment @segment_0 async  attributes {id = 3 : i32} {
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c8_0 = arith.constant 8 : index
        %async_token, %results = air.execute -> (memref<1x1x4x4x8x8xi32, 2>) {
          %alloc = memref.alloc() : memref<1x1x4x4x8x8xi32, 2>
          air.execute_terminator %alloc : memref<1x1x4x4x8x8xi32, 2>
        } {id = 1 : i32}
        %async_token_3, %results_4 = air.execute -> (memref<1x2x256x32xi32, 1>) {
          %alloc = memref.alloc() : memref<1x2x256x32xi32, 1>
          air.execute_terminator %alloc : memref<1x2x256x32xi32, 1>
        } {id = 3 : i32}
        %2 = scf.for %arg7 = %c1 to %c8_0 step %c1 iter_args(%arg8 = %async_token_3) -> (!air.async.token) {
          %3 = air.herd @herd_0 async [%arg8]  tile (%arg9, %arg10) in (%arg11=%c2, %arg12=%c2) args(%arg13=%results_4, %arg14=%results) : memref<1x2x256x32xi32, 1>, memref<1x1x4x4x8x8xi32, 2> attributes {id = 2 : i32} {
            %c1_11 = arith.constant 1 : index
            %c256 = arith.constant 256 : index
            %c8_12 = arith.constant 8 : index
            %c8192 = arith.constant 8192 : index
            %c16384 = arith.constant 16384 : index
            %c0 = arith.constant 0 : index
            %c32 = arith.constant 32 : index
            %c4 = arith.constant 4 : index
            %5 = air.wait_all async  {id = 4 : i32}
            %6 = scf.for %arg15 = %c0 to %c32 step %c4 iter_args(%arg16 = %5) -> (!air.async.token) {
              %async_token_13, %results_14 = air.execute [%arg16] -> (index) {
                %8 = affine.apply #map()[%arg15]
                air.execute_terminator %8 : index
              } {id = 7 : i32}
              %7 = air.dma_memcpy_nd async [%arg16, %async_token_13] (%arg14[] [] [], %arg13[%c0, %c0, %c0, %arg10, %results_14, %c0] [%c1_11, %c1_11, %c4, %c4, %c8_12, %c8_12] [%c16384, %c8192, %c8_12, %c256, %c32, %c1_11]) {broadcast_pattern = #set, id = 1 : i32} : (memref<1x1x4x4x8x8xi32, 2>, memref<1x2x256x32xi32, 1>)
              scf.yield %7 : !air.async.token
            }
            air.herd_terminator
          }
          %4 = air.wait_all async [%arg8, %3]  {id = 5 : i32}
          scf.yield %4 : !air.async.token
        }
        %async_token_8 = air.execute [%2] {
          memref.dealloc %results_4 : memref<1x2x256x32xi32, 1>
        } {id = 10 : i32}
        %async_token_10 = air.execute [%2] {
          memref.dealloc %results : memref<1x1x4x4x8x8xi32, 2>
        } {id = 12 : i32}
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
