//===- segment_loop_fusion.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-loop-fusion %s | FileCheck %s

// Fuse scf.for loops in air.segment.

// CHECK-LABEL: func.func @func0
// CHECK: air.launch
// CHECK: air.segment
// CHECK: memref.alloc()
// CHECK: scf.for
// CHECK: memref.alloc()
// CHECK: memref.alloc()
// CHECK: %[[EVENT0:.*]] = air.channel.get{{.*}}@channel_
// CHECK: scf.for{{.*}}iter_args({{.*}} = %[[EVENT0]])
// CHECK: %[[EVENT1:.*]] = air.channel.put{{.*}}@channel_
// CHECK: scf.for{{.*}}iter_args({{.*}} = %[[EVENT0]])
// CHECK: %[[EVENT2:.*]] = air.channel.put{{.*}}@channel_
// CHECK: %[[EVENT3:.*]] = air.channel.get{{.*}}@channel_
// CHECK: scf.for{{.*}}iter_args({{.*}} = %[[EVENT3]])
// CHECK: %[[EVENT4:.*]] = air.channel.put{{.*}}@channel_
// CHECK: scf.for{{.*}}iter_args({{.*}} = %[[EVENT3]])
// CHECK: %[[EVENT5:.*]] = air.channel.put{{.*}}@channel_

func.func @func0() {
  %c32 = arith.constant 32 : index
  %0 = air.launch async (%arg3, %arg4) in (%arg5=%c32, %arg6=%c32) {
    %6 = air.segment @segment_0 async {
      %c32_9 = arith.constant 32 : index
      %c256_10 = arith.constant 256 : index
      %c0_11 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c2048_13 = arith.constant 2048 : index
      %c64_14 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %7 = air.wait_all async 
      %8 = air.wait_all async 
      %async_token_15, %results_16 = air.execute -> (memref<64x64xi32, 1>) {
        %alloc = memref.alloc() : memref<64x64xi32, 1>
        air.execute_terminator %alloc : memref<64x64xi32, 1>
      }
      %async_token_17, %results_18 = air.execute [%async_token_15] -> (memref<64x256xi32, 1>) {
        %alloc = memref.alloc() : memref<64x256xi32, 1>
        air.execute_terminator %alloc : memref<64x256xi32, 1>
      }
      %async_token_19, %results_20 = air.execute [%async_token_17] -> (memref<256x64xi32, 1>) {
        %alloc = memref.alloc() : memref<256x64xi32, 1>
        air.execute_terminator %alloc : memref<256x64xi32, 1>
      }
      %9 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_17) -> (!air.async.token) {
        %18 = scf.for %arg12 = %c0_11 to %c256_10 step %c32_9 iter_args(%arg13 = %arg11) -> (!air.async.token) {
          %19 = air.channel.put async [%arg13]  @channel_0[] (%results_18[%c0_11, %arg12] [%c32_9, %c32_9] [%c256_10, %c1_12]) : (memref<64x256xi32, 1>)
          scf.yield %19 : !air.async.token
        }
        scf.yield %18 : !air.async.token
      }
      %10 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_17) -> (!air.async.token) {
        %18 = scf.for %arg12 = %c0_11 to %c256_10 step %c32_9 iter_args(%arg13 = %arg11) -> (!air.async.token) {
          %19 = air.channel.put async [%arg13]  @channel_1[] (%results_18[%c32_9, %arg12] [%c32_9, %c32_9] [%c256_10, %c1_12]) : (memref<64x256xi32, 1>)
          scf.yield %19 : !air.async.token
        }
        scf.yield %18 : !air.async.token
      }
      %11 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_19) -> (!air.async.token) {
        %18 = scf.for %arg12 = %c0_11 to %c256_10 step %c32_9 iter_args(%arg13 = %arg11) -> (!air.async.token) {
          %19 = air.channel.put async [%arg13]  @channel_2[] (%results_20[%arg12, %c0_11] [%c32_9, %c32_9] [%c64_14, %c1_12]) : (memref<256x64xi32, 1>)
          scf.yield %19 : !air.async.token
        }
        scf.yield %18 : !air.async.token
      }
      %12 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_19) -> (!air.async.token) {
        %18 = scf.for %arg12 = %c0_11 to %c256_10 step %c32_9 iter_args(%arg13 = %arg11) -> (!air.async.token) {
          %19 = air.channel.put async [%arg13]  @channel_3[] (%results_20[%arg12, %c32_9] [%c32_9, %c32_9] [%c64_14, %c1_12]) : (memref<256x64xi32, 1>)
          scf.yield %19 : !air.async.token
        }
        scf.yield %18 : !air.async.token
      }
      %13 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_17) -> (!air.async.token) {
        %18 = air.channel.get async [%arg11, %7]  @channel_4[] (%results_18[] [] []) : (memref<64x256xi32, 1>)
        scf.yield %18 : !air.async.token
      }
      %14 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_19) -> (!air.async.token) {
        %18 = air.channel.get async [%arg11, %8]  @channel_5[] (%results_20[] [] []) : (memref<256x64xi32, 1>)
        scf.yield %18 : !air.async.token
      }
      %async_token_21 = air.execute [%async_token_19] {
        memref.dealloc %results_20 : memref<256x64xi32, 1>
      }
      %async_token_22 = air.execute [%async_token_17] {
        memref.dealloc %results_18 : memref<64x256xi32, 1>
      }
      %async_token_23 = air.execute [%7, %8] {
        memref.dealloc %results_16 : memref<64x64xi32, 1>
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// Memref shrinkage via data access pattern analysis.

// CHECK-LABEL: func.func @func1
// CHECK: air.launch
// CHECK: air.segment
// CHECK: scf.for
// CHECK: memref.alloc() : memref<64x256xi32, 1>
// CHECK: %[[EVENT0:.*]] = air.channel.get {{.*}} : (memref<64x256xi32, 1>)
// CHECK: scf.for{{.*}}iter_args({{.*}} = %[[EVENT0]])
// CHECK: %[[EVENT1:.*]] = air.channel.put {{.*}} : (memref<64x256xi32, 1>)
// CHECK: scf.for{{.*}}iter_args({{.*}} = %[[EVENT0]])
// CHECK: %[[EVENT2:.*]] = air.channel.put {{.*}} : (memref<64x256xi32, 1>)

func.func @func1() {
  %c32 = arith.constant 32 : index
  %0 = air.launch async (%arg3, %arg4) in (%arg5=%c32, %arg6=%c32) {
    %6 = air.segment @segment_0 async {
      %c32_9 = arith.constant 32 : index
      %c256_10 = arith.constant 256 : index
      %c0_11 = arith.constant 0 : index
      %c1_12 = arith.constant 1 : index
      %c2048_13 = arith.constant 2048 : index
      %c64_14 = arith.constant 64 : index
      %c2 = arith.constant 2 : index
      %7 = air.wait_all async 
      %8 = air.wait_all async 
      %async_token_17, %results_18 = air.execute -> (memref<64x2048xi32, 1>) {
        %alloc = memref.alloc() : memref<64x2048xi32, 1>
        air.execute_terminator %alloc : memref<64x2048xi32, 1>
      }
      %9 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_17) -> (!air.async.token) {
        %18 = scf.for %arg12 = %c0_11 to %c256_10 step %c32_9 iter_args(%arg13 = %arg11) -> (!air.async.token) {
          %19 = air.channel.put async [%arg13]  @channel_0[] (%results_18[%c0_11, %arg12] [%c32_9, %c32_9] [%c2048_13, %c1_12]) : (memref<64x2048xi32, 1>)
          scf.yield %19 : !air.async.token
        }
        scf.yield %18 : !air.async.token
      }
      %10 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_17) -> (!air.async.token) {
        %18 = scf.for %arg12 = %c0_11 to %c256_10 step %c32_9 iter_args(%arg13 = %arg11) -> (!air.async.token) {
          %19 = air.channel.put async [%arg13]  @channel_1[] (%results_18[%c32_9, %arg12] [%c32_9, %c32_9] [%c2048_13, %c1_12]) : (memref<64x2048xi32, 1>)
          scf.yield %19 : !air.async.token
        }
        scf.yield %18 : !air.async.token
      }
      %13 = scf.for %arg10 = %c0_11 to %c2048_13 step %c256_10 iter_args(%arg11 = %async_token_17) -> (!air.async.token) {
        %18 = air.channel.get async [%arg11, %7]  @channel_4[] (%results_18[%c0_11, %c0_11] [%c64_14, %c256_10] [%c2048_13, %c1_12]) : (memref<64x2048xi32, 1>)
        scf.yield %18 : !air.async.token
      }
      %async_token_22 = air.execute [%async_token_17] {
        memref.dealloc %results_18 : memref<64x2048xi32, 1>
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}

// Multiple herds.

// CHECK-LABEL: func.func @func2
// CHECK: air.launch
// CHECK: air.segment
// CHECK: scf.for
// CHECK: memref.alloc() : memref<1x1x64x64xbf16, 1 : i32>
// CHECK: %[[EVENT0:.*]] = air.channel.get {{.*}} : (memref<1x1x64x64xbf16, 1 : i32>)
// CHECK-NEXT: %[[EVENT1:.*]] = air.channel.put {{.*}} : (memref<1x1x64x64xbf16, 1 : i32>)
// CHECK-NEXT: %[[EVENT2:.*]] = air.channel.put {{.*}} : (memref<1x1x64x64xbf16, 1 : i32>)
// CHECK: memref.dealloc %{{.*}} : memref<1x1x64x64xbf16, 1 : i32>

#map = affine_map<()[s0] -> (s0 * 128)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
#map2 = affine_map<()[s0] -> (s0 * 64)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
func.func @func2(%arg0: memref<512x1024xbf16>) {
  %c4 = arith.constant 4 : index
  %0 = air.launch async (%arg3, %arg4) in (%arg5=%c4, %arg6=%c4) args(%arg7=%arg0) : memref<512x1024xbf16> {
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    %c1024 = arith.constant 1024 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %async_token_2, %results_3 = air.execute -> (index) {
      %7 = affine.apply #map()[%arg3]
      air.execute_terminator %7 : index
    }
    %1 = air.wait_all async 
    %2 = scf.for %arg10 = %c0 to %c16 step %c1 iter_args(%arg11 = %1) -> (!air.async.token) {
      %async_token_10, %results_11 = air.execute [%arg11] -> (index) {
        %8 = affine.apply #map1()[%arg10]
        air.execute_terminator %8 : index
      }
      %7 = air.channel.put async [%async_token_10, %async_token_2]  @channel_10[] (%arg7[%results_3, %results_11] [%c128, %c64] [%c1024, %c1]) {id = 3 : i32} : (memref<512x1024xbf16>)
      scf.yield %7 : !air.async.token
    }
    %6 = air.segment @segment_0 async  attributes {id = 2 : i32} {
      %c8192 = arith.constant 8192 : index
      %c8 = arith.constant 8 : index
      %c256 = arith.constant 256 : index
      %c4_10 = arith.constant 4 : index
      %c16384 = arith.constant 16384 : index
      %c64_11 = arith.constant 64 : index
      %c0_14 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1_15 = arith.constant 1 : index
      %c16_16 = arith.constant 16 : index
      %async_token_19, %results_20 = air.execute -> (memref<1x1x8x16x4x8xbf16, 2 : i32>) {
        %alloc = memref.alloc() : memref<1x1x8x16x4x8xbf16, 2 : i32>
        air.execute_terminator %alloc : memref<1x1x8x16x4x8xbf16, 2 : i32>
      }
      %async_token_23, %results_24 = air.execute -> (memref<1x1x128x64xbf16, 1 : i32>) {
        %alloc = memref.alloc() : memref<1x1x128x64xbf16, 1 : i32>
        air.execute_terminator %alloc : memref<1x1x128x64xbf16, 1 : i32>
      }
      %7 = air.herd @herd_0 async [%async_token_19, %async_token_23]  tile (%arg10, %arg11) in (%arg12=%c2, %arg13=%c2) args(%arg14=%results_20) : memref<1x1x8x16x4x8xbf16, 2 : i32> {
        %cst = arith.constant 0.000000e+00 : bf16
        %async_token_35, %results_36 = air.execute -> (index) {
          %21 = affine.apply #map2()[%arg10]
          air.execute_terminator %21 : index
        }
        %async_token_37, %results_38 = air.execute -> (index) {
          %21 = affine.apply #map2()[%arg11]
          air.execute_terminator %21 : index
        }
        %19 = affine.if #set()[%arg10, %arg11] -> !air.async.token {
          %21 = air.channel.get async  @channel_4[%arg10, %arg11] (%arg14[] [] []) {id = 12 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
          affine.yield %21 : !air.async.token
        } else {
          %21 = air.channel.get async  @channel_5[%arg10, %arg11] (%arg14[] [] []) {id = 13 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
          affine.yield %21 : !air.async.token
        }
        air.herd_terminator
      }
      %8 = air.wait_all async [%async_token_23, %7] 
      %9 = scf.for %arg10 = %c0_14 to %c16_16 step %c1_15 iter_args(%arg11 = %async_token_23) -> (!air.async.token) {
        %19 = air.channel.get async [%arg11]  @channel_10[] (%results_24[] [] []) {id = 16 : i32} : (memref<1x1x128x64xbf16, 1 : i32>)
        scf.yield %19 : !air.async.token
      }
      %11 = scf.for %arg10 = %c0_14 to %c16_16 step %c1_15 iter_args(%arg11 = %async_token_23) -> (!air.async.token) {
        %19 = air.channel.put async [%async_token_23]  @channel_4[] (%results_24[%c0_14, %c0_14, %c0_14, %c0_14, %c0_14, %c0_14] [%c1_15, %c1_15, %c8, %c16_16, %c4_10, %c8] [%c8192, %c8192, %c8, %c256, %c64_11, %c1_15]) {id = 18 : i32} : (memref<1x1x128x64xbf16, 1 : i32>)
        scf.yield %19 : !air.async.token
      }
      %12 = scf.for %arg10 = %c0_14 to %c16_16 step %c1_15 iter_args(%arg11 = %async_token_23) -> (!air.async.token) {
        %19 = air.channel.put async [%async_token_23]  @channel_5[] (%results_24[%c0_14, %c0_14, %c0_14, %c0_14, %c64_11, %c0_14] [%c1_15, %c1_15, %c8, %c16_16, %c4_10, %c8] [%c8192, %c8192, %c8, %c256, %c64_11, %c1_15]) {id = 19 : i32} : (memref<1x1x128x64xbf16, 1 : i32>)
        scf.yield %19 : !air.async.token
      }
      %15 = air.herd @herd_0 async [%8]  tile (%arg10, %arg11) in (%arg12=%c2, %arg13=%c2) args(%arg14=%results_20) : memref<1x1x8x16x4x8xbf16, 2 : i32> {
        %c1_35 = arith.constant 1 : index
        %c16_36 = arith.constant 16 : index
        scf.for %arg17 = %c1_35 to %c16_36 step %c1_35 {
          %19 = affine.if #set()[%arg10, %arg11] -> !air.async.token {
            %21 = air.channel.get async  @channel_4[%arg10, %arg11] (%arg14[] [] []) {id = 22 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
            affine.yield %21 : !air.async.token
          } else {
            %21 = air.channel.get async  @channel_5[%arg10, %arg11] (%arg14[] [] []) {id = 23 : i32} : (memref<1x1x8x16x4x8xbf16, 2 : i32>)
            affine.yield %21 : !air.async.token
          }
        }
        air.herd_terminator
      }
      %async_token_31 = air.execute [%15] {
        memref.dealloc %results_24 : memref<1x1x128x64xbf16, 1 : i32>
      }
      %async_token_33 = air.execute [%15] {
        memref.dealloc %results_20 : memref<1x1x8x16x4x8xbf16, 2 : i32>
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
