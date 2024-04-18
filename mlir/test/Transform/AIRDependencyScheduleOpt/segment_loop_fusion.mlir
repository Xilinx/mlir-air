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

// CHECK: func.func private @linalg_fill_bf16(bf16, memref<1x1x16x16x4x4xbf16, 2 : i32>)
// CHECK-LABEL: func.func @func2
// CHECK: air.launch
// CHECK: air.segment
// CHECK: %[[EVENT1:.*]], %[[RESULT1:.*]] = air.execute -> (memref<1x1x16x16x4x4xbf16, 2 : i32>) {
// CHECK-NEXT: memref.alloc() : memref<1x1x16x16x4x4xbf16, 2 : i32>
// CHECK: %[[EVENT2:.*]] = air.herd @herd_0 async [%[[EVENT1]]]  tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2, %{{.*}}=%c2) args(%{{.*}}=%[[RESULT1]]) : memref<1x1x16x16x4x4xbf16, 2 : i32>
// CHECK: func.call @linalg_fill_bf16(%{{.*}}, %{{.*}}) : (bf16, memref<1x1x16x16x4x4xbf16, 2 : i32>) -> ()
// CHECK: air.herd_terminator
// CHECK: %[[EVENT3:.*]] = air.herd @herd_0 async [%[[EVENT2]]]  tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2, %{{.*}}=%c2) args(%{{.*}}=%[[RESULT1]]) : memref<1x1x16x16x4x4xbf16, 2 : i32>
// CHECK: func.call @linalg_fill_bf16(%{{.*}}, %{{.*}}) : (bf16, memref<1x1x16x16x4x4xbf16, 2 : i32>) -> ()
// CHECK: air.herd_terminator
// CHECK: %[[EVENT4:.*]] = air.herd @herd_0 async  tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2, %{{.*}}=%c2) args(%{{.*}}=%[[RESULT1]]) : memref<1x1x16x16x4x4xbf16, 2 : i32>
// CHECK-DAG: %[[CST256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[CST8192:.*]] = arith.constant 8192 : index
// CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[CST4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: air.channel.put async [{{.*}}]  @channel_12[%{{.*}}, %{{.*}}] (%{{.*}}[%[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]]] [%[[CST1]], %[[CST1]], %[[CST16]], %[[CST4]], %[[CST16]], %[[CST4]]] [%[[CST8192]], %[[CST8192]], %[[CST16]], %[[CST4]], %[[CST256]], %[[CST1]]]) {{.*}} : (memref<1x1x16x16x4x4xbf16, 2 : i32>)
// CHECK: air.herd_terminator
// CHECK: memref.dealloc %{{.*}} : memref<1x1x16x16x4x4xbf16, 2 : i32>

#map = affine_map<()[s0] -> (s0 * 128)>
air.channel @channel_12 [2, 2]
air.channel @channel_5 [1, 1] {broadcast_shape = [1, 2]}
air.channel @channel_4 [1, 1] {broadcast_shape = [1, 2]}
func.func private @linalg_fill_bf16(bf16, memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>) attributes {link_with = "mm.o", llvm.emit_c_interface}
func.func @func2() {
  %c4 = arith.constant 4 : index
  %0 = air.launch async (%arg3, %arg4) in (%arg5=%c4, %arg6=%c4) attributes {id = 1 : i32} {
    %1 = air.segment @segment_0 async  attributes {id = 2 : i32} {
      %c2 = arith.constant 2 : index
      %async_token, %results = air.execute -> (memref<1x1x32x32x4x4xbf16, 2 : i32>) {
        %alloc = memref.alloc() : memref<1x1x32x32x4x4xbf16, 2 : i32>
        air.execute_terminator %alloc : memref<1x1x32x32x4x4xbf16, 2 : i32>
      }
      %2 = air.herd @herd_0 async [%async_token]  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%results) : memref<1x1x32x32x4x4xbf16, 2 : i32> attributes {id = 3 : i32, link_with = "mm.o"} {
        %cst = arith.constant 0.000000e+00 : bf16
        %async_token_1, %results_2 = air.execute -> (index) {
          %5 = affine.apply #map()[%arg7]
          air.execute_terminator %5 : index
        }
        %async_token_3, %results_4 = air.execute -> (index) {
          %5 = affine.apply #map()[%arg8]
          air.execute_terminator %5 : index
        }
        %subview = memref.subview %arg11[0, 0, %results_4, %results_2, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
        %async_token_5 = air.execute {
          func.call @linalg_fill_bf16(%cst, %subview) : (bf16, memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>) -> ()
        }
        air.herd_terminator
      }
      %3 = air.herd @herd_0 async [%2]  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%results) : memref<1x1x32x32x4x4xbf16, 2 : i32> attributes {id = 4 : i32, link_with = "mm.o"} {
        %cst = arith.constant 0.000000e+00 : bf16
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        scf.for %arg12 = %c1 to %c16 step %c1 {
          %async_token_1, %results_2 = air.execute -> (index) {
            %5 = affine.apply #map()[%arg7]
            air.execute_terminator %5 : index
          }
          %async_token_3, %results_4 = air.execute -> (index) {
            %5 = affine.apply #map()[%arg8]
            air.execute_terminator %5 : index
          }
          %subview = memref.subview %arg11[0, 0, %results_4, %results_2, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x32x32x4x4xbf16, 2 : i32> to memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>
          %async_token_5 = air.execute [%async_token_1, %async_token_3] {
            func.call @linalg_fill_bf16(%cst, %subview) : (bf16, memref<1x1x16x16x4x4xbf16, strided<[16384, 16384, 512, 16, 4, 1], offset: ?>, 2 : i32>) -> ()
          }
        }
        air.herd_terminator
      }
      %4 = air.herd @herd_0 async  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%results) : memref<1x1x32x32x4x4xbf16, 2 : i32> attributes {id = 5 : i32} {
        %c1 = arith.constant 1 : index
        %c512 = arith.constant 512 : index
        %c4_1 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %c16384 = arith.constant 16384 : index
        %c0 = arith.constant 0 : index
        %async_token_2, %results_3 = air.execute -> (index) {
          %8 = affine.apply #map()[%arg7]
          air.execute_terminator %8 : index
        }
        %async_token_4, %results_5 = air.execute -> (index) {
          %8 = affine.apply #map()[%arg8]
          air.execute_terminator %8 : index
        }
        %5 = air.wait_all async 
        %6 = air.wait_all async 
        %7 = air.channel.put async [%async_token_2, %async_token_4, %5, %6]  @channel_12[%arg7, %arg8] (%arg11[%c0, %c0, %results_5, %results_3, %c0, %c0] [%c1, %c1, %c16, %c4_1, %c16, %c4_1] [%c16384, %c16384, %c16, %c4_1, %c512, %c1]) {id = 27 : i32} : (memref<1x1x32x32x4x4xbf16, 2 : i32>)
        air.herd_terminator
      }
      %async_token_0 = air.execute [%4] {
        memref.dealloc %results : memref<1x1x32x32x4x4xbf16, 2 : i32>
      }
      air.segment_terminator
    }
    air.launch_terminator
  }
  return
}
