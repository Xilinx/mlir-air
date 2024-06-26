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
    }
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
    }
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
// CHECK: %[[EVENT3:.*]] = air.herd @herd_0 async [%[[EVENT2]]]  tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2, %{{.*}}=%c2) args(%{{.*}}=%[[RESULT1]]) : memref<1x1x16x16x4x4xbf16, 2 : i32>
// CHECK: func.call @linalg_fill_bf16(%{{.*}}, %{{.*}}) : (bf16, memref<1x1x16x16x4x4xbf16, 2 : i32>) -> ()
// CHECK: %[[EVENT4:.*]] = air.herd @herd_0 async  tile (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2, %{{.*}}=%c2) args(%{{.*}}=%[[RESULT1]]) : memref<1x1x16x16x4x4xbf16, 2 : i32>
// CHECK-DAG: %[[CST256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[CST8192:.*]] = arith.constant 8192 : index
// CHECK-DAG: %[[CST1:.*]] = arith.constant 1 : index
// CHECK-DAG: %[[CST4:.*]] = arith.constant 4 : index
// CHECK-DAG: %[[CST16:.*]] = arith.constant 16 : index
// CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: air.channel.put async [{{.*}}]  @channel_12[%{{.*}}, %{{.*}}] (%{{.*}}[%[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]], %[[CST0]]] [%[[CST1]], %[[CST1]], %[[CST16]], %[[CST4]], %[[CST16]], %[[CST4]]] [%[[CST8192]], %[[CST8192]], %[[CST16]], %[[CST4]], %[[CST256]], %[[CST1]]]) {{.*}} : (memref<1x1x16x16x4x4xbf16, 2 : i32>)
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
      }
      %async_token_0 = air.execute [%4] {
        memref.dealloc %results : memref<1x1x32x32x4x4xbf16, 2 : i32>
      }
    }
  }
  return
}

// Vector transfer_read/write access to memref, L1 memref allocated globally.

// CHECK-LABEL: func.func @func3
// CHECK: memref.alloc() : memref<1x1x16x16x4x4xbf16, 2 : i32>
// CHECK: air.herd @herd_0
// CHECK: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: affine.apply #map{{.*}}()[%{{.*}}, %[[CST0]]]
// CHECK: affine.apply #map{{.*}}()[%{{.*}}, %[[CST0]]]
// CHECK: vector.transfer_read{{.*}}memref<1x1x16x16x4x4xbf16, 2 : i32>, vector<1x1x1x1x4x4xbf16>
// CHECK: vector.transfer_write{{.*}}vector<1x1x1x1x4x4xbf16>, memref<1x1x16x16x4x4xbf16, 2 : i32>

#map8 = affine_map<()[s0] -> (s0 * 16)>
#map9 = affine_map<()[s0, s1] -> (s0 + s1 * 16)>
func.func @func3() {
  %c8 = arith.constant 8 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c8, %arg7=%c8) attributes {id = 1 : i32} {
    %1 = air.segment @segment_0 async  attributes {id = 2 : i32} {
      %c2 = arith.constant 2 : index
      %async_token, %results = air.execute -> (memref<1x1x32x32x4x4xbf16, 2 : i32>) {
        %alloc = memref.alloc() : memref<1x1x32x32x4x4xbf16, 2 : i32>
        air.execute_terminator %alloc : memref<1x1x32x32x4x4xbf16, 2 : i32>
      }
      %2 = air.herd @herd_0 async  tile (%arg8, %arg9) in (%arg10=%c2, %arg11=%c2) args(%arg12=%results) : memref<1x1x32x32x4x4xbf16, 2 : i32> attributes {id = 5 : i32} {
        %c0 = arith.constant 0 : index
        %cst = arith.constant 0.000000e+00 : bf16
        %c8_1 = arith.constant 8 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %async_token_2, %results_3 = air.execute -> (index) {
          %5 = affine.apply #map8()[%arg8]
          air.execute_terminator %5 : index
        }
        %async_token_4, %results_5 = air.execute -> (index) {
          %5 = affine.apply #map8()[%arg9]
          air.execute_terminator %5 : index
        }
        %3 = air.wait_all async 
        %4 = scf.for %arg13 = %c0 to %c16 step %c1 iter_args(%arg14 = %3) -> (!air.async.token) {
          %5 = scf.for %arg15 = %c0 to %c16 step %c1 iter_args(%arg16 = %arg14) -> (!air.async.token) {
            %6 = scf.for %arg17 = %c0 to %c8_1 step %c1 iter_args(%arg18 = %arg16) -> (!air.async.token) {
              %async_token_8, %results_9 = air.execute [%arg18] -> (index) {
                %8 = affine.apply #map9()[%arg15, %arg9]
                air.execute_terminator %8 : index
              }
              %async_token_10, %results_11 = air.execute [%arg18] -> (index) {
                %8 = affine.apply #map9()[%arg13, %arg8]
                air.execute_terminator %8 : index
              }
              %async_token_12, %results_13 = air.execute [%async_token_10, %async_token_8] -> (vector<1x1x1x1x4x4xbf16>) {
                %8 = vector.transfer_read %arg12[%c0, %c0, %results_9, %results_11, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x1x32x32x4x4xbf16, 2 : i32>, vector<1x1x1x1x4x4xbf16>
                air.execute_terminator %8 : vector<1x1x1x1x4x4xbf16>
              }
              %async_token_14 = air.execute [%async_token_12] {
                vector.transfer_write %results_13, %arg12[%c0, %c0, %results_9, %results_11, %c0, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x4x4xbf16>, memref<1x1x32x32x4x4xbf16, 2 : i32>
              }
              %7 = air.wait_all async 
              scf.yield %7 : !air.async.token
            }
            scf.yield %6 : !air.async.token
          }
          scf.yield %5 : !air.async.token
        }
        %async_token_6, %results_7 = air.execute [%4, %async_token_4, %async_token_2] -> (vector<1x1x16x16x4x4xbf16>) {
          %5 = vector.transfer_read %arg12[%c0, %c0, %results_5, %results_3, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<1x1x32x32x4x4xbf16, 2 : i32>, vector<1x1x16x16x4x4xbf16>
          air.execute_terminator %5 : vector<1x1x16x16x4x4xbf16>
        }
      }
      %async_token_0 = air.execute {
        memref.dealloc %results : memref<1x1x32x32x4x4xbf16, 2 : i32>
      }
    }
  }
  return
}

// Vector transfer_read/write access to memref, L1 memref allocated globally. Alternative memref layout.

// CHECK-LABEL: func.func @func4
// CHECK: memref.alloc() : memref<1x1x16x16x4x4xf32, 2 : i32>
// CHECK: memref.alloc() : memref<1x1x16x8x8x4xbf16, 2 : i32>
// CHECK: memref.alloc() : memref<1x1x8x16x4x8xbf16, 2 : i32>
// CHECK: air.herd @herd_0
// CHECK: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: vector.transfer_read %{{.*}}[%[[CST0]], %[[CST0]], %{{.*}}, %{{.*}}, %[[CST0]], %[[CST0]]], %{{.*}} {in_bounds = [true, true, true, true, true, true]} : memref<1x1x8x16x4x8xbf16, 2 : i32>, vector<1x1x1x1x4x8xbf16>
// CHECK: vector.transfer_read %{{.*}}[%[[CST0]], %[[CST0]], %{{.*}}, %{{.*}}, %[[CST0]], %[[CST0]]], %{{.*}} {in_bounds = [true, true, true, true, true, true]} : memref<1x1x16x8x8x4xbf16, 2 : i32>, vector<1x1x1x1x8x4xbf16>
// CHECK: vector.transfer_read %{{.*}}[%[[CST0]], %[[CST0]], %{{.*}}, %{{.*}}, %[[CST0]], %[[CST0]]], %{{.*}} {in_bounds = [true, true, true, true, true, true]} : memref<1x1x16x16x4x4xf32, 2 : i32>, vector<1x1x1x1x4x4xf32>
// CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%[[CST0]], %[[CST0]], %{{.*}}, %{{.*}}, %[[CST0]], %[[CST0]]] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x4x4xf32>, memref<1x1x16x16x4x4xf32, 2 : i32>

#map10 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map11 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map12 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
func.func @func4(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>, %arg2: memref<512x512xf32>) {
  %c4 = arith.constant 4 : index
  %0 = air.launch async (%arg3, %arg4) in (%arg5=%c4, %arg6=%c4) attributes {id = 1 : i32} {
    %1 = air.segment @segment_0 async  attributes {id = 2 : i32} {
      %c2 = arith.constant 2 : index
      %async_token, %results = air.execute -> (memref<2x2x16x16x4x4xf32, 2 : i32>) {
        %alloc = memref.alloc() : memref<2x2x16x16x4x4xf32, 2 : i32>
        air.execute_terminator %alloc : memref<2x2x16x16x4x4xf32, 2 : i32>
      }
      %async_token_0, %results_1 = air.execute -> (memref<1x1x16x8x8x4xbf16, 2 : i32>) {
        %alloc = memref.alloc() : memref<1x1x16x8x8x4xbf16, 2 : i32>
        air.execute_terminator %alloc : memref<1x1x16x8x8x4xbf16, 2 : i32>
      }
      %async_token_2, %results_3 = air.execute -> (memref<1x1x8x16x4x8xbf16, 2 : i32>) {
        %alloc = memref.alloc() : memref<1x1x8x16x4x8xbf16, 2 : i32>
        air.execute_terminator %alloc : memref<1x1x8x16x4x8xbf16, 2 : i32>
      }
      %2 = air.herd @herd_0 async [%async_token, %async_token_0, %async_token_2]  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%results_3, %arg12=%results_1, %arg13=%results) : memref<1x1x8x16x4x8xbf16, 2 : i32>, memref<1x1x16x8x8x4xbf16, 2 : i32>, memref<2x2x16x16x4x4xf32, 2 : i32> attributes {id = 3 : i32} {
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %cst_4 = arith.constant 0.000000e+00 : bf16
        %c8 = arith.constant 8 : index
        %c1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %subview = memref.subview %arg13[%arg7, %arg8, 0, 0, 0, 0] [1, 1, 16, 16, 4, 4] [1, 1, 1, 1, 1, 1] : memref<2x2x16x16x4x4xf32, 2 : i32> to memref<1x1x16x16x4x4xf32, strided<[8192, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>
        %async_token_5 = air.execute {
          linalg.fill ins(%cst : f32) outs(%subview : memref<1x1x16x16x4x4xf32, strided<[8192, 4096, 256, 16, 4, 1], offset: ?>, 2 : i32>)
        }
        %3 = air.wait_all async 
        %4 = scf.for %arg14 = %c0 to %c16 step %c1 iter_args(%arg15 = %3) -> (!air.async.token) {
          %5 = scf.for %arg16 = %c0 to %c16 step %c1 iter_args(%arg17 = %arg15) -> (!air.async.token) {
            %6 = scf.for %arg18 = %c0 to %c8 step %c1 iter_args(%arg19 = %arg17) -> (!air.async.token) {
              %async_token_6, %results_7 = air.execute [%arg19] -> (vector<1x1x1x1x4x8xbf16>) {
                %11 = vector.transfer_read %arg11[%c0, %c0, %arg18, %arg14, %c0, %c0], %cst_4 {in_bounds = [true, true, true, true, true, true]} : memref<1x1x8x16x4x8xbf16, 2 : i32>, vector<1x1x1x1x4x8xbf16>
                air.execute_terminator %11 : vector<1x1x1x1x4x8xbf16>
              }
              %async_token_8, %results_9 = air.execute [%arg19] -> (vector<1x1x1x1x8x4xbf16>) {
                %11 = vector.transfer_read %arg12[%c0, %c0, %arg16, %arg18, %c0, %c0], %cst_4 {in_bounds = [true, true, true, true, true, true]} : memref<1x1x16x8x8x4xbf16, 2 : i32>, vector<1x1x1x1x8x4xbf16>
                air.execute_terminator %11 : vector<1x1x1x1x8x4xbf16>
              }
              %async_token_10, %results_11 = air.execute [%arg19] -> (vector<1x1x1x1x4x4xf32>) {
                %11 = vector.transfer_read %arg13[%arg7, %arg8, %arg16, %arg14, %c0, %c0], %cst {in_bounds = [true, true, true, true, true, true]} : memref<2x2x16x16x4x4xf32, 2 : i32>, vector<1x1x1x1x4x4xf32>
                air.execute_terminator %11 : vector<1x1x1x1x4x4xf32>
              }
              %7 = arith.extf %results_7 : vector<1x1x1x1x4x8xbf16> to vector<1x1x1x1x4x8xf32>
              %8 = arith.extf %results_9 : vector<1x1x1x1x8x4xbf16> to vector<1x1x1x1x8x4xf32>
              %9 = vector.contract {indexing_maps = [#map10, #map11, #map12], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %7, %8, %results_11 : vector<1x1x1x1x4x8xf32>, vector<1x1x1x1x8x4xf32> into vector<1x1x1x1x4x4xf32>
              %async_token_12 = air.execute [%async_token_10] {
                vector.transfer_write %9, %arg13[%arg7, %arg8, %arg16, %arg14, %c0, %c0] {in_bounds = [true, true, true, true, true, true]} : vector<1x1x1x1x4x4xf32>, memref<2x2x16x16x4x4xf32, 2 : i32>
              }
              %10 = air.wait_all async [%async_token_6, %async_token_8, %async_token_12] 
              scf.yield %10 : !air.async.token
            }
            scf.yield %6 : !air.async.token
          }
          scf.yield %5 : !air.async.token
        }
      }
    }
  }
  return
}

// Vector transfer_read/write access to memref, L1 memref allocated in herd.

// CHECK-LABEL: func.func @func5
// CHECK: air.herd @herd_0
// CHECK: memref.alloc() : memref<16x16x4x4xf32, 2 : i32>
// CHECK: memref.alloc() : memref<8x16x4x8xbf16, 2 : i32>
// CHECK: memref.alloc() : memref<16x8x8x4xbf16, 2 : i32>
// CHECK: vector.transfer_read{{.*}}memref<8x16x4x8xbf16, 2 : i32>, vector<1x1x4x8xbf16>
// CHECK: vector.transfer_read{{.*}}memref<16x8x8x4xbf16, 2 : i32>, vector<1x1x8x4xbf16>
// CHECK: vector.transfer_read{{.*}}memref<16x16x4x4xf32, 2 : i32>, vector<1x1x4x4xf32>
// CHECK: vector.transfer_write{{.*}}vector<1x1x4x4xf32>, memref<16x16x4x4xf32, 2 : i32>

#map5 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d0, d3, d5)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d2, d5, d4)>
#map7 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d0, d3, d4)>
air.channel @channel_6 [2, 2]
func.func @func5() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) attributes {id = 1 : i32} {
    %1 = air.segment @segment_0 async  attributes {id = 2 : i32} {
      %c2 = arith.constant 2 : index
      %2 = air.herd @herd_0 async  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) attributes {id = 3 : i32} {
        %c256 = arith.constant 256 : index
        %c4 = arith.constant 4 : index
        %c64 = arith.constant 64 : index
        %cst = arith.constant 0.000000e+00 : f32
        %c0 = arith.constant 0 : index
        %c8 = arith.constant 8 : index
        %c16 = arith.constant 16 : index
        %c1_0 = arith.constant 1 : index
        %cst_1 = arith.constant 0.000000e+00 : bf16
        %c288 = arith.constant 288 : index
        %async_token, %results = air.execute -> (memref<16x16x4x4xf32, 2 : i32>) {
          %alloc = memref.alloc() : memref<16x16x4x4xf32, 2 : i32>
          air.execute_terminator %alloc : memref<16x16x4x4xf32, 2 : i32>
        }
        %3 = scf.for %arg11 = %c0 to %c288 step %c8 iter_args(%arg12 = %async_token) -> (!air.async.token) {
          %async_token_3, %results_4 = air.execute -> (memref<8x16x4x8xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<8x16x4x8xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<8x16x4x8xbf16, 2 : i32>
          }
          %async_token_5, %results_6 = air.execute -> (memref<16x8x8x4xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<16x8x8x4xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<16x8x8x4xbf16, 2 : i32>
          }
          %5 = air.wait_all async [%arg12, %async_token_5, %async_token_3] 
          %6 = scf.for %arg13 = %c0 to %c16 step %c1_0 iter_args(%arg14 = %5) -> (!air.async.token) {
            %7 = scf.for %arg15 = %c0 to %c16 step %c1_0 iter_args(%arg16 = %arg14) -> (!air.async.token) {
              %8 = scf.for %arg17 = %c0 to %c8 step %c1_0 iter_args(%arg18 = %arg16) -> (!air.async.token) {
                %async_token_9, %results_10 = air.execute [%arg18] -> (vector<1x1x4x8xbf16>) {
                  %13 = vector.transfer_read %results_4[%arg17, %arg13, %c0, %c0], %cst_1 {in_bounds = [true, true, true, true]} : memref<8x16x4x8xbf16, 2 : i32>, vector<1x1x4x8xbf16>
                  air.execute_terminator %13 : vector<1x1x4x8xbf16>
                }
                %async_token_11, %results_12 = air.execute [%arg18] -> (vector<1x1x8x4xbf16>) {
                  %13 = vector.transfer_read %results_6[%arg15, %arg17, %c0, %c0], %cst_1 {in_bounds = [true, true, true, true]} : memref<16x8x8x4xbf16, 2 : i32>, vector<1x1x8x4xbf16>
                  air.execute_terminator %13 : vector<1x1x8x4xbf16>
                }
                %async_token_13, %results_14 = air.execute [%arg18] -> (vector<1x1x4x4xf32>) {
                  %13 = vector.transfer_read %results[%arg15, %arg13, %c0, %c0], %cst {in_bounds = [true, true, true, true]} : memref<16x16x4x4xf32, 2 : i32>, vector<1x1x4x4xf32>
                  air.execute_terminator %13 : vector<1x1x4x4xf32>
                }
                %9 = arith.extf %results_10 : vector<1x1x4x8xbf16> to vector<1x1x4x8xf32>
                %10 = arith.extf %results_12 : vector<1x1x8x4xbf16> to vector<1x1x8x4xf32>
                %11 = vector.contract {indexing_maps = [#map5, #map6, #map7], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"], kind = #vector.kind<add>} %9, %10, %results_14 : vector<1x1x4x8xf32>, vector<1x1x8x4xf32> into vector<1x1x4x4xf32>
                %async_token_15 = air.execute [%async_token_13] {
                  vector.transfer_write %11, %results[%arg15, %arg13, %c0, %c0] {in_bounds = [true, true, true, true]} : vector<1x1x4x4xf32>, memref<16x16x4x4xf32, 2 : i32>
                }
                %12 = air.wait_all async [%async_token_9, %async_token_11, %async_token_15] 
                scf.yield %12 : !air.async.token
              }
              scf.yield %8 : !air.async.token
            }
            scf.yield %7 : !air.async.token
          }
          %async_token_7 = air.execute [%6] {
            memref.dealloc %results_4 : memref<8x16x4x8xbf16, 2 : i32>
          }
          %async_token_8 = air.execute [%6] {
            memref.dealloc %results_6 : memref<16x8x8x4xbf16, 2 : i32>
          }
          scf.yield %6 : !air.async.token
        }
        %4 = air.channel.put async [%3]  @channel_6[%arg7, %arg8] (%results[%c0, %c0, %c0] [%c64, %c16, %c4] [%c4, %c256, %c1_0]) {id = 23 : i32} : (memref<16x16x4x4xf32, 2 : i32>)
        %async_token_2 = air.execute [%3] {
          memref.dealloc %results : memref<16x16x4x4xf32, 2 : i32>
        }
      }
    }
  }
  return
}

// Vectorization with linalg.generic.

// CHECK-LABEL: func.func @func6
// CHECK: memref.alloc() : memref<1x1x8x8x4x4xi32, 2 : i32>
// CHECK: memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
// CHECK: memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
// CHECK: air.herd @herd_0
// CHECK-DAG: %[[CST0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[CSTINT0:.*]] = arith.constant 0 : i32
// CHECK: linalg.fill ins(%[[CSTINT0]] : i32) outs(%{{.*}} : memref<1x1x8x8x4x4xi32, 2 : i32>)
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: memref.subview %{{.*}}[0, 0, %{{.*}}, %{{.*}}, 0, 0] [1, 1, 1, 1, 4, 8] [1, 1, 1, 1, 1, 1] : memref<1x1x4x8x4x8xi32, 2 : i32> to memref<1x1x1x1x4x8xi32, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>
// CHECK: memref.subview %{{.*}}[0, 0, %{{.*}}, %{{.*}}, 0, 0] [1, 1, 1, 1, 8, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x8x4x8x4xi32, 2 : i32> to memref<1x1x1x1x8x4xi32, strided<[1024, 1024, 128, 32, 4, 1], offset: ?>, 2 : i32>
// CHECK: memref.subview %{{.*}}[%[[CST0]], %[[CST0]], %{{.*}}, %{{.*}}, 0, 0] [1, 1, 1, 1, 4, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x8x8x4x4xi32, 2 : i32> to memref<1x1x1x1x4x4xi32, strided<[1024, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
// CHECK: linalg.generic

#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d2, d5, d3, d6, d8)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d2, d1, d4, d5, d8, d7)>
#map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d4, d3, d6, d7)>
func.func @func6(%arg0: memref<512x512xi32>, %arg1: memref<512x512xi32>, %arg2: memref<512x512xi32>) {
  %c8 = arith.constant 8 : index
  %0 = air.launch async (%arg3, %arg4) in (%arg5=%c8, %arg6=%c8) attributes {id = 1 : i32} {
    %1 = air.segment @segment_0 async  attributes {id = 2 : i32} {
      %c2 = arith.constant 2 : index
      %async_token, %results = air.execute -> (memref<2x2x8x8x4x4xi32, 2 : i32>) {
        %alloc = memref.alloc() : memref<2x2x8x8x4x4xi32, 2 : i32>
        air.execute_terminator %alloc : memref<2x2x8x8x4x4xi32, 2 : i32>
      }
      %async_token_0, %results_1 = air.execute -> (memref<1x1x8x4x8x4xi32, 2 : i32>) {
        %alloc = memref.alloc() : memref<1x1x8x4x8x4xi32, 2 : i32>
        air.execute_terminator %alloc : memref<1x1x8x4x8x4xi32, 2 : i32>
      }
      %async_token_2, %results_3 = air.execute -> (memref<1x1x4x8x4x8xi32, 2 : i32>) {
        %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2 : i32>
        air.execute_terminator %alloc : memref<1x1x4x8x4x8xi32, 2 : i32>
      }
      %2 = air.herd @herd_0 async [%async_token, %async_token_0, %async_token_2]  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%results_3, %arg12=%results_1, %arg13=%results) : memref<1x1x4x8x4x8xi32, 2 : i32>, memref<1x1x8x4x8x4xi32, 2 : i32>, memref<2x2x8x8x4x4xi32, 2 : i32> attributes {id = 3 : i32} {
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1 = arith.constant 1 : index
        %c8_4 = arith.constant 8 : index
        %subview = memref.subview %arg13[%arg7, %arg8, 0, 0, 0, 0] [1, 1, 8, 8, 4, 4] [1, 1, 1, 1, 1, 1] : memref<2x2x8x8x4x4xi32, 2 : i32> to memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
        %async_token_5 = air.execute {
          linalg.fill ins(%c0_i32 : i32) outs(%subview : memref<1x1x8x8x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>)
        }
        %3 = air.wait_all async 
        %4 = scf.for %arg14 = %c0 to %c8_4 step %c1 iter_args(%arg15 = %3) -> (!air.async.token) {
          %5 = scf.for %arg16 = %c0 to %c8_4 step %c1 iter_args(%arg17 = %arg15) -> (!air.async.token) {
            %6 = scf.for %arg18 = %c0 to %c4 step %c1 iter_args(%arg19 = %arg17) -> (!air.async.token) {
              %subview_6 = memref.subview %arg11[0, 0, %arg18, %arg14, 0, 0] [1, 1, 1, 1, 4, 8] [1, 1, 1, 1, 1, 1] : memref<1x1x4x8x4x8xi32, 2 : i32> to memref<1x1x1x1x4x8xi32, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>
              %subview_7 = memref.subview %arg12[0, 0, %arg16, %arg18, 0, 0] [1, 1, 1, 1, 8, 4] [1, 1, 1, 1, 1, 1] : memref<1x1x8x4x8x4xi32, 2 : i32> to memref<1x1x1x1x8x4xi32, strided<[1024, 1024, 128, 32, 4, 1], offset: ?>, 2 : i32>
              %subview_8 = memref.subview %arg13[%arg7, %arg8, %arg16, %arg14, 0, 0] [1, 1, 1, 1, 4, 4] [1, 1, 1, 1, 1, 1] : memref<2x2x8x8x4x4xi32, 2 : i32> to memref<1x1x1x1x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>
              %async_token_9 = air.execute [%arg19] {
                linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%subview_6, %subview_7 : memref<1x1x1x1x4x8xi32, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>, memref<1x1x1x1x8x4xi32, strided<[1024, 1024, 128, 32, 4, 1], offset: ?>, 2 : i32>) outs(%subview_8 : memref<1x1x1x1x4x4xi32, strided<[2048, 1024, 128, 16, 4, 1], offset: ?>, 2 : i32>) {
                ^bb0(%in: i32, %in_10: i32, %out: i32):
                  %7 = arith.muli %in, %in_10 : i32
                  %8 = arith.addi %out, %7 : i32
                  linalg.yield %8 : i32
                }
              }
              scf.yield %async_token_9 : !air.async.token
            }
            scf.yield %6 : !air.async.token
          }
          scf.yield %5 : !air.async.token
        }
      }
    }
  }
  return
}
