//===- dma_to_channel_nested_for_in_herd.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel -canonicalize -cse | FileCheck %s

// Hoisting external channel put/get op out of a herd with nested for loops

#map = affine_map<()[s0] -> (s0 * 32)>
module {
// CHECK-LABEL: func.func @sync
  func.func @sync(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) -> memref<64x64xi32> {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %alloc = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%alloc : memref<64x64xi32>)
    %alloc_0 = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
    memref.copy %alloc, %alloc_0 : memref<64x64xi32> to memref<64x64xi32>
// CHECK: %[[EVENT0:.*]] = scf.parallel (%[[VALUE0:.*]], %[[VALUE1:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put @channel_0[%[[VALUE0]], %[[VALUE1]]]
// CHECK: %[[EVENT1:.*]] = scf.parallel (%[[VALUE2:.*]], %[[VALUE3:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put @channel_1[%[[VALUE2]], %[[VALUE3]]]
// CHECK: %[[EVENT2:.*]] = scf.parallel (%[[VALUE4:.*]], %[[VALUE5:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put @channel_2[%[[VALUE4]], %[[VALUE5]]]
// CHECK: %[[EVENT3:.*]] = scf.parallel (%[[VALUE6:.*]], %[[VALUE7:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.get @channel_3[%[[VALUE6]], %[[VALUE7]]]
    air.herd @herd_0  tile (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) args(%arg6=%arg0, %arg7=%arg1, %arg8=%alloc_0) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      scf.for %newarg0 = %c0 to %c64 step %c32 {
        scf.for %newarg1 = %c0 to %c64 step %c32 {
            scf.for %arg9 = %c0 to %c64 step %c32 {
                %alloc_1 = memref.alloc() : memref<32x32xi32, 2>
                %alloc_2 = memref.alloc() : memref<32x32xi32, 2>
                %alloc_3 = memref.alloc() : memref<32x32xi32, 2>
                air.dma_memcpy_nd (%alloc_1[] [] [], %arg6[%newarg0, %arg9] [%c32, %c32] [%c64, %c1]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
                air.dma_memcpy_nd (%alloc_2[] [] [], %arg7[%arg9, %newarg1] [%c32, %c32] [%c64, %c1]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
                air.dma_memcpy_nd (%alloc_3[] [] [], %arg8[%newarg0, %newarg1] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
                linalg.matmul ins(%alloc_1, %alloc_2 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%alloc_3 : memref<32x32xi32, 2>)
                air.dma_memcpy_nd (%arg8[%newarg0, %newarg1] [%c32, %c32] [%c64, %c1], %alloc_3[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
                memref.dealloc %alloc_1 : memref<32x32xi32, 2>
                memref.dealloc %alloc_2 : memref<32x32xi32, 2>
                memref.dealloc %alloc_3 : memref<32x32xi32, 2>
            }
        }
      }
      air.herd_terminator
    }
    return %alloc_0 : memref<64x64xi32>
  }

// CHECK-LABEL: func.func @async
// CHECK: %[[EVENT0:.*]] = scf.parallel (%[[VALUE0:.*]], %[[VALUE1:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put async{{.*}}@channel_4[%[[VALUE0]], %[[VALUE1]]]
// CHECK: %[[EVENT1:.*]] = scf.parallel (%[[VALUE2:.*]], %[[VALUE3:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put async{{.*}}@channel_5[%[[VALUE2]], %[[VALUE3]]]
// CHECK: %[[EVENT2:.*]] = scf.parallel (%[[VALUE4:.*]], %[[VALUE5:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.put async{{.*}}@channel_6[%[[VALUE4]], %[[VALUE5]]]
// CHECK: %[[EVENT3:.*]] = scf.parallel (%[[VALUE6:.*]], %[[VALUE7:.*]]) ={{.*}}init
// CHECK: scf.for
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.get async{{.*}}@channel_7[%[[VALUE6]], %[[VALUE7]]]  
  func.func @async(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>) -> memref<64x64xi32> {
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %async_token, %results = air.execute -> (memref<64x64xi32>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
      air.execute_terminator %alloc : memref<64x64xi32>
    } {id = 1 : i32}
    %async_token_0 = air.execute [%async_token] {
      linalg.fill ins(%c0_i32 : i32) outs(%results : memref<64x64xi32>)
    } {id = 2 : i32}
    %async_token_1, %results_2 = air.execute -> (memref<64x64xi32>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
      air.execute_terminator %alloc : memref<64x64xi32>
    } {id = 3 : i32}
    %async_token_3 = air.execute [%async_token_1, %async_token_0] {
      memref.copy %results, %results_2 : memref<64x64xi32> to memref<64x64xi32>
    } {id = 4 : i32}
    %0 = air.herd @herd_0 async [%async_token_3]  tile (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) args(%arg6=%arg0, %arg7=%arg1, %arg8=%results_2) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> attributes {id = 1 : i32} {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %1 = air.wait_all async  {id = 6 : i32}
      %2 = scf.for %arg9 = %c0 to %c64 step %c32 iter_args(%arg10 = %1) -> (!air.async.token) {
        %c0_4 = arith.constant 0 : index
        %c64_5 = arith.constant 64 : index
        %c32_6 = arith.constant 32 : index
        %3 = air.wait_all async [%arg10]  {id = 4 : i32}
        %4 = scf.for %arg11 = %c0_4 to %c64_5 step %c32_6 iter_args(%arg12 = %3) -> (!air.async.token) {
          %c0_7 = arith.constant 0 : index
          %c64_8 = arith.constant 64 : index
          %c32_9 = arith.constant 32 : index
          %6 = air.wait_all async [%arg12]  {id = 2 : i32}
          %7 = scf.for %arg13 = %c0_7 to %c64_8 step %c32_9 iter_args(%arg14 = %6) -> (!air.async.token) {
            %c32_10 = arith.constant 32 : index
            %c64_11 = arith.constant 64 : index
            %c1_12 = arith.constant 1 : index
            %async_token_13, %results_14 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            } {id = 5 : i32}
            %async_token_15, %results_16 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            } {id = 6 : i32}
            %async_token_17, %results_18 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            } {id = 7 : i32}
            %9 = air.dma_memcpy_nd async [%arg14, %async_token_13] (%results_14[] [] [], %arg6[%arg9, %arg13] [%c32_10, %c32_10] [%c64_11, %c1_12]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
            %10 = air.dma_memcpy_nd async [%arg14, %async_token_15] (%results_16[] [] [], %arg7[%arg13, %arg11] [%c32_10, %c32_10] [%c64_11, %c1_12]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
            %11 = air.dma_memcpy_nd async [%arg14, %async_token_17] (%results_18[] [] [], %arg8[%arg9, %arg11] [%c32_10, %c32_10] [%c64_11, %c1_12]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
            %async_token_19 = air.execute [%arg14, %11, %10, %9] {
              linalg.matmul ins(%results_14, %results_16 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_18 : memref<32x32xi32, 2>)
            } {id = 8 : i32}
            %12 = air.dma_memcpy_nd async [%arg14, %async_token_19] (%arg8[%arg9, %arg11] [%c32_10, %c32_10] [%c64_11, %c1_12], %results_18[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
            %async_token_20 = air.execute [%async_token_19] {
              memref.dealloc %results_14 : memref<32x32xi32, 2>
            } {id = 9 : i32}
            %async_token_21 = air.execute [%async_token_19] {
              memref.dealloc %results_16 : memref<32x32xi32, 2>
            } {id = 10 : i32}
            %async_token_22 = air.execute [%12] {
              memref.dealloc %results_18 : memref<32x32xi32, 2>
            } {id = 11 : i32}
            %13 = air.wait_all async [%arg14, %12]  {id = 1 : i32}
            scf.yield %13 : !air.async.token
          }
          %8 = air.wait_all async [%arg12, %7]  {id = 3 : i32}
          scf.yield %8 : !air.async.token
        }
        %5 = air.wait_all async [%arg10, %4]  {id = 5 : i32}
        scf.yield %5 : !air.async.token
      }
      air.herd_terminator
    }
    return %results_2 : memref<64x64xi32>
  }
}
