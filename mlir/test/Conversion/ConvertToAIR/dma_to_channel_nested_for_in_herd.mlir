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
    }
    %async_token_0 = air.execute [%async_token] {
      linalg.fill ins(%c0_i32 : i32) outs(%results : memref<64x64xi32>)
    }
    %async_token_1, %results_2 = air.execute -> (memref<64x64xi32>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<64x64xi32>
      air.execute_terminator %alloc : memref<64x64xi32>
    }
    %async_token_3 = air.execute [%async_token_1, %async_token_0] {
      memref.copy %results, %results_2 : memref<64x64xi32> to memref<64x64xi32>
    }
    %0 = air.herd @herd_0 async [%async_token_3]  tile (%arg2, %arg3) in (%arg4=%c2, %arg5=%c2) args(%arg6=%arg0, %arg7=%arg1, %arg8=%results_2) : memref<64x64xi32>, memref<64x64xi32>, memref<64x64xi32> attributes {id = 1 : i32} {
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c32 = arith.constant 32 : index
      %1 = air.wait_all async 
      %2 = scf.for %arg9 = %c0 to %c64 step %c32 iter_args(%arg10 = %1) -> (!air.async.token) {
        %3 = scf.for %arg11 = %c0 to %c64 step %c32 iter_args(%arg12 = %arg10) -> (!air.async.token) {
          %4 = scf.for %arg13 = %c0 to %c64 step %c32 iter_args(%arg14 = %arg12) -> (!air.async.token) {
            %async_token_4, %results_5 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_6, %results_7 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %async_token_8, %results_9 = air.execute -> (memref<32x32xi32, 2>) {
              %alloc = memref.alloc() : memref<32x32xi32, 2>
              air.execute_terminator %alloc : memref<32x32xi32, 2>
            }
            %5 = air.dma_memcpy_nd async [%async_token_4, %arg14] (%results_5[] [] [], %arg6[%arg9, %arg13] [%c32, %c32] [%c64, %c1]) {id = 1 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
            %6 = air.dma_memcpy_nd async [%async_token_6, %arg14] (%results_7[] [] [], %arg7[%arg13, %arg11] [%c32, %c32] [%c64, %c1]) {id = 2 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
            %7 = air.dma_memcpy_nd async [%async_token_8, %arg14] (%results_9[] [] [], %arg8[%arg9, %arg11] [%c32, %c32] [%c64, %c1]) {id = 3 : i32} : (memref<32x32xi32, 2>, memref<64x64xi32>)
            %async_token_10 = air.execute [%7, %6, %5] {
              linalg.matmul ins(%results_5, %results_7 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%results_9 : memref<32x32xi32, 2>)
            }
            %8 = air.dma_memcpy_nd async [%async_token_10] (%arg8[%arg9, %arg11] [%c32, %c32] [%c64, %c1], %results_9[] [] []) {id = 4 : i32} : (memref<64x64xi32>, memref<32x32xi32, 2>)
            %async_token_11 = air.execute [%async_token_10] {
              memref.dealloc %results_5 : memref<32x32xi32, 2>
            }
            %async_token_12 = air.execute [%async_token_10] {
              memref.dealloc %results_7 : memref<32x32xi32, 2>
            }
            %async_token_13 = air.execute [%8] {
              memref.dealloc %results_9 : memref<32x32xi32, 2>
            }
            scf.yield %8 : !air.async.token
          }
          scf.yield %4 : !air.async.token
        }
        scf.yield %3 : !air.async.token
      }
      air.herd_terminator
    }
    return %results_2 : memref<64x64xi32>
  }
}
