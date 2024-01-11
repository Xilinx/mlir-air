//===- broadcast_dma.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency-canonicalize | FileCheck %s

// Prune redundant dependency edges when affine if-based broadcast pattern exists
// CHECK: %[[EVENT0:.*]] = scf.for{{.*}}iter_args(%[[EVENT1:.*]] = 
// CHECK: %[[EVENT2:.*]], %[[VALUE0:.*]] = air.execute [%[[EVENT1]]]
// CHECK: %[[EVENT3:.*]], %[[VALUE1:.*]] = air.execute [%[[EVENT1]]]
// CHECK: %[[EVENT4:.*]], %[[VALUE2:.*]] = air.execute [%[[EVENT1]]]
// CHECK-NOT: %[[EVENT5:.*]] = air.dma_memcpy_nd async [%[[EVENT2]], %[[EVENT1]]]
// CHECK: %[[EVENT5:.*]] = air.dma_memcpy_nd async [%[[EVENT2]]]
// CHECK-NOT: %[[EVENT6:.*]] = air.dma_memcpy_nd async [%[[EVENT3]], %[[EVENT1]]]
// CHECK: %[[EVENT6:.*]] = air.dma_memcpy_nd async [%[[EVENT3]]]
// CHECK-NOT: %[[EVENT7:.*]] = air.dma_memcpy_nd async [%[[EVENT4]], %[[EVENT1]]]
// CHECK: %[[EVENT7:.*]] = air.dma_memcpy_nd async [%[[EVENT4]]]

#map = affine_map<()[s0] -> (s0 * 32)>
#set0 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 1 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 == 0)>
#set3 = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 1 >= 0, s1 - 1 == 0)>
module {
  func.func @matmul(%arg0: memref<512x512xbf16>, %arg1: memref<512x512xbf16>, %arg2: memref<512x512xbf16>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c512 = arith.constant 512 : index
    %c64 = arith.constant 64 : index
    %async_token, %results = air.execute -> (memref<512x512xbf16>) {
      %1 = memref.alloc() {alignment = 128 : i64} : memref<512x512xbf16>
      air.execute_terminator %1 : memref<512x512xbf16>
    } {id = 1 : i32}
    %async_token_0 = air.execute [%async_token] {
      memref.copy %arg2, %results : memref<512x512xbf16> to memref<512x512xbf16>
    } {id = 2 : i32}
    %0 = scf.parallel (%arg3, %arg4) = (%c0, %c0) to (%c512, %c512) step (%c64, %c64) init (%async_token_0) -> !air.async.token {
      %1 = scf.for %arg5 = %c0 to %c512 step %c64 iter_args(%arg6 = %async_token_0) -> (!air.async.token) {
        %async_token_1, %results_2 = air.execute [%arg6] -> (memref<64x64xbf16, 1>) {
          %8 = memref.alloc() : memref<64x64xbf16, 1>
          air.execute_terminator %8 : memref<64x64xbf16, 1>
        } {id = 3 : i32}
        %async_token_3, %results_4 = air.execute [%arg6] -> (memref<64x64xbf16, 1>) {
          %8 = memref.alloc() : memref<64x64xbf16, 1>
          air.execute_terminator %8 : memref<64x64xbf16, 1>
        } {id = 4 : i32}
        %async_token_5, %results_6 = air.execute [%arg6] -> (memref<64x64xbf16, 1>) {
          %8 = memref.alloc() : memref<64x64xbf16, 1>
          air.execute_terminator %8 : memref<64x64xbf16, 1>
        } {id = 5 : i32}
        %2 = air.dma_memcpy_nd async [%async_token_1] (%results_2[] [] [], %arg0[%arg3, %arg5] [%c64, %c64] [%c512, %c1]) {id = 1 : i32} : (memref<64x64xbf16, 1>, memref<512x512xbf16>)
        %3 = air.dma_memcpy_nd async [%async_token_3] (%results_4[] [] [], %arg1[%arg5, %arg4] [%c64, %c64] [%c512, %c1]) {id = 2 : i32} : (memref<64x64xbf16, 1>, memref<512x512xbf16>)
        %4 = air.dma_memcpy_nd async [%async_token_5, %arg6] (%results_6[] [] [], %results[%arg3, %arg4] [%c64, %c64] [%c512, %c1]) {id = 3 : i32} : (memref<64x64xbf16, 1>, memref<512x512xbf16>)
        %5 = air.herd @herd_0 async [%3, %2, %4]  tile (%arg7, %arg8) in (%arg9=%c2, %arg10=%c2) args(%arg11=%results_2, %arg12=%results_4, %arg13=%results_6) : memref<64x64xbf16, 1>, memref<64x64xbf16, 1>, memref<64x64xbf16, 1> attributes {id = 1 : i32} {
          %c1_10 = arith.constant 1 : index
          %c64_11 = arith.constant 64 : index
          %c32 = arith.constant 32 : index
          %c0_12 = arith.constant 0 : index
          %async_token_13, %results_14 = air.execute -> (index) {
            %12 = affine.apply #map()[%arg7]
            air.execute_terminator %12 : index
          } {id = 6 : i32}
          %async_token_15, %results_16 = air.execute -> (index) {
            %12 = affine.apply #map()[%arg8]
            air.execute_terminator %12 : index
          } {id = 7 : i32}
          %8 = air.wait_all async [%async_token_13, %async_token_15] 
          %async_token_17, %results_18 = air.execute -> (memref<32x32xbf16, 2>) {
            %12 = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %12 : memref<32x32xbf16, 2>
          } {id = 10 : i32}
          %9 = air.dma_memcpy_nd async [%8, %async_token_17] (%results_18[] [] [], %arg13[%results_14, %results_16] [%c32, %c32] [%c64_11, %c1_10]) {id = 4 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
          %10 = scf.for %arg14 = %c0_12 to %c64_11 step %c32 iter_args(%arg15 = %9) -> (!air.async.token) {
            %async_token_20, %results_21 = air.execute [%arg15] -> (memref<32x32xbf16, 2>) {
              %15 = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %15 : memref<32x32xbf16, 2>
            } {id = 8 : i32}
            %async_token_22, %results_23 = air.execute [%arg15] -> (memref<32x32xbf16, 2>) {
              %15 = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %15 : memref<32x32xbf16, 2>
            } {id = 9 : i32}
            %12 = affine.if #set0()[%arg7, %arg8] -> !air.async.token {
              %c0_27 = arith.constant 0 : index
              %15 = air.dma_memcpy_nd async [%async_token_20, %arg15] (%results_21[] [] [], %arg11[%c0_27, %arg14] [%c32, %c32] [%c64_11, %c1_10]) {broadcast_set = #set0, id = 5 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              affine.yield %15 : !air.async.token
            } else {
              %c32_27 = arith.constant 32 : index
              %15 = air.dma_memcpy_nd async [%async_token_20, %arg15] (%results_21[] [] [], %arg11[%c32_27, %arg14] [%c32, %c32] [%c64_11, %c1_10]) {broadcast_set = #set1, id = 6 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              affine.yield %15 : !air.async.token
            }
            %13 = affine.if #set2()[%arg7, %arg8] -> !air.async.token {
              %c0_27 = arith.constant 0 : index
              %15 = air.dma_memcpy_nd async [%async_token_22, %arg15] (%results_23[] [] [], %arg12[%arg14, %c0_27] [%c32, %c32] [%c64_11, %c1_10]) {broadcast_set = #set2, id = 7 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              affine.yield %15 : !air.async.token
            } else {
              %c32_27 = arith.constant 32 : index
              %15 = air.dma_memcpy_nd async [%async_token_22, %arg15] (%results_23[] [] [], %arg12[%arg14, %c32_27] [%c32, %c32] [%c64_11, %c1_10]) {broadcast_set = #set3, id = 8 : i32} : (memref<32x32xbf16, 2>, memref<64x64xbf16, 1>)
              affine.yield %15 : !air.async.token
            }
            %async_token_24 = air.execute [%13, %arg15, %12] {
              linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%results_21, %results_23 : memref<32x32xbf16, 2>, memref<32x32xbf16, 2>) outs(%results_18 : memref<32x32xbf16, 2>)
            } {id = 11 : i32}
            %async_token_25 = air.execute [%async_token_24] {
              memref.dealloc %results_21 : memref<32x32xbf16, 2>
            } {id = 12 : i32}
            %async_token_26 = air.execute [%async_token_24] {
              memref.dealloc %results_23 : memref<32x32xbf16, 2>
            } {id = 13 : i32}
            %14 = air.wait_all async [%async_token_24, %async_token_25, %async_token_26] 
            scf.yield %14 : !air.async.token
          }
          %11 = air.dma_memcpy_nd async [%10] (%arg13[%results_14, %results_16] [%c32, %c32] [%c64_11, %c1_10], %results_18[] [] []) {broadcast = "both", id = 9 : i32} : (memref<64x64xbf16, 1>, memref<32x32xbf16, 2>)
          %async_token_19 = air.execute [%11] {
            memref.dealloc %results_18 : memref<32x32xbf16, 2>
          } {id = 14 : i32}
          air.herd_terminator
        }
        %6 = air.dma_memcpy_nd async [%5] (%results[%arg3, %arg4] [%c64, %c64] [%c512, %c1], %results_6[] [] []) {id = 10 : i32} : (memref<512x512xbf16>, memref<64x64xbf16, 1>)
        %async_token_7 = air.execute [%5] {
          memref.dealloc %results_2 : memref<64x64xbf16, 1>
        } {id = 15 : i32}
        %async_token_8 = air.execute [%5] {
          memref.dealloc %results_4 : memref<64x64xbf16, 1>
        } {id = 16 : i32}
        %async_token_9 = air.execute [%6] {
          memref.dealloc %results_6 : memref<64x64xbf16, 1>
        } {id = 17 : i32}
        %7 = air.wait_all async [%async_token_7, %async_token_8, %async_token_9] 
        scf.yield %7 : !air.async.token
      }
      scf.reduce(%1 : !air.async.token) {
      ^bb0(%arg5: !air.async.token, %arg6: !air.async.token):
        %2 = air.wait_all async [%arg5, %arg6] 
        scf.reduce.return %2 : !air.async.token
      }
    }
    return
  }
}