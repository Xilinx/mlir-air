//===- dma_broadcast.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// Air dma op with broadcast

// CHECK-COUNT-160: "name": "DmaMemcpyNdOp",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set1 = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set2 = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>
#set3 = affine_set<()[s0, s1] : (s0 - 3 == 0, s1 >= 0, -s1 + 3 >= 0)>
module {
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @test(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>, %arg3: memref<1024x1024xbf16>) -> memref<256x1024xbf16> {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %async_token, %results = air.execute -> (memref<256x1024xbf16>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<256x1024xbf16>
      air.execute_terminator %alloc : memref<256x1024xbf16>
    } {id = 1 : i32}
    %async_token_0 = air.execute [%async_token] {
      linalg.fill ins(%cst : bf16) outs(%results : memref<256x1024xbf16>)
    } {id = 2 : i32}
    %async_token_1, %results_2 = air.execute -> (memref<256x1024xbf16>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<256x1024xbf16>
      air.execute_terminator %alloc : memref<256x1024xbf16>
    } {id = 3 : i32}
    %async_token_3 = air.execute [%async_token_1, %async_token_0] {
      memref.copy %results, %results_2 : memref<256x1024xbf16> to memref<256x1024xbf16>
    } {id = 4 : i32}
    %0 = air.launch async [%async_token_3] (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%results_2) : memref<256x1024xbf16> {
      %1 = air.partition async  args(%arg9=%arg8) : memref<256x1024xbf16> attributes {column_usage = [4, 1]} {
        %c1_4 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %2 = air.wait_all async  {id = 8 : i32}
        %3 = scf.for %arg10 = %c0 to %c256 step %c128 iter_args(%arg11 = %2) -> (!air.async.token) {
          %4 = scf.for %arg12 = %c0 to %c1024 step %c128 iter_args(%arg13 = %arg11) -> (!air.async.token) {
            %async_token_5, %results_6 = air.execute -> (memref<128x128xbf16, 1>) {
              %alloc = memref.alloc() : memref<128x128xbf16, 1>
              air.execute_terminator %alloc : memref<128x128xbf16, 1>
            }
            %5 = air.dma_memcpy_nd async [%arg13, %async_token_5] (%results_6[] [] [], %arg9[%arg10, %arg12] [%c128, %c128] [%c1024, %c1_4]) : (memref<128x128xbf16, 1>, memref<256x1024xbf16>)
            %6 = air.herd @herd_0 async [%5]  tile (%arg14, %arg15) in (%arg16=%c4, %arg17=%c4) args(%arg18=%results_6) : memref<128x128xbf16, 1> {
              %c96 = arith.constant 96 : index
              %c64 = arith.constant 64 : index
              %c1_7 = arith.constant 1 : index
              %c0_8 = arith.constant 0 : index
              %c128_9 = arith.constant 128 : index
              %c32 = arith.constant 32 : index
              %async_token_10, %results_11 = air.execute -> (memref<32x32xbf16, 2>) {
                %alloc = memref.alloc() : memref<32x32xbf16, 2>
                air.execute_terminator %alloc : memref<32x32xbf16, 2>
              }
              %7 = affine.if #set()[%arg14, %arg15] -> !air.async.token {
                %8 = air.dma_memcpy_nd async [%async_token_10] (%results_11[] [] [], %arg18[%c0_8, %c0_8] [%c32, %c32] [%c128_9, %c1_7]) {broadcast_set = #set} : (memref<32x32xbf16, 2>, memref<128x128xbf16, 1>)
                affine.yield %8 : !air.async.token
              } else {
                %8 = affine.if #set1()[%arg14, %arg15] -> !air.async.token {
                  %9 = air.dma_memcpy_nd async [%async_token_10] (%results_11[] [] [], %arg18[%c32, %c0_8] [%c32, %c32] [%c128_9, %c1_7]) {broadcast_set = #set1} : (memref<32x32xbf16, 2>, memref<128x128xbf16, 1>)
                  affine.yield %9 : !air.async.token
                } else {
                  %9 = affine.if #set2()[%arg14, %arg15] -> !air.async.token {
                    %10 = air.dma_memcpy_nd async [%async_token_10] (%results_11[] [] [], %arg18[%c64, %c0_8] [%c32, %c32] [%c128_9, %c1_7]) {broadcast_set = #set2} : (memref<32x32xbf16, 2>, memref<128x128xbf16, 1>)
                    affine.yield %10 : !air.async.token
                  } else {
                    %10 = air.dma_memcpy_nd async [%async_token_10] (%results_11[] [] [], %arg18[%c96, %c0_8] [%c32, %c32] [%c128_9, %c1_7]) {broadcast_set = #set3} : (memref<32x32xbf16, 2>, memref<128x128xbf16, 1>)
                    affine.yield %10 : !air.async.token
                  }
                  affine.yield %9 : !air.async.token
                }
                affine.yield %8 : !air.async.token
              }
              air.herd_terminator
            }
            scf.yield %6 : !air.async.token
          }
          scf.yield %4 : !air.async.token
        }
        air.partition_terminator
      }
      air.launch_terminator
    }
    return %results_2 : memref<256x1024xbf16>
  }
}

