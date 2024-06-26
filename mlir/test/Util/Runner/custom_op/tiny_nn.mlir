//===- tiny_nn.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// AIR Runner simulation of a user hand-written tiny neural network kernel.
// User provides the latency of this kernel in arch.json (10480 cycles).

// RUN: air-runner %s -f tinyNN -m %S/arch.json | FileCheck %s

// CHECK: "name": "air.custom",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "B",
// CHECK: "ts": 0.[[#%d,TIME0:]],
// CHECK: "name": "air.custom",
// CHECK-NEXT: "cat": "layer",
// CHECK-NEXT: "ph": "E",
// CHECK: "ts": 11.[[#TIME0 + 10480 - 11000]],

module {
  func.func @tinyNN(%arg0: memref<14xi32>, %arg1: memref<442xi32>, %arg2: memref<32xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<14xi32>, memref<442xi32>, memref<32xi32> attributes {id = 3 : i32} {
      %1 = air.segment async  args(%arg10=%arg7, %arg11=%arg8, %arg12=%arg9) : memref<14xi32>, memref<442xi32>, memref<32xi32> attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c0_0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %async_token, %results = air.execute -> (memref<14xi32, 1>) {
          %alloc = memref.alloc() : memref<14xi32, 1>
          air.execute_terminator %alloc : memref<14xi32, 1>
        } {id = 1 : i32}
        %async_token_2, %results_3 = air.execute -> (memref<442xi32, 1>) {
          %alloc = memref.alloc() : memref<442xi32, 1>
          air.execute_terminator %alloc : memref<442xi32, 1>
        } {id = 2 : i32}
        %async_token_4, %results_5 = air.execute -> (memref<32xi32, 1>) {
          %alloc = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %alloc : memref<32xi32, 1>
        } {id = 3 : i32}
        %2 = air.dma_memcpy_nd async [%async_token] (%results[] [] [], %arg10[%c0_0] [%c0_0] [%c0_0]) {id = 1 : i32} : (memref<14xi32, 1>, memref<14xi32>)
        %3 = air.dma_memcpy_nd async [%async_token_2] (%results_3[] [] [], %arg11[%c0_0] [%c0_0] [%c0_0]) {id = 2 : i32} : (memref<442xi32, 1>, memref<442xi32>)
        %4 = air.herd async [%async_token_4, %3, %2]  tile (%arg13, %arg14) in (%arg15=%c1_1, %arg16=%c1_1) args(%arg17=%results, %arg18=%results_3, %arg19=%results_5) : memref<14xi32, 1>, memref<442xi32, 1>, memref<32xi32, 1> attributes {id = 1 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
          %c0_9 = arith.constant 0 : index
          %async_token_10, %results_11 = air.execute -> (memref<14xi32, 2>) {
            %alloc = memref.alloc() : memref<14xi32, 2>
            air.execute_terminator %alloc : memref<14xi32, 2>
          } {id = 4 : i32}
          %async_token_12, %results_13 = air.execute -> (memref<442xi32, 2>) {
            %alloc = memref.alloc() : memref<442xi32, 2>
            air.execute_terminator %alloc : memref<442xi32, 2>
          } {id = 5 : i32}
          %async_token_14, %results_15 = air.execute -> (memref<32xi32, 2>) {
            %alloc = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %alloc : memref<32xi32, 2>
          } {id = 6 : i32}
          %6 = air.dma_memcpy_nd async [%async_token_10] (%results_11[] [] [], %arg17[%c0_9] [%c0_9] [%c0_9]) {id = 3 : i32} : (memref<14xi32, 2>, memref<14xi32, 1>)
          %7 = air.dma_memcpy_nd async [%async_token_12] (%results_13[] [] [], %arg18[%c0_9] [%c0_9] [%c0_9]) {id = 4 : i32} : (memref<442xi32, 2>, memref<442xi32, 1>)
          %async_token_16 = air.execute [%async_token_14, %7, %6] {
            air.custom @nn  operands (%results_11, %results_13, %results_15) : memref<14xi32, 2>, memref<442xi32, 2>, memref<32xi32, 2> 
          } {id = 7 : i32}
          %8 = air.dma_memcpy_nd async [%async_token_16] (%arg19[%c0_9] [%c0_9] [%c0_9], %results_15[] [] []) {id = 5 : i32} : (memref<32xi32, 1>, memref<32xi32, 2>)
          %async_token_17 = air.execute [%async_token_16] {
            memref.dealloc %results_11 : memref<14xi32, 2>
          } {id = 8 : i32}
          %async_token_18 = air.execute [%async_token_16] {
            memref.dealloc %results_13 : memref<442xi32, 2>
          } {id = 9 : i32}
          %async_token_19 = air.execute [%8] {
            memref.dealloc %results_15 : memref<32xi32, 2>
          } {id = 10 : i32}
        }
        %5 = air.dma_memcpy_nd async [%4] (%arg12[%c0_0] [%c0_0] [%c0_0], %results_5[] [] []) {id = 6 : i32} : (memref<32xi32>, memref<32xi32, 1>)
        %async_token_6 = air.execute [%4] {
          memref.dealloc %results : memref<14xi32, 1>
        } {id = 11 : i32}
        %async_token_7 = air.execute [%4] {
          memref.dealloc %results_3 : memref<442xi32, 1>
        } {id = 12 : i32}
        %async_token_8 = air.execute [%5] {
          memref.dealloc %results_5 : memref<32xi32, 1>
        } {id = 13 : i32}
      }
    }
    return
  }
}
