//===- dma.mlir ------------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// Air dma op

// CHECK-COUNT-32: "name": "DmaMemcpyNdOp",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

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
      %1 = air.segment async  args(%arg9=%arg8) : memref<256x1024xbf16> attributes {x_loc = 0 : i64, x_size = 4 : i64, y_loc = 0 : i64, y_size = 4 : i64} {
        %c1_4 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %c1024 = arith.constant 1024 : index
        %c128 = arith.constant 128 : index
        %c256 = arith.constant 256 : index
        %2 = air.wait_all async  {id = 8 : i32}
        %3 = scf.for %arg10 = %c0 to %c256 step %c128 iter_args(%arg11 = %2) -> (!air.async.token) {
          %4 = scf.for %arg12 = %c0 to %c1024 step %c128 iter_args(%arg13 = %arg11) -> (!air.async.token) {
            %async_token_5, %results_6 = air.execute [%arg13] -> (memref<128x128xbf16, 1>) {
              %alloc = memref.alloc() : memref<128x128xbf16, 1>
              air.execute_terminator %alloc : memref<128x128xbf16, 1>
            }
            %6 = air.dma_memcpy_nd async [%async_token_5] (%results_6[] [] [], %arg9[%arg10, %arg12] [%c128, %c128] [%c1024, %c1_4]) : (memref<128x128xbf16, 1>, memref<256x1024xbf16>)
            %async_token_7 = air.execute [%6] {
              memref.dealloc %results_6 : memref<128x128xbf16, 1>
            }
            scf.yield %async_token_7 : !air.async.token
          }
          %5 = air.wait_all async [%arg11, %4] 
          scf.yield %5 : !air.async.token
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return %results_2 : memref<256x1024xbf16>
  }
}

