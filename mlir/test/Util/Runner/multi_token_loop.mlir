//===- multi_token_loop.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// Pipelined for loop with multiple async tokens

// CHECK-COUNT-32: ChannelGetOp

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
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
      %c1024 = arith.constant 1024 : index
      %c128 = arith.constant 128 : index
      %c1_4 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %1 = air.wait_all async 
      %2 = scf.for %arg9 = %c0 to %c1024 step %c128 iter_args(%arg10 = %1) -> (!air.async.token) {
        %4 = air.channel.put async [%arg10]  @channel_0[%arg4, %arg5] (%arg8[%c0, %arg9] [%c128, %c128] [%c1024, %c1_4]) : (memref<256x1024xbf16>)
        scf.yield %4 : !air.async.token
      }
      %3 = air.segment async  args(%arg9=%arg4, %arg10=%arg5) : index, index attributes {du_usage = [1, 1]} {
        %c0_5 = arith.constant 0 : index
        %c1024_6 = arith.constant 1024 : index
        %c128_7 = arith.constant 128 : index
        %cst_8 = arith.constant 2.000000e+00 : bf16
        %cst_9 = arith.constant 1.000000e+00 : bf16
        %cst_10 = arith.constant 5.000000e-01 : bf16
        %async_token_11, %results_12 = air.execute -> (memref<128x128xbf16, 1>) {
          %alloc = memref.alloc() : memref<128x128xbf16, 1>
          air.execute_terminator %alloc : memref<128x128xbf16, 1>
        }
        %4:2 = scf.for %arg11 = %c0_5 to %c1024_6 step %c128_7 iter_args(%arg12 = %async_token_11, %arg13 = %async_token_11) -> (!air.async.token, !air.async.token) {
          %async_token_13, %results_14 = air.execute [%arg12] -> (memref<128x128xbf16, 1>) {
            %alloc = memref.alloc() : memref<128x128xbf16, 1>
            air.execute_terminator %alloc : memref<128x128xbf16, 1>
          }
          %5 = air.channel.get async [%async_token_13]  @channel_0[%arg9, %arg10] (%results_14[] [] []) : (memref<128x128xbf16, 1>)
          %6 = air.channel.put async [%5]  @channel_1[%arg9, %arg10] (%results_14[] [] []) : (memref<128x128xbf16, 1>)
          %async_token_15 = air.execute [%6] {
            memref.dealloc %results_14 : memref<128x128xbf16, 1>
          }
          %async_token_16, %results_17 = air.execute [%arg13] -> (memref<128x128xbf16, 1>) {
            %alloc = memref.alloc() : memref<128x128xbf16, 1>
            air.execute_terminator %alloc : memref<128x128xbf16, 1>
          }
          %7 = air.channel.get async [%async_token_16]  @channel_1[%arg9, %arg10] (%results_17[] [] []) : (memref<128x128xbf16, 1>)
          %async_token_18 = air.execute [%7] {
            linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%results_17 : memref<128x128xbf16, 1>) outs(%results_12 : memref<128x128xbf16, 1>) {
            ^bb0(%in: bf16, %out: bf16):
              %8 = math.sqrt %cst_8 : bf16
              %9 = arith.divf %in, %8 : bf16
              %10 = math.erf %9 : bf16
              %11 = arith.addf %10, %cst_9 : bf16
              %12 = arith.mulf %11, %cst_10 : bf16
              %13 = arith.mulf %in, %12 : bf16
              linalg.yield %13 : bf16
            }
          }
          %async_token_19 = air.execute [%async_token_18] {
            memref.dealloc %results_17 : memref<128x128xbf16, 1>
          }
          scf.yield %async_token_15, %async_token_19 : !air.async.token, !air.async.token
        }
        %async_token_20 = air.execute [%4#1] {
          memref.dealloc %results_12 : memref<128x128xbf16, 1>
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return %results_2 : memref<256x1024xbf16>
  }
}

