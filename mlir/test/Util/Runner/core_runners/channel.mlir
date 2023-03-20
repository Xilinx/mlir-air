//===- channel.mlir --------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/../arch.json -g core | FileCheck %s

// Air channel ops, running each core in a herd

// CHECK-COUNT-4096: "name": "ChannelGetOp@channel_1(L1<--L2)",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

#map = affine_map<()[s0] -> (s0 * 32)>
module {
  air.channel @channel_1 [4, 4]
  air.channel @channel_0 [1, 1]
  ml_program.global private mutable @global_seed(dense<0> : tensor<i64>) : tensor<i64>
  func.func @test(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>, %arg3: memref<1024x1024xbf16>) -> memref<256x1024xbf16> {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %async_token, %results = air.execute -> (memref<256x1024xbf16>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<256x1024xbf16>
      air.execute_terminator %alloc : memref<256x1024xbf16>
    }
    %async_token_0 = air.execute [%async_token] {
      linalg.fill ins(%cst : bf16) outs(%results : memref<256x1024xbf16>)
    }
    %async_token_1, %results_2 = air.execute -> (memref<256x1024xbf16>) {
      %alloc = memref.alloc() {alignment = 128 : i64} : memref<256x1024xbf16>
      air.execute_terminator %alloc : memref<256x1024xbf16>
    }
    %async_token_3 = air.execute [%async_token_1, %async_token_0] {
      memref.copy %results, %results_2 : memref<256x1024xbf16> to memref<256x1024xbf16>
    }
    %0 = air.launch async [%async_token_3] (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%results_2) : memref<256x1024xbf16> {
      %c1_4 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c1024 = arith.constant 1024 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %1 = air.wait_all async 
      %2 = scf.for %arg9 = %c0 to %c256 step %c128 iter_args(%arg10 = %1) -> (!air.async.token) {
        %4 = scf.for %arg11 = %c0 to %c1024 step %c128 iter_args(%arg12 = %arg10) -> (!air.async.token) {
          %5 = air.channel.put async [%arg12]  @channel_0[] (%arg8[%arg9, %arg11] [%c128, %c128] [%c1024, %c1_4]) : (memref<256x1024xbf16>)
          scf.yield %5 : !air.async.token
        }
        scf.yield %4 : !air.async.token
      }
      %3 = air.partition async attributes {column_usage = [4, 1]} {
        %c32 = arith.constant 32 : index
        %c1_5 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c0_6 = arith.constant 0 : index
        %c1024_7 = arith.constant 1024 : index
        %c128_8 = arith.constant 128 : index
        %c256_9 = arith.constant 256 : index
        %4 = air.wait_all async 
        %5 = scf.for %arg9 = %c0_6 to %c256_9 step %c128_8 iter_args(%arg10 = %4) -> (!air.async.token) {
          %6 = scf.for %arg11 = %c0_6 to %c1024_7 step %c128_8 iter_args(%arg12 = %arg10) -> (!air.async.token) {
            %async_token_10, %results_11 = air.execute -> (memref<128x128xbf16, 1>) {
              %alloc = memref.alloc() : memref<128x128xbf16, 1>
              air.execute_terminator %alloc : memref<128x128xbf16, 1>
            }
            %7 = air.channel.get async [%async_token_10, %arg12]  @channel_0[] (%results_11[] [] []) : (memref<128x128xbf16, 1>)
            %8 = scf.for %arg13 = %c0_6 to %c1024_7 step %c128_8 iter_args(%arg14 = %7) -> (!air.async.token) {
              %async_token_12, %results_13 = air.execute [%arg14] -> (memref<128x128xbf16, 1>) {
                %alloc = memref.alloc() : memref<128x128xbf16, 1>
                air.execute_terminator %alloc : memref<128x128xbf16, 1>
              }
              %async_token_14, %results_15 = air.execute [%arg14] -> (memref<128x128xbf16, 1>) {
                %alloc = memref.alloc() : memref<128x128xbf16, 1>
                air.execute_terminator %alloc : memref<128x128xbf16, 1>
              }
              %9 = scf.parallel (%arg15, %arg16) = (%c0_6, %c0_6) to (%c4, %c4) step (%c1_5, %c1_5) init (%arg14) -> !air.async.token {
                %async_token_18, %results_19 = air.execute -> (index) {
                  %13 = affine.apply #map()[%arg15]
                  air.execute_terminator %13 : index
                }
                %async_token_20, %results_21 = air.execute -> (index) {
                  %13 = affine.apply #map()[%arg16]
                  air.execute_terminator %13 : index
                }
                %12 = air.channel.put async [%async_token_20, %async_token_18, %arg14]  @channel_1[%arg15, %arg16] (%results_11[%results_19, %results_21] [%c32, %c32] [%c128_8, %c1_5]) : (memref<128x128xbf16, 1>)
                scf.reduce(%12)  : !air.async.token {
                ^bb0(%arg17: !air.async.token, %arg18: !air.async.token):
                  %13 = air.wait_all async [%arg17, %arg18] 
                  scf.reduce.return %13 : !air.async.token
                }
                scf.yield
              }
              %10 = air.herd @herd_0 async [%arg14]  tile (%arg15, %arg16) in (%arg17=%c4, %arg18=%c4) {
                %12 = air.wait_all async 
                %async_token_18, %results_19 = air.execute -> (memref<32x32xbf16, 2>) {
                  %alloc = memref.alloc() : memref<32x32xbf16, 2>
                  air.execute_terminator %alloc : memref<32x32xbf16, 2>
                }
                %13 = air.channel.get async [%async_token_18, %12]  @channel_1[%arg15, %arg16] (%results_19[] [] []) : (memref<32x32xbf16, 2>)
                %async_token_21 = air.execute [%13] {
                  memref.dealloc %results_19 : memref<32x32xbf16, 2>
                }
                air.herd_terminator
              }
              %async_token_16 = air.execute [%10] {
                memref.dealloc %results_13 : memref<128x128xbf16, 1>
              }
              %async_token_17 = air.execute [%10] {
                memref.dealloc %results_15 : memref<128x128xbf16, 1>
              }
              %11 = air.wait_all async [%9, %10] 
              scf.yield %11 : !air.async.token
            }
            %async_token_22 = air.execute [%8] {
              memref.dealloc %results_11 : memref<128x128xbf16, 1>
            }
            scf.yield %async_token_22 : !air.async.token
          }
          scf.yield %6 : !air.async.token
        }
        air.partition_terminator
      }
      air.launch_terminator
    }
    return %results_2 : memref<256x1024xbf16>
  }
}

