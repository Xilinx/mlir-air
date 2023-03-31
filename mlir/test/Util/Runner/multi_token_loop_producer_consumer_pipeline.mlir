//===- multi_token_loop_produce_consumer_pipeline.mlir ---------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// Pipelined for loop with race condition for both producer and consumer

// CHECK-COUNT-64: ChannelGetOp

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "B",

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

#map = affine_map<()[s0] -> (s0 * 32)>
#map1 = affine_map<()[s0] -> (s0 + 32)>
module attributes {torch.debug_module_name = "mmult"} {
  air.channel @channel_3 [1, 4] {broadcast_shape = [4, 4]}
  air.channel @channel_2 [4, 1] {broadcast_shape = [4, 4]}
  air.channel @channel_1 [1, 4] {broadcast_shape = [4, 4]}
  air.channel @channel_0 [4, 1] {broadcast_shape = [4, 4]}
  func.func @test(%arg0: memref<512x512xf32>, %arg1: memref<512x512xf32>, %arg2: memref<512x512xf32>, %arg3: memref<512x512xf32>) -> memref<512x512xf32> {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %async_token, %results = air.execute -> (memref<512x512xf32>) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<512x512xf32>
      air.execute_terminator %alloc : memref<512x512xf32>
    } {id = 1 : i32}
    %async_token_0 = air.execute [%async_token] {
      linalg.fill ins(%cst : f32) outs(%results : memref<512x512xf32>)
    } {id = 2 : i32}
    %async_token_1, %results_2 = air.execute -> (memref<512x512xf32>) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<512x512xf32>
      air.execute_terminator %alloc : memref<512x512xf32>
    } {id = 3 : i32}
    %async_token_3 = air.execute [%async_token_1, %async_token_0] {
      memref.copy %results, %results_2 : memref<512x512xf32> to memref<512x512xf32>
    } {id = 4 : i32}
    %async_token_4, %results_5 = air.execute -> (memref<512x512xf32>) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<512x512xf32>
      air.execute_terminator %alloc : memref<512x512xf32>
    } {id = 5 : i32}
    %async_token_6 = air.execute [%async_token_4, %async_token_3] {
      memref.copy %results, %results_5 : memref<512x512xf32> to memref<512x512xf32>
    } {id = 6 : i32}
    %async_token_7, %results_8 = air.execute -> (memref<512x512xf32>) {
      %alloc = memref.alloc() {alignment = 64 : i64} : memref<512x512xf32>
      air.execute_terminator %alloc : memref<512x512xf32>
    } {id = 7 : i32}
    %async_token_9 = air.execute [%async_token_7, %async_token_6] {
      memref.copy %results, %results_8 : memref<512x512xf32> to memref<512x512xf32>
    } {id = 8 : i32}
    %0 = air.launch async [%async_token_9] (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) attributes {id = 7 : i32} {
      %1 = air.segment async  attributes {column_usage = [4, 1], id = 2 : i32} {
        %c64 = arith.constant 64 : index
        %c32 = arith.constant 32 : index
        %c1_10 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c0 = arith.constant 0 : index
        %c512 = arith.constant 512 : index
        %c128 = arith.constant 128 : index
        %async_token_11, %results_12 = air.execute -> (memref<128x128xf32, 1>) {
          %alloc = memref.alloc() : memref<128x128xf32, 1>
          air.execute_terminator %alloc : memref<128x128xf32, 1>
        } {id = 13 : i32}
        %2 = scf.for %arg8 = %c0 to %c512 step %c128 iter_args(%arg9 = %async_token_11) -> (!air.async.token) {
          %async_token_14, %results_15 = air.execute -> (memref<128x128xf32, 1>) {
            %alloc = memref.alloc() : memref<128x128xf32, 1>
            air.execute_terminator %alloc : memref<128x128xf32, 1>
          } {id = 11 : i32}
          %async_token_16, %results_17 = air.execute -> (memref<128x128xf32, 1>) {
            %alloc = memref.alloc() : memref<128x128xf32, 1>
            air.execute_terminator %alloc : memref<128x128xf32, 1>
          } {id = 12 : i32}
          %3 = air.wait_all async [%arg9, %async_token_14] 
          %4 = scf.parallel (%arg10) = (%c0) to (%c4) step (%c1_10) init (%3) -> !air.async.token {
            %async_token_20, %results_21 = air.execute -> (index) {
              %10 = affine.apply #map()[%arg10]
              air.execute_terminator %10 : index
            } {id = 14 : i32}
            %9 = scf.for %arg11 = %c0 to %c128 step %c64 iter_args(%arg12 = %3) -> (!air.async.token) {
              %10 = air.channel.put async [%arg12]  @channel_2[%arg10, %c0] (%results_15[%results_21, %arg11] [%c32, %c32] [%c128, %c1_10]) : (memref<128x128xf32, 1>)
              %async_token_22, %results_23 = air.execute -> (index) {
                %12 = affine.apply #map1()[%arg11]
                air.execute_terminator %12 : index
              }
              %11 = air.channel.put async [%10, %async_token_22]  @channel_0[%arg10, %c0] (%results_15[%results_21, %arg11] [%c32, %c32] [%c128, %c1_10]) : (memref<128x128xf32, 1>)
              scf.yield %11 : !air.async.token
            }
            scf.reduce(%9)  : !air.async.token {
            ^bb0(%arg11: !air.async.token, %arg12: !air.async.token):
              %10 = air.wait_all async [%arg11, %arg12] 
              scf.reduce.return %10 : !air.async.token
            }
            scf.yield
          }
          %5 = air.wait_all async [%arg9, %async_token_16] 
          %6 = scf.parallel (%arg10) = (%c0) to (%c4) step (%c1_10) init (%5) -> !air.async.token {
            %async_token_20, %results_21 = air.execute -> (index) {
              %10 = affine.apply #map()[%arg10]
              air.execute_terminator %10 : index
            } {id = 15 : i32}
            %9 = scf.for %arg11 = %c0 to %c128 step %c64 iter_args(%arg12 = %5) -> (!air.async.token) {
              %10 = air.channel.put async [%arg12]  @channel_3[%c0, %arg10] (%results_17[%arg11, %results_21] [%c32, %c32] [%c128, %c1_10]) : (memref<128x128xf32, 1>)
              %async_token_22, %results_23 = air.execute -> (index) {
                %12 = affine.apply #map1()[%arg11]
                air.execute_terminator %12 : index
              }
              %11 = air.channel.put async [%10, %async_token_22]  @channel_1[%c0, %arg10] (%results_17[%c0, %results_23] [%c32, %c32] [%c128, %c1_10]) : (memref<128x128xf32, 1>)
              scf.yield %11 : !air.async.token
            }
            scf.reduce(%9)  : !air.async.token {
            ^bb0(%arg11: !air.async.token, %arg12: !air.async.token):
              %10 = air.wait_all async [%arg11, %arg12] 
              scf.reduce.return %10 : !air.async.token
            }
            scf.yield
          }
          %7 = air.herd @herd_0 async [%arg9]  tile (%arg10, %arg11) in (%arg12=%c4, %arg13=%c4) attributes {id = 1 : i32} {
            %c0_20 = arith.constant 0 : index
            %c128_21 = arith.constant 128 : index
            %c64_22 = arith.constant 64 : index
            %async_token_23, %results_24 = air.execute -> (index) {
              %13 = affine.apply #map()[%arg10]
              air.execute_terminator %13 : index
            } {id = 14 : i32}
            %async_token_25, %results_26 = air.execute -> (index) {
              %13 = affine.apply #map()[%arg11]
              air.execute_terminator %13 : index
            } {id = 15 : i32}
            %9 = air.wait_all async [%async_token_23, %async_token_25]  {id = 2 : i32}
            %async_token_27, %results_28 = air.execute -> (memref<32x32xf32, 2>) {
              %alloc = memref.alloc() : memref<32x32xf32, 2>
              air.execute_terminator %alloc : memref<32x32xf32, 2>
            } {id = 18 : i32}
            %async_token_29, %results_30 = air.execute -> (memref<32x32xf32, 2>) {
              %alloc = memref.alloc() : memref<32x32xf32, 2>
              air.execute_terminator %alloc : memref<32x32xf32, 2>
            } {id = 16 : i32}
            %async_token_31, %results_32 = air.execute -> (memref<32x32xf32, 2>) {
              %alloc = memref.alloc() : memref<32x32xf32, 2>
              air.execute_terminator %alloc : memref<32x32xf32, 2>
            } {id = 17 : i32}
            %10 = air.wait_all async [%async_token_29, %async_token_31] 
            %async_token_33, %results_34 = air.execute -> (memref<32x32xf32, 2>) {
              %alloc = memref.alloc() : memref<32x32xf32, 2>
              air.execute_terminator %alloc : memref<32x32xf32, 2>
            } {id = 16 : i32}
            %async_token_35, %results_36 = air.execute -> (memref<32x32xf32, 2>) {
              %alloc = memref.alloc() : memref<32x32xf32, 2>
              air.execute_terminator %alloc : memref<32x32xf32, 2>
            } {id = 17 : i32}
            %11 = air.wait_all async [%async_token_33, %async_token_35] 
            %12:4 = scf.for %arg14 = %c0_20 to %c128_21 step %c64_22 iter_args(%arg15 = %10, %arg16 = %11, %arg17 = %11, %arg18 = %11) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
              %13 = air.channel.get async [%arg18, %arg15]  @channel_2[%arg10, %arg11] (%results_30[] [] []) : (memref<32x32xf32, 2>)
              %14 = air.channel.get async [%arg18, %arg15]  @channel_3[%arg10, %arg11] (%results_32[] [] []) : (memref<32x32xf32, 2>)
              %async_token_42 = air.execute [%14, %13, %arg17] {
                linalg.matmul ins(%results_30, %results_32 : memref<32x32xf32, 2>, memref<32x32xf32, 2>) outs(%results_28 : memref<32x32xf32, 2>)
              } {id = 19 : i32}
              %15 = air.channel.get async [%arg16, %14, %13]  @channel_0[%arg10, %arg11] (%results_34[] [] []) : (memref<32x32xf32, 2>)
              %16 = air.channel.get async [%arg16, %14, %13]  @channel_1[%arg10, %arg11] (%results_36[] [] []) : (memref<32x32xf32, 2>)
              %async_token_43 = air.execute [%16, %15, %async_token_42] {
                linalg.matmul ins(%results_34, %results_36 : memref<32x32xf32, 2>, memref<32x32xf32, 2>) outs(%results_28 : memref<32x32xf32, 2>)
              } {id = 19 : i32}
              %17 = air.wait_all async [%15, %16] 
              scf.yield %async_token_42, %async_token_43, %async_token_43, %17 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
            }
            %async_token_37 = air.execute [%12#0] {
              memref.dealloc %results_28 : memref<32x32xf32, 2>
            } {id = 20 : i32}
            %async_token_38 = air.execute [%12#0] {
              memref.dealloc %results_30 : memref<32x32xf32, 2>
            } {id = 20 : i32}
            %async_token_39 = air.execute [%12#0] {
              memref.dealloc %results_32 : memref<32x32xf32, 2>
            } {id = 21 : i32}
            %async_token_40 = air.execute [%12#1] {
              memref.dealloc %results_34 : memref<32x32xf32, 2>
            } {id = 20 : i32}
            %async_token_41 = air.execute [%12#1] {
              memref.dealloc %results_36 : memref<32x32xf32, 2>
            } {id = 21 : i32}
            air.herd_terminator
          }
          %async_token_18 = air.execute [%7] {
            memref.dealloc %results_15 : memref<128x128xf32, 1>
          } {id = 23 : i32}
          %async_token_19 = air.execute [%7] {
            memref.dealloc %results_17 : memref<128x128xf32, 1>
          } {id = 24 : i32}
          %8 = air.wait_all async [%6, %4, %7, %arg9] 
          scf.yield %8 : !air.async.token
        }
<<<<<<< HEAD
        %async_token_13 = air.execute [%2] {
          memref.dealloc %results_12 : memref<128x128xf32, 1>
        } {id = 24 : i32}
        air.partition_terminator
=======
        air.segment_terminator
>>>>>>> 0e7d30dab506796a6cd897330baafc1fce2a5944
      }
      air.launch_terminator
    }
    return %results_8 : memref<512x512xf32>
  }
}