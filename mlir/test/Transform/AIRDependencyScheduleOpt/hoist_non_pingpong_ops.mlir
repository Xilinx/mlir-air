//===- hoist_non_pingpong_ops.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-hoist-ops-not-using-ping-pong | FileCheck %s

// Hoist ops not related to ping-pong transformation out of the target scf.for loop.
// CHECK-LABEL: hoist_ops
// CHECK: %[[EVENT0:.*]] = scf.for {{.*}} iter_args(%[[EVENT1:.*]] = {{.*}}
// CHECK: %[[EVENT2:.*]] = air.herd @herd_0 async [%[[EVENT1]]]  tile
// CHECK: %[[EVENT3:.*]] = air.wait_all async [%[[EVENT2]]]
// CHECK: scf.yield %[[EVENT3]]
// CHECK: %[[EVENT4:.*]] = scf.for {{.*}} iter_args(%[[EVENT5:.*]] = {{.*}}
// CHECK: memref.alloc() {hoist_alloc = "true"}
// CHECK: air.channel.put async {{.*}} @channel_0
// CHECK: air.channel.get async {{.*}} @channel_1
// CHECK: memref.dealloc
// CHECK: scf.yield
// CHECK-NEXT: {isolated = true, unroll = 2 : i64}

air.channel @channel_1 [1, 1]
air.channel @channel_0 [1, 1]
func.func @hoist_ops(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>, %arg3: memref<1024x1024xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1) : memref<256x1024xbf16>, memref<1024x1024xbf16> attributes {id = 7 : i32} {
    %1 = air.segment async  args(%arg10=%arg4, %arg11=%arg5, %arg12=%arg6, %arg13=%arg7, %arg14=%arg8, %arg15=%arg9) : index, index, index, index, memref<256x1024xbf16>, memref<1024x1024xbf16> {
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c512 = arith.constant 512 : index
      %c64 = arith.constant 64 : index
      %30 = air.wait_all async 
      %31 = scf.for %arg20 = %c0 to %c2 step %c1_0 iter_args(%arg21 = %30) -> (!air.async.token) {
        %async_token, %results = air.execute [%arg21] -> (memref<32x32xbf16, 1>) {
          %alloc = memref.alloc() {hoist_alloc = "true"} : memref<32x32xbf16, 1>
          air.execute_terminator %alloc : memref<32x32xbf16, 1>
        }
        %2 = scf.for %arg16 = %c0 to %c512 step %c64 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %5 = air.channel.put async [%arg17]  @channel_0[] (%results[] [] []) : (memref<32x32xbf16, 1>)
          scf.yield %5 : !air.async.token
        }
        %3 = air.herd @herd_0 async [%async_token]  tile (%arg16, %arg17) in (%arg18=%c1_0, %arg19=%c1_0) {
          %c128 = arith.constant 128 : index
          %c0_2 = arith.constant 0 : index
          %c512_3 = arith.constant 512 : index
          %5 = air.wait_all async 
          %async_token_4, %results_5 = air.execute [%5] -> (memref<32x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x32xbf16, 2>
            air.execute_terminator %alloc : memref<32x32xbf16, 2>
          } 
          %async_token_9 = air.execute [%async_token_4] {
            memref.dealloc %results_5 : memref<32x32xbf16, 2>
          } 
        }
        %4 = scf.for %arg16 = %c0 to %c512 step %c64 iter_args(%arg17 = %async_token) -> (!air.async.token) {
          %5 = air.channel.get async [%arg17]  @channel_1[] (%results[] [] []) : (memref<32x32xbf16, 1>)
          scf.yield %5 : !air.async.token
        }
        %async_token_1 = air.execute [%4, %2] {
          memref.dealloc %results : memref<32x32xbf16, 1>
        }
        scf.yield %async_token_1 : !air.async.token
      } {unroll = 2}
    }
  }
  return
}
