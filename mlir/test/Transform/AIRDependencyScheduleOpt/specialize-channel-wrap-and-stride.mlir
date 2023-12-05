//===- specialize-channel-wrap-and-stride.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-specialize-channel-wrap-and-stride | FileCheck %s

// Specialize air.channel ops in perfectly nested scf.for in air.segment with wraps and strides.
// CHECK-LABEL: test
// CHECK: air.segment
// CHECK: %[[CST8:.*]] = arith.constant 8 : index
// CHECK: %[[CST0:.*]] = arith.constant 0 : index
// CHECK: %[[CST1:.*]] = arith.constant 1 : index
// CHECK: %[[CST16:.*]] = arith.constant 16 : index
// CHECK: %[[CST64:.*]] = arith.constant 64 : index
// CHECK: %[[CST512:.*]] = arith.constant 512 : index
// CHECK: %[[CST4:.*]] = arith.constant 4 : index
// CHECK: %[[EVENT0:.*]] = air.channel.put async [{{.*}}]  @channel_0[] (%[[VAL0:.*]][%[[CST0]], %[[CST0]]] [%[[CST8]], %[[CST4]], %[[CST4]]] [%[[CST64]], %[[CST16]], %[[CST1]]]) : (memref<8x16xi32, 1>)
// CHECK: scf.for{{.*}}iter_args(%[[EVENT1:.*]] = %[[EVENT0]])
// CHECK: air.herd
// CHECK: air.channel.get async{{.*}}@channel_0
// CHECK: air.herd_terminator
// CHECK: air.segment_terminator

module {
  air.channel @channel_0 [1, 1]
  func.func @test(%arg0: memref<256x1024xbf16>, %arg1: memref<1024x1024xbf16>, %arg2: memref<1024x1024xbf16>, %arg3: memref<1024x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1) : memref<256x1024xbf16>, memref<1024x1024xbf16> attributes {id = 7 : i32} {
      %1 = air.segment async  args(%arg15=%arg4, %arg16=%arg5, %arg17=%arg6, %arg18=%arg7, %arg19=%arg8, %arg20=%arg9) : index, index, index, index, memref<256x1024xbf16>, memref<1024x1024xbf16> {
        %c0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c16 = arith.constant 16 : index
        %c64 = arith.constant 64 : index
        %c512 = arith.constant 512 : index
        %c4 = arith.constant 4 : index
        %async_token_25, %results_26 = air.execute -> (memref<8x16xi32, 1>) {
          %alloc = memref.alloc() : memref<8x16xi32, 1>
          air.execute_terminator %alloc : memref<8x16xi32, 1>
        }
        %async_token_0 = air.wait_all async
        %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
          %async_token_3 = air.channel.put async [%arg11]  @channel_0[] (%results_26[%c0, %arg10] [%c4, %c4] [%c16, %c1_1]) : (memref<8x16xi32, 1>)
          scf.yield %async_token_3 : !air.async.token
        }
        %4 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %3) -> (!air.async.token) {
          %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
            %async_token_27, %results_28 = air.execute -> (memref<4x4xi32, 2>) {
              %alloc = memref.alloc() : memref<4x4xi32, 2>
              air.execute_terminator %alloc : memref<4x4xi32, 2>
            }
            %5 = air.channel.get async [%async_token_27]  @channel_0[%arg21, %arg22] (%results_28[] [] []) : (memref<4x4xi32, 2>)
            air.herd_terminator
          }
          scf.yield %2 : !air.async.token
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
