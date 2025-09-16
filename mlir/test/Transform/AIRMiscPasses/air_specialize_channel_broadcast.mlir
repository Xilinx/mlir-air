//===- air_specialize_channel_broadcast.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-specialize-dma-broadcast | FileCheck %s

// CHECK: [[$SET0:#set[0-9]*]] = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 3 >= 0)>
// CHECK: [[$SET1:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 - 1 == 0, s1 >= 0, -s1 + 3 >= 0)>
// CHECK: [[$SET2:#set[0-9]+]] = affine_set<()[s0, s1] : (s0 - 2 == 0, s1 >= 0, -s1 + 3 >= 0)>
// CHECK: air.channel @L2ToL1Chan1_0 [1, 1] {broadcast_shape = [1, 4]}
// CHECK: air.channel @L2ToL1Chan1_1 [1, 1] {broadcast_shape = [1, 4]}
// CHECK: air.channel @L2ToL1Chan1_2 [1, 1] {broadcast_shape = [1, 4]}
// CHECK: air.channel @L2ToL1Chan1_3 [1, 1] {broadcast_shape = [1, 4]}
// CHECK: air.channel.put async
// CHECK: @L2ToL1Chan1_0
// CHECK: air.channel.put async
// CHECK: @L2ToL1Chan1_1
// CHECK: air.channel.put async
// CHECK: @L2ToL1Chan1_2
// CHECK: air.channel.put async
// CHECK: @L2ToL1Chan1_3
// CHECK: air.herd @herd_0
// CHECK: affine.if [[$SET0]]
// CHECK: air.channel.get async
// CHECK: @L2ToL1Chan1_0
// CHECK: affine.yield
// CHECK: } else {
// CHECK: affine.if [[$SET1]]
// CHECK: air.channel.get async
// CHECK: @L2ToL1Chan1_1
// CHECK: affine.yield
// CHECK: } else {
// CHECK: affine.if [[$SET2]]
// CHECK: air.channel.get async
// CHECK: @L2ToL1Chan1_2
// CHECK: affine.yield
// CHECK: } else {
// CHECK: air.channel.get async
// CHECK: @L2ToL1Chan1_3
// CHECK: affine.yield

module {
  air.channel @L2ToL1Chan1 [4, 1] {broadcast_shape = [4, 4]}
  func.func @attention_bf16(%arg0: memref<128x64xbf16>, %arg1: memref<64x768xbf16>, %arg2: memref<768x64xbf16>, %arg3: memref<128x768xbf16>, %arg4: memref<128x64xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg5, %arg6) in (%arg7=%c1, %arg8=%c1) attributes {id = 3 : i32} {
      %1 = air.segment @attention_seg async  attributes {id = 2 : i32} {
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c1_0 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %c256 = arith.constant 256 : index
        %c4 = arith.constant 4 : index
        %c8 = arith.constant 8 : index
        %c0 = arith.constant 0 : index
        %async_token, %results = air.execute -> (memref<32x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<32x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<32x64xbf16, 1 : i32>
        } {id = 1 : i32}
        %async_token_1, %results_2 = air.execute -> (memref<32x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<32x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<32x64xbf16, 1 : i32>
        } {id = 2 : i32}
        %async_token_3, %results_4 = air.execute -> (memref<32x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<32x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<32x64xbf16, 1 : i32>
        } {id = 3 : i32}
        %async_token_5, %results_6 = air.execute -> (memref<32x64xbf16, 1 : i32>) {
          %alloc = memref.alloc() : memref<32x64xbf16, 1 : i32>
          air.execute_terminator %alloc : memref<32x64xbf16, 1 : i32>
        } {id = 4 : i32}
        %async_token_7, %results_8 = air.execute -> (memref<32x64xbf16, 2 : i32>) {
          %alloc = memref.alloc() : memref<32x64xbf16, 2 : i32>
          air.execute_terminator %alloc : memref<32x64xbf16, 2 : i32>
        } {id = 5 : i32}
        %2 = air.channel.put async [%async_token]  @L2ToL1Chan1[%c0, %c0] (%results[%c0, %c0, %c0, %c0] [%c8, %c8, %c4, %c8] [%c8, %c256, %c64, %c1_0]) {id = 1 : i32} : (memref<32x64xbf16, 1 : i32>)
        %3 = air.channel.put async [%async_token_1]  @L2ToL1Chan1[%c1_0, %c0] (%results_2[%c0, %c0, %c0, %c0] [%c8, %c8, %c4, %c8] [%c8, %c256, %c64, %c1_0]) {id = 2 : i32} : (memref<32x64xbf16, 1 : i32>)
        %4 = air.channel.put async [%async_token_3]  @L2ToL1Chan1[%c2, %c0] (%results_4[%c0, %c0, %c0, %c0] [%c8, %c8, %c4, %c8] [%c8, %c256, %c64, %c1_0]) {id = 3 : i32} : (memref<32x64xbf16, 1 : i32>)
        %5 = air.channel.put async [%async_token_5]  @L2ToL1Chan1[%c3, %c0] (%results_6[%c0, %c0, %c0, %c0] [%c8, %c8, %c4, %c8] [%c8, %c256, %c64, %c1_0]) {id = 4 : i32} : (memref<32x64xbf16, 1 : i32>)
        %6 = air.herd @herd_0 async [%async_token_7]  tile (%arg9, %arg10) in (%arg11=%c4, %arg12=%c4) args(%arg13=%results_8) : memref<32x64xbf16, 2 : i32> attributes {id = 1 : i32} {
          %7 = air.channel.get async  @L2ToL1Chan1[%arg9, %arg10] (%arg13[] [] []) {id = 5 : i32} : (memref<32x64xbf16, 2 : i32>)
        }
      }
    }
    return
  }
}
