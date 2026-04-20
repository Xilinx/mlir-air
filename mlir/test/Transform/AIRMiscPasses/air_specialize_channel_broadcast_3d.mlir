//===- air_specialize_channel_broadcast_3d.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-specialize-dma-broadcast | FileCheck %s

// Test 3D channel indices: channel with segment unroll index as extra leading
// dimension. The segment index is NOT a herd ID, so it must dispatch via scf.if
// rather than affine.if. The herd y-dimension dispatches via affine.if.

// CHECK-DAG: [[$SET0:#set[0-9]*]] = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>
// CHECK-DAG: [[$SET1:#set[0-9]*]] = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 1 == 0)>
// CHECK-DAG: [[$SET2:#set[0-9]*]] = affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 - 2 == 0)>

// Specialized channels: 2 segment iterations x 4 herd y positions = 8 channels.
// Each has broadcast_shape [1, 4, 1] — dim 1 (herd x) still broadcasts to 4.
// CHECK-DAG: air.channel @ch3d_0_0 [1, 1, 1] {broadcast_shape = [1, 4 : index, 1]}
// CHECK-DAG: air.channel @ch3d_0_1 [1, 1, 1] {broadcast_shape = [1, 4 : index, 1]}
// CHECK-DAG: air.channel @ch3d_0_2 [1, 1, 1] {broadcast_shape = [1, 4 : index, 1]}
// CHECK-DAG: air.channel @ch3d_0_3 [1, 1, 1] {broadcast_shape = [1, 4 : index, 1]}
// CHECK-DAG: air.channel @ch3d_1_0 [1, 1, 1] {broadcast_shape = [1, 4 : index, 1]}
// CHECK-DAG: air.channel @ch3d_1_1 [1, 1, 1] {broadcast_shape = [1, 4 : index, 1]}
// CHECK-DAG: air.channel @ch3d_1_2 [1, 1, 1] {broadcast_shape = [1, 4 : index, 1]}
// CHECK-DAG: air.channel @ch3d_1_3 [1, 1, 1] {broadcast_shape = [1, 4 : index, 1]}

// Puts: segment index dispatched via scf.if, herd y index is constant per put.
// CHECK-LABEL: func.func @segment_unroll_3d_broadcast
// CHECK: arith.cmpi eq
// CHECK: scf.if
// CHECK: air.channel.put async{{.*}}@ch3d_0_0
// CHECK: } else {
// CHECK: air.channel.put async{{.*}}@ch3d_1_0
// CHECK: arith.cmpi eq
// CHECK: scf.if
// CHECK: air.channel.put async{{.*}}@ch3d_0_1
// CHECK: } else {
// CHECK: air.channel.put async{{.*}}@ch3d_1_1

// Gets: segment index dispatched via scf.if, herd y via affine.if.
// CHECK: air.herd @herd_0
// CHECK: arith.cmpi eq
// CHECK: scf.if
// CHECK: affine.if [[$SET0]]
// CHECK: air.channel.get async{{.*}}@ch3d_0_0
// CHECK: } else {
// CHECK: affine.if [[$SET1]]
// CHECK: air.channel.get async{{.*}}@ch3d_0_1
// CHECK: } else {
// CHECK: affine.if [[$SET2]]
// CHECK: air.channel.get async{{.*}}@ch3d_0_2
// CHECK: } else {
// CHECK: air.channel.get async{{.*}}@ch3d_0_3
// CHECK: } else {
// CHECK: affine.if [[$SET0]]
// CHECK: air.channel.get async{{.*}}@ch3d_1_0
// CHECK: } else {
// CHECK: affine.if [[$SET1]]
// CHECK: air.channel.get async{{.*}}@ch3d_1_1

module {
  air.channel @ch3d [2, 1, 4] {broadcast_shape = [2 : index, 4 : index, 4 : index]}
  func.func @segment_unroll_3d_broadcast(%arg0: memref<128xbf16>, %arg1: memref<64xbf16, 2 : i32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%a0, %a1) in (%a2=%c1, %a3=%c1) args(%la0=%arg0, %la1=%arg1) : memref<128xbf16>, memref<64xbf16, 2 : i32> attributes {id = 1 : i32} {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %1 = air.segment @seg async unroll(%arg12, %arg13) in (%arg14=%c2, %arg15=%c1_0) args(%sa0=%la0, %sa1=%la1) : memref<128xbf16>, memref<64xbf16, 2 : i32> attributes {id = 2 : i32} {
        %c0 = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c3_s = arith.constant 3 : index
        %c4 = arith.constant 4 : index
        %t0 = air.channel.put async @ch3d[%arg12, %c0, %c0] (%sa0[] [] []) {id = 1 : i32} : (memref<128xbf16>)
        %t1 = air.channel.put async @ch3d[%arg12, %c0, %c1_s] (%sa0[] [] []) {id = 2 : i32} : (memref<128xbf16>)
        %t2 = air.channel.put async @ch3d[%arg12, %c0, %c2_s] (%sa0[] [] []) {id = 3 : i32} : (memref<128xbf16>)
        %t3 = air.channel.put async @ch3d[%arg12, %c0, %c3_s] (%sa0[] [] []) {id = 4 : i32} : (memref<128xbf16>)
        %2 = air.herd @herd_0 async tile (%arg16, %arg17) in (%arg18=%c4, %arg19=%c4) args(%buf=%sa1, %seg_idx=%arg12) : memref<64xbf16, 2 : i32>, index attributes {id = 3 : i32} {
          %3 = air.channel.get async @ch3d[%seg_idx, %arg16, %arg17] (%buf[] [] []) {id = 5 : i32} : (memref<64xbf16, 2 : i32>)
        }
      }
    }
    return
  }
}
