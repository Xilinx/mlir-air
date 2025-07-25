//===- affine_if.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// CHECK: %[[EVENT0:.*]] = air.wait_all async
// CHECK-NEXT: %[[EVENT1:.*]] = affine.if
// CHECK-NEXT: %[[EVENT2:.*]] = air.channel.put async [%[[EVENT0]]]
// CHECK-NEXT: %[[EVENT3:.*]] = air.wait_all async [%[[EVENT2]]]
// CHECK-NEXT: affine.yield %[[EVENT3]]
// CHECK-NEXT: } else {
// CHECK-NEXT: %[[EVENT4:.*]] = air.wait_all async [{{.*}}%[[EVENT0]]
// CHECK-NEXT: %[[EVENT5:.*]] = affine.if
// CHECK: %[[EVENT6:.*]] = air.channel.get async [%[[EVENT4]]]
// CHECK-NEXT: %[[EVENT7:.*]] = air.channel.put async [%[[EVENT6]]]
// CHECK-NEXT: %[[EVENT8:.*]] = air.wait_all async [%[[EVENT7]]]
// CHECK-NEXT: affine.yield %[[EVENT8]]
// CHECK-NEXT: } else {
// CHECK: %[[EVENT9:.*]] = air.channel.get async [%[[EVENT4]]]
// CHECK-NEXT: %[[EVENT10:.*]] = air.wait_all async [%[[EVENT9]]]
// CHECK-NEXT: affine.yield %[[EVENT10]]

module {
  air.channel @channel_0 [4]
  func.func @affine_if_1(%arg0: memref<1x1x2048xi32>, %arg1: memref<1x1x2048xi32>)  {
    %c4_1 = arith.constant 4 : index
    %c1_2 = arith.constant 1 : index
    air.herd @herd_0  tile (%arg4, %arg4_1) in (%arg6=%c4_1, %arg6_1=%c1_2) {
      %cst = arith.constant 0 : i32
      %cst_1 = arith.constant 1 : i32
      %2 = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
      affine.if affine_set<()[s0] : (s0 == 0)>()[%arg4] {
        air.channel.put  @channel_0[%arg4] (%2[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
      }
      else {
        affine.if affine_set<()[s0] : (s0 - 1 >= 0, -s0 + 2 >= 0)>()[%arg4] {
          %c1 = arith.constant 1 : index
          %iv_sub1 = arith.subi %arg4, %c1 : index
          air.channel.get  @channel_0[%iv_sub1] (%2[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
          air.channel.put  @channel_0[%arg4] (%2[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
        }
        else {
          %c1 = arith.constant 1 : index
          %iv_sub1 = arith.subi %arg4, %c1 : index
          air.channel.get  @channel_0[%iv_sub1] (%2[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
        }
      }
    }
    return
  }
}
