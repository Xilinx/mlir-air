//===- air_launch.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-to-std %s | FileCheck %s

// CHECK-LABEL:   func.func @launch_0(
// CHECK-SAME:               %{{.*}}: memref<16xf16>,
// CHECK-SAME:               %{{.*}}: memref<16xf16>) {
// CHECK:       affine.for %{{.*}} = 0 to 4 {
// CHECK:       affine.for %{{.*}} = 0 to 2 {
// CHECK:       affine.for %{{.*}} = 0 to 2 {
func.func @launch_0(%arg0: memref<16xf16>, %arg1: memref<16xf16>) {
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  air.launch (%arg2, %arg3, %arg4) in (%arg5=%c4, %arg6=%c2, %arg7=%c2) args(%arg8=%arg0, %arg9=%arg1) : memref<16xf16>, memref<16xf16> {
  }
  return
}

// CHECK-LABEL: launch_1
// CHECK: %[[VAL_1:.*]] = airrt.wait_all : !airrt.event
// CHECK: %[[VAL_2:.*]] = airrt.wait_all %[[VAL_1]] : !airrt.event
// CHECK: %[[VAL_3:.*]] = arith.constant 0 : index
// CHECK: %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK: %[[VAL_5:.*]] = airrt.wait_all %[[VAL_1]], %[[VAL_2]] : !airrt.event
// CHECK:           affine.for %[[VAL_6:.*]] = 0 to 1 {
func.func @launch_1() {
  %e0 = air.wait_all async
  %e1 = air.wait_all async [%e0]
  %t = air.launch async [%e0, %e1] () in () {
  }
  return
}

// Multi-dimensional air.launch, with async. air.channel consuming the induction vars.

// CHECK-LABEL: launch_2
// CHECK: affine.for %[[VAL_0:.*]] = 0 to 2 {
// CHECK: affine.for %[[VAL_1:.*]] = 0 to 2 {
// CHECK: affine.for %[[VAL_2:.*]] = 0 to 2 {
// CHECK: affine.for %[[VAL_3:.*]] = 0 to 2 {

air.channel @channel_3 [1, 1]
air.channel @channel_2 [1, 1]
air.channel @channel_1 [1, 1]
func.func @launch_2(%arg0: memref<2x32x6x6xi32>, %arg1: memref<4x32x3x3xi32>, %arg2: memref<2x4x4x4xi32>) {
  %c2 = arith.constant 2 : index
  %0 = air.launch async (%arg3, %arg4, %arg5, %arg6) in (%arg7=%c2, %arg8=%c2, %arg9=%c2, %arg10=%c2) args(%arg11=%arg0, %arg12=%arg2, %arg13=%arg1) : memref<2x32x6x6xi32>, memref<2x4x4x4xi32>, memref<4x32x3x3xi32> attributes {id = 1 : i32} {
    %c64 = arith.constant 64 : index
    %c1152 = arith.constant 1152 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %1 = air.channel.put async  @channel_1[] (%arg11[%arg3, %c0] [%c1, %c1152] [%c1152, %c1]) {id = 1 : i32} : (memref<2x32x6x6xi32>)
    %2 = air.channel.put async  @channel_2[] (%arg13[] [] []) {id = 2 : i32} : (memref<4x32x3x3xi32>)
    %3 = air.channel.get async  @channel_3[] (%arg12[%arg3, %c0] [%c1, %c64] [%c64, %c1]) {id = 3 : i32} : (memref<2x4x4x4xi32>)
    %4 = air.segment @segment_0 async  {
      %async_token, %results = air.execute -> (memref<1x32x6x6xi32, 1>) {
        %alloc = memref.alloc() : memref<1x32x6x6xi32, 1>
        air.execute_terminator %alloc : memref<1x32x6x6xi32, 1>
      }
      %5 = air.channel.get async [%async_token]  @channel_1[] (%results[] [] []) {id = 4 : i32} : (memref<1x32x6x6xi32, 1>)
      %async_token_0, %results_1 = air.execute -> (memref<4x32x3x3xi32, 1>) {
        %alloc = memref.alloc() : memref<4x32x3x3xi32, 1>
        air.execute_terminator %alloc : memref<4x32x3x3xi32, 1>
      }
      %6 = air.channel.get async [%async_token_0]  @channel_2[] (%results_1[] [] []) {id = 5 : i32} : (memref<4x32x3x3xi32, 1>)
      %async_token_2, %results_3 = air.execute -> (memref<1x4x4x4xi32, 1>) {
        %alloc = memref.alloc() : memref<1x4x4x4xi32, 1>
        air.execute_terminator %alloc : memref<1x4x4x4xi32, 1>
      }
      %7 = air.channel.put async [%6]  @channel_3[] (%results_3[] [] []) {id = 12 : i32} : (memref<1x4x4x4xi32, 1>)
    }
  }
  return
}
