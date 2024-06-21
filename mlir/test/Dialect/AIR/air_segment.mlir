//===- air_segment.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

module {

// CHECK-LABEL: module
// CHECK: func.func @test
func.func @test(%arg0 : memref<16x16xf32>, %arg1 : memref<16x16xf32>) -> () {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index

  // CHECK: air.segment attributes {foo = "bar"} {
  air.segment attributes {foo = "bar"} {
  }

  // CHECK: air.segment {
  air.segment args() {
  }

  // CHECK: air.segment unroll(%{{.*}}, %{{.*}}) in (%{{.*}}=%c1, %{{.*}}=%c2) attributes {foo = "bar"} {
  air.segment unroll (%tx, %ty) in (%size_x = %c1, %size_y = %c2) attributes {foo = "bar"} {
  }

  // CHECK: air.segment unroll(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}}=%c2, %{{.*}}=%c3, %{{.*}}=%c4) {
  air.segment unroll (%tx, %ty, %tz) in (%sx = %c2, %sy = %c3, %sz = %c4) attributes {  } {
  }

  // CHECK: air.segment async unroll(%{{.*}}) in (%{{.*}}=%c1)
  %t0 = air.segment async unroll (%tx) in (%size_x = %c1) {
  }

  // CHECK: %{{.*}} = air.segment async [%{{.*}}] unroll(%{{.*}}) in (%{{.*}}=%c2)
  %t1 = air.segment async [%t0] unroll (%tx) in (%size_x = %c2) {
  }
  
  // CHECK: %{{.*}} = air.segment @memcpy_nd async [%{{.*}}]
  %t2 = air.segment async [%t1] attributes {sym_name = "memcpy_nd"} {
  }

  // CHECK: air.segment [%{{.*}}, %{{.*}}] unroll(%{{.*}}) in (%{{.*}}=%c3)
  air.segment [%t0, %t1] unroll (%tx) in (%size_x = %c3) {
  }

  // CHECK: air.segment @memcpy_nd unroll(%{{.*}}, %{{.*}}) in (%{{.*}}=%c4, %{{.*}}=%c1) args(%{{.*}}=%{{.*}}) : memref<16x16xf32> {
  air.segment unroll (%arg2, %arg3) in (%size_x = %c4, %size_y = %c1) args(%arg4=%arg0) : memref<16x16xf32> attributes {sym_name = "memcpy_nd"} {
    %1 = memref.alloc() : memref<16x16xf32>
    air.dma_memcpy_nd (%1[] [] [], %arg4[] [] []) {id = 1 : i32} : (memref<16x16xf32>, memref<16x16xf32>)
    memref.dealloc %1 : memref<16x16xf32>
  }

  return
}

}