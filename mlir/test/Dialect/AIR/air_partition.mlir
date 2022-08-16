// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt %s | FileCheck %s

module {

// CHECK-LABEL: module
// CHECK: func.func @test
func.func @test(%arg0 : memref<16x16xf32>, %arg1 : memref<16x16xf32>) -> () {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index

  // CHECK: air.partition attributes {foo = "bar"} {
  air.partition attributes {foo = "bar"} {
    air.partition_terminator
  }

  // CHECK: air.partition unroll(%{{.*}}, %{{.*}}) in (%{{.*}}=%c1, %{{.*}}=%c2) attributes {foo = "bar"} {
  air.partition unroll (%tx, %ty) in (%size_x = %c1, %size_y = %c2) attributes {foo = "bar"} {
    air.partition_terminator
  }

  // CHECK: air.partition unroll(%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}}=%c2, %{{.*}}=%c3, %{{.*}}=%c4) {
  air.partition unroll (%tx, %ty, %tz) in (%sx = %c2, %sy = %c3, %sz = %c4) attributes {  } {
    air.partition_terminator
  }

  // CHECK: air.partition async unroll(%{{.*}}) in (%{{.*}}=%c1)
  %t0 = air.partition async unroll (%tx) in (%size_x = %c1) {
    air.partition_terminator
  }

  // CHECK: %{{.*}} = air.partition async [%{{.*}}] unroll(%{{.*}}) in (%{{.*}}=%c2)
  %t1 = air.partition async [%t0] unroll (%tx) in (%size_x = %c2) {
    air.partition_terminator
  }

  // CHECK: air.partition [%{{.*}}, %{{.*}}] unroll(%{{.*}}) in (%{{.*}}=%c3)
  air.partition [%t0, %t1] unroll (%tx) in (%size_x = %c3) {
    air.partition_terminator
  }

  // CHECK: air.partition unroll(%{{.*}}, %{{.*}}) in (%{{.*}}=%c4, %{{.*}}=%c1) args(%{{.*}}=%{{.*}}) : memref<16x16xf32> attributes {sym_name = "memcpy_nd"} {
  air.partition unroll (%arg2, %arg3) in (%size_x = %c4, %size_y = %c1) args(%arg4=%arg0) : memref<16x16xf32> attributes {sym_name = "memcpy_nd"} {
    %1 = memref.alloc() : memref<16x16xf32>
    air.dma_memcpy_nd (%1[] [] [], %arg4[] [] []) {id = 1 : i32} : (memref<16x16xf32>, memref<16x16xf32>)
    memref.dealloc %1 : memref<16x16xf32>
    air.partition_terminator
  }

  return
}

}