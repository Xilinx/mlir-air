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

  // CHECK: air.launch (%{{.*}}, %{{.*}}) in (%{{.*}}=%c1, %{{.*}}=%c2) attributes {foo = "bar"} {
  air.launch (%tx, %ty) in (%size_x = %c1, %size_y = %c2) attributes {foo = "bar"} {
    air.launch_terminator
  }

  // CHECK: air.launch (%{{.*}}, %{{.*}}, %{{.*}}) in (%{{.*}}=%c2, %{{.*}}=%c3, %{{.*}}=%c4) {
  air.launch (%tx, %ty, %tz) in (%sx = %c2, %sy = %c3, %sz = %c4) attributes {  } {
    air.launch_terminator
  }

  // CHECK: air.launch async (%{{.*}}) in (%{{.*}}=%c1) 
  %t0 = air.launch async (%tx) in (%size_x = %c1) {
    air.launch_terminator
  }

  // CHECK: %{{.*}} = air.launch async [%{{.*}}] (%{{.*}}) in (%{{.*}}=%c2) 
  %t1 = air.launch async [%t0] (%tx) in (%size_x = %c2) {
    air.launch_terminator
  }

  // CHECK: air.launch [%{{.*}}, %{{.*}}] (%{{.*}}) in (%{{.*}}=%c3)
  air.launch [%t0, %t1] (%tx) in (%size_x = %c3) {
    air.launch_terminator
  }

  return
}

}