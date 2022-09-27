//===- air_launch.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

module @module_1 {

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
  }

  // CHECK: air.launch async (%{{.*}}) in (%{{.*}}=%c1)
  %t0 = air.launch async (%tx) in (%size_x = %c1) {
    air.launch_terminator
  }

  // CHECK: %{{.*}} = air.launch async [%{{.*}}] (%{{.*}}) in (%{{.*}}=%c2)
  %t1 = air.launch async [%t0] (%tx) in (%size_x = %c2) {
  }

  // CHECK: air.launch [%{{.*}}, %{{.*}}] (%{{.*}}) in (%{{.*}}=%c3)
  air.launch [%t0, %t1] (%tx) in (%size_x = %c3) {
    air.launch_terminator
  }

  // CHECK: air.launch @memcpy_nd (%{{.*}}, %{{.*}}) in (%{{.*}}=%c4, %{{.*}}=%c1) args(%{{.*}}=%{{.*}}) : memref<16x16xf32> {
  air.launch (%arg2, %arg3) in (%size_x = %c4, %size_y = %c1) args(%arg4=%arg0) : memref<16x16xf32> attributes {sym_name = "memcpy_nd"} {
    %1 = memref.alloc() : memref<16x16xf32>
    air.dma_memcpy_nd (%1[] [] [], %arg4[] [] []) {id = 1 : i32} : (memref<16x16xf32>, memref<16x16xf32>)
    memref.dealloc %1 : memref<16x16xf32>
    air.launch_terminator
  }

  // CHECK: air.launch @mylaunch () in ()
  air.launch @mylaunch () in ()

  // CHECK: %[[E:.*]] = air.launch @function0 async (%{{.*}}) in (%{{.*}}=%c1) args(%{{.*}}=%{{.*}}, %{{.*}}=%{{.*}})
  // CHECK: air.launch @function1 async [%[[E]]] (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2, %{{.*}}=%c4) args(%{{.*}}=%{{.*}})
  %f0 = air.launch @function0 async (%t) in (%s = %c1) args(%a=%arg0, %b=%arg1) : memref<16x16xf32>, memref<16x16xf32>
  %f1 = air.launch @function1 async [%f0] (%t0, %t1) in (%s0 = %c2, %s1=%c4) args(%a=%arg1) : memref<16x16xf32>

  return
}

}