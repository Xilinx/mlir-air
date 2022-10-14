//===- air_partition_dma_memcpy_nd.mlir ------------------------*- MLIR -*-===//
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

// RUN: air-opt %s -air-dependency | FileCheck %s

module {

// CHECK-LABEL: module
func.func @memcpy_nd(%arg0: memref<4096xi32>) {
  %c0 = arith.constant 0 : index
  %c4096 = arith.constant 4096 : index
  %c128 = arith.constant 128 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  air.partition unroll (%arg1, %arg2) in (%size_x = %c4, %size_y = %c1) args(%arg3=%arg0) : memref<4096xi32> attributes {sym_name = "memcpy_nd"} {
  // CHECK: %[[EVENT0:.*]] = air.partition @memcpy_nd async unroll
    %c32 = arith.constant 32 : index
    %0 = arith.muli %arg1, %c32 : index
    // CHECK: %[[EVENT1:.*]], %[[EVENT2:.*]] = air.execute
    %1 = memref.alloc() : memref<32xi32, 2>
    // CHECK: %[[EVENT3:.*]], %[[EVENT4:.*]] = air.execute
    %c1_0 = arith.constant 1 : index
    air.dma_memcpy_nd (%1[] [] [], %arg3[%0] [%c32] [%c1_0]) {id = 1 : i32} : (memref<32xi32, 2>, memref<4096xi32>)
    // CHECK: %[[EVENT5:.*]] = air.dma_memcpy_nd async [{{.*}}%[[EVENT3]]{{.*}}, {{.*}}%[[EVENT1]]{{.*}}]
    air.dma_memcpy_nd (%arg3[%0] [%c32] [%c1_0], %1[] [] []) {id = 2 : i32} : (memref<4096xi32>, memref<32xi32, 2>)
    // CHECK: %[[EVENT6:.*]] = air.dma_memcpy_nd async [{{.*}}%[[EVENT5]]{{.*}}]
    memref.dealloc %1 : memref<32xi32, 2>
    // CHECK: %[[EVENT7:.*]] = air.execute [{{.*}}%[[EVENT6]]{{.*}}]
    air.partition_terminator
  }
  return
}

}