//===- airrt_alloc_dealloc.mlir --------------------------------*- MLIR -*-===//
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

// RUN: air-opt %s -airrt-to-llvm | FileCheck %s
// CHECK: %[[C64:.*]] = arith.constant 64 : index
// CHECK: call @air_alloc_L2_1d1i32(%[[C64]]) : (index) -> memref<?xi32, 1>
// CHECK: %[[C25:.*]] = arith.constant 25 : index
// CHECK: call @air_alloc_L2_2d1i32(%[[C25]]) : (index) -> memref<?x?xi32, 1>
// CHECK: %[[C6:.*]] = arith.constant 6 : index
// CHECK: call @air_alloc_L2_3d1i32(%[[C6]]) : (index) -> memref<?x?x?xi32, 1>
// CHECK: %[[C24:.*]] = arith.constant 24 : index
// CHECK: call @air_alloc_L2_4d1i32(%[[C24]]) : (index) -> memref<?x?x?x?xi32, 1>
// CHECK: call @air_dealloc_L2_1d1i32({{.*}}) : (memref<?xi32, 1>) -> ()
// CHECK: call @air_dealloc_L2_2d1i32({{.*}}) : (memref<?x?xi32, 1>) -> ()
// CHECK: call @air_dealloc_L2_3d1i32({{.*}}) : (memref<?x?x?xi32, 1>) -> ()
// CHECK: call @air_dealloc_L2_4d1i32({{.*}}) : (memref<?x?x?x?xi32, 1>) -> ()
module {
  func.func @f() {
    %1 = airrt.alloc : memref<64xi32, 1>
    %2 = airrt.alloc : memref<5x5xi32, 1>
    %3 = airrt.alloc : memref<1x2x3xi32, 1>
    %4 = airrt.alloc : memref<1x2x3x4xi32, 1>
    airrt.dealloc %1 : memref<64xi32, 1>
    airrt.dealloc %2 : memref<5x5xi32, 1>
    airrt.dealloc %3 : memref<1x2x3xi32, 1>
    airrt.dealloc %4 : memref<1x2x3x4xi32, 1>
    return
  }
}

