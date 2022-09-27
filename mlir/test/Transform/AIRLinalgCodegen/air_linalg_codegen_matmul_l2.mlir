//===- air_linalg_codegen_matmul_l2.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc.
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

// RUN: air-opt %s -air-linalg-codegen='l1-tile-size=32,32,32 l2-tile-size=64,64,64' | FileCheck %s

// CHECK-LABEL: matmul_on_memref
// CHECK: scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c128, %c128) step (%c64, %c64) {
// CHECK: scf.for %arg4 = %c0 to %c128 step %c64 {
// CHECK: memref.copy {{.*}} : memref<{{.*}}> to memref<64x64xi32, 1>
// CHECK: memref.copy {{.*}} : memref<{{.*}}> to memref<64x64xi32, 1>
// CHECK: memref.copy {{.*}} : memref<{{.*}}> to memref<64x64xi32, 1>
// CHECK: scf.parallel ({{.*}}) = (%c0, %c0) to (%c64, %c64) step (%c32, %c32) {
// CHECK: scf.for {{.*}} = %c0 to %c64 step %c32 {
// CHECK: memref.copy {{.*}} : memref<{{.*}}, 1> to memref<{{.*}}, 2>
// CHECK: memref.copy {{.*}} : memref<{{.*}}, 1> to memref<{{.*}}, 2>
// CHECK: memref.copy {{.*}} : memref<{{.*}}, 1> to memref<{{.*}}, 2>
// CHECK: memref.copy {{.*}} : memref<{{.*}}, 2> to memref<{{.*}}, 1>
// CHECK: scf.yield
// CHECK: memref.copy {{.*}} : memref<{{.*}}, 1> to memref<{{.*}}>
func.func @matmul_on_memref(%arg0: memref<128x128xi32>, %arg1: memref<128x128xi32>) -> memref<128x128xi32> {
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.alloc() : memref<128x128xi32>
    linalg.fill ins(%c0_i32 : i32) outs(%0 : memref<128x128xi32>)
    %1 = memref.alloc() : memref<128x128xi32>
    linalg.copy ins(%0 : memref<128x128xi32>) outs(%1 : memref<128x128xi32>)
    linalg.matmul ins(%arg0, %arg1 : memref<128x128xi32>, memref<128x128xi32>) outs(%1 : memref<128x128xi32>)
    return %1 : memref<128x128xi32>
  }
