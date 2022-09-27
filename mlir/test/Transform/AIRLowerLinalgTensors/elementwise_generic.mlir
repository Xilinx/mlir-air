//===- elementwise_generic.mlir --------------------------------*- MLIR -*-===//
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

// RUN: air-opt %s -air-lower-linalg-tensors | FileCheck %s
// CHECK: %[[BUF2:.*]] = AIE.buffer(%0) {sym_name = "buf2"} : memref<32x32xi32, 2>
// CHECK: %[[BUF1:.*]] = AIE.buffer(%0) {sym_name = "buf1"} : memref<32x32xi32, 2>
// CHECK: %[[BUF0:.*]] = AIE.buffer(%0) {sym_name = "buf0"} : memref<32x32xi32, 2>
// CHECK: affine.for %arg0 = 0 to 32 {
// CHECK:     affine.for %arg1 = 0 to 32 {
// CHECK:       %{{.*}} = affine.load %[[BUF0]][%arg0, %arg1] : memref<32x32xi32, 2>
// CHECK:       %{{.*}} = affine.load %[[BUF1]][%arg0, %arg1] : memref<32x32xi32, 2>
// CHECK:       %{{.*}} = arith.muli %{{.*}}, %{{.*}} : i32
// CHECK:       affine.store %{{.*}}, %[[BUF2]][%arg0, %arg1] : memref<32x32xi32, 2>
// CHECK:     }
// CHECK:   }
#map = affine_map<(d0, d1) -> (d0, d1)>
module @aie.0  {
  %0 = AIE.tile(0, 0)
  %1 = AIE.lock(%0, 2)
  %2 = AIE.lock(%0, 1)
  %3 = AIE.lock(%0, 0)
  %4 = AIE.buffer(%0) {sym_name = "buf2"} : memref<32x32xi32, 2>
  %5 = AIE.buffer(%0) {sym_name = "buf1"} : memref<32x32xi32, 2>
  %6 = AIE.buffer(%0) {sym_name = "buf0"} : memref<32x32xi32, 2>
  %7 = AIE.core(%0)  {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    AIE.useLock(%3, Acquire, 1)
    AIE.useLock(%2, Acquire, 1)
    AIE.useLock(%1, Acquire, 0)
    %8 = linalg.init_tensor [32, 32] : tensor<32x32xi32>
    %9 = bufferization.to_tensor %6 : memref<32x32xi32, 2>
    %10 = bufferization.to_tensor %5 : memref<32x32xi32, 2>
    %11 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%9, %10 : tensor<32x32xi32>, tensor<32x32xi32>) outs(%8 : tensor<32x32xi32>) {
    ^bb0(%arg0: i32, %arg1: i32, %arg2: i32):  // no predecessors
      %12 = arith.muli %arg0, %arg1 : i32
      linalg.yield %12 : i32
    } -> tensor<32x32xi32>
    memref.tensor_store %11, %4 : memref<32x32xi32, 2>
    AIE.useLock(%3, Release, 0)
    AIE.useLock(%2, Release, 0)
    AIE.useLock(%1, Release, 1)
    AIE.end
  }
}
