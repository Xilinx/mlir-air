//===- air_rm_alloc_linalg_copy.mlir ---------------------------*- MLIR -*-===//
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

// RUN: air-opt %s -air-linalg-codegen=test-patterns | FileCheck %s
// CHECK: func.func @forward(%[[VAL_0:.*]]: memref<4096xi32>, %[[VAL_1:.*]]: memref<4096xi32>, %[[VAL_2:.*]]: memref<?xi32>) {
// CHECK:   %[[VAL_3:.*]] = memref.cast %[[VAL_2]] : memref<?xi32> to memref<4096xi32>
// CHECK:   linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel"]} ins(%[[VAL_0]], %[[VAL_1]] : memref<4096xi32>, memref<4096xi32>) outs(%[[VAL_3]] : memref<4096xi32>) {
// CHECK:   ^bb0(%[[VAL_4:.*]]: i32, %[[VAL_5:.*]]: i32, %[[VAL_6:.*]]: i32):
// CHECK:     %[[VAL_7:.*]] = arith.muli %[[VAL_4]], %[[VAL_5]] : i32
// CHECK:     linalg.yield %[[VAL_7]] : i32
// CHECK:   }
// CHECK:   return
// CHECK: }
// XFAIL: *
module attributes {torch.debug_module_name = "model"}  {
  func.func @forward(%arg0: memref<4096xi32>, %arg1: memref<4096xi32>, %arg2: memref<?xi32>) {
    %0 = memref.alloc() : memref<4096xi32>
    linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%arg0, %arg1 : memref<4096xi32>, memref<4096xi32>) outs(%0 : memref<4096xi32>) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i32):  // no predecessors
      %2 = arith.muli %arg3, %arg4 : i32
      linalg.yield %2 : i32
    }
    %1 = memref.cast %0 : memref<4096xi32> to memref<?xi32>
    memref.copy %1, %arg2 : memref<?xi32> to memref<?xi32>
    return
  }
}