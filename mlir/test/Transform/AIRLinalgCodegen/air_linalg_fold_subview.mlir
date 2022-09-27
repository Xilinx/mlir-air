//===- air_linalg_fold_subview.mlir ----------------------------*- MLIR -*-===//
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

// RUN: air-opt %s -air-linalg-codegen=test-patterns

// CHECK: scf.for %[[VAL_7:.*]] = {{.*}} {
// CHECK:   scf.for %[[VAL_8:.*]] = {{.*}} {
// CHECK:     scf.for %[[VAL_9:.*]] = {{.*}} {
// CHECK:       scf.parallel (%[[VAL_10:.*]], %[[VAL_11:.*]]) = ({{.*}}) to ({{.*}}) step ({{.*}}) {
// CHECK:         %[[VAL_12:.*]] = arith.addi %[[VAL_7]], %[[VAL_10]] : index
// CHECK:         %[[VAL_13:.*]] = memref.subview %arg0{{\[}}%[[VAL_12]], %[[VAL_9]]] [16, 64] [1, 1] : memref<1024x1024xf32> to memref<16x64xf32, #map>
// CHECK:         %[[VAL_14:.*]] = arith.addi %[[VAL_8]], %[[VAL_11]] : index
// CHECK:         %[[VAL_15:.*]] = memref.subview %arg1{{\[}}%[[VAL_9]], %[[VAL_14]]] [64, 16] [1, 1] : memref<1024x1024xf32> to memref<64x16xf32, #map>
// CHECK:         %[[VAL_16:.*]] = arith.addi %[[VAL_7]], %[[VAL_10]] : index
// CHECK:         %[[VAL_17:.*]] = arith.addi %[[VAL_8]], %[[VAL_11]] : index
// CHECK:         %[[VAL_18:.*]] = memref.subview %0{{\[}}%[[VAL_16]], %[[VAL_17]]] [16, 16] [1, 1] : memref<1024x1024xf32> to memref<16x16xf32, #map>
#map = affine_map<(d0, d1)[s0] -> (d0 * 1024 + s0 + d1)>
module  {
  func.func @myFunc(%arg0: memref<1024x1024xf32>, %arg1: memref<1024x1024xf32>) -> memref<1024x1024xf32> {
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %c64 = arith.constant 64 : index
    %0 = memref.alloc() : memref<1024x1024xf32>
    scf.for %arg2 = %c0 to %c1024 step %c64 {
      scf.for %arg3 = %c0 to %c1024 step %c64 {
        scf.for %arg4 = %c0 to %c1024 step %c64 {
          %1 = memref.subview %arg0[%arg2, %arg4] [64, 64] [1, 1] : memref<1024x1024xf32> to memref<64x64xf32, #map>
          %2 = memref.subview %arg1[%arg4, %arg3] [64, 64] [1, 1] : memref<1024x1024xf32> to memref<64x64xf32, #map>
          %3 = memref.subview %0[%arg2, %arg3] [64, 64] [1, 1] : memref<1024x1024xf32> to memref<64x64xf32, #map>
          scf.parallel (%arg5, %arg6) = (%c0, %c0) to (%c64, %c64) step (%c16, %c16) {
            %4 = memref.subview %1[%arg5, 0] [16, 64] [1, 1] : memref<64x64xf32, #map> to memref<16x64xf32, #map>
            %5 = memref.subview %2[0, %arg6] [64, 16] [1, 1] : memref<64x64xf32, #map> to memref<64x16xf32, #map>
            %6 = memref.subview %3[%arg5, %arg6] [16, 16] [1, 1] : memref<64x64xf32, #map> to memref<16x16xf32, #map>
            %7 = memref.alloc() : memref<16x64xf32, 2>
            %8 = memref.alloc() : memref<64x16xf32, 2>
            %9 = memref.alloc() : memref<16x16xf32, 2>
            linalg.copy ins(%4 :  memref<16x64xf32, #map>) outs(%7 : memref<16x64xf32, 2>)
            linalg.copy ins(%5 :  memref<64x16xf32, #map>) outs(%8 : memref<64x16xf32, 2>)
            linalg.copy ins(%6 :  memref<16x16xf32, #map>) outs(%9 : memref<16x16xf32, 2>)
            linalg.matmul ins(%7, %8 : memref<16x64xf32, 2>, memref<64x16xf32, 2>) outs(%9 : memref<16x16xf32, 2>)
            linalg.copy ins(%9 : memref<16x16xf32, 2>) outs(%6 : memref<16x16xf32, #map>)
            memref.dealloc %7 : memref<16x64xf32, 2>
            memref.dealloc %8 : memref<64x16xf32, 2>
            memref.dealloc %9 : memref<16x16xf32, 2>
            scf.yield
          }
        }
      }
    }
    return %0 : memref<1024x1024xf32>
  }
}