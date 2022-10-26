//===- air_linalg_rm_view.mlir ---------------------------------*- MLIR -*-===//
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

// RUN: air-opt %s -air-linalg-codegen=test-patterns | FileCheck %s
// CHECK:  %[[VAL_13:.*]] = memref.subview %{{.*}}[0, %{{.*}}, %{{.*}}, %{{.*}} [1, 16, 18, 18] [1, 1, 1, 1] : memref<1x64x66x66xf32> to memref<1x16x18x18xf32, #map>
// CHECK:  %[[VAL_14:.*]] = memref.subview %{{.*}}{{\[}}%{{.*}}, %{{.*}}, 0, 0] [32, 16, 3, 3] [1, 1, 1, 1] : memref<128x64x3x3xf32> to memref<32x16x3x3xf32, #map1>
// CHECK:  %[[VAL_15:.*]] = memref.subview %{{.*}}[0, %{{.*}}, %{{.*}}, %{{.*}} [1, 32, 16, 16] [1, 1, 1, 1] : memref<1x128x64x64xf32> to memref<1x32x16x16xf32, #map2>
// CHECK:  %[[VAL_16:.*]] = memref.alloc() : memref<1x16x18x18xf32, 1>
// CHECK:  %[[VAL_17:.*]] = memref.alloc() : memref<32x16x3x3xf32, 1>
// CHECK:  %[[VAL_18:.*]] = memref.alloc() : memref<1x32x16x16xf32, 1>
// CHECK:  linalg.copy ins(%[[VAL_13]] : memref<1x16x18x18xf32, #map>) outs(%[[VAL_16]] : memref<1x16x18x18xf32, 1>)
// CHECK:  linalg.copy ins(%[[VAL_14]] : memref<32x16x3x3xf32, #map1>) outs(%[[VAL_17]] : memref<32x16x3x3xf32, 1>)
// CHECK:  linalg.copy ins(%[[VAL_15]] : memref<1x32x16x16xf32, #map2>) outs(%[[VAL_18]] : memref<1x32x16x16xf32, 1>)
// CHECK:  linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%[[VAL_16]], %[[VAL_17]] : memref<1x16x18x18xf32, 1>, memref<32x16x3x3xf32, 1>) outs(%[[VAL_18]] : memref<1x32x16x16xf32, 1>)
// CHECK:  linalg.copy ins(%[[VAL_18]] : memref<1x32x16x16xf32, 1>) outs(%[[VAL_15]] : memref<1x32x16x16xf32, #map2>)
// CHECK:  memref.dealloc %[[VAL_16]] : memref<1x16x18x18xf32, 1>
// CHECK:  memref.dealloc %[[VAL_17]] : memref<32x16x3x3xf32, 1>
// CHECK:  memref.dealloc %[[VAL_18]] : memref<1x32x16x16xf32, 1>

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 278784 + d1 * 4356 + d2 * 66 + d3 + 67)>
#map1 = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 278784 + s0 + d1 * 4356 + d2 * 66 + d3)>
#map2 = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 576 + s0 + d1 * 9 + d2 * 3 + d3)>
#map3 = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 524288 + s0 + d1 * 4096 + d2 * 64 + d3)>
module attributes {torch.debug_module_name = "Conv2D"}  {
  memref.global "private" constant @__constant_128x64x3x3xf32 : memref<128x64x3x3xf32> = dense<1.000000e+00>
  func.func @forward(%arg0: memref<1x64x64x64xf32>, %arg1: memref<1x128x64x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %true = arith.constant true
    %0 = memref.get_global @__constant_128x64x3x3xf32 : memref<128x64x3x3xf32>
    %2 = memref.alloc() : memref<1x64x66x66xf32>
    %c32 = arith.constant 32 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c64 = arith.constant 64 : index
    %c0 = arith.constant 0 : index
    scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c64, %c128) step (%c16, %c32) {
      scf.for %arg4 = %c0 to %c64 step %c16 {
        scf.for %arg5 = %c0 to %c64 step %c16 {
          %4 = memref.subview %2[0, %arg5, %arg4, %arg2] [1, 16, 18, 18] [1, 1, 1, 1] : memref<1x64x66x66xf32> to memref<1x16x18x18xf32, #map1>
          %5 = memref.subview %0[%arg3, %arg5, 0, 0] [32, 16, 3, 3] [1, 1, 1, 1] : memref<128x64x3x3xf32> to memref<32x16x3x3xf32, #map2>
          %6 = memref.subview %arg1[0, %arg3, %arg4, %arg2] [1, 32, 16, 16] [1, 1, 1, 1] : memref<1x128x64x64xf32> to memref<1x32x16x16xf32, #map3>
          %7 = memref.alloc() : memref<20736xi8>
          %8 = memref.view %7[%c0][] : memref<20736xi8> to memref<1x16x18x18xf32>
          %9 = memref.alloc() : memref<18432xi8>
          %10 = memref.view %9[%c0][] : memref<18432xi8> to memref<32x16x3x3xf32>
          %11 = memref.alloc() : memref<32768xi8>
          %12 = memref.view %11[%c0][] : memref<32768xi8> to memref<1x32x16x16xf32>
          linalg.copy ins(%4 : memref<1x16x18x18xf32, #map1>) outs(%8 : memref<1x16x18x18xf32>)
          linalg.copy ins(%5 : memref<32x16x3x3xf32, #map2>) outs(%10 : memref<32x16x3x3xf32>)
          linalg.copy ins(%6 : memref<1x32x16x16xf32, #map3>) outs(%12 : memref<1x32x16x16xf32>)
          linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%8, %10 : memref<1x16x18x18xf32>, memref<32x16x3x3xf32>) outs(%12 : memref<1x32x16x16xf32>)
          linalg.copy ins(%12 : memref<1x32x16x16xf32>) outs(%6 : memref<1x32x16x16xf32, #map3>)
          memref.dealloc %7 : memref<20736xi8>
          memref.dealloc %9 : memref<18432xi8>
          memref.dealloc %11 : memref<32768xi8>
        }
      }
      scf.yield
    }
    return
  }
}

