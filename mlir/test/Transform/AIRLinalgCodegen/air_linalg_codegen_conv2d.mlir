//===- air_linalg_codegen_conv2d.mlir --------------------------*- MLIR -*-===//
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

// RUN: air-opt %s -air-linalg-codegen='herd-size=4,4' | FileCheck %s
// XFAIL: *
// CHECK:    scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c64, %c128) step (%c16, %c32) {
// CHECK:      scf.for %arg4 = %c0 to %c64 step %c16 {
// CHECK:        scf.for %arg5 = %c0 to %c64 step %c16 {
// CHECK:          %4 = memref.subview %2[0, %arg5, %arg4, %arg2] [1, 16, 18, 18] [1, 1, 1, 1] : memref<1x64x66x66xf32> to memref<1x16x18x18xf32, #map1>
// CHECK:          %5 = memref.subview %0[%arg3, %arg5, 0, 0] [32, 16, 3, 3] [1, 1, 1, 1] : memref<128x64x3x3xf32> to memref<32x16x3x3xf32, #map2>
// CHECK:          %6 = memref.subview %arg1[0, %arg3, %arg4, %arg2] [1, 32, 16, 16] [1, 1, 1, 1] : memref<1x128x64x64xf32> to memref<1x32x16x16xf32, #map3>
// CHECK:          %7 = memref.alloc() : memref<1x16x18x18xf32, 2>
// CHECK:          %8 = memref.alloc() : memref<32x16x3x3xf32, 2>
// CHECK:          %9 = memref.alloc() : memref<1x32x16x16xf32, 2>
// CHECK:          linalg.copy(%4, %7) : memref<1x16x18x18xf32, #map1>, memref<1x16x18x18xf32, 2> 
// CHECK:          linalg.copy(%5, %8) : memref<32x16x3x3xf32, #map2>, memref<32x16x3x3xf32, 2> 
// CHECK:          linalg.copy(%6, %9) : memref<1x32x16x16xf32, #map3>, memref<1x32x16x16xf32, 2> 
// CHECK:          linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%7, %8 : memref<1x16x18x18xf32, 2>, memref<32x16x3x3xf32, 2>) outs(%9 : memref<1x32x16x16xf32, 2>)
// CHECK:          linalg.copy(%9, %6) : memref<1x32x16x16xf32, 2>, memref<1x32x16x16xf32, #map3> 
// CHECK:          memref.dealloc %7 : memref<1x16x18x18xf32, 2>
// CHECK:          memref.dealloc %8 : memref<32x16x3x3xf32, 2>
// CHECK:          memref.dealloc %9 : memref<1x32x16x16xf32, 2>

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 278784 + d1 * 4356 + d2 * 66 + d3 + 67)>
module attributes {torch.debug_module_name = "Conv2D"}  {
  memref.global "private" constant @__constant_128xf32 : memref<128xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_128x64x3x3xf32 : memref<128x64x3x3xf32> = dense<1.000000e+00>
  func.func @forward(%arg0: memref<1x64x64x64xf32>, %arg1: memref<1x128x64x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_128x64x3x3xf32 : memref<128x64x3x3xf32>
    %1 = memref.get_global @__constant_128xf32 : memref<128xf32>
    %true = arith.constant true
    assert %true, "expect groups to be 1"
    %2 = memref.alloc() : memref<1x64x66x66xf32>
    linalg.fill(%cst, %2) : f32, memref<1x64x66x66xf32> 
    %3 = memref.alloc() : memref<1x64x66x66xf32>
    linalg.copy(%2, %3) : memref<1x64x66x66xf32>, memref<1x64x66x66xf32> 
    %4 = memref.subview %3[0, 0, 1, 1] [1, 64, 64, 64] [1, 1, 1, 1] : memref<1x64x66x66xf32> to memref<1x64x64x64xf32, #map0>
    linalg.copy(%arg0, %4) : memref<1x64x64x64xf32>, memref<1x64x64x64xf32, #map0> 
    linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%3, %0 : memref<1x64x66x66xf32>, memref<128x64x3x3xf32>) outs(%arg1 : memref<1x128x64x64xf32>)
    return
  }
}

