//===- air_linalg_kernelPartialResultAccum.mlir ----------------*- MLIR -*-===//
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
// CHECK: scf.for %arg6 = %c0 to %c64 step %c16 {
// CHECK:     %6 = memref.subview %arg1[0, %4, %arg6, %5] [1, 32, 16, 16] [1, 1, 1, 1] : memref<1x128x64x64xf32> to memref<1x32x16x16xf32, #map1>
// CHECK:     %7 = memref.alloc() : memref<1x32x16x16xf32, 2>
// CHECK:     linalg.copy ins(%6 : memref<1x32x16x16xf32, #map1>) outs(%7 : memref<1x32x16x16xf32, 2>)
// CHECK:     scf.for %arg7 = %c0 to %c64 step %c16 {
// CHECK:         %8 = memref.subview %2[0, %arg7, %arg6, %5] [1, 16, 18, 18] [1, 1, 1, 1] : memref<1x64x66x66xf32> to memref<1x16x18x18xf32, #map2>
// CHECK:         %9 = memref.subview %0[%4, %arg7, 0, 0] [32, 16, 3, 3] [1, 1, 1, 1] : memref<128x64x3x3xf32> to memref<32x16x3x3xf32, #map3>
// CHECK:         %10 = memref.alloc() : memref<1x16x18x18xf32, 2>
// CHECK:         %11 = memref.alloc() : memref<32x16x3x3xf32, 2>
// CHECK:         linalg.copy ins(%8 : memref<1x16x18x18xf32, #map2>) outs(%10 : memref<1x16x18x18xf32, 2>)
// CHECK:         linalg.copy ins(%9 : memref<32x16x3x3xf32, #map3>) outs(%11 : memref<32x16x3x3xf32, 2>)
// CHECK:         linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%10, %11 : memref<1x16x18x18xf32, 2>, memref<32x16x3x3xf32, 2>) outs(%7 : memref<1x32x16x16xf32, 2>)
// CHECK:         memref.dealloc %10 : memref<1x16x18x18xf32, 2>
// CHECK:         memref.dealloc %11 : memref<32x16x3x3xf32, 2>
// CHECK:     }
// CHECK:     linalg.copy ins(%7 : memref<1x32x16x16xf32, 2>) outs(%6 : memref<1x32x16x16xf32, #map1>)
// CHECK:     memref.dealloc %7 : memref<1x32x16x16xf32, 2>
// CHECK: }

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 278784 + d1 * 4356 + d2 * 66 + d3 + 67)>
#map1 = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 524288 + s0 + d1 * 4096 + d2 * 64 + d3)>
#map2 = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 278784 + s0 + d1 * 4356 + d2 * 66 + d3)>
#map3 = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 576 + s0 + d1 * 9 + d2 * 3 + d3)>
module attributes {torch.debug_module_name = "Conv2D"} {
  memref.global "private" constant @__constant_128xf32 : memref<128xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_128x64x3x3xf32 : memref<128x64x3x3xf32> = dense<1.000000e+00>
  func.func @forward(%arg0: memref<1x64x64x64xf32>, %arg1: memref<1x128x64x64xf32>) {
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c128 = arith.constant 128 : index
    %c16 = arith.constant 16 : index
    %cst = arith.constant 0.000000e+00 : f32
    %0 = memref.get_global @__constant_128x64x3x3xf32 : memref<128x64x3x3xf32>
    %1 = memref.alloc() : memref<1x64x66x66xf32>
    linalg.fill ins(%cst : f32) outs(%1 : memref<1x64x66x66xf32>)
    %2 = memref.alloc() : memref<1x64x66x66xf32>
    linalg.copy ins(%1 : memref<1x64x66x66xf32>) outs(%2 : memref<1x64x66x66xf32>)
    %3 = memref.subview %2[0, 0, 1, 1] [1, 64, 64, 64] [1, 1, 1, 1] : memref<1x64x66x66xf32> to memref<1x64x64x64xf32, #map0>
    linalg.copy ins(%arg0 : memref<1x64x64x64xf32>) outs(%3 : memref<1x64x64x64xf32, #map0>)
    scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c32, %c64) step (%c16, %c32) {
      scf.for %arg4 = %c0 to %c128 step %c64 {
        %4 = arith.addi %arg4, %arg3 : index
        scf.for %arg5 = %c0 to %c64 step %c32 {
          %5 = arith.addi %arg5, %arg2 : index
          scf.for %arg6 = %c0 to %c64 step %c16 {
            %6 = memref.subview %arg1[0, %4, %arg6, %5] [1, 32, 16, 16] [1, 1, 1, 1] : memref<1x128x64x64xf32> to memref<1x32x16x16xf32, #map1>
            scf.for %arg7 = %c0 to %c64 step %c16 {
              %7 = memref.subview %2[0, %arg7, %arg6, %5] [1, 16, 18, 18] [1, 1, 1, 1] : memref<1x64x66x66xf32> to memref<1x16x18x18xf32, #map2>
              %8 = memref.subview %0[%4, %arg7, 0, 0] [32, 16, 3, 3] [1, 1, 1, 1] : memref<128x64x3x3xf32> to memref<32x16x3x3xf32, #map3>
              %9 = memref.alloc() : memref<1x16x18x18xf32, 2>
              %10 = memref.alloc() : memref<32x16x3x3xf32, 2>
              %11 = memref.alloc() : memref<1x32x16x16xf32, 2>
              linalg.copy ins(%7 : memref<1x16x18x18xf32, #map2>) outs(%9 : memref<1x16x18x18xf32, 2>)
              linalg.copy ins(%8 : memref<32x16x3x3xf32, #map3>) outs(%10 : memref<32x16x3x3xf32, 2>)
              linalg.copy ins(%6 : memref<1x32x16x16xf32, #map1>) outs(%11 : memref<1x32x16x16xf32, 2>)
              linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%9, %10 : memref<1x16x18x18xf32, 2>, memref<32x16x3x3xf32, 2>) outs(%11 : memref<1x32x16x16xf32, 2>)
              linalg.copy ins(%11 : memref<1x32x16x16xf32, 2>) outs(%6 : memref<1x32x16x16xf32, #map1>)
              memref.dealloc %9 : memref<1x16x18x18xf32, 2>
              memref.dealloc %10 : memref<32x16x3x3xf32, 2>
              memref.dealloc %11 : memref<1x32x16x16xf32, 2>
            }
          }
        }
      }
      scf.yield
    }
    return
  }
}