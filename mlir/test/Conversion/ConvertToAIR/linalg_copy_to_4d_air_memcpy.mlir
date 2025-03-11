//===- linalg_copy_to_4d_air_memcpy.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-copy-to-dma  -cse -canonicalize | FileCheck %s

//CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%c0{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] [%c1{{.*}}, %c16{{.*}}, %c18{{.*}}, %c18{{.*}}] [%c278784{{.*}}, %c4356{{.*}}, %c66{{.*}}, %c1{{.*}}])
//CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%{{.*}}, %{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c32{{.*}}, %c16{{.*}}, %c3{{.*}}, %c3{{.*}}] [%c576{{.*}}, %c9{{.*}}, %c3{{.*}}, %c1{{.*}}])
//CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%c0{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] [%c1{{.*}}, %c32{{.*}}, %c16{{.*}}, %c16{{.*}}] [%c524288{{.*}}, %c4096{{.*}}, %c64{{.*}}, %c1{{.*}}])
//CHECK: air.dma_memcpy_nd (%{{.*}}[%c0{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] [%c1{{.*}}, %c32{{.*}}, %c16{{.*}}, %c16{{.*}}] [%c524288{{.*}}, %c4096{{.*}}, %c64{{.*}}, %c1{{.*}}], %{{.*}}[] [] [])

#map0 = affine_map<(d0, d1, d2, d3) -> (d0 * 278784 + d1 * 4356 + d2 * 66 + d3 + 67)>
#map1 = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 278784 + s0 + d1 * 4356 + d2 * 66 + d3)>
#map2 = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 576 + s0 + d1 * 9 + d2 * 3 + d3)>
#map3 = affine_map<(d0, d1, d2, d3)[s0] -> (d0 * 524288 + s0 + d1 * 4096 + d2 * 64 + d3)>
module attributes {torch.debug_module_name = "Conv2D"}  {
  memref.global "private" constant @__constant_128xf32 : memref<128xf32> = dense<1.000000e+00>
  memref.global "private" constant @__constant_128x64x3x3xf32 : memref<128x64x3x3xf32> = dense<1.000000e+00>
  func.func @forward(%arg0: memref<1x64x64x64xf32>, %arg1: memref<1x128x64x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    %true = arith.constant true
    %0 = memref.get_global @__constant_128x64x3x3xf32 : memref<128x64x3x3xf32>
    // assert %true, "expect groups to be 1"
    %1 = memref.alloc() : memref<1x64x66x66xf32>
    linalg.fill ins(%cst : f32) outs(%1 : memref<1x64x66x66xf32>)
    %2 = memref.alloc() : memref<1x64x66x66xf32>
    linalg.copy ins(%1 : memref<1x64x66x66xf32>) outs(%2 : memref<1x64x66x66xf32> )
    %3 = memref.subview %2[0, 0, 1, 1] [1, 64, 64, 64] [1, 1, 1, 1] : memref<1x64x66x66xf32> to memref<1x64x64x64xf32, #map0>
    linalg.copy ins(%arg0 : memref<1x64x64x64xf32>) outs(%3 : memref<1x64x64x64xf32, #map0> )
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
          %7 = memref.alloc() : memref<1x16x18x18xf32, 2>
          %8 = memref.alloc() : memref<32x16x3x3xf32, 2>
          %9 = memref.alloc() : memref<1x32x16x16xf32, 2>
          linalg.copy ins(%4 : memref<1x16x18x18xf32, #map1>) outs(%7 : memref<1x16x18x18xf32, 2> )
          linalg.copy ins(%5 : memref<32x16x3x3xf32, #map2>) outs(%8 : memref<32x16x3x3xf32, 2> )
          linalg.copy ins(%6 : memref<1x32x16x16xf32, #map3>) outs(%9 : memref<1x32x16x16xf32, 2> )
          linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%7, %8 : memref<1x16x18x18xf32, 2>, memref<32x16x3x3xf32, 2>) outs(%9 : memref<1x32x16x16xf32, 2>)
          linalg.copy ins(%9 : memref<1x32x16x16xf32, 2>) outs(%6 : memref<1x32x16x16xf32, #map3> )
          memref.dealloc %7 : memref<1x16x18x18xf32, 2>
          memref.dealloc %8 : memref<32x16x3x3xf32, 2>
          memref.dealloc %9 : memref<1x32x16x16xf32, 2>
        }
      }
      scf.reduce
    }
    return
  }
}


