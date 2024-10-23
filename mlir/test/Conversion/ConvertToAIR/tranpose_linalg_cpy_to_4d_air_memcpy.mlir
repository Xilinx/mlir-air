//===- tranpose_linalg_copy_to_4d_air_memcpy.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===-------------------------------------------------------------------------------===//

// RUN: air-opt %s -air-copy-to-dma -canonicalize | FileCheck %s

// CHECK: func.func @test(%[[ARG0:.*]]
// CHECK: scf.for %[[ARG1:.*]] = %c0 to %c128 step %c32 {
// CHECK: air.dma_memcpy_nd (%alloc[%[[ARG1:.*]], %c0{{.*}}, %c0{{.*}}, %c0{{.*}}] [%c32{{.*}}, %c8{{.*}}, %c8{{.*}}, %c16] [%c1024, %c128{{.*}}, %c16{{.*}}, %c1], %[[ARG0:.*]][%[[ARG1:.*]], %c0{{.*}}, %c0{{.*}}, %0] [%c32{{.*}}, %c8{{.*}}, %c8{{.*}}, %c16{{.*}}] [%c4096, %c64, %c512, %c1{{.*}}]) 
#map = affine_map<(d0) -> (d0 * 16)>
module {
  func.func @test(%arg1: memref<128x8x8x64xbf16>) -> memref<128x8x8x64xbf16> {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.forall (%arg3, %arg4) in (4, 4) {
      %alloc_0 = memref.alloc() : memref<128x8x8x16xbf16, 1>
      scf.for %arg5 = %c0 to %c128 step %c32 {
        %2 = affine.apply #map(%arg4)
        %subview_3 = memref.subview %arg1[%arg5, 0, 0, %2] [32, 8, 8, 16] [1, 1, 1, 1] : memref<128x8x8x64xbf16> to memref<32x8x8x16xbf16, strided<[4096, 512, 64, 1], offset: ?>>
        %transpose = memref.transpose %subview_3 (d0, d1, d2, d3) -> (d0, d2, d1, d3) : memref<32x8x8x16xbf16, strided<[4096, 512, 64, 1], offset: ?>> to memref<32x8x8x16xbf16, strided<[4096, 64, 512, 1], offset: ?>>
        %subview_4 = memref.subview %alloc_0[%arg5, 0, 0, 0] [32, 8, 8, 16] [1, 1, 1, 1] : memref<128x8x8x16xbf16, 1> to memref<32x8x8x16xbf16, strided<[1024, 128, 16, 1], offset: ?>, 1>
        linalg.copy ins(%transpose : memref<32x8x8x16xbf16, strided<[4096, 64, 512, 1], offset: ?>>) outs(%subview_4 : memref<32x8x8x16xbf16, strided<[1024, 128, 16, 1], offset: ?>, 1>)
      }
      scf.forall (%arg5, %arg6) in (2, 2) {
        scf.for %arg7 = %c0 to %c128 step %c8 {
          %alloc_8 = memref.alloc() : memref<8x8x4x16xbf16, 2>
          %subview_6 = memref.subview %alloc_0[%arg7, 0, %arg6, 0] [8, 8, 4, 16] [1, 1, 1, 1] : memref<128x8x8x16xbf16, 1> to memref<8x8x4x16xbf16, strided<[1024, 128, 16, 1], offset: ?>, 1>
          memref.copy %subview_6, %alloc_8 : memref<8x8x4x16xbf16, strided<[1024, 128, 16, 1], offset: ?>, 1> to memref<8x8x4x16xbf16, 2>
          memref.dealloc %alloc_8 : memref<8x8x4x16xbf16, 2>
        }
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
      memref.dealloc %alloc_0 : memref<128x8x8x16xbf16, 1>
    }
    return %arg1 : memref<128x8x8x64xbf16>
  }
}
