//===- air_linalg_to_func.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===--------------------------------------------------------------===//

// RUN: air-opt %s -air-linalg-to-func | FileCheck %s


//-----
//
// Testing the regular functionality of the pass:
// linalg ops being replaced with corresponding library calls.
//
module {
  // CHECK: func.func private @linalg_fill_f32_view64x64xf32as2(f32, memref<64x64xf32, 2>) 
  // CHECK: func.func private @linalg_matmul_view64x64xbf16as2_view64x64xbf16as2_view64x64xf32as2(memref<64x64xbf16, 2>, memref<64x64xbf16, 2>, memref<64x64xf32, 2>) 
  func.func @test1(%arg0: memref<512x512xf32>) -> memref<512x512xf32> {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f32
    scf.forall (%arg1, %arg2) in (4, 4) {
      scf.forall (%arg3, %arg4) in (2, 2) {
        %alloc_0 = memref.alloc() : memref<64x64xf32, 2>
        // CHECK: linalg_fill_f32_view64x64xf32as2
        linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<64x64xf32, 2>)
        scf.for %arg7 = %c0 to %c32 step %c16 {
          %alloc_1 = memref.alloc() : memref<64x64xbf16, 2>
          %alloc_2 = memref.alloc() : memref<64x64xbf16, 2>
          // CHECK: linalg_matmul_view64x64xbf16as2_view64x64xbf16as2_view64x64xf32as2
          linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%alloc_1, %alloc_2 : memref<64x64xbf16, 2>, memref<64x64xbf16, 2>) outs(%alloc_0 : memref<64x64xf32, 2>)
          memref.dealloc %alloc_1 : memref<64x64xbf16, 2>
          memref.dealloc %alloc_2 : memref<64x64xbf16, 2>
        }
        memref.dealloc %alloc_0 : memref<64x64xf32, 2>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    }
    return %arg0 : memref<512x512xf32>
  }
}


//-----
//
// Testing change in signature of library call 
// as a result of a shape-altering operation.
//
module {
  // CHECK: func.func private @linalg_fill_f32_view64x64xf32as2(f32, memref<64x64xf32, 2>) 
  // CHECK: func.func private @linalg_matmul_view64x64xbf16as2_view64x64xbf16as2_view64x64xf32as2(memref<64x64xbf16, 2>, memref<16x4x64xbf16, 2>, memref<64x64xf32, 2>) 
  func.func @test2(%arg0: memref<512x512xf32>) -> memref<512x512xf32> {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %cst = arith.constant 0.000000e+00 : f32
    scf.forall (%arg1, %arg2) in (4, 4) {
      scf.forall (%arg3, %arg4) in (2, 2) {
        %alloc_0 = memref.alloc() : memref<64x64xf32, 2>
        // CHECK: linalg_fill_f32_view64x64xf32as2
        linalg.fill ins(%cst : f32) outs(%alloc_0 : memref<64x64xf32, 2>)
        scf.for %arg7 = %c0 to %c32 step %c16 {
          %alloc_1 = memref.alloc() : memref<64x64xbf16, 2>
          %alloc_2 = memref.alloc() : memref<16x4x64xbf16, 2>
          %collapse_shape = memref.collapse_shape %alloc_2 [[0, 1], [2]] : memref<16x4x64xbf16, 2> into memref<64x64xbf16, 2>
          // CHECK: linalg_matmul_view64x64xbf16as2_view64x64xbf16as2_view64x64xf32as2(%alloc_0, %alloc_1, %alloc) : (memref<64x64xbf16, 2>, memref<16x4x64xbf16, 2>, memref<64x64xf32, 2>) -> ()
          linalg.matmul {cast = #linalg.type_fn<cast_signed>} ins(%alloc_1, %collapse_shape : memref<64x64xbf16, 2>, memref<64x64xbf16, 2>) outs(%alloc_0 : memref<64x64xf32, 2>)
          memref.dealloc %alloc_1 : memref<64x64xbf16, 2>
          memref.dealloc %alloc_2 : memref<16x4x64xbf16, 2>
        }
        memref.dealloc %alloc_0 : memref<64x64xf32, 2>
      } {mapping = [#gpu.thread<y>, #gpu.thread<x>]}
    }
    return %arg0 : memref<512x512xf32>
  }
}
