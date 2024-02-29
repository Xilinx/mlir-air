//===- air_linalg_to_func.mlir  --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-linalg-to-func | FileCheck %s

// CHECK-LABEL: test_0
// CHECK: call @linalg_fill_i16_view64x64xi16
// CHECK: call @linalg_matmul_view64x256xi16_view256x64xi16_view64x64xi16
func.func @test_0(%arg0: memref<64x256xi16>, %arg1: memref<256x64xi16>, %arg2: memref<64x64xi16>) {
  %cst = arith.constant 0 : i16
  linalg.fill ins(%cst : i16) outs(%arg2 : memref<64x64xi16>)
  linalg.matmul ins(%arg0, %arg1 : memref<64x256xi16>, memref<256x64xi16>) outs(%arg2 : memref<64x64xi16>)
  return
}

// CHECK-LABEL: test_1
// CHECK: call @zero_f32
// CHECK: call @matmul_f32
#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @test_1(%arg0: memref<64x256xf32>, %arg1: memref<256x64xf32>, %arg2: memref<64x64xf32>) {
    %cst = arith.constant 0.000000e+00 : f32
    linalg.generic {library_call = "zero_f32", indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : f32) outs(%arg2 : memref<64x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    linalg.generic {library_call = "matmul_f32", indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : memref<64x256xf32>, memref<256x64xf32>) outs(%arg2 : memref<64x64xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %0 = arith.mulf %in, %in_0 : f32
      %1 = arith.addf %out, %0 : f32
      linalg.yield %1 : f32
    }
    return
  }
}

