//===- air_linalg_rm_subview.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2021-2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-linalg-codegen=test-patterns | FileCheck %s

// CHECK-LABEL:   func.func @myFunc(
// CHECK-SAME:                      %[[VAL_0:.*]]: memref<64x64xf32>,
// CHECK-SAME:                      %[[VAL_1:.*]]: memref<64x64xf32>) -> memref<64x64xf32> {
// CHECK:           %[[VAL_2:.*]] = arith.constant 64 : index
// CHECK:           %[[VAL_3:.*]] = arith.constant 16 : index
// CHECK:           %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<64x64xf32>
// CHECK:           scf.parallel (%[[VAL_6:.*]], %[[VAL_7:.*]]) = (%[[VAL_4]], %[[VAL_4]]) to (%[[VAL_2]], %[[VAL_2]]) step (%[[VAL_3]], %[[VAL_3]]) {
// CHECK:             %[[VAL_8:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_6]], 0] [16, 64] [1, 1] : memref<64x64xf32> to memref<16x64xf32, #{{.*}}>
// CHECK:             %[[VAL_9:.*]] = memref.subview %[[VAL_1]][0, %[[VAL_7]]] [64, 16] [1, 1] : memref<64x64xf32> to memref<64x16xf32, #{{.*}}>
// CHECK:             %[[VAL_10:.*]] = memref.subview %[[VAL_5]]{{\[}}%[[VAL_6]], %[[VAL_7]]] [16, 16] [1, 1] : memref<64x64xf32> to memref<16x16xf32, #{{.*}}>
// CHECK:             %[[VAL_11:.*]] = memref.alloc() : memref<16x64xf32, 1>
// CHECK:             %[[VAL_12:.*]] = memref.alloc() : memref<64x16xf32, 1>
// CHECK:             %[[VAL_13:.*]] = memref.alloc() : memref<16x16xf32, 1>
// CHECK:             linalg.copy ins(%[[VAL_8]] : memref<16x64xf32, #{{.*}}>) outs(%[[VAL_11]] : memref<16x64xf32, 1>)
// CHECK:             linalg.copy ins(%[[VAL_9]] : memref<64x16xf32, #{{.*}}>) outs(%[[VAL_12]] : memref<64x16xf32, 1>)
// CHECK:             linalg.copy ins(%[[VAL_10]] : memref<16x16xf32, #{{.*}}>) outs(%[[VAL_13]] : memref<16x16xf32, 1>)
// CHECK:             linalg.matmul ins(%[[VAL_11]], %[[VAL_12]] : memref<16x64xf32, 1>, memref<64x16xf32, 1>) outs(%[[VAL_13]] : memref<16x16xf32, 1>)
// CHECK:             linalg.copy ins(%[[VAL_13]] : memref<16x16xf32, 1>) outs(%[[VAL_10]] : memref<16x16xf32, #{{.*}}>)
// CHECK:             memref.dealloc %[[VAL_11]] : memref<16x64xf32, 1>
// CHECK:             memref.dealloc %[[VAL_12]] : memref<64x16xf32, 1>
// CHECK:             memref.dealloc %[[VAL_13]] : memref<16x16xf32, 1>
#map0 = affine_map<(d0, d1)[s0] -> (d0 * 64 + s0 + d1)>
#map1 = affine_map<(d0, d1) -> (d0 * 64 + d1)>
#map2 = affine_map<(d0, d1) -> (d0 * 16 + d1)>
module  {
  func.func @myFunc(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>) -> memref<64x64xf32> {
    %c4096 = arith.constant 4096 : index
    %c64 = arith.constant 64 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1024 = arith.constant 1024 : index
    %0 = memref.alloc() : memref<64x64xf32>
    scf.parallel (%arg2, %arg3) = (%c0, %c0) to (%c64, %c64) step (%c16, %c16) {
      %1 = memref.subview %arg0[%arg2, 0] [16, 64] [1, 1] : memref<64x64xf32> to memref<16x64xf32, #map0>
      %2 = memref.subview %arg1[0, %arg3] [64, 16] [1, 1] : memref<64x64xf32> to memref<64x16xf32, #map0>
      %3 = memref.subview %0[%arg2, %arg3] [16, 16] [1, 1] : memref<64x64xf32> to memref<16x16xf32, #map0>
      %4 = memref.alloc(%c4096) : memref<?xi8>
      %5 = memref.view %4[%c0][] : memref<?xi8> to memref<16x64xf32>
      %6 = memref.subview %5[0, 0] [16, 64] [1, 1] : memref<16x64xf32> to memref<16x64xf32, #map1>
      %7 = memref.alloc(%c4096) : memref<?xi8>
      %8 = memref.view %7[%c0][] : memref<?xi8> to memref<64x16xf32>
      %9 = memref.subview %8[0, 0] [64, 16] [1, 1] : memref<64x16xf32> to memref<64x16xf32, #map2>
      %10 = memref.alloc(%c1024) : memref<?xi8>
      %11 = memref.view %10[%c0][] : memref<?xi8> to memref<16x16xf32>
      %12 = memref.subview %11[0, 0] [16, 16] [1, 1] : memref<16x16xf32> to memref<16x16xf32, #map2>
      linalg.copy ins(%1 : memref<16x64xf32, #map0>) outs(%6 : memref<16x64xf32, #map1>)
      linalg.copy ins(%2 : memref<64x16xf32, #map0>) outs(%9 : memref<64x16xf32, #map2>)
      linalg.copy ins(%3 : memref<16x16xf32, #map0>) outs(%12 : memref<16x16xf32, #map2>)
      linalg.matmul ins(%6, %9 : memref<16x64xf32, #map1>, memref<64x16xf32, #map2>) outs(%12 : memref<16x16xf32, #map2>)
      linalg.copy ins(%12 : memref<16x16xf32, #map2>) outs(%3 : memref<16x16xf32, #map0>)
      memref.dealloc %4 : memref<?xi8>
      memref.dealloc %7 : memref<?xi8>
      memref.dealloc %10 : memref<?xi8>
      scf.yield
    }
    return %0 : memref<64x64xf32>
  }
}