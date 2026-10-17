//===- wrap_func_with_parallel_one_point.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-wrap-func-with-parallel='loop-bounds=1,1,1' %s | FileCheck %s
// RUN: air-opt -air-wrap-func-with-parallel='loop-bounds=4,1,1' %s | FileCheck %s --check-prefix=GRID

// The body reads no grid index. On a one-point grid it is still wrapped, once;
// on a larger grid it is left alone.

// CHECK-LABEL: func.func @no_grid_index
// CHECK: scf.parallel
// CHECK: linalg.fill
// CHECK-NOT: scf.parallel
// CHECK: return

// GRID-LABEL: func.func @no_grid_index
// GRID-NOT: scf.parallel
// GRID: linalg.fill
func.func @no_grid_index(%arg0: memref<*xf32>, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
  %cst = arith.constant 1.000000e+00 : f32
  %c0 = arith.constant 0 : index
  %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [%c0], sizes: [16], strides: [1] : memref<*xf32> to memref<16xf32, strided<[1], offset: ?>>
  linalg.fill ins(%cst : f32) outs(%reinterpret_cast : memref<16xf32, strided<[1], offset: ?>>)
  return
}
