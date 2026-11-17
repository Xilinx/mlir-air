//===- gpu_kernel_outline.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-gpu-outlining | FileCheck %s

// Each gpu.launch becomes a gpu.func in its own gpu.module. Constants used in
// the body are rebuilt in the kernel; other values become kernel arguments.

// CHECK-LABEL: func.func @two_launches
// CHECK-SAME: (%[[A:.*]]: memref<256xf32>, %[[N:.*]]: index, %[[S:.*]]: f32)
// CHECK: gpu.launch_func @two_launches_module::@two_launches_module {{.*}} args(%[[A]] : memref<256xf32>, %[[S]] : f32)
// CHECK: gpu.launch_func @two_launches_module_0::@two_launches_module blocks in (%[[N]], {{.*}} args(%[[A]] : memref<256xf32>)

// CHECK: gpu.module @two_launches_module {
// CHECK: gpu.func @two_launches_module(%{{.*}}: memref<256xf32>, %{{.*}}: f32) kernel
// CHECK-SAME: known_block_size = array<i32: 64, 1, 1>, known_grid_size = array<i32: 4, 1, 1>
// CHECK-DAG: arith.constant 64 : index
// CHECK-DAG: arith.constant 2.000000e+00 : f32
// CHECK: gpu.return

// CHECK: gpu.module @two_launches_module_0 {
// CHECK: gpu.func @two_launches_module(%{{.*}}: memref<256xf32>) kernel attributes {known_block_size = array<i32: 64, 1, 1>} {
// CHECK: gpu.return
module {
  func.func @two_launches(%a: memref<256xf32>, %n: index, %s: f32) {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c64 = arith.constant 64 : index
    %f2 = arith.constant 2.0 : f32
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %c4, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c64, %sy = %c1, %sz = %c1) {
      %i0 = arith.muli %bx, %c64 : index
      %i = arith.addi %i0, %tx : index
      %v = memref.load %a[%i] : memref<256xf32>
      %w = arith.mulf %v, %f2 : f32
      %x = arith.addf %w, %s : f32
      memref.store %x, %a[%i] : memref<256xf32>
      gpu.terminator
    }
    gpu.launch blocks(%bx, %by, %bz) in (%gx = %n, %gy = %c1, %gz = %c1)
               threads(%tx, %ty, %tz) in (%sx = %c64, %sy = %c1, %sz = %c1) {
      %v = memref.load %a[%tx] : memref<256xf32>
      memref.store %v, %a[%bx] : memref<256xf32>
      gpu.terminator
    }
    return
  }
}
