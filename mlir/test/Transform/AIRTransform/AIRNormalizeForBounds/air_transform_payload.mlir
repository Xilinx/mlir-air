//===- air_transform_payload.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' -verify-diagnostics %s | FileCheck %s

#map_mul8 = affine_map<(d0) -> (d0 * 8)>
#map_add16 = affine_map<(d0) -> (d0 + 16)>
#map_mul4_add8 = affine_map<(d0) -> (d0 * 4 + 8)>

// Test case 1: Basic affine.apply multiplication on loop induction variable
// The loop goes from 0 to 64 step 8, and applies d0 * 8 to the induction variable
// After normalization, the loop should go from 0 to 512 step 64 (8*8=64, 64*8=512)
// CHECK-LABEL: @basic_multiply
// CHECK: arith.constant 512 : index
// CHECK: arith.constant 64 : index
// CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-NOT: affine.apply
// CHECK: memref.load %{{.*}}[%[[IV]]]
func.func @basic_multiply(%arg0: memref<512xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c8 = arith.constant 8 : index
  scf.for %arg1 = %c0 to %c64 step %c8 {
    %0 = affine.apply #map_mul8(%arg1)
    %val = memref.load %arg0[%0] : memref<512xf32>
  }
  return
}

// Test case 2: Affine.apply with addition on loop induction variable
// The loop goes from 0 to 32 step 4, and applies d0 + 16 to the induction variable
// After normalization, the loop should go from 16 to 48 step 4
// CHECK-LABEL: @basic_add
// CHECK: arith.constant 16 : index
// CHECK: arith.constant 48 : index
// CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-NOT: affine.apply
// CHECK: memref.load %{{.*}}[%[[IV]]]
func.func @basic_add(%arg0: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c4 = arith.constant 4 : index
  scf.for %arg1 = %c0 to %c32 step %c4 {
    %0 = affine.apply #map_add16(%arg1)
    %val = memref.load %arg0[%0] : memref<64xf32>
  }
  return
}

// Test case 3: Multiple affine.apply ops on the same loop induction variable
// Only the first affine.apply can be folded (since it's the only one with single use of IV)
// CHECK-LABEL: @multiple_affine_apply
func.func @multiple_affine_apply(%arg0: memref<512xf32>, %arg1: memref<512xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c8 = arith.constant 8 : index
  // CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
  scf.for %arg2 = %c0 to %c64 step %c8 {
    %0 = affine.apply #map_mul8(%arg2)
    %1 = affine.apply #map_mul8(%arg2)
    %val0 = memref.load %arg0[%0] : memref<512xf32>
    %val1 = memref.load %arg1[%1] : memref<512xf32>
  }
  return
}

// Test case 4: Nested loops with affine.apply on each level
// CHECK-LABEL: @nested_loops
func.func @nested_loops(%arg0: memref<512x512xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c8 = arith.constant 8 : index
  // CHECK: scf.for %[[IV_OUTER:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
  // CHECK: scf.for %[[IV_INNER:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
  scf.for %arg1 = %c0 to %c64 step %c8 {
    scf.for %arg2 = %c0 to %c64 step %c8 {
      %0 = affine.apply #map_mul8(%arg1)
      %1 = affine.apply #map_mul8(%arg2)
      %val = memref.load %arg0[%0, %1] : memref<512x512xf32>
    }
  }
  return
}

// Test case 5: Affine.apply with combined multiplication and addition
// The loop goes from 0 to 32 step 4, and applies d0 * 4 + 8 to the induction variable
// After normalization: lb = 0*4+8=8, ub = 32*4+8=136, step = 4*4=16
// CHECK-LABEL: @mul_and_add
// CHECK: arith.constant 8 : index
// CHECK: arith.constant 136 : index
// CHECK: arith.constant 16 : index
// CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
// CHECK-NOT: affine.apply
// CHECK: memref.load %{{.*}}[%[[IV]]]
func.func @mul_and_add(%arg0: memref<256xf32>) {
  %c0 = arith.constant 0 : index
  %c32 = arith.constant 32 : index
  %c4 = arith.constant 4 : index
  scf.for %arg1 = %c0 to %c32 step %c4 {
    %0 = affine.apply #map_mul4_add8(%arg1)
    %val = memref.load %arg0[%0] : memref<256xf32>
  }
  return
}

// Test case 6: Loop with no affine.apply (should remain unchanged)
// CHECK-LABEL: @no_affine_apply
func.func @no_affine_apply(%arg0: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c8 = arith.constant 8 : index
  // CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}}
  // CHECK: memref.load %{{.*}}[%[[IV]]]
  scf.for %arg1 = %c0 to %c64 step %c8 {
    %val = memref.load %arg0[%arg1] : memref<64xf32>
  }
  return
}
