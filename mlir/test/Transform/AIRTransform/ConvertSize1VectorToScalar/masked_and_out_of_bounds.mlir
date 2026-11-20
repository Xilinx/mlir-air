//===- masked_and_out_of_bounds.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -transform-interpreter | FileCheck %s

// Size-1 transfers become a scalar load or store only when every element is
// accessed: a masked-off or out-of-bounds element reads as the padding value
// and is not written, which a load and a store would not respect.

// CHECK-LABEL: func.func @in_bounds
// CHECK-NOT: vector.transfer_read
// CHECK: %[[A:.*]] = memref.load %{{.*}}[] : memref<f32>
// CHECK: %[[B:.*]] = memref.load %{{.*}}[%{{.*}}] : memref<8xf32>
// CHECK: memref.store %{{.*}}, %{{.*}}[] : memref<f32>
// CHECK: memref.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<8xf32>
// CHECK-NOT: vector.transfer_write
func.func @in_bounds(%s: memref<f32>, %m: memref<8xf32>, %i: index) {
  %p = arith.constant 0.0 : f32
  %a = vector.transfer_read %s[], %p : memref<f32>, vector<f32>
  %b = vector.transfer_read %m[%i], %p {in_bounds = [true]} : memref<8xf32>, vector<1xf32>
  vector.transfer_write %a, %s[] : vector<f32>, memref<f32>
  vector.transfer_write %b, %m[%i] {in_bounds = [true]} : vector<1xf32>, memref<8xf32>
  return
}

// CHECK-LABEL: func.func @masked
// CHECK-NOT: memref.load
// CHECK: vector.transfer_read %{{.*}}, %{{.*}}, %{{.*}} {in_bounds = [true]}
// CHECK: vector.transfer_write %{{.*}}, %{{.*}}, %{{.*}} {in_bounds = [true]}
// CHECK-NOT: memref.store
func.func @masked(%m: memref<8xf32>, %i: index, %k: vector<1xi1>) {
  %p = arith.constant 0.0 : f32
  %b = vector.transfer_read %m[%i], %p, %k {in_bounds = [true]} : memref<8xf32>, vector<1xf32>
  vector.transfer_write %b, %m[%i], %k {in_bounds = [true]} : vector<1xf32>, memref<8xf32>
  return
}

// CHECK-LABEL: func.func @out_of_bounds
// CHECK-NOT: memref.load
// CHECK: vector.transfer_read %{{.*}}[%{{.*}}], %{{.*}} : memref<8xf32>, vector<1xf32>
// CHECK: vector.transfer_write %{{.*}}, %{{.*}}[%{{.*}}] : vector<1xf32>, memref<8xf32>
// CHECK-NOT: memref.store
func.func @out_of_bounds(%m: memref<8xf32>, %i: index) {
  %p = arith.constant 0.0 : f32
  %b = vector.transfer_read %m[%i], %p : memref<8xf32>, vector<1xf32>
  vector.transfer_write %b, %m[%i] : vector<1xf32>, memref<8xf32>
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %r = transform.air.convert_size1_vector_to_scalar %f : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
