//===- share_fold.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -transform-interpreter -fold-memref-alias-ops -canonicalize %s | FileCheck %s

// Vector transfers through the reshapes of a shared buffer fold into the kept
// buffer, its row and column split out of the replaced buffer's indices.
// CHECK-LABEL: @fold
// CHECK: %[[A:.*]] = memref.alloc() : memref<8x64x8x8xbf16, 2>
// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.expand_shape
// CHECK-NOT: memref.collapse_shape
// CHECK: vector.transfer_write %{{.*}}, %[[A]][%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] {{.*}} : vector<8x8xbf16>, memref<8x64x8x8xbf16, 2>
// CHECK: vector.transfer_read %[[A]][%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}], {{.*}} : memref<8x64x8x8xbf16, 2>, vector<8x8xbf16>
func.func @fold(%i: index, %j: index, %v: vector<8x8xbf16>) -> vector<8x8xbf16> {
  %c0 = arith.constant 0 : index
  %pad = arith.constant 0.0 : bf16
  %a = memref.alloc() : memref<8x64x8x8xbf16, 2>
  vector.transfer_write %v, %a[%i, %j, %c0, %c0] {in_bounds = [true, true]} : vector<8x8xbf16>, memref<8x64x8x8xbf16, 2>
  %b = memref.alloc() : memref<16x32x8x8xbf16, 2>
  %r = vector.transfer_read %b[%i, %j, %c0, %c0], %pad {in_bounds = [true, true]} : memref<16x32x8x8xbf16, 2>, vector<8x8xbf16>
  return %r : vector<8x8xbf16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %allocs = transform.structured.match ops{["memref.alloc"]} in %arg1 : (!transform.any_op) -> !transform.any_op
    %one = transform.air.share_alloc %allocs : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
