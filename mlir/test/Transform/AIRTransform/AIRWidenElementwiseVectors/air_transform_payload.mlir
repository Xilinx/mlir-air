//===- air_transform_payload.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/air_transform.mlir' %s | FileCheck %s

// An 8-lane chain of elementwise ops on 16 lanes: the inputs repeated by a
// shuffle, a splat constant made wider, one extraction of the low lanes at
// the end.
// CHECK-LABEL: @widen
// CHECK: %[[C:.*]] = arith.constant dense<{{.*}}> : vector<16xf32>
// CHECK: %[[A:.*]] = vector.shuffle %arg0, %{{.*}} [0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7]
// CHECK: %[[E:.*]] = arith.extf %[[A]] : vector<16xbf16> to vector<16xf32>
// CHECK: %[[M:.*]] = arith.mulf %[[E]], %[[C]] : vector<16xf32>
// CHECK: %[[X:.*]] = math.exp2 %[[M]] : vector<16xf32>
// CHECK: %[[R:.*]] = vector.extract_strided_slice %[[X]] offsets = [0], sizes = [8], strides = [1] : vector<16xf32> to vector<8xf32>
// CHECK: return %[[R]]
func.func @widen(%a: vector<8xbf16>) -> vector<8xf32> {
  %c = arith.constant dense<0.5> : vector<8xf32>
  %e = arith.extf %a : vector<8xbf16> to vector<8xf32>
  %m = arith.mulf %e, %c : vector<8xf32>
  %x = math.exp2 %m : vector<8xf32>
  return %x : vector<8xf32>
}

// Lane counts already at the width, or not dividing it, are left.
// CHECK-LABEL: @keep
// CHECK: arith.addf %arg0, %arg0 : vector<16xf32>
// CHECK: arith.addf %arg1, %arg1 : vector<6xf32>
func.func @keep(%a: vector<16xf32>, %b: vector<6xf32>) -> (vector<16xf32>, vector<6xf32>) {
  %x = arith.addf %a, %a : vector<16xf32>
  %y = arith.addf %b, %b : vector<6xf32>
  return %x, %y : vector<16xf32>, vector<6xf32>
}
