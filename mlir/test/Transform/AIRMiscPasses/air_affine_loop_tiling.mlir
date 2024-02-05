//===- air_affine_loop_tiling.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-affine-loop-tile="tile-sizes=4,4" %s | FileCheck %s
// CHECK: [[$MAP0:#map[0-9]*]] = affine_map<(d0) -> (d0)>
// CHECK: [[$MAP1:#map[0-9]+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-LABEL: @func0
// CHECK: affine.for %[[VAL0:.*]] = 0 to 32 step 4 {
// CHECK: affine.for %[[VAL1:.*]] = 0 to 32 step 4 {
// CHECK: affine.for %[[VAL2:.*]] = [[$MAP0]](%[[VAL0]]) to [[$MAP1]](%[[VAL0]]) {
// CHECK: affine.for %[[VAL3:.*]] = [[$MAP0]](%[[VAL1]]) to [[$MAP1]](%[[VAL1]]) {
func.func @func0(){
  affine.for %arg0 = 0 to 32 {
    affine.for %arg1 = 0 to 32 {
      %c0 = arith.constant 0 : index
    }
  }
  return
}
