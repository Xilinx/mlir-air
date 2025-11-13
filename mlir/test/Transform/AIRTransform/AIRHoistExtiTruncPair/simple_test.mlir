//===- simple_test.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-transform='filename=%S/simple_transform.mlir' -verify-diagnostics | FileCheck %s

// Simple test case 1: Basic hoisting without shape_cast
// CHECK-LABEL: @hoist_simple_no_shapecast
func.func @hoist_simple_no_shapecast(%arg0: vector<64xi16>) -> vector<64xi16> {
  %c0 = arith.constant 0 : index
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  
  // CHECK: %[[INIT_EXT:.*]] = arith.extsi %{{.*}} : vector<64xi16> to vector<64xi32>
  // CHECK-NEXT: %[[LOOP:.*]] = scf.for %{{.*}} = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ITER:.*]] = %[[INIT_EXT]]) -> (vector<64xi32>)
  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %arg0) -> (vector<64xi16>) {
    // CHECK-NOT: arith.extsi
    %acc_ext = arith.extsi %acc : vector<64xi16> to vector<64xi32>
    
    // Some computation
    %val = arith.addi %acc_ext, %acc_ext : vector<64xi32>
    
    // CHECK-NOT: arith.trunci
    %val_trunc = arith.trunci %val : vector<64xi32> to vector<64xi16>
    scf.yield %val_trunc : vector<64xi16>
    // CHECK: scf.yield %{{.*}} : vector<64xi32>
  }
  // CHECK: %[[RESULT:.*]] = arith.trunci %[[LOOP]] : vector<64xi32> to vector<64xi16>
  // CHECK: return %[[RESULT]]
  return %result : vector<64xi16>
}
