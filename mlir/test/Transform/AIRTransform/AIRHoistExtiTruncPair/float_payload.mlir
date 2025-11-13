//===- float_payload.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-transform='filename=%S/float_transform.mlir' -verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func.func @hoist_simple_extf_truncf
// CHECK: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<64xbf16>
// CHECK: %[[INIT_EXT:.*]] = arith.extf %[[INIT]] : vector<64xbf16> to vector<64xf32>
// CHECK: %[[LOOP_RESULT:.*]] = scf.for {{.*}} iter_args(%[[ARG:.*]] = %[[INIT_EXT]]) -> (vector<64xf32>)
// CHECK-NOT: arith.extf
// CHECK: vector.fma
// CHECK-NOT: arith.truncf
// CHECK: scf.yield %{{.*}} : vector<64xf32>
// CHECK: %[[FINAL:.*]] = arith.truncf %[[LOOP_RESULT]] : vector<64xf32> to vector<64xbf16>
func.func @hoist_simple_extf_truncf() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst_bf16 = arith.constant dense<0.0> : vector<64xbf16>
  %cst_f32 = arith.constant dense<1.0> : vector<64xf32>
  
  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%arg = %cst_bf16) -> (vector<64xbf16>) {
    %arg_ext = arith.extf %arg : vector<64xbf16> to vector<64xf32>
    %fma = vector.fma %arg_ext, %cst_f32, %arg_ext : vector<64xf32>
    %result_bf16 = arith.truncf %fma : vector<64xf32> to vector<64xbf16>
    scf.yield %result_bf16 : vector<64xbf16>
  }
  return
}

// CHECK-LABEL: func.func @hoist_extf_truncf_with_shape_cast
// CHECK: %[[INIT:.*]] = arith.constant dense<0.000000e+00> : vector<64xf16>
// CHECK: %[[INIT_EXT:.*]] = arith.extf %[[INIT]] : vector<64xf16> to vector<64xf32>
// CHECK: %[[LOOP_RESULT:.*]] = scf.for {{.*}} iter_args(%[[ARG:.*]] = %[[INIT_EXT]]) -> (vector<64xf32>)
// CHECK: %[[ARG_SHAPED:.*]] = vector.shape_cast %[[ARG]] : vector<64xf32> to vector<1x1x8x8xf32>
// CHECK-NOT: arith.extf
// CHECK: arith.addf
// CHECK-NOT: arith.truncf
// CHECK: %[[RESULT_FLAT:.*]] = vector.shape_cast {{.*}} : vector<1x1x8x8xf32> to vector<64xf32>
// CHECK: scf.yield %[[RESULT_FLAT]] : vector<64xf32>
// CHECK: %[[FINAL:.*]] = arith.truncf %[[LOOP_RESULT]] : vector<64xf32> to vector<64xf16>
func.func @hoist_extf_truncf_with_shape_cast() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %cst_f16 = arith.constant dense<0.0> : vector<64xf16>
  %a = arith.constant dense<1.0> : vector<1x1x8x8xf32>
  
  %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%arg = %cst_f16) -> (vector<64xf16>) {
    %arg_shaped = vector.shape_cast %arg : vector<64xf16> to vector<1x1x8x8xf16>
    %arg_ext = arith.extf %arg_shaped : vector<1x1x8x8xf16> to vector<1x1x8x8xf32>
    %add = arith.addf %arg_ext, %a : vector<1x1x8x8xf32>
    %result_f16 = arith.truncf %add : vector<1x1x8x8xf32> to vector<1x1x8x8xf16>
    %result_flat = vector.shape_cast %result_f16 : vector<1x1x8x8xf16> to vector<64xf16>
    scf.yield %result_flat : vector<64xf16>
  }
  return
}
