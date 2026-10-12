//===- fold_iter_args_into_induction_var.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -allow-unregistered-dialect -transform-interpreter -split-input-file -canonicalize %s | FileCheck %s

// Pointer offsets advanced by a fixed step, as a Triton K loop carries them,
// with an i32 induction variable and index offsets. The offsets are computed
// from the induction variable and the loop carries nothing.

// CHECK-LABEL: func.func @offsets
// CHECK-SAME: (%[[A:.*]]: index, %[[B:.*]]: index)
// CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
// CHECK-DAG: %[[C512:.*]] = arith.constant 512 : index
// CHECK: scf.for %[[IV:.*]] = %{{.*}} to %{{.*}} step %{{.*}} : i32 {
// CHECK-DAG: arith.index_cast %[[IV]] : i32 to index
// CHECK-DAG: %[[DA:.*]] = arith.muli %{{.*}}, %[[C256]]
// CHECK-DAG: %[[OA:.*]] = arith.addi %[[A]], %[[DA]]
// CHECK-DAG: %[[DB:.*]] = arith.muli %{{.*}}, %[[C512]]
// CHECK-DAG: %[[OB:.*]] = arith.addi %[[B]], %[[DB]]
// CHECK: "test.use"(%[[OA]], %[[OB]])
// CHECK: }
// CHECK: %[[EA:.*]] = arith.addi %[[A]], %c1024
// CHECK: return %[[EA]]
func.func @offsets(%a: index, %b: index) -> index {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %c256 = arith.constant 256 : index
  %c512 = arith.constant 512 : index
  %r:2 = scf.for %k = %c0 to %c4 step %c1 iter_args(%oa = %a, %ob = %b) -> (index, index) : i32 {
    "test.use"(%oa, %ob) : (index, index) -> ()
    %na = arith.addi %oa, %c256 : index
    %nb = arith.addi %c512, %ob : index
    scf.yield %na, %nb : index, index
  }
  return %r#0 : index
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %new = transform.air.fold_iter_args_into_induction_var %loop : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A lower bound and step other than 0 and 1: the iteration number is
// (iv - lb) / step.

// CHECK-LABEL: func.func @strided
// CHECK: scf.for %[[IV:.*]] = %c2 to %c10 step %c3 {
// CHECK: %[[S:.*]] = arith.subi %[[IV]], %c2
// CHECK: %[[N:.*]] = arith.divui %[[S]], %c3
// CHECK: %[[D:.*]] = arith.muli %[[N]], %c7
// CHECK: %[[V:.*]] = arith.addi %{{.*}}, %[[D]]
// CHECK: "test.use"(%[[V]])
func.func @strided(%x: index) {
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c7 = arith.constant 7 : index
  %c10 = arith.constant 10 : index
  %r = scf.for %k = %c2 to %c10 step %c3 iter_args(%v = %x) -> (index) {
    "test.use"(%v) : (index) -> ()
    %n = arith.addi %v, %c7 : index
    scf.yield %n : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %new = transform.air.fold_iter_args_into_induction_var %loop : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// A step computed in the loop, and a tensor, are left as they are.

// CHECK-LABEL: func.func @not_linear
// CHECK: scf.for {{.*}} iter_args(%[[V:.*]] = %{{.*}}, %[[T:.*]] = %{{.*}}) -> (index, tensor<4xf32>)
// CHECK: %[[STEP:.*]] = arith.muli %[[V]], %[[V]]
// CHECK: arith.addi %[[V]], %[[STEP]]
func.func @not_linear(%x: index, %t: tensor<4xf32>) -> (index, tensor<4xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r:2 = scf.for %k = %c0 to %c4 step %c1 iter_args(%v = %x, %u = %t) -> (index, tensor<4xf32>) {
    %step = arith.muli %v, %v : index
    %n = arith.addi %v, %step : index
    %e = math.exp %u : tensor<4xf32>
    scf.yield %n, %e : index, tensor<4xf32>
  }
  return %r#0, %r#1 : index, tensor<4xf32>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %new = transform.air.fold_iter_args_into_induction_var %loop : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The advanced value is also read in the body, before it goes round the
// loop. That read sees the next iteration's offset, not the initial one.

// CHECK-LABEL: func.func @next_read_in_body
// CHECK-SAME: (%[[X:.*]]: index)
// CHECK: scf.for %[[IV:.*]] = %c0 to %c4 step %c1 {
// CHECK: %[[D:.*]] = arith.muli %[[IV]], %c64
// CHECK: %[[V:.*]] = arith.addi %[[X]], %[[D]]
// CHECK: %[[N:.*]] = arith.addi %[[V]], %c64
// CHECK: "test.use"(%[[N]])
func.func @next_read_in_body(%x: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c64 = arith.constant 64 : index
  %r = scf.for %k = %c0 to %c4 step %c1 iter_args(%v = %x) -> (index) {
    %n = arith.addi %v, %c64 : index
    "test.use"(%n) : (index) -> ()
    scf.yield %n : index
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %new = transform.air.fold_iter_args_into_induction_var %loop : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
