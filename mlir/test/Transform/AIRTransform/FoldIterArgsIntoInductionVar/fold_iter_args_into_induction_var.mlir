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

// -----

// An i64 offset advanced in a loop over an i32 induction variable: the
// iteration count is sign-extended to i64.

// CHECK-LABEL: func.func @widen
// CHECK-SAME: (%[[X:.*]]: i64, %[[N:.*]]: i32)
// CHECK: scf.for %[[IV:.*]] = %{{.*}} to %[[N]] step %{{.*}} : i32 {
// CHECK: %[[E:.*]] = arith.extsi %[[IV]] : i32 to i64
// CHECK: %[[D:.*]] = arith.muli %[[E]], %c8_i64
// CHECK: %[[V:.*]] = arith.addi %[[X]], %[[D]]
// CHECK: "test.use"(%[[V]])
// CHECK: }
// CHECK: %[[T:.*]] = arith.extsi %{{.*}} : i32 to i64
// CHECK: %[[TD:.*]] = arith.muli %[[T]], %c8_i64
// CHECK: %[[R:.*]] = arith.addi %[[X]], %[[TD]]
// CHECK: return %[[R]]
func.func @widen(%x: i64, %n: i32) -> i64 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c8 = arith.constant 8 : i64
  %r = scf.for %k = %c0 to %n step %c1 iter_args(%v = %x) -> (i64) : i32 {
    "test.use"(%v) : (i64) -> ()
    %next = arith.addi %v, %c8 : i64
    scf.yield %next : i64
  }
  return %r : i64
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %new = transform.air.fold_iter_args_into_induction_var %loop : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// An i32 offset advanced in a loop over an i64 induction variable: the
// iteration count is truncated to i32.

// CHECK-LABEL: func.func @narrow
// CHECK-SAME: (%[[X:.*]]: i32, %[[N:.*]]: i64)
// CHECK: scf.for %[[IV:.*]] = %{{.*}} to %[[N]] step %{{.*}} : i64 {
// CHECK: %[[E:.*]] = arith.trunci %[[IV]] : i64 to i32
// CHECK: %[[D:.*]] = arith.muli %[[E]], %c8_i32
// CHECK: %[[V:.*]] = arith.addi %[[X]], %[[D]]
// CHECK: "test.use"(%[[V]])
// CHECK: }
// CHECK: %[[T:.*]] = arith.trunci %{{.*}} : i64 to i32
// CHECK: %[[TD:.*]] = arith.muli %[[T]], %c8_i32
// CHECK: %[[R:.*]] = arith.addi %[[X]], %[[TD]]
// CHECK: return %[[R]]
func.func @narrow(%x: i32, %n: i64) -> i32 {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c8 = arith.constant 8 : i32
  %r = scf.for %k = %c0 to %n step %c1 iter_args(%v = %x) -> (i32) : i64 {
    "test.use"(%v) : (i32) -> ()
    %next = arith.addi %v, %c8 : i32
    scf.yield %next : i32
  }
  return %r : i32
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %loop = transform.structured.match ops{["scf.for"]} in %arg0 : (!transform.any_op) -> !transform.any_op
    %new = transform.air.fold_iter_args_into_induction_var %loop : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
