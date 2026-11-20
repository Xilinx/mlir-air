//===- hoist_cases.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -allow-unregistered-dialect -transform-interpreter -split-input-file %s | FileCheck %s

// The new loop keeps the attributes of the old one.

// CHECK-LABEL: func.func @loop_attribute
// CHECK: scf.for {{.*}} -> (vector<64xi32>) {
// CHECK: } {keep_me}
func.func @loop_attribute(%arg0: vector<64xi16>) -> vector<64xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %arg0) -> (vector<64xi16>) {
    %w = arith.extsi %acc : vector<64xi16> to vector<64xi32>
    %s = arith.addi %w, %w : vector<64xi32>
    %n = arith.trunci %s : vector<64xi32> to vector<64xi16>
    scf.yield %n : vector<64xi16>
  } {keep_me}
  return %r : vector<64xi16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} attributes{sym_name = "loop_attribute"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %f : (!transform.any_op) -> !transform.any_op
    %ext = transform.structured.match ops{["arith.extsi"]} in %loop : (!transform.any_op) -> !transform.any_op
    %trunc = transform.structured.match ops{["arith.trunci"]} in %loop : (!transform.any_op) -> !transform.any_op
    %new = transform.air.hoist_cast_pair %ext, %trunc, %loop : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The narrow accumulator has another user in the body: it reads a truncation
// of the wide iter_arg.

// CHECK-LABEL: func.func @narrow_iter_arg_used_elsewhere
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (vector<64xi32>) {
// CHECK-NEXT: %[[N:.*]] = arith.trunci %[[ACC]] : vector<64xi32> to vector<64xi16>
// CHECK-NEXT: %[[S:.*]] = arith.addi %[[ACC]], %[[ACC]] : vector<64xi32>
// CHECK-NEXT: "test.use"(%[[N]]) : (vector<64xi16>) -> ()
// CHECK-NEXT: scf.yield %[[S]] : vector<64xi32>
func.func @narrow_iter_arg_used_elsewhere(%arg0: vector<64xi16>) -> vector<64xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %arg0) -> (vector<64xi16>) {
    %w = arith.extsi %acc : vector<64xi16> to vector<64xi32>
    %s = arith.addi %w, %w : vector<64xi32>
    "test.use"(%acc) : (vector<64xi16>) -> ()
    %n = arith.trunci %s : vector<64xi32> to vector<64xi16>
    scf.yield %n : vector<64xi16>
  }
  return %r : vector<64xi16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} attributes{sym_name = "narrow_iter_arg_used_elsewhere"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %f : (!transform.any_op) -> !transform.any_op
    %ext = transform.structured.match ops{["arith.extsi"]} in %loop : (!transform.any_op) -> !transform.any_op
    %trunc = transform.structured.match ops{["arith.trunci"]} in %loop : (!transform.any_op) -> !transform.any_op
    %new = transform.air.hoist_cast_pair %ext, %trunc, %loop : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The truncated value has another user in the body: the truncation stays for
// it, and the loop carries the wide value.

// CHECK-LABEL: func.func @truncation_used_elsewhere
// CHECK: scf.for {{.*}} iter_args(%[[ACC:.*]] = %{{.*}}) -> (vector<64xi32>) {
// CHECK-NEXT: %[[S:.*]] = arith.addi %[[ACC]], %[[ACC]] : vector<64xi32>
// CHECK-NEXT: %[[N:.*]] = arith.trunci %[[S]] : vector<64xi32> to vector<64xi16>
// CHECK-NEXT: "test.use"(%[[N]]) : (vector<64xi16>) -> ()
// CHECK-NEXT: scf.yield %[[S]] : vector<64xi32>
func.func @truncation_used_elsewhere(%arg0: vector<64xi16>) -> vector<64xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %arg0) -> (vector<64xi16>) {
    %w = arith.extsi %acc : vector<64xi16> to vector<64xi32>
    %s = arith.addi %w, %w : vector<64xi32>
    %n = arith.trunci %s : vector<64xi32> to vector<64xi16>
    "test.use"(%n) : (vector<64xi16>) -> ()
    scf.yield %n : vector<64xi16>
  }
  return %r : vector<64xi16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} attributes{sym_name = "truncation_used_elsewhere"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %f : (!transform.any_op) -> !transform.any_op
    %ext = transform.structured.match ops{["arith.extsi"]} in %loop : (!transform.any_op) -> !transform.any_op
    %trunc = transform.structured.match ops{["arith.trunci"]} in %loop : (!transform.any_op) -> !transform.any_op
    %new = transform.air.hoist_cast_pair %ext, %trunc, %loop : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

// The truncation consumes the extension directly: the loop just carries the
// wide value.

// CHECK-LABEL: func.func @truncation_of_extension
// CHECK: %[[INIT:.*]] = arith.extsi %arg0 : vector<64xi16> to vector<64xi32>
// CHECK: %[[LOOP:.*]] = scf.for {{.*}} iter_args(%[[ACC:.*]] = %[[INIT]]) -> (vector<64xi32>) {
// CHECK-NEXT: scf.yield %[[ACC]] : vector<64xi32>
// CHECK-NEXT: }
// CHECK-NEXT: arith.trunci %[[LOOP]] : vector<64xi32> to vector<64xi16>
func.func @truncation_of_extension(%arg0: vector<64xi16>) -> vector<64xi16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %arg0) -> (vector<64xi16>) {
    %w = arith.extsi %acc : vector<64xi16> to vector<64xi32>
    %n = arith.trunci %w : vector<64xi32> to vector<64xi16>
    scf.yield %n : vector<64xi16>
  }
  return %r : vector<64xi16>
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg0: !transform.any_op {transform.readonly}) {
    %f = transform.structured.match ops{["func.func"]} attributes{sym_name = "truncation_of_extension"} in %arg0 : (!transform.any_op) -> !transform.any_op
    %loop = transform.structured.match ops{["scf.for"]} in %f : (!transform.any_op) -> !transform.any_op
    %ext = transform.structured.match ops{["arith.extsi"]} in %loop : (!transform.any_op) -> !transform.any_op
    %trunc = transform.structured.match ops{["arith.trunci"]} in %loop : (!transform.any_op) -> !transform.any_op
    %new = transform.air.hoist_cast_pair %ext, %trunc, %loop : (!transform.any_op, !transform.any_op, !transform.any_op) -> !transform.any_op
    transform.yield
  }
}
