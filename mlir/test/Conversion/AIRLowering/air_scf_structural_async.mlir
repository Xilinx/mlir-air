//===- air_scf_structural_async.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-to-std %s | FileCheck %s

// An async token carried through scf.while becomes an !airrt.event in the
// loop's operands, both regions and its results.

// CHECK-LABEL: func.func @while_token
// CHECK: %[[T0:.*]] = airrt.wait_all : !airrt.event
// CHECK: %[[R:.*]]:2 = scf.while (%{{.*}} = %{{.*}}, %[[T:.*]] = %[[T0]]) : (index, !airrt.event) -> (index, !airrt.event)
// CHECK: scf.condition(%{{.*}}) %{{.*}}, %[[T]] : index, !airrt.event
// CHECK: ^bb0(%{{.*}}: index, %[[U:.*]]: !airrt.event):
// CHECK: %[[T1:.*]] = airrt.wait_all %[[U]] : !airrt.event
// CHECK: scf.yield %{{.*}}, %[[T1]] : index, !airrt.event
// CHECK: airrt.wait_all %[[R]]#1
func.func @while_token(%n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %t0 = air.wait_all async
  %r:2 = scf.while (%i = %c0, %t = %t0) : (index, !air.async.token) -> (index, !air.async.token) {
    %cond = arith.cmpi slt, %i, %n : index
    scf.condition(%cond) %i, %t : index, !air.async.token
  } do {
  ^bb0(%j: index, %u: !air.async.token):
    %t1 = air.wait_all async [%u]
    %k = arith.addi %j, %c1 : index
    scf.yield %k, %t1 : index, !air.async.token
  }
  air.wait_all [%r#1]
  return
}

// An scf.if carrying a token keeps its attributes.
// CHECK-LABEL: func.func @if_token_attrs
// CHECK: scf.if %{{.*}} -> (!airrt.event)
// CHECK: } {keep_me}
func.func @if_token_attrs(%c: i1) {
  %t0 = air.wait_all async
  %r = scf.if %c -> (!air.async.token) {
    %t1 = air.wait_all async [%t0]
    scf.yield %t1 : !air.async.token
  } else {
    scf.yield %t0 : !air.async.token
  } {keep_me}
  air.wait_all [%r]
  return
}
