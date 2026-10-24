//===- lower_scf_branch_air_token.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='test-patterns=lower-scf-tokens' | FileCheck %s

// scf.if and scf.index_switch lose their air.async.token results, as scf.for
// does: each region waits on the token it yielded, a dropped result becomes a
// dependency-free air.wait_all, and the other results and the cases stay.

// CHECK-LABEL: func.func @branches
// CHECK: %[[IF:.*]] = scf.if %{{.*}} -> (index) {
// CHECK: %[[T1:.*]] = air.wait_all async
// CHECK: air.wait_all [%[[T1]]]
// CHECK: scf.yield %{{.*}} : index
// CHECK: } else {
// CHECK: %[[T2:.*]] = air.wait_all async
// CHECK: air.wait_all [%[[T2]]]
// CHECK: scf.yield %{{.*}} : index
// CHECK: %[[SW:.*]] = scf.index_switch %{{.*}} -> index
// CHECK: case 0 {
// CHECK: scf.yield %{{.*}} : index
// CHECK: case 2 {
// CHECK: scf.yield %{{.*}} : index
// CHECK: default {
// CHECK: scf.yield %{{.*}} : index
// CHECK: arith.addi %[[IF]], %[[SW]] : index
// CHECK-NOT: !air.async.token
// CHECK: return

func.func @branches(%arg0: memref<32xi32, 2>) {
  %c1 = arith.constant 1 : index
  air.herd @herd_0 tile (%x, %y) in (%sx=%c1, %sy=%c1) args(%buf=%arg0) : memref<32xi32, 2> attributes {id = 1 : i32} {
    %c0 = arith.constant 0 : index
    %one = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %t0 = air.wait_all async
    %cond = arith.cmpi eq, %x, %c0 : index
    %if:2 = scf.if %cond -> (!air.async.token, index) {
      %t1 = air.wait_all async [%t0]
      scf.yield %t1, %c0 : !air.async.token, index
    } else {
      %t2 = air.wait_all async [%t0]
      scf.yield %t2, %c2 : !air.async.token, index
    }
    %sw:2 = scf.index_switch %y -> !air.async.token, index
    case 0 {
      %t3 = air.wait_all async [%if#0]
      scf.yield %t3, %c0 : !air.async.token, index
    }
    case 2 {
      %t4 = air.wait_all async [%if#0]
      scf.yield %t4, %c2 : !air.async.token, index
    }
    default {
      %t5 = air.wait_all async [%if#0]
      scf.yield %t5, %one : !air.async.token, index
    }
    %v = arith.addi %if#1, %sw#1 : index
    %i = arith.index_cast %v : index to i32
    memref.store %i, %buf[%v] : memref<32xi32, 2>
    air.wait_all [%sw#0]
  }
  return
}
