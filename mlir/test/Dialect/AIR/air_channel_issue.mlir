//===- air_channel_issue.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -verify-diagnostics | FileCheck %s

// A core issues: offset only, async, and with a bundle index.
// CHECK-LABEL: func.func @core
// CHECK: air.channel.issue @c[] offset(%{{.*}} : i64) {issue_id = 3 : i64}
// CHECK: air.channel.issue async [%{{.*}}] @c2[%{{.*}}] offset(%{{.*}} : index) {issue_id = 4 : i64}
air.channel @c [1]
air.channel @c2 [2]
func.func @core() {
  %c1 = arith.constant 1 : index
  air.herd tile (%x, %y) in (%sx = %c1, %sy = %c1) {
    %e = arith.constant 64 : i64
    %i = arith.constant 0 : index
    air.channel.issue @c[] offset(%e : i64) {issue_id = 3 : i64}
    %t = air.wait_all async
    %t2 = air.channel.issue async [%t] @c2[%i] offset(%i : index) {issue_id = 4 : i64}
  }
  return
}

// -----

// The host issues: a run-time length, outside any segment.
// CHECK-LABEL: func.func @host
// CHECK: air.channel.issue @c[] length(%{{.*}}) {issue_id = 0 : i64}
air.channel @c [1]
func.func @host(%n: index) {
  %c1 = arith.constant 1 : index
  air.launch (%x) in (%sx = %c1) args(%len = %n) : index {
    air.channel.issue @c[] length(%len) {issue_id = 0 : i64}
  }
  return
}

// -----

air.channel @c [1]
func.func @nothing() {
  // expected-error@+1 {{patches nothing}}
  air.channel.issue @c[] {issue_id = 0 : i64}
  return
}

// -----

air.channel @c [1]
func.func @host_offset(%n: index) {
  %c1 = arith.constant 1 : index
  air.launch (%x) in (%sx = %c1) args(%o = %n) : index {
    %e = arith.index_cast %o : index to i64
    // expected-error@+1 {{can set a length but not an offset}}
    air.channel.issue @c[] offset(%e : i64) {issue_id = 0 : i64}
  }
  return
}
