//===- air_channel_issue_host.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-std | FileCheck %s

// A host-issued length in a segment stays in the host function, for
// airrt-to-npu to lower; the launch and segment around it are lowered.

// CHECK-LABEL: func.func @host_length
// CHECK-NOT: air.launch
// CHECK-NOT: air.segment
// CHECK: %[[N:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: air.channel.issue @kv[] length(%[[N]]) {air.length_bound = 8 : i64, issue_id = 0 : i64}
module {
  air.channel @kv []
  func.func @host_length(%arg0: i32) {
    air.launch () in () args(%n0=%arg0) : i32 {
      air.segment @seg args(%n=%n0) : i32 {
        %len = arith.index_cast %n : i32 to index
        air.channel.issue @kv[] length(%len) {air.length_bound = 8 : i64, issue_id = 0 : i64}
      }
    }
    return
  }
}
