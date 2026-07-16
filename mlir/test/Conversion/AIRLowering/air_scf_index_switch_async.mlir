//===- air_scf_index_switch_async.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-to-std %s | FileCheck %s

// A launch-scope scf.index_switch that carries an async drain token (used to
// select per-iteration host feeds in a fused multi-iteration launch) must have
// its async-token results converted to !airrt.event, mirroring the scf.for /
// scf.if conversions. Without the conversion the switch is left carrying an
// !air.async.token result, which strands the downstream drain.

// CHECK-LABEL: func.func @scf_index_switch
// CHECK: scf.index_switch %{{.*}} -> !airrt.event
// CHECK: scf.yield %{{.*}} : !airrt.event
// CHECK: scf.yield %{{.*}} : !airrt.event
// CHECK-NOT: !air.async.token
func.func @scf_index_switch(%arg0: index) {
  %0 = air.wait_all async
  %1 = scf.index_switch %arg0 -> !air.async.token
  case 0 {
    %2 = air.wait_all async [%0]
    scf.yield %2 : !air.async.token
  }
  default {
    %3 = air.wait_all async [%0]
    scf.yield %3 : !air.async.token
  }
  air.wait_all [%1]
  return
}
