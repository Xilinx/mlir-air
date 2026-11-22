//===- branch_token_position.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency-canonicalize | FileCheck %s

// An scf.if yields a value and two tokens; the write after it depends on the
// execute inside, whose token the if yields third. The write must wait on
// that result, not on the if's first token or its first result.

// CHECK-LABEL: func.func @two_tokens
// CHECK: %[[IF:.*]]:3 = scf.if
// CHECK: air.execute [{{.*}}%[[IF]]#2{{.*}}] {
// CHECK-NEXT: vector.transfer_write %[[IF]]#0
func.func @two_tokens(%a: memref<64xbf16>, %c: i1) {
  %i = arith.constant 0 : index
  %z = arith.constant 0.0 : bf16
  %t0 = air.wait_all async
  %r:3 = scf.if %c -> (vector<64xbf16>, !air.async.token, !air.async.token) {
    %t1 = air.wait_all async
    %t2, %w = air.execute [%t0] -> (vector<64xbf16>) {
      %x = vector.transfer_read %a[%i], %z {in_bounds = [true]} : memref<64xbf16>, vector<64xbf16>
      air.execute_terminator %x : vector<64xbf16>
    }
    scf.yield %w, %t1, %t2 : vector<64xbf16>, !air.async.token, !air.async.token
  } else {
    %v0 = arith.constant dense<0.0> : vector<64xbf16>
    %t1 = air.wait_all async
    scf.yield %v0, %t1, %t1 : vector<64xbf16>, !air.async.token, !air.async.token
  }
  %t3 = air.execute [%r#2] {
    vector.transfer_write %r#0, %a[%i] {in_bounds = [true]} : vector<64xbf16>, memref<64xbf16>
  }
  return
}
