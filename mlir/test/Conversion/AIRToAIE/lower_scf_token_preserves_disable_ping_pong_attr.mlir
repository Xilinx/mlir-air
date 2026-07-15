//===- lower_scf_token_preserves_disable_ping_pong_attr.mlir ---*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='test-patterns=lower-scf-tokens' | FileCheck %s

// LowerScfTokenPattern strips !air.async.token iter_args from scf.for by
// building a fresh sync scf.for. The user-facing ping-pong opt-out attr
// `air.disable_ping_pong` must survive that rewrite.

module attributes {torch.debug_module_name = "mmult"} {
  func.func @lower_scf_token_preserves_disable_ping_pong_attr(%arg0: memref<32x32xi32, 2>, %arg1: memref<32x32xi32, 2>, %arg2: memref<32x32xi32, 2>) {
    %c1 = arith.constant 1 : index
    air.herd @herd_0 tile (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<32x32xi32, 2>, memref<32x32xi32, 2>, memref<32x32xi32, 2> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %0 = air.wait_all async
      // CHECK: scf.for %{{.*}} = %c0 to %c64 step %c32 {
      // CHECK: } {air.disable_ping_pong}
      %1 = scf.for %arg10 = %c0 to %c64 step %c32 iter_args(%arg11 = %0) -> (!air.async.token) {
        %asyncToken = air.execute [%arg11] {
          linalg.matmul ins(%arg7, %arg8 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%arg9 : memref<32x32xi32, 2>)
          air.execute_terminator
        } {id = 1 : i32}
        %2 = air.wait_all async [%asyncToken]
        scf.yield %2 : !air.async.token
      } {air.disable_ping_pong}
      air.wait_all [%1]
    }
    return
  }
}
