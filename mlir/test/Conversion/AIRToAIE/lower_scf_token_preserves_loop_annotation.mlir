//===- lower_scf_token_preserves_loop_annotation.mlir ----------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='test-patterns=lower-scf-tokens' | FileCheck %s

// LowerScfTokenPattern strips !air.async.token iter_args from scf.for by
// building a fresh sync scf.for and hand-copying a few attrs. A user-attached
// llvm.loop_annotation was dropped. Verify it survives.

#loop_unroll = #llvm.loop_unroll<disable = true>
#loop_annotation = #llvm.loop_annotation<unroll = #loop_unroll, mustProgress = true>

// CHECK-DAG: #[[$LOOP_UNROLL:.*]] = #llvm.loop_unroll<disable = true>
// CHECK-DAG: #[[$LOOP_ANNOT:.*]] = #llvm.loop_annotation<unroll = #[[$LOOP_UNROLL]], mustProgress = true>

module attributes {torch.debug_module_name = "mmult"} {
  func.func @lower_scf_token_preserves_loop_annotation(%arg0: memref<32x32xi32, 2>, %arg1: memref<32x32xi32, 2>, %arg2: memref<32x32xi32, 2>) {
    %c1 = arith.constant 1 : index
    air.herd @herd_0 tile (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<32x32xi32, 2>, memref<32x32xi32, 2>, memref<32x32xi32, 2> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %0 = air.wait_all async
      // CHECK: scf.for %{{.*}} = %c0 to %c64 step %c32 {
      // CHECK: } {loop_annotation = #[[$LOOP_ANNOT]]}
      %1 = scf.for %arg10 = %c0 to %c64 step %c32 iter_args(%arg11 = %0) -> (!air.async.token) {
        %asyncToken = air.execute [%arg11] {
          linalg.matmul ins(%arg7, %arg8 : memref<32x32xi32, 2>, memref<32x32xi32, 2>) outs(%arg9 : memref<32x32xi32, 2>)
          air.execute_terminator
        } {id = 1 : i32}
        %2 = air.wait_all async [%asyncToken]
        scf.yield %2 : !air.async.token
      } {loop_annotation = #loop_annotation}
      air.wait_all [%1]
    }
    return
  }
}
