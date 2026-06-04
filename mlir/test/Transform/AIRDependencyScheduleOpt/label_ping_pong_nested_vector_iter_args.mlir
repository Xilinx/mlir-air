//===- label_ping_pong_nested_vector_iter_args.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression: an scf.for must NOT be labeled for ping-pong if its body
// contains a nested scf.for with a non-token iter_arg (e.g. a vector
// accumulator). Ping-pong's body duplication produces two body copies,
// each carrying the nested scf.for; when the nested for has a vector
// iter_arg, the duplicated copies are read/yielded in a way that the
// downstream async-dependency rewrite + AIE lowering miscompiles (odd-
// iteration outputs come out as garbage on the order of ~1e7). Until
// the underlying unroll + async-rewrite path is fixed end-to-end, skip
// ping-pong for such loops -- the user can refactor to memref-based
// accumulators if they want PP overlap.

// RUN: air-opt %s -air-label-scf-for-to-ping-pong | FileCheck %s

// =============================================================================
// Case 1 (NEGATIVE): outer loop has the per-iter L1 alloc that would
// normally qualify for ping-pong (single channel.get/iter, etc.), but its
// body contains an inner scf.for whose iter_args carry a vector<32xbf16>
// accumulator. The outer loop must NOT be labeled (no hoist_alloc, no
// unroll).
// =============================================================================

// CHECK-LABEL: func.func @nested_vector_iter_args_rejects
// CHECK:       scf.for
// CHECK-NOT:   hoist_alloc
// CHECK-NOT:   } {unroll
// CHECK:       return

module {
  air.channel @load_chan [1]
  func.func @nested_vector_iter_args_rejects(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %cst0_bf16 = arith.constant 0.000000e+00 : bf16
          %cst_zero_vec = arith.constant dense<0.000000e+00> : vector<32xbf16>
          %async_token_0 = air.wait_all async
          %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
            // Outer-loop direct alloc -- would normally be the candidate
            // for ping-pong (single channel.get/iter).
            %async_token_a, %results_a = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc_a = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc_a : memref<32x32xbf16, 2>
            }
            %fill = air.channel.get async [%async_token_a] @load_chan[] (%results_a[] [] []) : (memref<32x32xbf16, 2>)
            // Nested scf.for with a VECTOR iter_arg -- this is what
            // disqualifies the outer loop from ping-pong labeling.
            %inner = scf.for %arg12 = %c0 to %c8 step %c1_h iter_args(%acc = %cst_zero_vec) -> (vector<32xbf16>) {
              %v = vector.transfer_read %results_a[%arg12, %c0], %cst0_bf16 {in_bounds = [true]} : memref<32x32xbf16, 2>, vector<32xbf16>
              %sum = arith.addf %acc, %v : vector<32xbf16>
              scf.yield %sum : vector<32xbf16>
            }
            // Sink so the inner result isn't DCE'd.
            vector.transfer_write %inner, %results_a[%c0, %c0] {in_bounds = [true]} : vector<32xbf16>, memref<32x32xbf16, 2>
            %async_token_d = air.execute [%fill] {
              memref.dealloc %results_a : memref<32x32xbf16, 2>
            }
            scf.yield %async_token_d : !air.async.token
          }
        }
      }
    }
    return
  }

// =============================================================================
// Case 2 (POSITIVE control): same outer-loop / alloc structure as Case 1
// but the inner scf.for carries only an !air.async.token iter_arg (no
// vector). The outer loop MUST still be labeled (hoist_alloc + unroll=2).
// This locks the predicate against over-rejection.
// =============================================================================

// CHECK-LABEL: func.func @nested_token_only_iter_args_labels
// CHECK:       scf.for
// CHECK:       memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
// CHECK:       } {unroll = 2 : i32}
// CHECK:       return

  func.func @nested_token_only_iter_args_labels(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 2 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_1 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %async_token_0 = air.wait_all async
          %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
            %async_token_a, %results_a = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc_a = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc_a : memref<32x32xbf16, 2>
            }
            %fill = air.channel.get async [%async_token_a] @load_chan[] (%results_a[] [] []) : (memref<32x32xbf16, 2>)
            // Inner scf.for with ONLY an async-token iter_arg -- this
            // does NOT disqualify the outer loop.
            %inner = scf.for %arg12 = %c0 to %c8 step %c1_h iter_args(%tok = %fill) -> (!air.async.token) {
              %t = air.wait_all async [%tok]
              scf.yield %t : !air.async.token
            }
            %async_token_d = air.execute [%inner] {
              memref.dealloc %results_a : memref<32x32xbf16, 2>
            }
            scf.yield %async_token_d : !air.async.token
          }
        }
      }
    }
    return
  }
}
