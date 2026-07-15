//===- label_ping_pong_disable_opt_out.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// User-facing opt-out: an scf.for carrying the air.disable_ping_pong attr
// in the input IR must NOT be labeled for ping-pong, regardless of any
// other eligibility checks. Used by designs that know PP is unsafe or
// unprofitable for a specific loop (e.g. a body that contains a nested
// scf.for with vector iter_args, which today miscompiles through the
// unroll + async-rewrite + AIE-lowering path).

// RUN: air-opt %s -air-label-scf-for-to-ping-pong | FileCheck %s

// =============================================================================
// Case 1 (NEGATIVE): outer-loop alloc that would normally qualify for
// ping-pong (single channel.get/iter, etc.), with air.disable_ping_pong
// attached. Outer must NOT be labeled (no hoist_alloc, no unroll).
// =============================================================================

// CHECK-LABEL: func.func @opt_out_rejects
// CHECK:       scf.for
// CHECK-NOT:   hoist_alloc
// CHECK-NOT:   } {unroll
// CHECK:       return

module {
  air.channel @load_chan [1]
  func.func @opt_out_rejects(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %async_token_0 = air.wait_all async
          %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
            %async_token_a, %results_a = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc_a = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc_a : memref<32x32xbf16, 2>
            }
            %fill = air.channel.get async [%async_token_a] @load_chan[] (%results_a[] [] []) : (memref<32x32xbf16, 2>)
            %async_token_d = air.execute [%fill] {
              memref.dealloc %results_a : memref<32x32xbf16, 2>
            }
            scf.yield %async_token_d : !air.async.token
          } {air.disable_ping_pong}
        }
      }
    }
    return
  }

// =============================================================================
// Case 2 (POSITIVE control): identical body to Case 1, but without
// air.disable_ping_pong. Outer MUST be labeled (hoist_alloc + unroll=2).
// Locks the predicate against unconditional rejection.
// =============================================================================

// CHECK-LABEL: func.func @no_opt_out_labels
// CHECK:       scf.for
// CHECK:       memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
// CHECK:       } {unroll = 2 : i32}
// CHECK:       return

  func.func @no_opt_out_labels(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 2 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_1 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %async_token_0 = air.wait_all async
          %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
            %async_token_a, %results_a = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc_a = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc_a : memref<32x32xbf16, 2>
            }
            %fill = air.channel.get async [%async_token_a] @load_chan[] (%results_a[] [] []) : (memref<32x32xbf16, 2>)
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
// Case 3 (NEGATIVE, nested): air.disable_ping_pong sits on a NESTED scf.for
// inside the candidate's body. The outer candidate must still NOT be labeled
// -- the walk in isPingPongCandidate honors the attr at any depth.
// =============================================================================

// CHECK-LABEL: func.func @opt_out_on_nested_for_rejects_outer
// CHECK:       scf.for
// CHECK-NOT:   hoist_alloc
// CHECK-NOT:   } {unroll
// CHECK:       return

  func.func @opt_out_on_nested_for_rejects_outer(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 3 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_2 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %cst0_bf16 = arith.constant 0.000000e+00 : bf16
          %async_token_0 = air.wait_all async
          %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
            // Outer-loop direct alloc -- candidate for PP labeling.
            %async_token_a, %results_a = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc_a = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc_a : memref<32x32xbf16, 2>
            }
            %fill = air.channel.get async [%async_token_a] @load_chan[] (%results_a[] [] []) : (memref<32x32xbf16, 2>)
            // Nested compute loop. PP opt-out attached here.
            %async_token_compute = air.execute [%fill] {
              scf.for %arg12 = %c0 to %c8 step %c1_h {
                %v = vector.transfer_read %results_a[%arg12, %c0], %cst0_bf16 {in_bounds = [true]} : memref<32x32xbf16, 2>, vector<32xbf16>
                vector.transfer_write %v, %results_a[%arg12, %c0] {in_bounds = [true]} : vector<32xbf16>, memref<32x32xbf16, 2>
              } {air.disable_ping_pong}
            }
            %async_token_d = air.execute [%async_token_compute] {
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
