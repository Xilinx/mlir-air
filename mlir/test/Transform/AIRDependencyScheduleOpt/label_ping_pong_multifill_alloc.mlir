//===- label_ping_pong_multifill_alloc.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression: an scf.for whose direct-child memref.alloc is filled by more
// than one air.channel.get per single outer iteration must NOT be labeled
// for ping-pong. The K>=2 fills per buffer per iteration are sequential
// overwrites of the same memory -- doubling the buffer doesn't decouple
// them. Pre-fix the label-ping-pong pass still labeled the outer loop
// (hoist_alloc=true, unroll=2), and downstream ping-pong-transform then
// emitted N_buffers lock-init plus per-buffer-grouped BD chains; the
// writer raced past the reader and silently corrupted the output.

// RUN: air-opt %s -air-label-scf-for-to-ping-pong | FileCheck %s

// =============================================================================
// Case 1: outer-loop alloc filled K=4 times per outer iter via an inner
// static scf.for. Pre-fix: outer labeled (hoist + unroll). Post-fix: the
// outer must NOT be labeled (and the inner has no candidate alloc, so it
// is not labeled either).
// =============================================================================

// CHECK-LABEL: func.func @multifill_outer_loop
// CHECK:       scf.for
// CHECK-NOT:   hoist_alloc
// CHECK-NOT:   } {unroll
// CHECK:       return

module {
  air.channel @gather_chan [1]
  func.func @multifill_outer_loop(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c4_inner = arith.constant 4 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %async_token_0 = air.wait_all async
          %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
            // Outer-loop direct alloc -- the candidate for ping-pong.
            %async_token_a, %results_a = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc_a = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc_a : memref<32x32xbf16, 2>
            }
            // Inner static for loop with K=4 channel.gets into the SAME
            // outer alloc -- 4 sequential overwrites per outer iteration.
            // This is what makes the outer loop NOT ping-pong-eligible.
            %inner = scf.for %arg12 = %c0 to %c4_inner step %c1_h iter_args(%arg13 = %async_token_a) -> (!air.async.token) {
              %g = air.channel.get async [%arg13] @gather_chan[] (%results_a[] [] []) {id = 1 : i32} : (memref<32x32xbf16, 2>)
              scf.yield %g : !air.async.token
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

// -----

// =============================================================================
// Case 2: control case. Same shape, same outer alloc, but exactly ONE
// channel.get per outer iter (no inner gather loop). The outer loop IS a
// valid ping-pong candidate and MUST still be labeled. Guards against the
// fix over-rejecting and breaking the common single-fill case.
// =============================================================================

// CHECK-LABEL: func.func @singlefill_outer_loop
// CHECK:       scf.for
// CHECK:       memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
// CHECK:       } {unroll = 2 : i32}

module {
  air.channel @single_chan [1]
  func.func @singlefill_outer_loop(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %async_token_0 = air.wait_all async
          %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
            %async_token_a, %results_a = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc_a = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc_a : memref<32x32xbf16, 2>
            }
            // Exactly one channel.get per outer iter -- ping-pong-safe.
            %g = air.channel.get async [%async_token_a] @single_chan[] (%results_a[] [] []) {id = 1 : i32} : (memref<32x32xbf16, 2>)
            %async_token_d = air.execute [%g] {
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

// -----

// =============================================================================
// Case 3: alloc is wrapped in an air.execute, and the channel.gets reference
// the execute's yielded result, not the bare memref.alloc result. Tests
// that the predicate correctly resolves the alloc identity through the
// air.execute terminator -- the typical IR shape produced by the AIR
// frontend lowering.
// =============================================================================

// CHECK-LABEL: func.func @multifill_via_execute_yield
// CHECK:       scf.for
// CHECK-NOT:   hoist_alloc
// CHECK-NOT:   } {unroll
// CHECK:       return

module {
  air.channel @gather_chan_v [1]
  func.func @multifill_via_execute_yield(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_inner = arith.constant 2 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %async_token_0 = air.wait_all async
          %3 = scf.for %arg10 = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
            %async_token_a, %results_a = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc_a = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc_a : memref<32x32xbf16, 2>
            }
            // K=2 gets per outer iter via static inner loop -- still
            // disqualifies the outer for ping-pong.
            %inner = scf.for %arg12 = %c0 to %c2_inner step %c1_h iter_args(%arg13 = %async_token_a) -> (!air.async.token) {
              %g = air.channel.get async [%arg13] @gather_chan_v[] (%results_a[] [] []) {id = 1 : i32} : (memref<32x32xbf16, 2>)
              scf.yield %g : !air.async.token
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
