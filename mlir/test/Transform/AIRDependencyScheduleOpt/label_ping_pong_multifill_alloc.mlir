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

// -----

// =============================================================================
// Case 4 (positive): channel.gets inside mutually-exclusive `affine.if`
// branches (the IR shape produced by `air-specialize-dma-broadcast` for
// index-dispatched broadcasts) all write to the SAME outer-loop alloc, but
// only ONE branch runs per outer iter. Pre-fix the multifill predicate
// counted these as N gets and rejected the loop. Post-fix: dedup by the
// outermost `affine.if` between the get and the for, so the chain registers
// as a single per-iter occurrence and the loop is labeled.
// =============================================================================

// CHECK-LABEL: func.func @affine_if_broadcast_branches_eligible
// CHECK:       scf.for
// CHECK:       memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
// CHECK:       } {unroll = 2 : i32}

#set_a = affine_set<()[s0] : (s0 == 0)>
#set_b = affine_set<()[s0] : (s0 - 1 == 0)>
#set_c = affine_set<()[s0] : (s0 - 2 == 0)>

module {
  air.channel @chA [1, 1] {broadcast_shape = [1, 4]}
  air.channel @chB [1, 1] {broadcast_shape = [1, 4]}
  air.channel @chC [1, 1] {broadcast_shape = [1, 4]}
  air.channel @chD [1, 1] {broadcast_shape = [1, 4]}

  func.func @affine_if_broadcast_branches_eligible(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async tile (%col, %row) in (%cs=%c4, %rs=%c4) {
          %c0 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %async_token_0 = air.wait_all async
          %3 = scf.for %iv = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
            %async_token_a, %results_a = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc_a = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc_a : memref<32x32xbf16, 2>
            }
            // N=4 mutually-exclusive affine.if branches dispatch on the herd
            // column index. Each branch reads via a distinct channel into the
            // same alloc. Only one branch runs per iter, so this is logically
            // a single get-per-iter, not 4.
            %g = affine.if #set_a()[%col] -> !air.async.token {
              %ga = air.channel.get async [%async_token_a] @chA[] (%results_a[] [] []) {id = 1 : i32} : (memref<32x32xbf16, 2>)
              affine.yield %ga : !air.async.token
            } else {
              %gb = affine.if #set_b()[%col] -> !air.async.token {
                %gbb = air.channel.get async [%async_token_a] @chB[] (%results_a[] [] []) {id = 2 : i32} : (memref<32x32xbf16, 2>)
                affine.yield %gbb : !air.async.token
              } else {
                %gc = affine.if #set_c()[%col] -> !air.async.token {
                  %gcc = air.channel.get async [%async_token_a] @chC[] (%results_a[] [] []) {id = 3 : i32} : (memref<32x32xbf16, 2>)
                  affine.yield %gcc : !air.async.token
                } else {
                  %gd = air.channel.get async [%async_token_a] @chD[] (%results_a[] [] []) {id = 4 : i32} : (memref<32x32xbf16, 2>)
                  affine.yield %gd : !air.async.token
                }
                affine.yield %gc : !air.async.token
              }
              affine.yield %gb : !air.async.token
            }
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
// Case 5 (negative): two SIBLING top-level `affine.if`s, each independently
// gating its own `air.channel.get` into the same outer-loop alloc. The two
// affine.ifs are NOT mutually exclusive (no common affine.if ancestor places
// them in different regions), so both gets can execute in the same iter ->
// genuine multi-fill -> reject. Guards against the dedup over-broadening.
// =============================================================================

// CHECK-LABEL: func.func @sibling_affine_ifs_rejected
// CHECK:       scf.for
// CHECK-NOT:   hoist_alloc
// CHECK-NOT:   } {unroll
// CHECK:       return

#set_x = affine_set<()[s0] : (s0 == 0)>
#set_y = affine_set<()[s0] : (s0 - 1 == 0)>

module {
  air.channel @chX [1, 1]
  air.channel @chY [1, 1]

  func.func @sibling_affine_ifs_rejected(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async tile (%col, %row) in (%cs=%c4, %rs=%c4) {
          %c0 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %async_token_0 = air.wait_all async
          %3 = scf.for %iv = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
            %async_token_a, %results_a = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc_a = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc_a : memref<32x32xbf16, 2>
            }
            // Two TOP-LEVEL affine.ifs back-to-back. Their predicates may
            // both evaluate true in the same iter; nothing makes them
            // mutually exclusive. Must reject.
            %g1 = affine.if #set_x()[%col] -> !air.async.token {
              %gx = air.channel.get async [%async_token_a] @chX[] (%results_a[] [] []) {id = 1 : i32} : (memref<32x32xbf16, 2>)
              affine.yield %gx : !air.async.token
            } else {
              affine.yield %async_token_a : !air.async.token
            }
            %g2 = affine.if #set_y()[%col] -> !air.async.token {
              %gy = air.channel.get async [%g1] @chY[] (%results_a[] [] []) {id = 2 : i32} : (memref<32x32xbf16, 2>)
              affine.yield %gy : !air.async.token
            } else {
              affine.yield %g1 : !air.async.token
            }
            %async_token_d = air.execute [%g2] {
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
// Case 6 (negative): two gets to the same alloc inside the same `then`
// block of a single affine.if. Both run when the predicate is true ->
// genuine multi-fill in that branch -> reject. Guards against per-(alloc,
// topIf) dedup ignoring within-branch multiplicity.
// =============================================================================

// CHECK-LABEL: func.func @same_branch_two_gets_rejected
// CHECK:       scf.for
// CHECK-NOT:   hoist_alloc
// CHECK-NOT:   } {unroll
// CHECK:       return

#set_p = affine_set<()[s0] : (s0 == 0)>

module {
  air.channel @chP [1, 1]
  air.channel @chQ [1, 1]

  func.func @same_branch_two_gets_rejected(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async tile (%col, %row) in (%cs=%c4, %rs=%c4) {
          %c0 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c512 = arith.constant 512 : index
          %async_token_0 = air.wait_all async
          %3 = scf.for %iv = %c0 to %c512 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
            %async_token_a, %results_a = air.execute [%arg11] -> (memref<32x32xbf16, 2>) {
              %alloc_a = memref.alloc() : memref<32x32xbf16, 2>
              air.execute_terminator %alloc_a : memref<32x32xbf16, 2>
            }
            // ONE affine.if whose `then` body contains TWO sequential gets
            // to the same alloc. They are NOT mutually exclusive (both run
            // when the predicate is true).
            %g = affine.if #set_p()[%col] -> !air.async.token {
              %gp = air.channel.get async [%async_token_a] @chP[] (%results_a[] [] []) {id = 1 : i32} : (memref<32x32xbf16, 2>)
              %gq = air.channel.get async [%gp] @chQ[] (%results_a[] [] []) {id = 2 : i32} : (memref<32x32xbf16, 2>)
              affine.yield %gq : !air.async.token
            } else {
              affine.yield %async_token_a : !air.async.token
            }
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
