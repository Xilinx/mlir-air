//===- label_ping_pong_odd_trip_count.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Unroll-by-2 rotates the producer through two L1 buffers. An odd trip count
// peels a remainder whose copy allocates a third, and AIRToAIE chains one BD
// per static put site, so the consumer round-robins three against two:
//
//   producer  A B  A B  A B  A B  C      (4 unrolled trips + peeled tail)
//   consumer  A B C  A B C  A B C        (aie.mem next_bd chain)
//
// Transfers 3 and 6 send C, which the core has not written this trip. Locks
// still balance, so nothing deadlocks or warns -- the DMA just moves stale
// bytes.
//
// So label only a provably even trip count. air::isTripCountDivisibleByFactor
// is the predicate the unroller itself uses, so the two cannot disagree.
// "Provably" cuts both ways: 2*J labels (Case 5), unknown parity does not
// (Case 4).

// RUN: air-opt %s -air-label-scf-for-to-ping-pong | FileCheck %s

// =============================================================================
// Case 1 (NEGATIVE): 9 trips (0 to 576 step 64). Odd -- must NOT be labeled.
// =============================================================================

// CHECK-LABEL: func.func @odd_trip_count_rejects
// CHECK:       scf.for
// CHECK-NOT:   hoist_alloc
// CHECK-NOT:   } {unroll
// CHECK:       return

module {
  air.channel @load_chan [1]
  func.func @odd_trip_count_rejects(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 1 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_0 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c64 = arith.constant 64 : index
          %c576 = arith.constant 576 : index
          %async_token_0 = air.wait_all async
          %3 = scf.for %arg10 = %c0 to %c576 step %c64 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
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
// Case 2 (POSITIVE control): identical body, 8 trips (0 to 512 step 64). Even,
// so it MUST still be labeled. Pins the predicate against rejecting
// everything.
// =============================================================================

// CHECK-LABEL: func.func @even_trip_count_labels
// CHECK:       scf.for
// CHECK:       memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
// CHECK:       } {unroll = 2 : i32}
// CHECK:       return

  func.func @even_trip_count_labels(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 2 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_1 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
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
// Case 3 (NEGATIVE): unit step, so 9 trips is the bound itself. Case 1 reaches
// 9 by striding 64 across an even bound, where only the derived count is odd.
// Both spellings are pinned.
// =============================================================================

// CHECK-LABEL: func.func @odd_trip_count_unit_step_rejects
// CHECK:       scf.for
// CHECK-NOT:   hoist_alloc
// CHECK-NOT:   } {unroll
// CHECK:       return

  func.func @odd_trip_count_unit_step_rejects(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 3 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_2 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c9 = arith.constant 9 : index
          %async_token_0 = air.wait_all async
          %3 = scf.for %arg10 = %c0 to %c9 step %c1_h iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
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
// Case 4 (NEGATIVE): dynamic bound of unprovable parity -- a herd-size block
// argument. Declined: if it is odd at runtime the remainder allocates a third
// buffer exactly as Case 1, and the failure is silent wrong data.
// =============================================================================

// CHECK-LABEL: func.func @dynamic_unprovable_parity_rejects
// CHECK:       scf.for
// CHECK-NOT:   hoist_alloc
// CHECK-NOT:   } {unroll
// CHECK:       return

  func.func @dynamic_unprovable_parity_rejects(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 4 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_3 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %async_token_0 = air.wait_all async
          // %arg23 is the herd-size block argument: a runtime value, and
          // nothing in the IR constrains its parity.
          %3 = scf.for %arg10 = %c0 to %arg23 step %c1_h iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
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
// Case 5 (POSITIVE control for Case 4): `0 to 2*%arg23 step 1` is dynamic but
// provably even, so it must still be labeled. Without this, Case 4 would be
// satisfied by a predicate that refused every dynamic bound.
// =============================================================================

// CHECK-LABEL: func.func @dynamic_provably_even_labels
// CHECK:       scf.for
// CHECK:       memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
// CHECK:       } {unroll = 2 : i32}
// CHECK:       return

  func.func @dynamic_provably_even_labels(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 5 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_4 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_h = arith.constant 2 : index
          %ub = arith.muli %arg23, %c2_h : index
          %async_token_0 = air.wait_all async
          %3 = scf.for %arg10 = %c0 to %ub step %c1_h iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
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
}
