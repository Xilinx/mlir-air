//===- label_ping_pong_odd_trip_count.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Ping-pong labeling unrolls the body by 2 and alternates the duplicated L1
// buffers, so the PRODUCER rotates through exactly two slots. An ODD trip count
// leaves a peeled remainder iteration whose copy allocates a THIRD buffer.
// AIRToAIE then chains one BD per static put site, so the CONSUMING DMA
// round-robins three slots against a producer that only ever alternates two:
//
//   producer  A B  A B  A B  A B  C      (4 unrolled trips + peeled tail)
//   consumer  A B C  A B C  A B C        (aie.mem next_bd chain)
//
// Transfers 3 and 6 send C, which the core has not written at that point --
// zeros on the first pass over the ring, the previous trip's data after. The
// lock counts still balance, so nothing deadlocks and nothing warns; the DMA
// simply moves the wrong bytes. Measured on gemma4-e2b's 9-trip vocab egress,
// where it silently zeroed logit blocks 2 and 5 of the first chunk.
//
// So an odd-trip-count loop must not be labeled.

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
// so the unrolled body consumes the ring a whole number of times and the two
// sequences stay in step -- MUST still be labeled. Locks the new predicate
// against rejecting everything.
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
// Case 3 (NEGATIVE): a non-unit step that still yields an odd trip count --
// 0 to 9 step 1 is 9 trips. Guards the trip-count arithmetic rather than a
// bound-value pattern.
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
// Case 4 (POSITIVE, documents a KNOWN LIMIT): the trip count is not a
// compile-time constant, so the guard cannot see it and the loop is still
// labeled. A dynamic loop that turns out to be odd at runtime peels into a
// third buffer exactly as Case 1 would. Fixing that needs the peeled remainder
// to reuse the ping slot instead of allocating a new buffer; this test exists
// so the gap is recorded rather than assumed closed.
// =============================================================================

// CHECK-LABEL: func.func @dynamic_trip_count_still_labels
// CHECK:       scf.for
// CHECK:       memref.alloc() {hoist_alloc = true} : memref<32x32xbf16, 2>
// CHECK:       } {unroll = 2 : i32}
// CHECK:       return

  func.func @dynamic_trip_count_still_labels(%arg0: memref<256x1024xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4) in (%arg6=%c1) attributes {id = 4 : i32} {
      %1 = air.segment async {
        %c4 = arith.constant 4 : index
        %2 = air.herd @herd_3 async tile (%arg21, %arg22) in (%arg23=%c4, %arg24=%c4) {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %async_token_0 = air.wait_all async
          // %arg23 is the herd-size block argument: a runtime value here, so
          // getConstantIntValue cannot fold it and the guard does not fire.
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
}
