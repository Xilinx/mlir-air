//===- herd_merge_dep_propagation.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-isolate-async-dma-loop-nests="scope=func" --split-input-file | FileCheck %s

// Verify that MergeAIRHerdsPattern correctly propagates inter-herd async
// dependencies as intra-herd ordering when collapsing same-named herds into
// a single herd body. Five scenarios:
//   1. Linear chain  — herd_0 → herd_0 → herd_0 (transitive ordering)
//   2. Negative      — same-named herds with NO inter-herd dep (no
//                      spurious barrier should be inserted)
//   3. Multi-predecessor — herd_0 depends on TWO independent predecessor
//                          herds (barrier must aggregate both tokens)
//   4. Fan-out join  — predecessor herd's body ends with TWO parallel
//                      async ops; the next herd's barrier must wait on
//                      BOTH branches, not just the lexically last one
//   5. Free-only container — predecessor herd ends with an scf.for whose
//                            body only deallocates; the next herd must
//                            wait on the real-work fill, NOT on the
//                            dealloc loop's "freed memory" token

//===----------------------------------------------------------------------===//
// Scenario 1: linear chain. Three @herd_0 instances, each depending on the
// previous. After merge, the body must preserve the ordering as intra-herd
// async deps. The single-source barrier wait_all is folded away by
// canonicalisation, leaving direct token edges between the cloned execute
// ops.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @chain
// Single merged herd (all three @herd_0 collapse into one):
// CHECK: air.herd @herd_0 async
// CHECK-NOT: air.herd @herd_0
// First fill (from producer herd):
// CHECK: %[[FILL0:.*]] = air.execute
// CHECK: linalg.fill ins(%c0_i16
// Second fill (from middle herd) must depend on the first:
// CHECK: %[[FILL1:.*]] = air.execute {{\[}}%[[FILL0]]{{\]}}
// CHECK: linalg.fill ins(%c1_i16
// channel.put (from output herd) must depend transitively on the second:
// CHECK: air.channel.put async {{\[}}%[[FILL1]]{{\]}}

module {
  air.channel @channel_out_a [1, 1]
  func.func @chain(%arg0: memref<32x32xi16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) args(%arg5=%arg0) : memref<32x32xi16> attributes {id = 1 : i32} {
      %1 = air.segment @seg async attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        %async_token_0, %alloc_L1 = air.execute -> (memref<1x1x4x8x4x8xi16, 2 : i32>) {
          %a = memref.alloc() : memref<1x1x4x8x4x8xi16, 2 : i32>
          air.execute_terminator %a : memref<1x1x4x8x4x8xi16, 2 : i32>
        } {id = 1 : i32}
        // Herd 1: fill with 0
        %2 = air.herd @herd_0 async [%async_token_0] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%buf=%alloc_L1) : memref<1x1x4x8x4x8xi16, 2 : i32> attributes {id = 3 : i32} {
          %cst = arith.constant 0 : i16
          %subview = memref.subview %buf[%tx, %ty, 0, 0, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : memref<1x1x4x8x4x8xi16, 2 : i32> to memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>
          %t = air.execute {
            linalg.fill ins(%cst : i16) outs(%subview : memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>)
          } {id = 4 : i32}
        }
        // Herd 2: fill with 1, depends on herd 1
        %3 = air.herd @herd_0 async [%2] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%buf=%alloc_L1) : memref<1x1x4x8x4x8xi16, 2 : i32> attributes {id = 5 : i32} {
          %cst = arith.constant 1 : i16
          %subview = memref.subview %buf[%tx, %ty, 0, 0, 0, 0] [1, 1, 4, 8, 4, 8] [1, 1, 1, 1, 1, 1] : memref<1x1x4x8x4x8xi16, 2 : i32> to memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>
          %t = air.execute {
            linalg.fill ins(%cst : i16) outs(%subview : memref<1x1x4x8x4x8xi16, strided<[1024, 1024, 256, 32, 8, 1], offset: ?>, 2 : i32>)
          } {id = 6 : i32}
        }
        // Herd 3: channel.put, depends on herd 2
        %4 = air.herd @herd_0 async [%3] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%buf=%alloc_L1) : memref<1x1x4x8x4x8xi16, 2 : i32> attributes {id = 7 : i32} {
          %ci1 = arith.constant 1 : index
          %ci8 = arith.constant 8 : index
          %ci32 = arith.constant 32 : index
          %ci256 = arith.constant 256 : index
          %ci1024 = arith.constant 1024 : index
          %5 = air.channel.put async @channel_out_a[%tx, %ty] (%buf[%tx, %ty, %tx, %tx, %tx, %tx] [%ci1, %ci1, %ci8, %ci8, %ci8, %ci8] [%ci1024, %ci1024, %ci32, %ci8, %ci256, %ci1]) {id = 8 : i32} : (memref<1x1x4x8x4x8xi16, 2 : i32>)
        }
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Scenario 2: negative. Two same-named herds operating on DISJOINT buffers
// with NO inter-herd dependency. Merge must collapse them into a single
// herd body without inserting a spurious barrier or serialising the two
// independent fills.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @negative_no_dep
// Single merged herd:
// CHECK: air.herd @herd_n async
// CHECK-NOT: air.herd @herd_n
// First fill has no async deps (empty bracket list elided):
// CHECK: air.execute {
// CHECK: linalg.fill ins(%c7_i16
// No barrier wait_all between the two independent fills:
// CHECK-NOT: air.wait_all
// Second fill also has no async deps — it must NOT have been pessimised
// to depend on the first:
// CHECK: air.execute {
// CHECK: linalg.fill ins(%c9_i16

module {
  func.func @negative_no_dep() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) attributes {id = 1 : i32} {
      %1 = air.segment @seg async attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        %async_token_a, %allocA = air.execute -> (memref<8xi16, 2 : i32>) {
          %a = memref.alloc() : memref<8xi16, 2 : i32>
          air.execute_terminator %a : memref<8xi16, 2 : i32>
        } {id = 1 : i32}
        %async_token_b, %allocB = air.execute -> (memref<8xi16, 2 : i32>) {
          %b = memref.alloc() : memref<8xi16, 2 : i32>
          air.execute_terminator %b : memref<8xi16, 2 : i32>
        } {id = 2 : i32}
        // Two independent same-named herds; no inter-herd token dep.
        %2 = air.herd @herd_n async [%async_token_a] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%bufA=%allocA) : memref<8xi16, 2 : i32> attributes {id = 3 : i32} {
          %cst = arith.constant 7 : i16
          %t = air.execute {
            linalg.fill ins(%cst : i16) outs(%bufA : memref<8xi16, 2 : i32>)
          } {id = 4 : i32}
        }
        %3 = air.herd @herd_n async [%async_token_b] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%bufB=%allocB) : memref<8xi16, 2 : i32> attributes {id = 5 : i32} {
          %cst = arith.constant 9 : i16
          %t = air.execute {
            linalg.fill ins(%cst : i16) outs(%bufB : memref<8xi16, 2 : i32>)
          } {id = 6 : i32}
        }
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Scenario 3: multi-predecessor. Herd 3 depends on BOTH herd 1 AND herd 2
// (independent predecessors). Merge must aggregate both predecessors'
// tokens into the dependent group's barrier — exercising the
// `barrierDeps` SmallVector code path with size > 1.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @multi_predecessor
// CHECK: air.herd @herd_m async
// CHECK-NOT: air.herd @herd_m
// Two independent producer fills (different buffers):
// CHECK: %[[FILL_A:.*]] = air.execute
// CHECK: linalg.fill ins(%c1_i16
// CHECK: %[[FILL_B:.*]] = air.execute
// CHECK: linalg.fill ins(%c2_i16
// Consumer fill must depend on BOTH predecessors. The dep list is order-
// dependent on canonicalisation; the current pipeline emits B then A.
// CHECK: air.execute {{\[}}%[[FILL_B]], %[[FILL_A]]{{\]}}
// CHECK: linalg.fill ins(%c3_i16

module {
  func.func @multi_predecessor() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) attributes {id = 1 : i32} {
      %1 = air.segment @seg async attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        %async_token_a, %allocA = air.execute -> (memref<8xi16, 2 : i32>) {
          %a = memref.alloc() : memref<8xi16, 2 : i32>
          air.execute_terminator %a : memref<8xi16, 2 : i32>
        } {id = 1 : i32}
        %async_token_b, %allocB = air.execute -> (memref<8xi16, 2 : i32>) {
          %b = memref.alloc() : memref<8xi16, 2 : i32>
          air.execute_terminator %b : memref<8xi16, 2 : i32>
        } {id = 2 : i32}
        %async_token_c, %allocC = air.execute -> (memref<8xi16, 2 : i32>) {
          %c = memref.alloc() : memref<8xi16, 2 : i32>
          air.execute_terminator %c : memref<8xi16, 2 : i32>
        } {id = 3 : i32}
        // Two independent producer herds (different buffers, no shared deps).
        %2 = air.herd @herd_m async [%async_token_a] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%bufA=%allocA) : memref<8xi16, 2 : i32> attributes {id = 4 : i32} {
          %cst = arith.constant 1 : i16
          %t = air.execute {
            linalg.fill ins(%cst : i16) outs(%bufA : memref<8xi16, 2 : i32>)
          } {id = 5 : i32}
        }
        %3 = air.herd @herd_m async [%async_token_b] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%bufB=%allocB) : memref<8xi16, 2 : i32> attributes {id = 6 : i32} {
          %cst = arith.constant 2 : i16
          %t = air.execute {
            linalg.fill ins(%cst : i16) outs(%bufB : memref<8xi16, 2 : i32>)
          } {id = 7 : i32}
        }
        // Consumer herd depends on BOTH predecessor herds.
        %4 = air.herd @herd_m async [%2, %3, %async_token_c] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%bufC=%allocC) : memref<8xi16, 2 : i32> attributes {id = 8 : i32} {
          %cst = arith.constant 3 : i16
          %t = air.execute {
            linalg.fill ins(%cst : i16) outs(%bufC : memref<8xi16, 2 : i32>)
          } {id = 9 : i32}
        }
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Scenario 4: parallel fan-out at body end. The producer herd's body ends
// with TWO independent async ops (parallel branches that write to disjoint
// buffers). The consumer herd's first op must wait on BOTH branches.
//
// This is a true regression test: with the original single-overwrite
// `lastToken` (only the lexically last op recorded as the predecessor's
// completion token), the consumer would only depend on ONE branch and the
// other would race. Data-flow canonicalisation does NOT restore the lost
// edge here because the lost producer writes to a buffer the consumer
// never reads.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @fanout_join
// CHECK: air.herd @herd_f async
// CHECK-NOT: air.herd @herd_f
// Two parallel producer fills (no inter-dep, disjoint buffers):
// CHECK: %[[FILL_P1:.*]] = air.execute
// CHECK: linalg.fill ins(%c1_i16
// CHECK: %[[FILL_P2:.*]] = air.execute
// CHECK: linalg.fill ins(%c2_i16
// Consumer fill must depend on BOTH parallel producer tokens. The
// pipeline emits the second producer's token first in the dep list.
// CHECK: air.execute {{\[}}%[[FILL_P2]], %[[FILL_P1]]{{\]}}
// CHECK: linalg.fill ins(%c9_i16

module {
  func.func @fanout_join() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) attributes {id = 1 : i32} {
      %1 = air.segment @seg async attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        %async_token_a, %allocA = air.execute -> (memref<8xi16, 2 : i32>) {
          %a = memref.alloc() : memref<8xi16, 2 : i32>
          air.execute_terminator %a : memref<8xi16, 2 : i32>
        } {id = 1 : i32}
        %async_token_b, %allocB = air.execute -> (memref<8xi16, 2 : i32>) {
          %b = memref.alloc() : memref<8xi16, 2 : i32>
          air.execute_terminator %b : memref<8xi16, 2 : i32>
        } {id = 2 : i32}
        %async_token_c, %allocC = air.execute -> (memref<8xi16, 2 : i32>) {
          %c = memref.alloc() : memref<8xi16, 2 : i32>
          air.execute_terminator %c : memref<8xi16, 2 : i32>
        } {id = 3 : i32}
        // Producer herd body ends with TWO parallel fills on disjoint buffers.
        %2 = air.herd @herd_f async [%async_token_a, %async_token_b] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%bufA=%allocA, %bufB=%allocB) : memref<8xi16, 2 : i32>, memref<8xi16, 2 : i32> attributes {id = 4 : i32} {
          %cst1 = arith.constant 1 : i16
          %cst2 = arith.constant 2 : i16
          %t1 = air.execute {
            linalg.fill ins(%cst1 : i16) outs(%bufA : memref<8xi16, 2 : i32>)
          } {id = 5 : i32}
          %t2 = air.execute {
            linalg.fill ins(%cst2 : i16) outs(%bufB : memref<8xi16, 2 : i32>)
          } {id = 6 : i32}
        }
        // Consumer herd writes to a buffer NEITHER producer fill touched —
        // so memref-flow canonicalisation cannot re-derive the deps. The
        // explicit token chain is the only ordering signal.
        %3 = air.herd @herd_f async [%2, %async_token_c] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%bufC=%allocC) : memref<8xi16, 2 : i32> attributes {id = 7 : i32} {
          %cst3 = arith.constant 9 : i16
          %t = air.execute {
            linalg.fill ins(%cst3 : i16) outs(%bufC : memref<8xi16, 2 : i32>)
          } {id = 8 : i32}
        }
      }
    }
    return
  }
}

// -----

//===----------------------------------------------------------------------===//
// Scenario 5: free-only container at end of producer body. The producer
// herd's body ends with an scf.for whose body only deallocates a buffer.
// The scf.for is async and has a token, but its only memory effect is
// `Free`. The consumer herd's first op must wait on the producer's REAL
// WORK (the fill), NOT on the dealloc-loop's "freed memory" token.
//
// True regression test for the generalised free-only filter: the original
// narrow check only inspected `air.execute` bodies for `memref.dealloc`,
// missing scf.for / affine.if / etc. wrapping a dealloc, and would pick
// the dealloc loop's token as the group's completion signal.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @free_only_container
// CHECK: air.herd @herd_d async
// CHECK-NOT: air.herd @herd_d
// Producer fill on its own buffer:
// CHECK: %[[FILL_TOK:.*]] = air.execute
// CHECK: linalg.fill ins(%c7_i16
// Producer's scf.for-of-deallocs (still emitted, just not chosen as the
// group's live-out):
// CHECK: scf.for
// CHECK: memref.dealloc
// Consumer fill must depend on the producer FILL token, NOT on the
// scf.for token.
// CHECK: air.execute {{\[}}%[[FILL_TOK]]{{\]}}
// CHECK: linalg.fill ins(%c9_i16

module {
  func.func @free_only_container() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) attributes {id = 1 : i32} {
      %1 = air.segment @seg async attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        %async_token_a, %allocA = air.execute -> (memref<8xi16, 2 : i32>) {
          %a = memref.alloc() : memref<8xi16, 2 : i32>
          air.execute_terminator %a : memref<8xi16, 2 : i32>
        } {id = 1 : i32}
        %async_token_b, %allocB = air.execute -> (memref<8xi16, 2 : i32>) {
          %b = memref.alloc() : memref<8xi16, 2 : i32>
          air.execute_terminator %b : memref<8xi16, 2 : i32>
        } {id = 2 : i32}
        // Producer: fill, then async scf.for body of only dealloc.
        %2 = air.herd @herd_d async [%async_token_a] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%bufA=%allocA) : memref<8xi16, 2 : i32> attributes {id = 3 : i32} {
          %cstv = arith.constant 7 : i16
          %ci0 = arith.constant 0 : index
          %ci1 = arith.constant 1 : index
          %ci2 = arith.constant 2 : index
          %tfill = air.execute {
            linalg.fill ins(%cstv : i16) outs(%bufA : memref<8xi16, 2 : i32>)
          } {id = 4 : i32}
          %seed = air.wait_all async [%tfill]
          %loop = scf.for %iv = %ci0 to %ci2 step %ci1 iter_args(%tok = %seed) -> (!air.async.token) {
            %tdealloc = air.execute [%tok] {
              memref.dealloc %bufA : memref<8xi16, 2 : i32>
            } {id = 99 : i32}
            scf.yield %tdealloc : !air.async.token
          }
        }
        // Consumer.
        %3 = air.herd @herd_d async [%2, %async_token_b] tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%bufB=%allocB) : memref<8xi16, 2 : i32> attributes {id = 5 : i32} {
          %cst = arith.constant 9 : i16
          %t = air.execute {
            linalg.fill ins(%cst : i16) outs(%bufB : memref<8xi16, 2 : i32>)
          } {id = 6 : i32}
        }
      }
    }
    return
  }
}
