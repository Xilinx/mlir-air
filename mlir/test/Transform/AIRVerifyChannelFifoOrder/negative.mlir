//===- negative.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-verify-channel-fifo-order -split-input-file -verify-diagnostics

// Cases the verifier must reject: ops addressing the same channel slot with no
// ordering between them.

// -----
// Two gets on one slot, different buffers, no edge between them.
module {
  air.channel @channel_0 [1]
  func.func @neg_unordered_gets() {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) {
      air.segment @seg {
        %c0 = arith.constant 0 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        // expected-note@+1 {{earlier op addressing the same slot}}
        %g0 = air.channel.get async [%t0] @channel_0[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
        // expected-error@+2 {{addresses the same channel slot as an earlier op but is not ordered after it}}
        // expected-note@+1 {{ops on one channel share a FIFO and must be totally ordered}}
        %g1 = air.channel.get async [%t1] @channel_0[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}

// -----
// Same, for puts.
module {
  air.channel @channel_1 [1]
  func.func @neg_unordered_puts() {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) {
      air.segment @seg {
        %c0 = arith.constant 0 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        // expected-note@+1 {{earlier op addressing the same slot}}
        %p0 = air.channel.put async [%t0] @channel_1[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
        // expected-error@+2 {{addresses the same channel slot as an earlier op but is not ordered after it}}
        // expected-note@+1 {{ops on one channel share a FIFO and must be totally ordered}}
        %p1 = air.channel.put async [%t1] @channel_1[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}

// -----
// Two endpoints in sibling affine.if guards, as broadcast lowering produces. A
// conditional is not a scope boundary, so the two must still be ordered.
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @channel_2 [1, 1] {broadcast_shape = [1, 2]}
  func.func @neg_unordered_across_affine_if() {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        %c2_0 = arith.constant 2 : index
        air.herd @h tile (%x, %y) in (%sx=%c1_0, %sy=%c2_0) {
          %t0, %r0 = air.execute -> (memref<8xi32, 2 : i32>) {
            %m = memref.alloc() : memref<8xi32, 2 : i32>
            air.execute_terminator %m : memref<8xi32, 2 : i32>
          }
          %t1, %r1 = air.execute -> (memref<8xi32, 2 : i32>) {
            %m = memref.alloc() : memref<8xi32, 2 : i32>
            air.execute_terminator %m : memref<8xi32, 2 : i32>
          }
          %w = air.wait_all async
          %i0 = affine.if #set()[%x, %y] -> !air.async.token {
            // expected-note@+1 {{earlier op addressing the same slot}}
            %g0 = air.channel.get async [%t0] @channel_2[%x, %y] (%r0[] [] []) : (memref<8xi32, 2 : i32>)
            affine.yield %g0 : !air.async.token
          } else {
            affine.yield %w : !air.async.token
          }
          %i1 = affine.if #set()[%x, %y] -> !air.async.token {
            // expected-error@+2 {{addresses the same channel slot as an earlier op but is not ordered after it}}
            // expected-note@+1 {{ops on one channel share a FIFO and must be totally ordered}}
            %g1 = air.channel.get async [%t1] @channel_2[%x, %y] (%r1[] [] []) : (memref<8xi32, 2 : i32>)
            affine.yield %g1 : !air.async.token
          } else {
            affine.yield %w : !air.async.token
          }
        }
      }
    }
    return
  }
}
