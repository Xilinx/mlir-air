//===- positive.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-verify-channel-fifo-order -split-input-file -verify-diagnostics

// Cases the verifier must accept. A silent run means the ordering holds.

// -----
// Directly ordered: the second get depends on the first.
module {
  air.channel @channel_0 [1]
  func.func @pos_ordered_gets() {
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
        %g0 = air.channel.get async [%t0] @channel_0[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
        %g1 = air.channel.get async [%t1, %g0] @channel_0[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}

// -----
// Ordered transitively through an air.wait_all. A path suffices, so the walk
// has to look through join nodes.
module {
  air.channel @channel_1 [1]
  func.func @pos_ordered_transitively() {
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
        %g0 = air.channel.get async [%t0] @channel_1[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
        %w = air.wait_all async [%g0, %t1]
        %g1 = air.channel.get async [%w] @channel_1[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}

// -----
// Different indices address different sub-channels: no ordering required.
module {
  air.channel @channel_2 [2]
  func.func @pos_different_indices() {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) {
      air.segment @seg {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %g0 = air.channel.get async [%t0] @channel_2[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
        %g1 = air.channel.get async [%t1] @channel_2[%c1_0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}

// -----
// Opposite directions are not ordered against each other; the order is per
// direction.
module {
  air.channel @channel_3 [1]
  func.func @pos_put_get_indep() {
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
        %p0 = air.channel.put async [%t0] @channel_3[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
        %g0 = air.channel.get async [%t1] @channel_3[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}

// -----
// Ordered across sibling affine.if guards, naming the first conditional's token
// result.
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @channel_4 [1, 1] {broadcast_shape = [1, 2]}
  func.func @pos_ordered_across_affine_if() {
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
            %g0 = air.channel.get async [%t0] @channel_4[%x, %y] (%r0[] [] []) : (memref<8xi32, 2 : i32>)
            affine.yield %g0 : !air.async.token
          } else {
            affine.yield %w : !air.async.token
          }
          %i1 = affine.if #set()[%x, %y] -> !air.async.token {
            %g1 = air.channel.get async [%t1, %i0] @channel_4[%x, %y] (%r1[] [] []) : (memref<8xi32, 2 : i32>)
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

// -----
// Only one branch of a conditional ever runs, so there is no ordering to
// require. Demanding one would be unsatisfiable: the only value both endpoints
// could name is the enclosing affine.if's own result.
#set1 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @channel_5 [1, 1] {broadcast_shape = [1, 2]}
  func.func @pos_exclusive_branches() {
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
          %i0 = affine.if #set1()[%x, %y] -> !air.async.token {
            %g0 = air.channel.get async [%t0] @channel_5[%x, %y] (%r0[] [] []) : (memref<8xi32, 2 : i32>)
            affine.yield %g0 : !air.async.token
          } else {
            %g1 = air.channel.get async [%t1] @channel_5[%x, %y] (%r1[] [] []) : (memref<8xi32, 2 : i32>)
            affine.yield %g1 : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----
// Distinct loops are distinct scopes. The verifier has to match the emitter
// here, or it rejects IR the emitter is content to produce.
module {
  air.channel @channel_6 [1]
  func.func @pos_distinct_loops() {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) {
      air.segment @seg {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %l0 = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%it = %t0) -> (!air.async.token) {
          %g0 = air.channel.get async [%it] @channel_6[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %g0 : !air.async.token
        }
        %l1 = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%it = %t1) -> (!air.async.token) {
          %g1 = air.channel.get async [%it] @channel_6[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %g1 : !air.async.token
        }
      }
    }
    return
  }
}

// -----
// Endpoints under unrelated conditions. No ordering is required: on the arm
// that does not run, a conditional's token result carries whatever the other
// arm yields, which says nothing about the endpoint inside.
module {
  air.channel @channel_7 [1]
  func.func @pos_distinct_guards(%p0: i1, %p1: i1) {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) args(%lp0=%p0, %lp1=%p1) : i1, i1 {
      air.segment @seg args(%q0=%lp0, %q1=%lp1) : i1, i1 {
        %c0 = arith.constant 0 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %w = air.wait_all async
        %i0 = scf.if %q0 -> (!air.async.token) {
          %g0 = air.channel.get async [%t0] @channel_7[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %g0 : !air.async.token
        } else {
          scf.yield %w : !air.async.token
        }
        %i1 = scf.if %q1 -> (!air.async.token) {
          %g1 = air.channel.get async [%t1] @channel_7[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
          scf.yield %g1 : !air.async.token
        } else {
          scf.yield %w : !air.async.token
        }
      }
    }
    return
  }
}

// -----
// The path from the second endpoint to the first runs through the arm of an
// affine.if. Reading the branch op's dependency list alone would stop at the
// branch, since an affine.if's operands are index values.
#set2 = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @channel_8 [1, 1] {broadcast_shape = [1, 2]}
  func.func @pos_path_through_affine_if_arm() {
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
          %i0 = affine.if #set2()[%x, %y] -> !air.async.token {
            %g0 = air.channel.get async [%t0] @channel_8[%x, %y] (%r0[] [] []) : (memref<8xi32, 2 : i32>)
            affine.yield %g0 : !air.async.token
          } else {
            affine.yield %w : !air.async.token
          }
          %i1 = affine.if #set2()[%x, %y] -> !air.async.token {
            %j = air.wait_all async [%i0]
            %g1 = air.channel.get async [%t1, %j] @channel_8[%x, %y] (%r1[] [] []) : (memref<8xi32, 2 : i32>)
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

// -----
// The same shape, but every arm of the branch carries the earlier endpoint, so
// the later one is ordered after it whichever arm runs.
module {
  air.channel @channel_9 [1]
  func.func @pos_ordering_on_every_arm(%p: i1) {
    %c1 = arith.constant 1 : index
    air.launch (%a, %b) in (%ax=%c1, %ay=%c1) args(%lp=%p) : i1 {
      air.segment @seg args(%q=%lp) : i1 {
        %c0 = arith.constant 0 : index
        %t0, %r0 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %t1, %r1 = air.execute -> (memref<8xi32, 1 : i32>) {
          %m = memref.alloc() : memref<8xi32, 1 : i32>
          air.execute_terminator %m : memref<8xi32, 1 : i32>
        }
        %g0 = air.channel.get async [%t0] @channel_9[%c0] (%r0[] [] []) : (memref<8xi32, 1 : i32>)
        %i = scf.if %q -> (!air.async.token) {
          %j = air.wait_all async [%g0]
          scf.yield %j : !air.async.token
        } else {
          %k = air.wait_all async [%g0]
          scf.yield %k : !air.async.token
        }
        %g1 = air.channel.get async [%t1, %i] @channel_9[%c0] (%r1[] [] []) : (memref<8xi32, 1 : i32>)
      }
    }
    return
  }
}
