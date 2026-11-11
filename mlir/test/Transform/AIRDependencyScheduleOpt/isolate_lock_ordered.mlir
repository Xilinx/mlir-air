//===- isolate_lock_ordered.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-isolate-async-dma-loop-nests --split-input-file | FileCheck %s

// The loops created by splitting a loop are not ordered against each other.
// An op may be separated from a loop it depends on when both only move data
// through channels and the dependency is on an L2 buffer, whose locks keep the
// order. Other dependencies keep the ops in one loop.

// A: puts that read an L2 buffer depend on the loop that fills it. Each goes
// to its own loop, and each put waits on its own loop's iteration token.
// CHECK-LABEL: func.func @lock_ordered_split
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}@fill
// CHECK: scf.for {{.*}} iter_args(%[[IT0:.*]] = %{{.*}})
// CHECK-NEXT: air.channel.put async [%[[IT0]]]  @out0
// CHECK: scf.for {{.*}} iter_args(%[[IT1:.*]] = %{{.*}})
// CHECK-NEXT: air.channel.put async [%[[IT1]]]  @out1

module {
  air.channel @fill [1, 1]
  air.channel @out0 [1, 1]
  air.channel @out1 [1, 1]
  func.func @lock_ordered_split() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%a, %b) in (%sa=%c1, %sb=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %t0, %buf = air.execute -> (memref<2x64xbf16, 1>) {
          %m = memref.alloc() : memref<2x64xbf16, 1>
          air.execute_terminator %m : memref<2x64xbf16, 1>
        }
        %2 = scf.for %k = %c0 to %c4 step %c1_0 iter_args(%it = %t0) -> (!air.async.token) {
          %f = scf.for %h = %c0 to %c2 step %c1_0 iter_args(%t = %it) -> (!air.async.token) {
            %g = air.channel.get async [%t] @fill[%c0, %c0] (%buf[%h, %c0] [%c1_0, %c4] [%c4, %c1_0]) : (memref<2x64xbf16, 1>)
            scf.yield %g : !air.async.token
          }
          %p0 = air.channel.put async [%f] @out0[%c0, %c0] (%buf[%c0, %c0] [%c1_0, %c4] [%c4, %c1_0]) : (memref<2x64xbf16, 1>)
          %p1 = air.channel.put async [%f] @out1[%c0, %c0] (%buf[%c1_0, %c0] [%c1_0, %c4] [%c4, %c1_0]) : (memref<2x64xbf16, 1>)
          %w = air.wait_all async [%p0, %p1]
          scf.yield %w : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// B: an execute writes an L2 buffer that a get also writes, with no token
// between them. The execute is not a data mover, so they stay in one loop.
// CHECK-LABEL: func.func @non_mover_conflict
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}@fill
// CHECK-NOT: scf.for
// CHECK: linalg.fill

module {
  air.channel @fill [1, 1]
  func.func @non_mover_conflict() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%a, %b) in (%sa=%c1, %sb=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %cst = arith.constant 0.0 : bf16
        %t0, %buf = air.execute -> (memref<64xbf16, 1>) {
          %m = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %m : memref<64xbf16, 1>
        }
        %2 = scf.for %k = %c0 to %c4 step %c1_0 iter_args(%it = %t0) -> (!air.async.token) {
          %g = air.channel.get async [%it] @fill[%c0, %c0] (%buf[] [] []) : (memref<64xbf16, 1>)
          %e = air.execute [%it] {
            linalg.fill ins(%cst : bf16) outs(%buf : memref<64xbf16, 1>)
          }
          %w = air.wait_all async [%g, %e]
          scf.yield %w : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// C: two writers of one L2 buffer, ordered by a token. Locks do not order two
// writers, so they stay in one loop.
// CHECK-LABEL: func.func @two_writers
// CHECK: scf.for
// CHECK-NEXT: %[[G0:.*]] = air.channel.get{{.*}}@in0
// CHECK-NEXT: air.channel.get async [%[[G0]]]  @in1

module {
  air.channel @in0 [1, 1]
  air.channel @in1 [1, 1]
  func.func @two_writers() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%a, %b) in (%sa=%c1, %sb=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %t0, %buf = air.execute -> (memref<64xbf16, 1>) {
          %m = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %m : memref<64xbf16, 1>
        }
        %2 = scf.for %k = %c0 to %c4 step %c1_0 iter_args(%it = %t0) -> (!air.async.token) {
          %g0 = air.channel.get async [%it] @in0[%c0, %c0] (%buf[] [] []) : (memref<64xbf16, 1>)
          %g1 = air.channel.get async [%g0] @in1[%c0, %c0] (%buf[] [] []) : (memref<64xbf16, 1>)
          scf.yield %g1 : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// D: a read of L3 memory depends on a write to it. Only the token orders
// them, so they stay in one loop.
// CHECK-LABEL: func.func @host_order
// CHECK: scf.for
// CHECK-NEXT: %[[G:.*]] = air.channel.get{{.*}}@drain
// CHECK-NEXT: air.channel.put async [%[[G]]]  @feed

module {
  air.channel @drain [1, 1]
  air.channel @feed [1, 1]
  func.func @host_order(%arg0: memref<64xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %t = air.wait_all async
    %2 = scf.for %k = %c0 to %c4 step %c1 iter_args(%it = %t) -> (!air.async.token) {
      %g = air.channel.get async [%it] @drain[%c0, %c0] (%arg0[] [] []) : (memref<64xbf16>)
      %p = air.channel.put async [%g] @feed[%c0, %c0] (%arg0[] [] []) : (memref<64xbf16>)
      scf.yield %p : !air.async.token
    }
    return
  }
}

// -----

// E: the put depends on the fill loop through a use inside its own loop, not
// through that loop's init. The dependency is still seen, the two are split,
// the use inside the loop is rewritten, and the get is not cloned into the
// put's loop.
// CHECK-LABEL: func.func @nested_use
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}@fill
// CHECK-NOT: air.channel.get{{.*}}@fill
// CHECK: scf.for {{.*}} iter_args(%[[IT:.*]] = %{{.*}})
// CHECK: scf.for
// CHECK: air.channel.put async [%[[IT]]]  @out

module {
  air.channel @fill [1, 1]
  air.channel @out [1, 1]
  func.func @nested_use() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%a, %b) in (%sa=%c1, %sb=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %t0, %buf = air.execute -> (memref<64xbf16, 1>) {
          %m = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %m : memref<64xbf16, 1>
        }
        %2 = scf.for %k = %c0 to %c4 step %c1_0 iter_args(%it = %t0) -> (!air.async.token) {
          %g = scf.for %h = %c0 to %c4 step %c1_0 iter_args(%t = %it) -> (!air.async.token) {
            %off = arith.muli %h, %c16 : index
            %f = air.channel.get async [%t] @fill[%c0, %c0] (%buf[%off] [%c16] [%c1_0]) : (memref<64xbf16, 1>)
            scf.yield %f : !air.async.token
          }
          %w = air.wait_all async [%it]
          %p = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%t = %w) -> (!air.async.token) {
            %off = arith.muli %i, %c16 : index
            %put = air.channel.put async [%g] @out[%c0, %c0] (%buf[%off] [%c16] [%c1_0]) : (memref<64xbf16, 1>)
            scf.yield %put : !air.async.token
          }
          %y = air.wait_all async [%g, %p]
          scf.yield %y : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// F: a get into an L2 buffer and a put that forwards it, the put depending on
// the get directly. They stay in one loop.
// CHECK-LABEL: func.func @relay_stays_together
// CHECK: scf.for
// CHECK-NEXT: air.channel.get{{.*}}@in
// CHECK-NEXT: air.channel.put{{.*}}@out
// CHECK-NOT: scf.for

module {
  air.channel @in [1, 1]
  air.channel @out [1, 1]
  func.func @relay_stays_together() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%a, %b) in (%sa=%c1, %sb=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %t0, %buf = air.execute -> (memref<64xbf16, 1>) {
          %m = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %m : memref<64xbf16, 1>
        }
        %2 = scf.for %k = %c0 to %c4 step %c1_0 iter_args(%it = %t0) -> (!air.async.token) {
          %g = air.channel.get async [%it] @in[%c0, %c0] (%buf[] [] []) : (memref<64xbf16, 1>)
          %p = air.channel.put async [%g] @out[%c0, %c0] (%buf[] [] []) : (memref<64xbf16, 1>)
          scf.yield %p : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// G: the L2 buffer is allocated inside the loop and used only inside the
// regions of the fill loop and of the loop around the put. After the split
// both loops must use one buffer, so its alloc is hoisted out of the loop
// instead of being cloned into each new loop.
// CHECK-LABEL: func.func @shared_inner_alloc
// CHECK: air.segment
// CHECK-COUNT-1: memref.alloc() : memref<64xbf16, 1>
// CHECK-NOT: memref.alloc() : memref<64xbf16, 1>
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}@fill
// CHECK: scf.for
// CHECK: air.channel.put{{.*}}@out

module {
  air.channel @fill [1, 1]
  air.channel @out [1, 1]
  func.func @shared_inner_alloc() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%a, %b) in (%sa=%c1, %sb=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %c32 = arith.constant 32 : index
        %w0 = air.wait_all async
        %2 = scf.for %k = %c0 to %c4 step %c1_0 iter_args(%it = %w0) -> (!air.async.token) {
          %t0, %buf = air.execute [%it] -> (memref<64xbf16, 1>) {
            %m = memref.alloc() : memref<64xbf16, 1>
            air.execute_terminator %m : memref<64xbf16, 1>
          }
          %f = scf.for %h = %c0 to %c2 step %c1_0 iter_args(%t = %t0) -> (!air.async.token) {
            %off = arith.muli %h, %c32 : index
            %g = air.channel.get async [%t] @fill[%c0, %c0] (%buf[%off] [%c32] [%c1_0]) : (memref<64xbf16, 1>)
            scf.yield %g : !air.async.token
          }
          %p = scf.for %h = %c0 to %c1_0 step %c1_0 iter_args(%t = %f) -> (!air.async.token) {
            %q = air.channel.put async [%t] @out[%c0, %c0] (%buf[] [] []) : (memref<64xbf16, 1>)
            scf.yield %q : !air.async.token
          }
          %d = air.execute [%p] {
            memref.dealloc %buf : memref<64xbf16, 1>
          }
          scf.yield %d : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// H: the loop around the put takes an index from the fill loop as well as its
// token. A non-token value orders them, so they stay in one loop.
// CHECK-LABEL: func.func @non_token_result
// CHECK: scf.for
// CHECK: %[[F:.*]]:2 = scf.for
// CHECK: air.channel.get{{.*}}@fill
// CHECK-NOT: air.channel.get
// CHECK: scf.for {{.*}} = %[[F]]#1
// CHECK: air.channel.put{{.*}}@out
// CHECK-NOT: air.channel.get{{.*}}@fill

module {
  air.channel @fill [1, 1]
  air.channel @out [1, 1]
  func.func @non_token_result() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%a, %b) in (%sa=%c1, %sb=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %c32 = arith.constant 32 : index
        %t0, %buf = air.execute -> (memref<64xbf16, 1>) {
          %m = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %m : memref<64xbf16, 1>
        }
        %2 = scf.for %k = %c0 to %c4 step %c1_0 iter_args(%it = %t0) -> (!air.async.token) {
          %f:2 = scf.for %h = %c0 to %c2 step %c1_0 iter_args(%t = %it, %n = %c0) -> (!air.async.token, index) {
            %off = arith.muli %h, %c32 : index
            %g = air.channel.get async [%t] @fill[%c0, %c0] (%buf[%off] [%c32] [%c1_0]) : (memref<64xbf16, 1>)
            %n1 = arith.addi %n, %c1_0 : index
            scf.yield %g, %n1 : !air.async.token, index
          }
          %p = scf.for %h = %f#1 to %c2 step %c1_0 iter_args(%t = %f#0) -> (!air.async.token) {
            %q = air.channel.put async [%t] @out[%c0, %c0] (%buf[] [] []) : (memref<64xbf16, 1>)
            scf.yield %q : !air.async.token
          }
          scf.yield %p : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// I: the fill loop also carries an index initialized outside the loop body.
// When the put's loop is split from it, only tokens go into the put loop's
// rebuilt dependencies.
// CHECK-LABEL: func.func @index_init_arg
// CHECK: scf.for
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}@fill
// CHECK: scf.for {{.*}} iter_args(%[[IT:.*]] = %{{.*}}) -> (!air.async.token)
// CHECK: air.channel.put async [%[[IT]]]  @out

module {
  air.channel @fill [1, 1]
  air.channel @out [1, 1]
  func.func @index_init_arg() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%a, %b) in (%sa=%c1, %sb=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %c32 = arith.constant 32 : index
        %t0, %buf = air.execute -> (memref<64xbf16, 1>) {
          %m = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %m : memref<64xbf16, 1>
        }
        %2 = scf.for %k = %c0 to %c4 step %c1_0 iter_args(%it = %t0) -> (!air.async.token) {
          %f:2 = scf.for %h = %c0 to %c2 step %c1_0 iter_args(%t = %it, %n = %c0) -> (!air.async.token, index) {
            %off = arith.muli %n, %c32 : index
            %g = air.channel.get async [%t] @fill[%c0, %c0] (%buf[%off] [%c32] [%c1_0]) : (memref<64xbf16, 1>)
            %n1 = arith.addi %n, %c1_0 : index
            scf.yield %g, %n1 : !air.async.token, index
          }
          %w = air.wait_all async [%f#0]
          %p = scf.for %h = %c0 to %c1_0 step %c1_0 iter_args(%t = %w) -> (!air.async.token) {
            %q = air.channel.put async [%t] @out[%c0, %c0] (%buf[] [] []) : (memref<64xbf16, 1>)
            scf.yield %q : !air.async.token
          }
          scf.yield %p : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// J: four cores write parts of an L2 buffer through an scf.parallel, and a put
// sends the whole buffer on. The buffer has several writers, so the gather and
// the put stay in one loop.
// CHECK-LABEL: func.func @gather_then_put
// CHECK: scf.for
// CHECK: scf.parallel
// CHECK: air.channel.get{{.*}}@gather
// CHECK-NOT: scf.for
// CHECK: air.channel.put{{.*}}@out

module {
  air.channel @gather [1, 4]
  air.channel @out [1, 1]
  func.func @gather_then_put() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%a, %b) in (%sa=%c1, %sb=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c4 = arith.constant 4 : index
        %c16 = arith.constant 16 : index
        %t0, %buf = air.execute -> (memref<64xbf16, 1>) {
          %m = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %m : memref<64xbf16, 1>
        }
        %2 = scf.for %k = %c0 to %c4 step %c1_0 iter_args(%it = %t0) -> (!air.async.token) {
          %g = scf.parallel (%i) = (%c0) to (%c4) step (%c1_0) init (%it) -> !air.async.token {
            %off = arith.muli %i, %c16 : index
            %get = air.channel.get async [%it] @gather[%c0, %i] (%buf[%off] [%c16] [%c1_0]) : (memref<64xbf16, 1>)
            scf.reduce(%get : !air.async.token) {
            ^bb0(%l: !air.async.token, %r: !air.async.token):
              %j = air.wait_all async [%l, %r]
              scf.reduce.return %j : !air.async.token
            }
          }
          %p = air.channel.put async [%g] @out[%c0, %c0] (%buf[] [] []) : (memref<64xbf16, 1>)
          scf.yield %p : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// J: a join: two fill loops write the two halves of an L2 buffer, and a put
// reads it whole. Locks do not order the two writers, so the buffer's lock
// counts come from the body they share: the three stay in one loop.
// CHECK-LABEL: func.func @join_stays
// CHECK: scf.for
// CHECK: %[[F0:.*]] = scf.for
// CHECK: air.channel.get{{.*}}@fill[%c0, %c0]
// CHECK: %[[F1:.*]] = scf.for
// CHECK: air.channel.get{{.*}}@fill[%c0, %c1
// CHECK: air.channel.put async [%[[F0]], %[[F1]]]  @out
// CHECK-NOT: scf.for

module {
  air.channel @fill [1, 2]
  air.channel @out [1, 1]
  func.func @join_stays() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%a, %b) in (%sa=%c1, %sb=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c4 = arith.constant 4 : index
        %t0, %buf = air.execute -> (memref<2x64xbf16, 1>) {
          %m = memref.alloc() : memref<2x64xbf16, 1>
          air.execute_terminator %m : memref<2x64xbf16, 1>
        }
        %2 = scf.for %k = %c0 to %c4 step %c1_0 iter_args(%it = %t0) -> (!air.async.token) {
          %f0 = scf.for %h = %c0 to %c2 step %c1_0 iter_args(%t = %it) -> (!air.async.token) {
            %g = air.channel.get async [%t] @fill[%c0, %c0] (%buf[%c0, %h] [%c1_0, %c4] [%c4, %c1_0]) : (memref<2x64xbf16, 1>)
            scf.yield %g : !air.async.token
          }
          %f1 = scf.for %h = %c0 to %c2 step %c1_0 iter_args(%t = %it) -> (!air.async.token) {
            %g = air.channel.get async [%t] @fill[%c0, %c1_0] (%buf[%c1_0, %h] [%c1_0, %c4] [%c4, %c1_0]) : (memref<2x64xbf16, 1>)
            scf.yield %g : !air.async.token
          }
          %p = air.channel.put async [%f0, %f1] @out[%c0, %c0] (%buf[] [] []) : (memref<2x64xbf16, 1>)
          scf.yield %p : !air.async.token
        }
      }
    }
    return
  }
}
