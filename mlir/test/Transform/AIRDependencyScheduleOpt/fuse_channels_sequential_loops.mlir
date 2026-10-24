//===- fuse_channels_sequential_loops.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels="sequential-loops=true" -split-input-file | FileCheck %s

// Two loops that run one after the other, each streaming from its own memtile
// buffer into the same two core buffers: one channel carries both loops'
// transfers. A third channel, into another core buffer, stays separate.

// CHECK-LABEL: func.func @two_loops
// CHECK-COUNT-2: air.channel.put {{.*}} @chan_a[]
// CHECK-COUNT-2: air.channel.put {{.*}} @chan_a[]
// CHECK: air.channel.put {{.*}} @chan_c[]
// CHECK: air.herd
// CHECK-COUNT-2: air.channel.get {{.*}} @chan_a[]
// CHECK-COUNT-2: air.channel.get {{.*}} @chan_a[]
// CHECK: air.channel.get {{.*}} @chan_c[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  air.channel @chan_c [1, 1]
  func.func @two_loops(%nq: index, %nb: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%aq=%nq, %ab=%nb) : index, index {
      %1 = air.segment @seg async args(%sq=%aq, %sb=%ab) : index, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %tq, %p = air.execute -> (memref<2x2560xi16, 1>) {
          %a = memref.alloc() : memref<2x2560xi16, 1>
          air.execute_terminator %a : memref<2x2560xi16, 1>
        }
        %tb, %z = air.execute -> (memref<8x2560xi16, 1>) {
          %a = memref.alloc() : memref<8x2560xi16, 1>
          air.execute_terminator %a : memref<8x2560xi16, 1>
        }
        %2 = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[0, 0] [1, 2560] [2560, 1]) : (memref<2x2560xi16, 1>)
          %p1 = air.channel.put async [%p0] @chan_a[] (%p[1, 0] [1, 2560] [2560, 1]) : (memref<2x2560xi16, 1>)
          scf.yield %p1 : !air.async.token
        }
        %3 = scf.for %k = %c0 to %sb step %c1_0 iter_args(%d = %tb) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%z[0, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
          %p1 = air.channel.put async [%p0] @chan_b[] (%z[1, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
          scf.yield %p1 : !air.async.token
        }
        %4 = air.channel.put async [%3] @chan_c[] (%z[2, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%hq=%sq, %hb=%sb) : index, index {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %ta, %ba = air.execute -> (memref<2560xi16, 2>) {
            %a = memref.alloc() : memref<2560xi16, 2>
            air.execute_terminator %a : memref<2560xi16, 2>
          }
          %tb2, %bb = air.execute -> (memref<2560xi16, 2>) {
            %a = memref.alloc() : memref<2560xi16, 2>
            air.execute_terminator %a : memref<2560xi16, 2>
          }
          %6 = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_a[] (%ba[] [] []) : (memref<2560xi16, 2>)
            %g1 = air.channel.get async [%g0] @chan_a[] (%bb[] [] []) : (memref<2560xi16, 2>)
            scf.yield %g1 : !air.async.token
          }
          %7 = scf.for %k = %c0_1 to %hb step %c1_1 iter_args(%d = %6) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_b[] (%ba[] [] []) : (memref<2560xi16, 2>)
            %g1 = air.channel.get async [%g0] @chan_b[] (%bb[] [] []) : (memref<2560xi16, 2>)
            scf.yield %g1 : !air.async.token
          }
          %8 = air.channel.get async [%7] @chan_c[] (%bb[] [] []) : (memref<2560xi16, 2>)
        }
      }
    }
    return
  }
}

// -----

// The memtile side runs chan_a's loop first, the core chan_b's: one channel
// would hand the core chan_a's data in chan_b's loop. The channels stay
// apart.

// CHECK-LABEL: func.func @reversed_order
// CHECK: air.channel.put {{.*}} @chan_a[]
// CHECK: air.channel.put {{.*}} @chan_b[]
// CHECK: air.herd
// CHECK: air.channel.get {{.*}} @chan_b[]
// CHECK: air.channel.get {{.*}} @chan_a[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  air.channel @chan_c [1, 1]
  func.func @reversed_order(%nq: index, %nb: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%aq=%nq, %ab=%nb) : index, index {
      %1 = air.segment @seg async args(%sq=%aq, %sb=%ab) : index, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %tq, %p = air.execute -> (memref<2x2560xi16, 1>) {
          %a = memref.alloc() : memref<2x2560xi16, 1>
          air.execute_terminator %a : memref<2x2560xi16, 1>
        }
        %tb, %z = air.execute -> (memref<8x2560xi16, 1>) {
          %a = memref.alloc() : memref<8x2560xi16, 1>
          air.execute_terminator %a : memref<8x2560xi16, 1>
        }
        %2 = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[0, 0] [1, 2560] [2560, 1]) : (memref<2x2560xi16, 1>)
          %p1 = air.channel.put async [%p0] @chan_a[] (%p[1, 0] [1, 2560] [2560, 1]) : (memref<2x2560xi16, 1>)
          scf.yield %p1 : !air.async.token
        }
        %3 = scf.for %k = %c0 to %sb step %c1_0 iter_args(%d = %tb) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%z[0, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
          %p1 = air.channel.put async [%p0] @chan_b[] (%z[1, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
          scf.yield %p1 : !air.async.token
        }
        %4 = air.channel.put async [%3] @chan_c[] (%z[2, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%hq=%sq, %hb=%sb) : index, index {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %ta, %ba = air.execute -> (memref<2560xi16, 2>) {
            %a = memref.alloc() : memref<2560xi16, 2>
            air.execute_terminator %a : memref<2560xi16, 2>
          }
          %tb2, %bb = air.execute -> (memref<2560xi16, 2>) {
            %a = memref.alloc() : memref<2560xi16, 2>
            air.execute_terminator %a : memref<2560xi16, 2>
          }
          %6 = scf.for %k = %c0_1 to %hb step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_b[] (%ba[] [] []) : (memref<2560xi16, 2>)
            %g1 = air.channel.get async [%g0] @chan_b[] (%bb[] [] []) : (memref<2560xi16, 2>)
            scf.yield %g1 : !air.async.token
          }
          %7 = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %6) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_a[] (%ba[] [] []) : (memref<2560xi16, 2>)
            %g1 = air.channel.get async [%g0] @chan_a[] (%bb[] [] []) : (memref<2560xi16, 2>)
            scf.yield %g1 : !air.async.token
          }
          %8 = air.channel.get async [%7] @chan_c[] (%bb[] [] []) : (memref<2560xi16, 2>)
        }
      }
    }
    return
  }
}

// -----

// chan_b's loop body was unrolled: its gets go through the same two buffers
// in the same order, twice per trip. The channels merge.

// CHECK-LABEL: func.func @unrolled_rotation
// CHECK-NOT: @chan_b
// CHECK: air.channel.get {{.*}} @chan_c[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  air.channel @chan_c [1, 1]
  func.func @unrolled_rotation(%nq: index, %nb: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%aq=%nq, %ab=%nb) : index, index {
      %1 = air.segment @seg async args(%sq=%aq, %sb=%ab) : index, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %tq, %p = air.execute -> (memref<2x2560xi16, 1>) {
          %a = memref.alloc() : memref<2x2560xi16, 1>
          air.execute_terminator %a : memref<2x2560xi16, 1>
        }
        %tb, %z = air.execute -> (memref<8x2560xi16, 1>) {
          %a = memref.alloc() : memref<8x2560xi16, 1>
          air.execute_terminator %a : memref<8x2560xi16, 1>
        }
        %2 = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[0, 0] [1, 2560] [2560, 1]) : (memref<2x2560xi16, 1>)
          %p1 = air.channel.put async [%p0] @chan_a[] (%p[1, 0] [1, 2560] [2560, 1]) : (memref<2x2560xi16, 1>)
          scf.yield %p1 : !air.async.token
        }
        %3 = scf.for %k = %c0 to %sb step %c1_0 iter_args(%d = %tb) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%z[0, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
          %p1 = air.channel.put async [%p0] @chan_b[] (%z[1, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
          %p2 = air.channel.put async [%p1] @chan_b[] (%z[3, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
          %p3 = air.channel.put async [%p2] @chan_b[] (%z[4, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
          scf.yield %p3 : !air.async.token
        }
        %4 = air.channel.put async [%3] @chan_c[] (%z[2, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%hq=%sq, %hb=%sb) : index, index {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %ta, %ba = air.execute -> (memref<2560xi16, 2>) {
            %a = memref.alloc() : memref<2560xi16, 2>
            air.execute_terminator %a : memref<2560xi16, 2>
          }
          %tb2, %bb = air.execute -> (memref<2560xi16, 2>) {
            %a = memref.alloc() : memref<2560xi16, 2>
            air.execute_terminator %a : memref<2560xi16, 2>
          }
          %6 = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_a[] (%ba[] [] []) : (memref<2560xi16, 2>)
            %g1 = air.channel.get async [%g0] @chan_a[] (%bb[] [] []) : (memref<2560xi16, 2>)
            scf.yield %g1 : !air.async.token
          }
          %7 = scf.for %k = %c0_1 to %hb step %c1_1 iter_args(%d = %6) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_b[] (%ba[] [] []) : (memref<2560xi16, 2>)
            %g1 = air.channel.get async [%g0] @chan_b[] (%bb[] [] []) : (memref<2560xi16, 2>)
            %g2 = air.channel.get async [%g1] @chan_b[] (%ba[] [] []) : (memref<2560xi16, 2>)
            %g3 = air.channel.get async [%g2] @chan_b[] (%bb[] [] []) : (memref<2560xi16, 2>)
            scf.yield %g3 : !air.async.token
          }
          %8 = air.channel.get async [%7] @chan_c[] (%bb[] [] []) : (memref<2560xi16, 2>)
        }
      }
    }
    return
  }
}

// -----

// chan_b's loop fills the core's buffers in the other order. The channels
// stay apart.

// CHECK-LABEL: func.func @swapped_rotation
// CHECK: air.herd
// CHECK: air.channel.get {{.*}} @chan_a[]
// CHECK: air.channel.get {{.*}} @chan_b[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  air.channel @chan_c [1, 1]
  func.func @swapped_rotation(%nq: index, %nb: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%aq=%nq, %ab=%nb) : index, index {
      %1 = air.segment @seg async args(%sq=%aq, %sb=%ab) : index, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %tq, %p = air.execute -> (memref<2x2560xi16, 1>) {
          %a = memref.alloc() : memref<2x2560xi16, 1>
          air.execute_terminator %a : memref<2x2560xi16, 1>
        }
        %tb, %z = air.execute -> (memref<8x2560xi16, 1>) {
          %a = memref.alloc() : memref<8x2560xi16, 1>
          air.execute_terminator %a : memref<8x2560xi16, 1>
        }
        %2 = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[0, 0] [1, 2560] [2560, 1]) : (memref<2x2560xi16, 1>)
          %p1 = air.channel.put async [%p0] @chan_a[] (%p[1, 0] [1, 2560] [2560, 1]) : (memref<2x2560xi16, 1>)
          scf.yield %p1 : !air.async.token
        }
        %3 = scf.for %k = %c0 to %sb step %c1_0 iter_args(%d = %tb) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%z[0, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
          %p1 = air.channel.put async [%p0] @chan_b[] (%z[1, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
          scf.yield %p1 : !air.async.token
        }
        %4 = air.channel.put async [%3] @chan_c[] (%z[2, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%hq=%sq, %hb=%sb) : index, index {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %ta, %ba = air.execute -> (memref<2560xi16, 2>) {
            %a = memref.alloc() : memref<2560xi16, 2>
            air.execute_terminator %a : memref<2560xi16, 2>
          }
          %tb2, %bb = air.execute -> (memref<2560xi16, 2>) {
            %a = memref.alloc() : memref<2560xi16, 2>
            air.execute_terminator %a : memref<2560xi16, 2>
          }
          %6 = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_a[] (%ba[] [] []) : (memref<2560xi16, 2>)
            %g1 = air.channel.get async [%g0] @chan_a[] (%bb[] [] []) : (memref<2560xi16, 2>)
            scf.yield %g1 : !air.async.token
          }
          %7 = scf.for %k = %c0_1 to %hb step %c1_1 iter_args(%d = %6) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_b[] (%bb[] [] []) : (memref<2560xi16, 2>)
            %g1 = air.channel.get async [%g0] @chan_b[] (%ba[] [] []) : (memref<2560xi16, 2>)
            scf.yield %g1 : !air.async.token
          }
          %8 = air.channel.get async [%7] @chan_c[] (%bb[] [] []) : (memref<2560xi16, 2>)
        }
      }
    }
    return
  }
}

// -----

// Three loops into one core buffer, run A, B, C on the memtile side and A, C,
// B on the core. B joins A; C is then ordered against A and B together and
// stays apart.

// CHECK-LABEL: func.func @three_loops
// CHECK: air.channel.put {{.*}} @chan_a[]
// CHECK: air.channel.put {{.*}} @chan_a[]
// CHECK: air.channel.put {{.*}} @chan_c[]
// CHECK: air.herd
// CHECK: air.channel.get {{.*}} @chan_a[]
// CHECK: air.channel.get {{.*}} @chan_c[]
// CHECK: air.channel.get {{.*}} @chan_a[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  air.channel @chan_c [1, 1]
  func.func @three_loops(%nq: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%aq=%nq) : index {
      %1 = air.segment @seg async args(%sq=%aq) : index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %tb, %z = air.execute -> (memref<8x2560xi16, 1>) {
          %a = memref.alloc() : memref<8x2560xi16, 1>
          air.execute_terminator %a : memref<8x2560xi16, 1>
        }
        %pa = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %tb) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%z[0, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %pb = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %pa) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%z[1, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %pc = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %pb) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_c[] (%z[2, 0] [1, 2560] [2560, 1]) : (memref<8x2560xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%hq=%sq) : index {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %ta, %ba = air.execute -> (memref<2560xi16, 2>) {
            %a = memref.alloc() : memref<2560xi16, 2>
            air.execute_terminator %a : memref<2560xi16, 2>
          }
          %ga = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_a[] (%ba[] [] []) : (memref<2560xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
          %gc = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %ga) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_c[] (%ba[] [] []) : (memref<2560xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
          %gb = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %gc) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_b[] (%ba[] [] []) : (memref<2560xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// Two loops bring buffers A and B from L3 into L2 at the same time. Neither
// the launch's puts nor the memtile's gets wait for the other loop. A single
// channel would have no fixed order for the two loops' transfers, so the
// channels are not fused.

// CHECK-LABEL: func.func @concurrent_loops
// CHECK: air.channel.put {{.*}} @chan_a[]
// CHECK: air.channel.put {{.*}} @chan_b[]
// CHECK: air.channel.get {{.*}} @chan_a[]
// CHECK: air.channel.get {{.*}} @chan_b[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @concurrent_loops(%in_a: memref<4x2560xi16>, %in_b: memref<4x2560xi16>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%la=%in_a, %lb=%in_b, %ln=%n) : memref<4x2560xi16>, memref<4x2560xi16>, index {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2560 = arith.constant 2560 : index
      %w = air.wait_all async
      %put_a = scf.for %i = %c0 to %ln step %c1_0 iter_args(%d = %w) -> (!air.async.token) {
        %p = air.channel.put async [%d] @chan_a[] (%la[%c0, %c0] [%c1_0, %c2560] [%c2560, %c1_0]) : (memref<4x2560xi16>)
        scf.yield %p : !air.async.token
      }
      %put_b = scf.for %i = %c0 to %ln step %c1_0 iter_args(%d = %w) -> (!air.async.token) {
        %p = air.channel.put async [%d] @chan_b[] (%lb[%c0, %c0] [%c1_0, %c2560] [%c2560, %c1_0]) : (memref<4x2560xi16>)
        scf.yield %p : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_1 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %ta, %buf_a = air.execute -> (memref<2560xi16, 1>) {
          %a = memref.alloc() : memref<2560xi16, 1>
          air.execute_terminator %a : memref<2560xi16, 1>
        }
        %tb, %buf_b = air.execute -> (memref<2560xi16, 1>) {
          %a = memref.alloc() : memref<2560xi16, 1>
          air.execute_terminator %a : memref<2560xi16, 1>
        }
        %get_a = scf.for %i = %c0_1 to %sn step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
          %g = air.channel.get async [%d] @chan_a[] (%buf_a[] [] []) : (memref<2560xi16, 1>)
          scf.yield %g : !air.async.token
        }
        %get_b = scf.for %i = %c0_1 to %sn step %c1_1 iter_args(%d = %tb) -> (!air.async.token) {
          %g = air.channel.get async [%d] @chan_b[] (%buf_b[] [] []) : (memref<2560xi16, 1>)
          scf.yield %g : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// The same two loops, but the launch starts the puts of B only after the
// puts of A are done. The memtile's gets still run concurrently. The fused
// channel carries all transfers of A and then all of B, the order in which
// the launch sends them.

// CHECK-LABEL: func.func @dependent_loops
// CHECK: air.channel.put {{.*}} @chan_a[]
// CHECK: air.channel.put {{.*}} @chan_a[]
// CHECK: air.channel.get {{.*}} @chan_a[]
// CHECK: air.channel.get {{.*}} @chan_a[]
// CHECK-NOT: @chan_b[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @dependent_loops(%in_a: memref<4x2560xi16>, %in_b: memref<4x2560xi16>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%la=%in_a, %lb=%in_b, %ln=%n) : memref<4x2560xi16>, memref<4x2560xi16>, index {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2560 = arith.constant 2560 : index
      %w = air.wait_all async
      %put_a = scf.for %i = %c0 to %ln step %c1_0 iter_args(%d = %w) -> (!air.async.token) {
        %p = air.channel.put async [%d] @chan_a[] (%la[%c0, %c0] [%c1_0, %c2560] [%c2560, %c1_0]) : (memref<4x2560xi16>)
        scf.yield %p : !air.async.token
      }
      %put_b = scf.for %i = %c0 to %ln step %c1_0 iter_args(%d = %put_a) -> (!air.async.token) {
        %p = air.channel.put async [%d] @chan_b[] (%lb[%c0, %c0] [%c1_0, %c2560] [%c2560, %c1_0]) : (memref<4x2560xi16>)
        scf.yield %p : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_1 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %ta, %buf_a = air.execute -> (memref<2560xi16, 1>) {
          %a = memref.alloc() : memref<2560xi16, 1>
          air.execute_terminator %a : memref<2560xi16, 1>
        }
        %tb, %buf_b = air.execute -> (memref<2560xi16, 1>) {
          %a = memref.alloc() : memref<2560xi16, 1>
          air.execute_terminator %a : memref<2560xi16, 1>
        }
        %get_a = scf.for %i = %c0_1 to %sn step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
          %g = air.channel.get async [%d] @chan_a[] (%buf_a[] [] []) : (memref<2560xi16, 1>)
          scf.yield %g : !air.async.token
        }
        %get_b = scf.for %i = %c0_1 to %sn step %c1_1 iter_args(%d = %tb) -> (!air.async.token) {
          %g = air.channel.get async [%d] @chan_b[] (%buf_b[] [] []) : (memref<2560xi16, 1>)
          scf.yield %g : !air.async.token
        }
      }
    }
    return
  }
}

// -----

// As @concurrent_loops, but the launch has no put loop for B. With no puts
// to order against, the order of the gets is the only one, so the channels
// are fused.

// CHECK-LABEL: func.func @loops_without_puts
// CHECK: air.channel.put {{.*}} @chan_a[]
// CHECK: air.channel.get {{.*}} @chan_a[]
// CHECK: air.channel.get {{.*}} @chan_a[]
// CHECK-NOT: @chan_b[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @loops_without_puts(%in_a: memref<4x2560xi16>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%la=%in_a, %ln=%n) : memref<4x2560xi16>, index {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2560 = arith.constant 2560 : index
      %w = air.wait_all async
      %put_a = scf.for %i = %c0 to %ln step %c1_0 iter_args(%d = %w) -> (!air.async.token) {
        %p = air.channel.put async [%d] @chan_a[] (%la[%c0, %c0] [%c1_0, %c2560] [%c2560, %c1_0]) : (memref<4x2560xi16>)
        scf.yield %p : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_1 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %ta, %buf_a = air.execute -> (memref<2560xi16, 1>) {
          %a = memref.alloc() : memref<2560xi16, 1>
          air.execute_terminator %a : memref<2560xi16, 1>
        }
        %tb, %buf_b = air.execute -> (memref<2560xi16, 1>) {
          %a = memref.alloc() : memref<2560xi16, 1>
          air.execute_terminator %a : memref<2560xi16, 1>
        }
        %get_a = scf.for %i = %c0_1 to %sn step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
          %g = air.channel.get async [%d] @chan_a[] (%buf_a[] [] []) : (memref<2560xi16, 1>)
          scf.yield %g : !air.async.token
        }
        %get_b = scf.for %i = %c0_1 to %sn step %c1_1 iter_args(%d = %tb) -> (!air.async.token) {
          %g = air.channel.get async [%d] @chan_b[] (%buf_b[] [] []) : (memref<2560xi16, 1>)
          scf.yield %g : !air.async.token
        }
      }
    }
    return
  }
}
