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

// The loops stream from L2 buffers pinned to different memtiles, which move
// them on their own DMA channels. The channels stay apart.

// CHECK-LABEL: func.func @pinned_apart
// CHECK: air.channel.put {{.*}} @chan_a[]
// CHECK: air.channel.put {{.*}} @chan_b[]
// CHECK: air.herd
// CHECK: air.channel.get {{.*}} @chan_a[]
// CHECK: air.channel.get {{.*}} @chan_b[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @pinned_apart(%nq: index, %nb: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%aq=%nq, %ab=%nb) : index, index {
      %1 = air.segment @seg async args(%sq=%aq, %sb=%ab) : index, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %tq, %p = air.execute -> (memref<16xi16, 1>) {
          %a = memref.alloc() {air.memtile_col = 0 : i32} : memref<16xi16, 1>
          air.execute_terminator %a : memref<16xi16, 1>
        }
        %tz, %z = air.execute -> (memref<16xi16, 1>) {
          %a = memref.alloc() {air.memtile_col = 1 : i32} : memref<16xi16, 1>
          air.execute_terminator %a : memref<16xi16, 1>
        }
        %2 = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[] [] []) : (memref<16xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %3 = scf.for %k = %c0 to %sb step %c1_0 iter_args(%d = %tz) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%z[] [] []) : (memref<16xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%hq=%sq, %hb=%sb) : index, index {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %ta, %ba = air.execute -> (memref<16xi16, 2>) {
            %a = memref.alloc() : memref<16xi16, 2>
            air.execute_terminator %a : memref<16xi16, 2>
          }
          %6 = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_a[] (%ba[] [] []) : (memref<16xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
          %7 = scf.for %k = %c0_1 to %hb step %c1_1 iter_args(%d = %6) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_b[] (%ba[] [] []) : (memref<16xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// As above, with both L2 buffers pinned to the same memtile. The channels
// merge.

// CHECK-LABEL: func.func @pinned_together
// CHECK-NOT: @chan_b
// CHECK: air.herd
// CHECK-COUNT-2: air.channel.get {{.*}} @chan_a[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @pinned_together(%nq: index, %nb: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%aq=%nq, %ab=%nb) : index, index {
      %1 = air.segment @seg async args(%sq=%aq, %sb=%ab) : index, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %tq, %p = air.execute -> (memref<16xi16, 1>) {
          %a = memref.alloc() {air.memtile_col = 0 : i32} : memref<16xi16, 1>
          air.execute_terminator %a : memref<16xi16, 1>
        }
        %tz, %z = air.execute -> (memref<16xi16, 1>) {
          %a = memref.alloc() {air.memtile_col = 0 : i32} : memref<16xi16, 1>
          air.execute_terminator %a : memref<16xi16, 1>
        }
        %2 = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[] [] []) : (memref<16xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %3 = scf.for %k = %c0 to %sb step %c1_0 iter_args(%d = %tz) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%z[] [] []) : (memref<16xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%hq=%sq, %hb=%sb) : index, index {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %ta, %ba = air.execute -> (memref<16xi16, 2>) {
            %a = memref.alloc() : memref<16xi16, 2>
            air.execute_terminator %a : memref<16xi16, 2>
          }
          %6 = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_a[] (%ba[] [] []) : (memref<16xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
          %7 = scf.for %k = %c0_1 to %hb step %c1_1 iter_args(%d = %6) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_b[] (%ba[] [] []) : (memref<16xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// The core's gets move the same elements of the same buffer, spelled with a
// unit dimension in one loop and not in the other, from offsets computed
// alike in each loop. The channels merge.

// CHECK-LABEL: func.func @same_elements_spelled_apart
// CHECK-NOT: @chan_b
// CHECK: air.channel.get {{.*}} @chan_a[] (%{{.*}}[%{{.*}}, %{{.*}}] [2, 8] [8, 1])
// CHECK: air.channel.get {{.*}} @chan_a[] (%{{.*}}[%{{.*}}, %{{.*}}, %{{.*}}] [2, 1, 8] [8, 8, 1])

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @same_elements_spelled_apart(%nq: index, %nb: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%aq=%nq, %ab=%nb) : index, index {
      %1 = air.segment @seg async args(%sq=%aq, %sb=%ab) : index, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %tq, %p = air.execute -> (memref<16xi16, 1>) {
          %a = memref.alloc() : memref<16xi16, 1>
          air.execute_terminator %a : memref<16xi16, 1>
        }
        %2 = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[] [] []) : (memref<16xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %3 = scf.for %k = %c0 to %sb step %c1_0 iter_args(%d = %2) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%p[] [] []) : (memref<16xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%hq=%sq, %hb=%sb) : index, index {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %ta, %ba = air.execute -> (memref<8x8xi16, 2>) {
            %a = memref.alloc() : memref<8x8xi16, 2>
            air.execute_terminator %a : memref<8x8xi16, 2>
          }
          %6 = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %o = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%y]
            %g0 = air.channel.get async [%d] @chan_a[] (%ba[%o, %c0_1] [2, 8] [8, 1]) : (memref<8x8xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
          %7 = scf.for %k = %c0_1 to %hb step %c1_1 iter_args(%d = %6) -> (!air.async.token) {
            %o = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%y]
            %g0 = air.channel.get async [%d] @chan_b[] (%ba[%o, %c0_1, %c0_1] [2, 1, 8] [8, 8, 1]) : (memref<8x8xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// As above, but the second loop's gets start a row further on: other
// elements. The channels stay apart.

// CHECK-LABEL: func.func @other_elements
// CHECK: air.channel.get {{.*}} @chan_a[]
// CHECK: air.channel.get {{.*}} @chan_b[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @other_elements(%nq: index, %nb: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%aq=%nq, %ab=%nb) : index, index {
      %1 = air.segment @seg async args(%sq=%aq, %sb=%ab) : index, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %tq, %p = air.execute -> (memref<16xi16, 1>) {
          %a = memref.alloc() : memref<16xi16, 1>
          air.execute_terminator %a : memref<16xi16, 1>
        }
        %2 = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[] [] []) : (memref<16xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %3 = scf.for %k = %c0 to %sb step %c1_0 iter_args(%d = %2) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%p[] [] []) : (memref<16xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%hq=%sq, %hb=%sb) : index, index {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %ta, %ba = air.execute -> (memref<8x8xi16, 2>) {
            %a = memref.alloc() : memref<8x8xi16, 2>
            air.execute_terminator %a : memref<8x8xi16, 2>
          }
          %6 = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %o = affine.apply affine_map<()[s0] -> (s0 * 2)>()[%y]
            %g0 = air.channel.get async [%d] @chan_a[] (%ba[%o, %c0_1] [2, 8] [8, 1]) : (memref<8x8xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
          %7 = scf.for %k = %c0_1 to %hb step %c1_1 iter_args(%d = %6) -> (!air.async.token) {
            %o = affine.apply affine_map<()[s0] -> (s0 * 2 + 1)>()[%y]
            %g0 = air.channel.get async [%d] @chan_b[] (%ba[%o, %c0_1, %c0_1] [2, 1, 8] [8, 8, 1]) : (memref<8x8xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// One loop's get fills the whole core buffer with no access pattern, the
// other's spells it out as rows. The channels merge.

// CHECK-LABEL: func.func @whole_buffer_spelled_apart
// CHECK-NOT: @chan_b
// CHECK: air.channel.get {{.*}} @chan_a[] (%{{.*}}[] [] [])
// CHECK: air.channel.get {{.*}} @chan_a[] (%{{.*}}[%{{.*}}, %{{.*}}] [8, 8] [8, 1])

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @whole_buffer_spelled_apart(%nq: index, %nb: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%aq=%nq, %ab=%nb) : index, index {
      %1 = air.segment @seg async args(%sq=%aq, %sb=%ab) : index, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %tq, %p = air.execute -> (memref<64xi16, 1>) {
          %a = memref.alloc() : memref<64xi16, 1>
          air.execute_terminator %a : memref<64xi16, 1>
        }
        %2 = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[] [] []) : (memref<64xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %3 = scf.for %k = %c0 to %sb step %c1_0 iter_args(%d = %2) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%p[] [] []) : (memref<64xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%hq=%sq, %hb=%sb) : index, index {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %ta, %ba = air.execute -> (memref<8x8xi16, 2>) {
            %a = memref.alloc() : memref<8x8xi16, 2>
            air.execute_terminator %a : memref<8x8xi16, 2>
          }
          %6 = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_a[] (%ba[] [] []) : (memref<8x8xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
          %7 = scf.for %k = %c0_1 to %hb step %c1_1 iter_args(%d = %6) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_b[] (%ba[%c0_1, %c0_1] [8, 8] [8, 1]) : (memref<8x8xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// Offsets left out in front are 0: the first loop's get reads elements 0-7
// and the second's elements 4-11, of the same eight. The channels stay apart.

// CHECK-LABEL: func.func @omitted_offsets_differ
// CHECK: air.channel.get {{.*}} @chan_a[]
// CHECK: air.channel.get {{.*}} @chan_b[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @omitted_offsets_differ(%nq: index, %nb: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%aq=%nq, %ab=%nb) : index, index {
      %1 = air.segment @seg async args(%sq=%aq, %sb=%ab) : index, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %tq, %p = air.execute -> (memref<16xi16, 1>) {
          %a = memref.alloc() : memref<16xi16, 1>
          air.execute_terminator %a : memref<16xi16, 1>
        }
        %2 = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[] [] []) : (memref<16xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %3 = scf.for %k = %c0 to %sb step %c1_0 iter_args(%d = %2) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%p[] [] []) : (memref<16xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%hq=%sq, %hb=%sb) : index, index {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %c2_1 = arith.constant 2 : index
          %ta, %ba = air.execute -> (memref<16xi16, 2>) {
            %a = memref.alloc() : memref<16xi16, 2>
            air.execute_terminator %a : memref<16xi16, 2>
          }
          %6 = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_a[] (%ba[] [1, 2, 2, 2] [2, 4, 2, 1]) : (memref<16xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
          %7 = scf.for %k = %c0_1 to %hb step %c1_1 iter_args(%d = %6) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_b[] (%ba[%c1_1, %c2_1] [2, 2, 2] [4, 2, 1]) : (memref<16xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// One loop's get leaves its offsets out, the other's spells them as 0. The
// channels merge.

// CHECK-LABEL: func.func @omitted_offsets_alike
// CHECK-NOT: @chan_b
// CHECK: air.channel.get {{.*}} @chan_a[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @omitted_offsets_alike(%nq: index, %nb: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%aq=%nq, %ab=%nb) : index, index {
      %1 = air.segment @seg async args(%sq=%aq, %sb=%ab) : index, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %tq, %p = air.execute -> (memref<16xi16, 1>) {
          %a = memref.alloc() : memref<16xi16, 1>
          air.execute_terminator %a : memref<16xi16, 1>
        }
        %2 = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[] [] []) : (memref<16xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %3 = scf.for %k = %c0 to %sb step %c1_0 iter_args(%d = %2) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%p[] [] []) : (memref<16xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%hq=%sq, %hb=%sb) : index, index {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %ta, %ba = air.execute -> (memref<16xi16, 2>) {
            %a = memref.alloc() : memref<16xi16, 2>
            air.execute_terminator %a : memref<16xi16, 2>
          }
          %6 = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_a[] (%ba[] [2, 2, 2] [4, 2, 1]) : (memref<16xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
          %7 = scf.for %k = %c0_1 to %hb step %c1_1 iter_args(%d = %6) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_b[] (%ba[%c0_1, %c0_1, %c0_1] [2, 2, 2] [4, 2, 1]) : (memref<16xi16, 2>)
            scf.yield %g0 : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// Each loop moves a tile of 8 and then one of 4 through one channel, into
// the same core buffers. The channels merge.

// CHECK-LABEL: func.func @two_volumes
// CHECK-NOT: @chan_b
// CHECK: air.herd
// CHECK-COUNT-4: air.channel.get {{.*}} @chan_a[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @two_volumes(%nq: index, %nb: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%aq=%nq, %ab=%nb) : index, index {
      %1 = air.segment @seg async args(%sq=%aq, %sb=%ab) : index, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %tq, %p = air.execute -> (memref<12xi16, 1>) {
          %a = memref.alloc() : memref<12xi16, 1>
          air.execute_terminator %a : memref<12xi16, 1>
        }
        %2 = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[0] [8] [1]) : (memref<12xi16, 1>)
          %p1 = air.channel.put async [%p0] @chan_a[] (%p[8] [4] [1]) : (memref<12xi16, 1>)
          scf.yield %p1 : !air.async.token
        }
        %3 = scf.for %k = %c0 to %sb step %c1_0 iter_args(%d = %2) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%p[0] [8] [1]) : (memref<12xi16, 1>)
          %p1 = air.channel.put async [%p0] @chan_b[] (%p[8] [4] [1]) : (memref<12xi16, 1>)
          scf.yield %p1 : !air.async.token
        }
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%hq=%sq, %hb=%sb) : index, index {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %ta, %bk = air.execute -> (memref<8xi16, 2>) {
            %a = memref.alloc() : memref<8xi16, 2>
            air.execute_terminator %a : memref<8xi16, 2>
          }
          %tb, %bv = air.execute -> (memref<4xi16, 2>) {
            %a = memref.alloc() : memref<4xi16, 2>
            air.execute_terminator %a : memref<4xi16, 2>
          }
          %6 = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_a[] (%bk[] [] []) : (memref<8xi16, 2>)
            %g1 = air.channel.get async [%g0] @chan_a[] (%bv[] [] []) : (memref<4xi16, 2>)
            scf.yield %g1 : !air.async.token
          }
          %7 = scf.for %k = %c0_1 to %hb step %c1_1 iter_args(%d = %6) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_b[] (%bk[] [] []) : (memref<8xi16, 2>)
            %g1 = air.channel.get async [%g0] @chan_b[] (%bv[] [] []) : (memref<4xi16, 2>)
            scf.yield %g1 : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// As above, but the second loop moves the tile of 4 first: the core would
// take its data into the wrong buffers. The channels stay apart.

// CHECK-LABEL: func.func @volumes_in_other_order
// CHECK: air.herd
// CHECK-COUNT-2: air.channel.get {{.*}} @chan_a[]
// CHECK-COUNT-2: air.channel.get {{.*}} @chan_b[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @volumes_in_other_order(%nq: index, %nb: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%aq=%nq, %ab=%nb) : index, index {
      %1 = air.segment @seg async args(%sq=%aq, %sb=%ab) : index, index {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %tq, %p = air.execute -> (memref<12xi16, 1>) {
          %a = memref.alloc() : memref<12xi16, 1>
          air.execute_terminator %a : memref<12xi16, 1>
        }
        %2 = scf.for %k = %c0 to %sq step %c1_0 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[0] [8] [1]) : (memref<12xi16, 1>)
          %p1 = air.channel.put async [%p0] @chan_a[] (%p[8] [4] [1]) : (memref<12xi16, 1>)
          scf.yield %p1 : !air.async.token
        }
        %3 = scf.for %k = %c0 to %sb step %c1_0 iter_args(%d = %2) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%p[8] [4] [1]) : (memref<12xi16, 1>)
          %p1 = air.channel.put async [%p0] @chan_b[] (%p[0] [8] [1]) : (memref<12xi16, 1>)
          scf.yield %p1 : !air.async.token
        }
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) args(%hq=%sq, %hb=%sb) : index, index {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %ta, %bk = air.execute -> (memref<8xi16, 2>) {
            %a = memref.alloc() : memref<8xi16, 2>
            air.execute_terminator %a : memref<8xi16, 2>
          }
          %tb, %bv = air.execute -> (memref<4xi16, 2>) {
            %a = memref.alloc() : memref<4xi16, 2>
            air.execute_terminator %a : memref<4xi16, 2>
          }
          %6 = scf.for %k = %c0_1 to %hq step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_a[] (%bk[] [] []) : (memref<8xi16, 2>)
            %g1 = air.channel.get async [%g0] @chan_a[] (%bv[] [] []) : (memref<4xi16, 2>)
            scf.yield %g1 : !air.async.token
          }
          %7 = scf.for %k = %c0_1 to %hb step %c1_1 iter_args(%d = %6) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_b[] (%bv[] [] []) : (memref<4xi16, 2>)
            %g1 = air.channel.get async [%g0] @chan_b[] (%bk[] [] []) : (memref<8xi16, 2>)
            scf.yield %g1 : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// Results out of the memtile, one loop after the other. The first loop runs
// no times for this function's sizes, and its launch-side get was folded
// away: the launch side is compared on the second loop alone. The channels
// merge.

// CHECK-LABEL: func.func @launch_get_folded
// CHECK-NOT: @chan_b
// CHECK: air.channel.get {{.*}} @chan_a[]
// CHECK: air.segment
// CHECK: air.channel.put {{.*}} @chan_a[]
// CHECK: air.channel.put {{.*}} @chan_a[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @launch_get_folded(%out: memref<64xi16>, %nb: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) args(%lo=%out, %ab=%nb) : memref<64xi16>, index {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c0_i = arith.constant 0 : index
      %w = air.wait_all async
      %2 = scf.for %k = %c0 to %ab step %c1_0 iter_args(%d = %w) -> (!air.async.token) {
        %g = air.channel.get async [%d] @chan_b[] (%lo[] [] []) : (memref<64xi16>)
        scf.yield %g : !air.async.token
      }
      %1 = air.segment @seg async args(%sq=%c0_i, %sb=%ab) : index, index {
        %c0_2 = arith.constant 0 : index
        %c1_2 = arith.constant 1 : index
        %tq, %p = air.execute -> (memref<64xi16, 1>) {
          %a = memref.alloc() : memref<64xi16, 1>
          air.execute_terminator %a : memref<64xi16, 1>
        }
        %3 = scf.for %k = %c0_2 to %sq step %c1_2 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[] [] []) : (memref<64xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %4 = scf.for %k = %c0_2 to %sb step %c1_2 iter_args(%d = %3) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%p[] [] []) : (memref<64xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
      }
    }
    return
  }
}
