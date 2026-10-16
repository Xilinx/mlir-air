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
