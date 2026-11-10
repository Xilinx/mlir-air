//===- fuse_channels_sibling_loop_use.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels="aggressive-mode=L1" -split-input-file | FileCheck %s

// On the core, two loops with equal bounds run one after the other, each
// receiving a row into its own buffer and reading it in the same trip.
// Merging chan_b into chan_a would move chan_b's get into the first loop:
// the second loop's trips would then read whatever row arrived last. The
// channels stay apart.

// CHECK-LABEL: func.func @sibling_use
// CHECK: scf.for
// CHECK: air.channel.put {{.*}} @chan_a[]
// CHECK: scf.for
// CHECK: air.channel.put {{.*}} @chan_b[]
// CHECK: air.herd
// CHECK: scf.for
// CHECK-NEXT: air.channel.get {{.*}} @chan_a[]
// CHECK-NOT: air.channel.get
// CHECK: memref.copy
// CHECK: scf.for
// CHECK-NEXT: air.channel.get {{.*}} @chan_b[]
// CHECK: memref.copy

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @sibling_use() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %tq, %p = air.execute -> (memref<2x256xi16, 1>) {
          %a = memref.alloc() : memref<2x256xi16, 1>
          air.execute_terminator %a : memref<2x256xi16, 1>
        }
        %tb, %z = air.execute -> (memref<2x256xi16, 1>) {
          %a = memref.alloc() : memref<2x256xi16, 1>
          air.execute_terminator %a : memref<2x256xi16, 1>
        }
        %2 = scf.for %k = %c0 to %c2 step %c1_0 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[%k, 0] [1, 256] [256, 1]) : (memref<2x256xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %3 = scf.for %k = %c0 to %c2 step %c1_0 iter_args(%d = %tb) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%z[%k, 0] [1, 256] [256, 1]) : (memref<2x256xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %c2_1 = arith.constant 2 : index
          %ta, %ba = air.execute -> (memref<256xi16, 2>) {
            %a = memref.alloc() : memref<256xi16, 2>
            air.execute_terminator %a : memref<256xi16, 2>
          }
          %tb2, %bb = air.execute -> (memref<256xi16, 2>) {
            %a = memref.alloc() : memref<256xi16, 2>
            air.execute_terminator %a : memref<256xi16, 2>
          }
          %tc, %bc = air.execute -> (memref<256xi16, 2>) {
            %a = memref.alloc() : memref<256xi16, 2>
            air.execute_terminator %a : memref<256xi16, 2>
          }
          %6 = scf.for %k = %c0_1 to %c2_1 step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_a[] (%ba[] [] []) : (memref<256xi16, 2>)
            %u0 = air.execute [%g0] {
              memref.copy %ba, %bc : memref<256xi16, 2> to memref<256xi16, 2>
            }
            scf.yield %u0 : !air.async.token
          }
          %7 = scf.for %k = %c0_1 to %c2_1 step %c1_1 iter_args(%d = %6) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_b[] (%bb[] [] []) : (memref<256xi16, 2>)
            %u0 = air.execute [%g0] {
              memref.copy %bb, %bc : memref<256xi16, 2> to memref<256xi16, 2>
            }
            scf.yield %u0 : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// As above, the second loop reading chan_b's buffer through a view made
// before the loops: a use all the same. The channels stay apart.

// CHECK-LABEL: func.func @view_use
// CHECK: air.channel.put {{.*}} @chan_a[]
// CHECK: air.channel.put {{.*}} @chan_b[]
// CHECK: air.herd
// CHECK: air.channel.get {{.*}} @chan_a[]
// CHECK: air.channel.get {{.*}} @chan_b[]

module {
  air.channel @chan_a [1, 1]
  air.channel @chan_b [1, 1]
  func.func @view_use() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %tq, %p = air.execute -> (memref<2x256xi16, 1>) {
          %a = memref.alloc() : memref<2x256xi16, 1>
          air.execute_terminator %a : memref<2x256xi16, 1>
        }
        %tb, %z = air.execute -> (memref<2x256xi16, 1>) {
          %a = memref.alloc() : memref<2x256xi16, 1>
          air.execute_terminator %a : memref<2x256xi16, 1>
        }
        %2 = scf.for %k = %c0 to %c2 step %c1_0 iter_args(%d = %tq) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_a[] (%p[%k, 0] [1, 256] [256, 1]) : (memref<2x256xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %3 = scf.for %k = %c0 to %c2 step %c1_0 iter_args(%d = %tb) -> (!air.async.token) {
          %p0 = air.channel.put async [%d] @chan_b[] (%z[%k, 0] [1, 256] [256, 1]) : (memref<2x256xi16, 1>)
          scf.yield %p0 : !air.async.token
        }
        %5 = air.herd @herd async tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %c0_1 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %c2_1 = arith.constant 2 : index
          %ta, %ba = air.execute -> (memref<256xi16, 2>) {
            %a = memref.alloc() : memref<256xi16, 2>
            air.execute_terminator %a : memref<256xi16, 2>
          }
          %tb2, %bb = air.execute -> (memref<256xi16, 2>) {
            %a = memref.alloc() : memref<256xi16, 2>
            air.execute_terminator %a : memref<256xi16, 2>
          }
          %tc, %bc = air.execute -> (memref<256xi16, 2>) {
            %a = memref.alloc() : memref<256xi16, 2>
            air.execute_terminator %a : memref<256xi16, 2>
          }
          %bv = memref.subview %bb[0] [128] [1] : memref<256xi16, 2> to memref<128xi16, strided<[1]>, 2>
          %6 = scf.for %k = %c0_1 to %c2_1 step %c1_1 iter_args(%d = %ta) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_a[] (%ba[] [] []) : (memref<256xi16, 2>)
            %u0 = air.execute [%g0] {
              memref.copy %ba, %bc : memref<256xi16, 2> to memref<256xi16, 2>
            }
            scf.yield %u0 : !air.async.token
          }
          %7 = scf.for %k = %c0_1 to %c2_1 step %c1_1 iter_args(%d = %6) -> (!air.async.token) {
            %g0 = air.channel.get async [%d] @chan_b[] (%bb[] [] []) : (memref<256xi16, 2>)
            %u0 = air.execute [%g0] {
              %cv = memref.subview %bc[0] [128] [1] : memref<256xi16, 2> to memref<128xi16, strided<[1]>, 2>
              memref.copy %bv, %cv : memref<128xi16, strided<[1]>, 2> to memref<128xi16, strided<[1]>, 2>
            }
            scf.yield %u0 : !air.async.token
          }
        }
      }
    }
    return
  }
}
