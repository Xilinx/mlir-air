//===- air_split_l2_memref_slots_per_channel.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-split-l2-memref="tiles-per-l2-tile=1" | FileCheck %s

// Each row of the buffer goes out on its own channel, in two transfers, one
// per slot of dim 1. Transfers on one channel go to the same consumer, so the
// slots do not mark the split: the buffer is split by row, into four buffers
// that each keep both slots.
// CHECK-LABEL: func.func @rows_with_slots
// CHECK-COUNT-4: memref.alloc() : memref<1x2x64xbf16, 1>
// CHECK-NOT: memref.alloc() : memref<4x1x64xbf16, 1>

air.channel @in [1, 1]
air.channel @c0 [1, 1]
air.channel @c1 [1, 1]
air.channel @c2 [1, 1]
air.channel @c3 [1, 1]
func.func @rows_with_slots(%arg0: memref<512xbf16>) {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%l) in (%ls=%c1) args(%src=%arg0) : memref<512xbf16> {
    %c0_l = arith.constant 0 : index
    %c1_l = arith.constant 1 : index
    %c2_l = arith.constant 2 : index
    %c4_l = arith.constant 4 : index
    %c64_l = arith.constant 64 : index
    %c128_l = arith.constant 128 : index
    %w = air.wait_all async
    %pl = scf.for %j = %c0_l to %c2_l step %c1_l iter_args(%t = %w) -> (!air.async.token) {
      %o = arith.muli %j, %c64_l : index
      %p = air.channel.put async [%t] @in[] (%src[%c0_l, %o] [%c4_l, %c64_l] [%c128_l, %c1_l]) : (memref<512xbf16>)
      scf.yield %p : !air.async.token
    }
    %1 = air.segment @seg async {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %ta, %a = air.execute -> (memref<4x2x64xbf16, 1>) {
        %x = memref.alloc() : memref<4x2x64xbf16, 1>
        air.execute_terminator %x : memref<4x2x64xbf16, 1>
      }
      %g = scf.for %j = %c0 to %c2 step %c1_0 iter_args(%t = %ta) -> (!air.async.token) {
        %q = air.channel.get async [%t] @in[] (%a[%c0, %j, %c0] [%c4, %c1_0, %c64] [%c128, %c64, %c1_0]) : (memref<4x2x64xbf16, 1>)
        scf.yield %q : !air.async.token
      }
      %c0_r = arith.constant 0 : index
      %p0a = air.channel.put async [%g] @c0[] (%a[%c0_r, %c0, %c0] [%c1_0, %c1_0, %c64] [%c128, %c64, %c1_0]) : (memref<4x2x64xbf16, 1>)
      %p0b = air.channel.put async [%p0a] @c0[] (%a[%c0_r, %c1_0, %c0] [%c1_0, %c1_0, %c64] [%c128, %c64, %c1_0]) : (memref<4x2x64xbf16, 1>)
      %c1_r = arith.constant 1 : index
      %p1a = air.channel.put async [%g] @c1[] (%a[%c1_r, %c0, %c0] [%c1_0, %c1_0, %c64] [%c128, %c64, %c1_0]) : (memref<4x2x64xbf16, 1>)
      %p1b = air.channel.put async [%p1a] @c1[] (%a[%c1_r, %c1_0, %c0] [%c1_0, %c1_0, %c64] [%c128, %c64, %c1_0]) : (memref<4x2x64xbf16, 1>)
      %c2_r = arith.constant 2 : index
      %p2a = air.channel.put async [%g] @c2[] (%a[%c2_r, %c0, %c0] [%c1_0, %c1_0, %c64] [%c128, %c64, %c1_0]) : (memref<4x2x64xbf16, 1>)
      %p2b = air.channel.put async [%p2a] @c2[] (%a[%c2_r, %c1_0, %c0] [%c1_0, %c1_0, %c64] [%c128, %c64, %c1_0]) : (memref<4x2x64xbf16, 1>)
      %c3_r = arith.constant 3 : index
      %p3a = air.channel.put async [%g] @c3[] (%a[%c3_r, %c0, %c0] [%c1_0, %c1_0, %c64] [%c128, %c64, %c1_0]) : (memref<4x2x64xbf16, 1>)
      %p3b = air.channel.put async [%p3a] @c3[] (%a[%c3_r, %c1_0, %c0] [%c1_0, %c1_0, %c64] [%c128, %c64, %c1_0]) : (memref<4x2x64xbf16, 1>)
      %w2 = air.wait_all async [%p0b, %p1b, %p2b, %p3b]
      %d = air.execute [%w2] {
        memref.dealloc %a : memref<4x2x64xbf16, 1>
      }
      %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c4, %sy=%c1_0) {
        %tl, %l1 = air.execute -> (memref<64xbf16, 2>) {
          %x = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %x : memref<64xbf16, 2>
        }
        %g0a = air.channel.get async [%tl] @c0[] (%l1[] [] []) : (memref<64xbf16, 2>)
        %g0b = air.channel.get async [%g0a] @c0[] (%l1[] [] []) : (memref<64xbf16, 2>)
        %g1a = air.channel.get async [%tl] @c1[] (%l1[] [] []) : (memref<64xbf16, 2>)
        %g1b = air.channel.get async [%g1a] @c1[] (%l1[] [] []) : (memref<64xbf16, 2>)
        %g2a = air.channel.get async [%tl] @c2[] (%l1[] [] []) : (memref<64xbf16, 2>)
        %g2b = air.channel.get async [%g2a] @c2[] (%l1[] [] []) : (memref<64xbf16, 2>)
        %g3a = air.channel.get async [%tl] @c3[] (%l1[] [] []) : (memref<64xbf16, 2>)
        %g3b = air.channel.get async [%g3a] @c3[] (%l1[] [] []) : (memref<64xbf16, 2>)
        %dl = air.execute [%g0b, %g1b, %g2b, %g3b] {
          memref.dealloc %l1 : memref<64xbf16, 2>
        }
      }
    }
  }
  return
}
