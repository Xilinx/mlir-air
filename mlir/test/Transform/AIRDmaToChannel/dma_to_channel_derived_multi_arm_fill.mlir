//===- dma_to_channel_derived_multi_arm_fill.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel -split-input-file | FileCheck %s

// A derived far half goes where the buffer is FILLED, and one buffer is
// commonly filled in several MUTUALLY EXCLUSIVE places -- a vocab arm and a
// decode arm of the same scf.index_switch, each with its own feed loop.
//
// Those are not competing candidates to choose between. The transfer belongs
// in every one of them, the same way an anchored hoist replicates across arms:
// an arm changes only WHETHER the transfer is issued, never how many times.
//
// Picking one is silent. The arms that lose keep their consumers, which then
// wait on a producer emitted somewhere they never execute, and the design
// hangs with no missing-endpoint diagnostic anywhere. That is exactly how
// fused_decode's @inX lost its memtile producer: two feed loops, one derived
// put, and decode timed out several dispatches in.

// CHECK-LABEL: func.func @two_arms
// Each arm's fill gets its own pair of derived puts.
// CHECK: scf.index_switch
// CHECK: air.channel.get{{.*}}@f
// CHECK: air.channel.put{{.*}}@x{{.*}}[0] [256] [1]
// CHECK: air.channel.put{{.*}}@x{{.*}}[256] [256] [1]
// CHECK: air.channel.get{{.*}}@f
// CHECK: air.channel.put{{.*}}@x{{.*}}[0] [256] [1]
// CHECK: air.channel.put{{.*}}@x{{.*}}[256] [256] [1]

air.channel @x [1]
air.channel @f [1]
func.func @two_arms(%arg0: memref<64xbf16>, %sel: index) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0, %ls2=%sel) : memref<64xbf16>, index {
    air.segment @seg args(%sa=%la, %ss=%ls2) : memref<64xbf16>, index {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %l2 = memref.alloc() : memref<512xbf16, 1>
      scf.index_switch %ss
      case 0 {
        scf.for %r = %c0 to %c2 step %c1_s {
          air.channel.get @f[%c0] (%l2[] [] []) : (memref<512xbf16, 1>)
        }
        scf.yield
      }
      default {
        scf.for %r = %c0 to %c4 step %c1_s {
          air.channel.get @f[%c0] (%l2[] [] []) : (memref<512xbf16, 1>)
        }
        scf.yield
      }
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%b=%l2) : memref<512xbf16, 1> {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2_h = arith.constant 2 : index
        scf.for %j = %c0_h to %c2_h step %c1_h {
          %a = memref.alloc() : memref<256xbf16, 2>
          air.dma_memcpy_nd (%a[] [] [], %b[] [] []) {id = 1 : i32, channel = @x, channel_indices = array<i64: 0>} : (memref<256xbf16, 2>, memref<512xbf16, 1>)
          memref.dealloc %a : memref<256xbf16, 2>
        }
      }
    }
  }
  return
}

// -----

// NEGATIVE CONTROL: one fill site is still one placement. Replicating per
// REGION must not turn a single feed loop into two producers -- that would
// double the lock releases the consumer counts, which is the same failure in
// the other direction.

// CHECK-LABEL: func.func @one_arm
// CHECK: air.channel.put{{.*}}@x2{{.*}}[0] [256] [1]
// CHECK: air.channel.put{{.*}}@x2{{.*}}[256] [256] [1]
// CHECK-NOT: air.channel.put{{.*}}@x2

air.channel @x2 [1]
air.channel @f2 [1]
func.func @one_arm(%arg0: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64xbf16> {
    air.segment @seg args(%sa=%la) : memref<64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %l2 = memref.alloc() : memref<512xbf16, 1>
      scf.for %r = %c0 to %c2 step %c1_s {
        air.channel.get @f2[%c0] (%l2[] [] []) : (memref<512xbf16, 1>)
      }
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%b=%l2) : memref<512xbf16, 1> {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2_h = arith.constant 2 : index
        scf.for %j = %c0_h to %c2_h step %c1_h {
          %a = memref.alloc() : memref<256xbf16, 2>
          air.dma_memcpy_nd (%a[] [] [], %b[] [] []) {id = 1 : i32, channel = @x2, channel_indices = array<i64: 0>} : (memref<256xbf16, 2>, memref<512xbf16, 1>)
          memref.dealloc %a : memref<256xbf16, 2>
        }
      }
    }
  }
  return
}
