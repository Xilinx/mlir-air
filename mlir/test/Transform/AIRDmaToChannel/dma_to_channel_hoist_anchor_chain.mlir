//===- dma_to_channel_hoist_anchor_chain.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Two rules that only bite once more than one feed in a design is spelled as an
// air.dma_memcpy_nd. Both were found by porting a real design's whole feed
// block, and neither is reachable with a single anchored transfer.

// RUN: air-opt %s -air-dependency -air-dma-to-channel -split-input-file | FileCheck %s

// RULE 1 -- an anchor is honoured only on the LAST hop of the walk outwards.
//
// @staged is an L3 -> L2 -> L1 feed, so it has endpoints at BOTH launch and
// segment level. An L3-sourced DMA in the herd hoists herd -> segment -> launch.
// Resolving the anchor on the first hop matches @staged's SEGMENT-level endpoint
// and pins the transfer one level short of where it belongs; the anchor is then
// consumed, the remaining hop runs unanchored, and the transfer lands at the
// hierarchy's position after all -- the exact thing the anchor exists to stop.
// So @c must come out at LAUNCH level, immediately before the launch-scope
// @staged put, and NOT inside the segment.
// CHECK-LABEL: func.func @anchor_only_on_last_hop
// CHECK: air.channel.put{{.*}}@c
// CHECK-NEXT: air.channel.put{{.*}}@staged
// CHECK: air.segment
air.channel @c []
air.channel @staged [1]
air.channel @staged_l1 [1]
func.func @anchor_only_on_last_hop(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1_l = arith.constant 1 : index
    air.channel.put @staged[%c0] (%la[%c0, %c0] [%c32, %c32] [%c64, %c1_l]) : (memref<64x64xi32>)
    air.segment @seg args(%sa=%la) : memref<64x64xi32> {
      %c0_s = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c32_s = arith.constant 32 : index
      %l2 = memref.alloc() : memref<32x32xi32, 1>
      // The segment-level half of the staged feed. This is what a first-hop
      // anchor resolution would latch onto.
      air.channel.get @staged[%c0_s] (%l2[] [] []) : (memref<32x32xi32, 1>)
      air.channel.put @staged_l1[%c0_s] (%l2[] [] []) : (memref<32x32xi32, 1>)
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%a=%sa) : memref<64x64xi32> {
        %c0_h = arith.constant 0 : index
        %c32_h = arith.constant 32 : index
        %c64_h = arith.constant 64 : index
        %c1_h = arith.constant 1 : index
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.dma_memcpy_nd (%alloc[] [] [], %a[%c0_h, %c0_h] [%c32_h, %c32_h] [%c64_h, %c1_h]) {id = 1 : i32, channel = @c, hoist_before = @staged} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        memref.dealloc %alloc : memref<32x32xi32, 2>
      }
      memref.dealloc %l2 : memref<32x32xi32, 1>
    }
  }
  return
}

// -----

// RULE 2 -- anchors CHAIN. @b is anchored to @a, and @a's own external half is
// derived by this same pass, so @a is not a fixed landmark the way a
// hand-written producer is. Pinning the resulting order matters because a real
// design ends up with a whole feed block derived, each feed anchored behind the
// previous one, and the shim BD queue follows program order.
// Required order: @base (hand-written) then @a then @b.
// CHECK-LABEL: func.func @anchor_chain
// CHECK: air.channel.put{{.*}}@base
// CHECK-NEXT: air.channel.put{{.*}}@a
// CHECK-NEXT: air.channel.put{{.*}}@b
air.channel @a []
air.channel @b []
air.channel @base [1]
air.channel @tail [1]
func.func @anchor_chain(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1_l = arith.constant 1 : index
    air.channel.put @base[%c0] (%la[%c0, %c0] [%c32, %c32] [%c64, %c1_l]) : (memref<64x64xi32>)
    air.channel.put @tail[%c0] (%la[%c0, %c0] [%c32, %c32] [%c64, %c1_l]) : (memref<64x64xi32>)
    air.segment @seg args(%sa=%la) : memref<64x64xi32> {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%a=%sa) : memref<64x64xi32> {
        %c0_h = arith.constant 0 : index
        %c32_h = arith.constant 32 : index
        %c64_h = arith.constant 64 : index
        %c1_h = arith.constant 1 : index
        %alloc_a = memref.alloc() : memref<32x32xi32, 2>
        %alloc_b = memref.alloc() : memref<32x32xi32, 2>
        // Written b-then-a so the fix cannot be mistaken for source order.
        air.dma_memcpy_nd (%alloc_b[] [] [], %a[%c0_h, %c0_h] [%c32_h, %c32_h] [%c64_h, %c1_h]) {id = 2 : i32, channel = @b, hoist_after = @a} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        air.dma_memcpy_nd (%alloc_a[] [] [], %a[%c0_h, %c0_h] [%c32_h, %c32_h] [%c64_h, %c1_h]) {id = 1 : i32, channel = @a, hoist_after = @base} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        memref.dealloc %alloc_a : memref<32x32xi32, 2>
        memref.dealloc %alloc_b : memref<32x32xi32, 2>
      }
    }
  }
  return
}
