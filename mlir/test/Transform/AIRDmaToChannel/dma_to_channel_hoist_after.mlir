//===- dma_to_channel_hoist_after.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// air-dma-to-channel derives the external half of a DMA at the CONSUMER
// hierarchy's position, which is after every producer the front end wrote by
// hand. On a launch carrying air.preserve_shim_dma_order that reorders the shim
// BD queue, and the reordering is load-bearing: in fused_decode it moves the
// rope LUT feed from BD slot 6 to slot 18, behind the whole weight stream, and
// the rope core deadlocks waiting on a LUT that no longer arrives first.
//
// `hoist_after` names a channel to be issued after. The derived op is placed
// immediately after that channel's last endpoint, inheriting its position AND
// its control context -- no guard is synthesised, so if the anchor sits in a
// switch arm the derived op lands in that same arm.

// RUN: air-opt %s -air-dependency -air-dma-to-channel -split-input-file | FileCheck %s

// The put must land between @anchor and @after, not at the end of the block.
// CHECK-LABEL: func.func @anchored
// CHECK: air.channel.put{{.*}}@anchor
// CHECK-NEXT: air.channel.put{{.*}}@c
// CHECK: air.channel.put{{.*}}@after
air.channel @c []
air.channel @anchor [1]
air.channel @after [1]
func.func @anchored(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1_l = arith.constant 1 : index
    air.channel.put @anchor[%c0] (%la[%c0, %c0] [%c32, %c32] [%c64, %c1_l]) : (memref<64x64xi32>)
    air.channel.put @after[%c0] (%la[%c0, %c0] [%c32, %c32] [%c64, %c1_l]) : (memref<64x64xi32>)
    air.segment @seg args(%sa=%la) : memref<64x64xi32> {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%a=%sa) : memref<64x64xi32> {
        %c0_h = arith.constant 0 : index
        %c32_h = arith.constant 32 : index
        %c64_h = arith.constant 64 : index
        %c1_h = arith.constant 1 : index
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.dma_memcpy_nd (%alloc[] [] [], %a[%c0_h, %c0_h] [%c32_h, %c32_h] [%c64_h, %c1_h]) {id = 1 : i32, channel = @c, hoist_after = @anchor} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        memref.dealloc %alloc : memref<32x32xi32, 2>
      }
    }
  }
  return
}

// -----

// Without the anchor the same design puts the derived op at the hierarchy, i.e.
// after BOTH hand-written producers. This is the behaviour hoist_after exists to
// override, so pin it too.
// CHECK-LABEL: func.func @unanchored
// CHECK: air.channel.put{{.*}}@anchor
// CHECK-NEXT: air.channel.put{{.*}}@after
// CHECK: air.channel.put{{.*}}@c
air.channel @c []
air.channel @anchor [1]
air.channel @after [1]
func.func @unanchored(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1_l = arith.constant 1 : index
    air.channel.put @anchor[%c0] (%la[%c0, %c0] [%c32, %c32] [%c64, %c1_l]) : (memref<64x64xi32>)
    air.channel.put @after[%c0] (%la[%c0, %c0] [%c32, %c32] [%c64, %c1_l]) : (memref<64x64xi32>)
    air.segment @seg args(%sa=%la) : memref<64x64xi32> {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%a=%sa) : memref<64x64xi32> {
        %c0_h = arith.constant 0 : index
        %c32_h = arith.constant 32 : index
        %c64_h = arith.constant 64 : index
        %c1_h = arith.constant 1 : index
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.dma_memcpy_nd (%alloc[] [] [], %a[%c0_h, %c0_h] [%c32_h, %c32_h] [%c64_h, %c1_h]) {id = 1 : i32, channel = @c} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        memref.dealloc %alloc : memref<32x32xi32, 2>
      }
    }
  }
  return
}

// -----

// hoist_before is the mirror: place the derived op immediately BEFORE the
// anchor's endpoint. Needed when the hand-written producer was FIRST in its
// block, where there is nothing to sit after.
// CHECK-LABEL: func.func @anchored_before
// CHECK: air.channel.put{{.*}}@c
// CHECK-NEXT: air.channel.put{{.*}}@anchor
air.channel @c []
air.channel @anchor [1]
func.func @anchored_before(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    %c0 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c1_l = arith.constant 1 : index
    air.channel.put @anchor[%c0] (%la[%c0, %c0] [%c32, %c32] [%c64, %c1_l]) : (memref<64x64xi32>)
    air.segment @seg args(%sa=%la) : memref<64x64xi32> {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%a=%sa) : memref<64x64xi32> {
        %c0_h = arith.constant 0 : index
        %c32_h = arith.constant 32 : index
        %c64_h = arith.constant 64 : index
        %c1_h = arith.constant 1 : index
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.dma_memcpy_nd (%alloc[] [] [], %a[%c0_h, %c0_h] [%c32_h, %c32_h] [%c64_h, %c1_h]) {id = 1 : i32, channel = @c, hoist_before = @anchor} : (memref<32x32xi32, 2>, memref<64x64xi32>)
        memref.dealloc %alloc : memref<32x32xi32, 2>
      }
    }
  }
  return
}
