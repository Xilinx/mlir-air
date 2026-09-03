//===- dma_to_channel_index_switch.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Hoisting the external half of a DMA out of an scf.index_switch arm has to
// REBUILD the switch. cloneOpsInBlock turns every op it is not told to hoist
// into a wait_all, so before this was handled the switch was replaced wholesale
// and the external put inside it went with it -- silently. The failure surfaced
// ~35 passes later in air-verify-hierarchy-locality as "found channel op not in
// pairs", pointing at the surviving get rather than at the arm the put was lost
// from.
//
// This matters because a per-layer-type feed is written exactly this way: one
// arm per layer type, switched on a runtime arm index.
//
// Both spellings are checked. air-dependency runs BEFORE air-dma-to-channel in
// aircc, so the async form -- where every arm yields a token -- is the one that
// actually occurs; flattening it would make a copy written on one arm issue on
// every arm.

// RUN: air-opt %s -air-dma-to-channel | FileCheck %s
// RUN: air-opt %s -air-dependency -air-dma-to-channel | FileCheck %s --check-prefix=ASYNC

// The put is hoisted to the launch, and it is still INSIDE a switch.
// CHECK-LABEL: func.func @f
// CHECK: air.launch
// CHECK: scf.index_switch
// CHECK: air.channel.put{{.*}}@c
// CHECK: air.herd
// CHECK: scf.index_switch
// CHECK: air.channel.get{{.*}}@c

// ASYNC-LABEL: func.func @f
// ASYNC: scf.index_switch{{.*}}-> !air.async.token
// ASYNC: air.channel.put async{{.*}}@c
// ASYNC: air.herd
// ASYNC: air.channel.get async{{.*}}@c

air.channel @c []
func.func @f(%arg0: memref<64x64xi32>, %sw: index) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0, %s=%sw) : memref<64x64xi32>, index {
    air.segment args(%sa=%la, %ss=%s) : memref<64x64xi32>, index {
      %c1_0 = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) args(%a=%sa, %sv=%ss) : memref<64x64xi32>, index {
        %c0 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c64 = arith.constant 64 : index
        %cst1 = arith.constant 1 : index
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        scf.index_switch %sv
        case 0 {
          scf.yield
        }
        default {
          air.dma_memcpy_nd (%alloc[] [] [], %a[%c0, %c0] [%c32, %c32] [%c64, %cst1]) {id = 1 : i32, channel = @c} : (memref<32x32xi32, 2>, memref<64x64xi32>)
          scf.yield
        }
        memref.dealloc %alloc : memref<32x32xi32, 2>
      }
    }
  }
  return
}
