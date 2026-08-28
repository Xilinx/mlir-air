//===- dma_to_channel_index_switch_nested_xfail.mlir -----------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// KNOWN LIMITATION, recorded so it is not rediscovered from a 5000-line builder.
//
// One level of scf.index_switch around the external half is handled (see
// dma_to_channel_index_switch.mlir). TWO are not: here the DMA sits in a switch
// arm inside a herd, and that herd sits in a switch arm of its own inside the
// segment. The put has to be rebuilt through both, and the second hop fails
// with "operand #1 does not dominate this use" -- the cloned switch's arg is
// still the inner hierarchy's block argument, because the remap handed to the
// clone does not carry the hierarchy operand mapping for that second hop.
//
// The same gap should exist for a doubly-nested scf.if, since
// cloneScfIfUsingRemap resolves its condition the same way; it has simply never
// been reached.
//
// This is the shape a per-layer-type feed takes in the fused_decode builder
// (arm switch at the segment, arm switch again inside the herd), so it is what
// blocks porting those feeds to air.dma_memcpy_nd.

// XFAIL: *
// RUN: air-opt %s -air-dependency -air-dma-to-channel | FileCheck %s

// CHECK-LABEL: func.func @f
// CHECK: air.channel.put{{.*}}@c

air.channel @c []
func.func @f(%arg0: memref<64x64xi32>, %sw: index) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0, %s=%sw) : memref<64x64xi32>, index {
    air.segment args(%sa=%la, %ss=%s) : memref<64x64xi32>, index {
      %c1_0 = arith.constant 1 : index
      scf.index_switch %ss
      case 0 {
        scf.yield
      }
      default {
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
        scf.yield
      }
    }
  }
  return
}
