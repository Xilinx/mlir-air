//===- dma_to_channel_merge_hoisted_guard.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel | FileCheck %s

// The guarded form of dma_to_channel_fuse_into_producer_loop.mlir, and the one
// a real design actually has: the producer loop is inside an scf.index_switch,
// and so is the herd-side transfer. cloneIndexSwitchUsingRemap rebuilds the
// guard around the hoisted copy -- correctly, since flattening it would make a
// copy written on one arm issue on every arm -- but it rebuilds unconditionally,
// so the fill lands in one switch's arm and the derived drain in another's.
//
// Loop fusion cannot repair that: a value defined in one scf.index_switch arm
// is invisible from a sibling switch's arm, so the producer loop's result can
// never stand in for the hoisted loop's. The guards have to become one first.
//
// Note the two switches are NOT on the same SSA value by the time the hoist is
// done -- the condition is cloned along with everything else the transfer
// depends on, and CSE does not run until two passes later. Matching them is
// therefore structural.
//
// Expect ONE segment-level switch out, its default arm holding one loop that
// both fills and drains, with the drain naming the fill.

// CHECK-LABEL: func.func @kv
// CHECK: air.segment
// CHECK: scf.index_switch
// CHECK: scf.for
// CHECK: %[[GET:.*]] = air.channel.get async{{.*}}@fill
// CHECK: air.channel.put async [{{.*}}%[[GET]]{{.*}}@drain
// CHECK-NOT: scf.index_switch
// CHECK: air.herd

air.channel @fill [1]
air.channel @drain [1]
func.func @kv(%arg0: memref<4096xbf16>, %sel: index) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0, %lsel=%sel) : memref<4096xbf16>, index {
    air.segment @seg args(%ssel=%lsel) : index {
      %c0_s = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c2_s = arith.constant 2 : index
      %buf = memref.alloc() : memref<4096xbf16, 1>
      scf.index_switch %ssel
      case 0 {
        scf.yield
      }
      default {
        scf.for %i = %c0_s to %c2_s step %c1_s {
          air.channel.get @fill[%c0_s] (%buf[] [] []) : (memref<4096xbf16, 1>)
        }
        scf.yield
      }
      air.herd @h tile (%tx, %ty) in (%sx=%c2_s, %sy=%c1_s) args(%b=%buf, %hsel=%ssel) : memref<4096xbf16, 1>, index {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2_h = arith.constant 2 : index
        %is0 = arith.cmpi eq, %tx, %c0_h : index
        scf.if %is0 {
          scf.index_switch %hsel
          case 0 {
            scf.yield
          }
          default {
            scf.for %j = %c0_h to %c2_h step %c1_h {
              %l1 = memref.alloc() : memref<2048xbf16, 2>
              air.dma_memcpy_nd (%l1[] [] [], %b[%c0_h] [%c2_h] [%c1_h]) {id = 1 : i32, channel = @drain, channel_indices = array<i64: 0>} : (memref<2048xbf16, 2>, memref<4096xbf16, 1>)
              memref.dealloc %l1 : memref<2048xbf16, 2>
            }
            scf.yield
          }
        }
      }
      memref.dealloc %buf : memref<4096xbf16, 1>
    }
  }
  return
}
