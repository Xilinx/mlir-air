//===- broadcast_detection_scf_if_guard.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-broadcast-detection -split-input-file | FileCheck %s

// A GUARD is a dependence on whatever it tests. A transfer under
// `scf.if (tx == 0)` happens on one tile and not another, so it is not
// invariant across the herd however constant its own operands look.
//
// affine.if was already traced here; scf.if was not. That matters because a
// guard is the ONLY way a front end can select between per-tile BUFFERS -- a
// memref is an SSA operand, not arithmetic, so it cannot be an expression. The
// memref itself is never traced either (only scalars are), so with the guard
// also unseen such a transfer looked completely herd-invariant.
//
// Broadcast detection then calls it broadcastable and specialization merges
// transfers that read DIFFERENT buffers into one producer. fused_decode's
// weight fan is exactly that shape.

// The transfer varies with tx (a different buffer per column) and not with ty,
// so the only correct pattern broadcasts along ty with tx bound to the symbol.
// The all-tiles set is what it used to get, and that is the bug: one producer
// for four cores that read TWO different buffers.
// CHECK-DAG: #[[ROWVAR:.*]] = affine_set<(d0, d1)[s0] : (d0 - s0 == 0, d1 >= 0, -d1 + 1 >= 0, s0 >= 0, -s0 + 1 >= 0)>
// CHECK-LABEL: func.func @guarded_by_tile
// CHECK: broadcast_pattern = #[[ROWVAR]]
func.func @guarded_by_tile(%arg0: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64xbf16> {
    air.segment @seg args(%sa=%la) : memref<64xbf16> {
      %c2 = arith.constant 2 : index
      %b0 = memref.alloc() : memref<256xbf16, 1>
      %b1 = memref.alloc() : memref<256xbf16, 1>
      air.herd @h tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%p0=%b0, %p1=%b1) : memref<256xbf16, 1>, memref<256xbf16, 1> {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %a = memref.alloc() : memref<256xbf16, 2>
        %is0 = arith.cmpi eq, %tx, %c0_h : index
        scf.if %is0 {
          air.dma_memcpy_nd (%a[] [] [], %p0[] [] []) {id = 1 : i32} : (memref<256xbf16, 2>, memref<256xbf16, 1>)
        }
        %is1 = arith.cmpi eq, %tx, %c1_h : index
        scf.if %is1 {
          air.dma_memcpy_nd (%a[] [] [], %p1[] [] []) {id = 2 : i32} : (memref<256xbf16, 2>, memref<256xbf16, 1>)
        }
        memref.dealloc %a : memref<256xbf16, 2>
      }
      memref.dealloc %b0 : memref<256xbf16, 1>
      memref.dealloc %b1 : memref<256xbf16, 1>
    }
  }
  return
}

// -----

// NEGATIVE CONTROL: the SAME transfer with no guard really is herd-invariant,
// and must still be detected. Every core reads the same buffer, so one
// broadcast producer is right and declining would leave four separate feeds.

// CHECK-LABEL: func.func @unguarded
// CHECK: broadcast_pattern
func.func @unguarded(%arg0: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64xbf16> {
    air.segment @seg args(%sa=%la) : memref<64xbf16> {
      %c2 = arith.constant 2 : index
      %b0 = memref.alloc() : memref<256xbf16, 1>
      air.herd @h tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%p0=%b0) : memref<256xbf16, 1> {
        %a = memref.alloc() : memref<256xbf16, 2>
        air.dma_memcpy_nd (%a[] [] [], %p0[] [] []) {id = 1 : i32} : (memref<256xbf16, 2>, memref<256xbf16, 1>)
        memref.dealloc %a : memref<256xbf16, 2>
      }
      memref.dealloc %b0 : memref<256xbf16, 1>
    }
  }
  return
}

// -----

// NEGATIVE CONTROL: a guard on something that is NOT a herd index says nothing
// about tiles. A runtime parameter reads the same on every core, so the
// transfer is still invariant and must still be detected -- "guarded" is not by
// itself a reason to decline.

// CHECK-LABEL: func.func @guarded_by_rtp
// CHECK: broadcast_pattern
func.func @guarded_by_rtp(%arg0: memref<64xbf16>, %rtp: index) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0, %lr=%rtp) : memref<64xbf16>, index {
    air.segment @seg args(%sa=%la, %sr=%lr) : memref<64xbf16>, index {
      %c2 = arith.constant 2 : index
      %b0 = memref.alloc() : memref<256xbf16, 1>
      air.herd @h tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%p0=%b0, %pr=%sr) : memref<256xbf16, 1>, index {
        %c0_h = arith.constant 0 : index
        %a = memref.alloc() : memref<256xbf16, 2>
        %g = arith.cmpi eq, %pr, %c0_h : index
        scf.if %g {
          air.dma_memcpy_nd (%a[] [] [], %p0[] [] []) {id = 1 : i32} : (memref<256xbf16, 2>, memref<256xbf16, 1>)
        }
        memref.dealloc %a : memref<256xbf16, 2>
      }
      memref.dealloc %b0 : memref<256xbf16, 1>
    }
  }
  return
}

// -----

// Guarded on BOTH tile indices: the transfer is distinct on every core, so
// there is no dimension left to broadcast along and no pattern is right. This
// is fused_decode's weight fan, where the row index also picks the slice.

// CHECK-LABEL: func.func @guarded_by_both
// CHECK-NOT: broadcast_pattern
func.func @guarded_by_both(%arg0: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64xbf16> {
    air.segment @seg args(%sa=%la) : memref<64xbf16> {
      %c2 = arith.constant 2 : index
      %b0 = memref.alloc() : memref<256xbf16, 1>
      %b1 = memref.alloc() : memref<256xbf16, 1>
      air.herd @h tile (%tx, %ty) in (%sx=%c2, %sy=%c2) args(%p0=%b0, %p1=%b1) : memref<256xbf16, 1>, memref<256xbf16, 1> {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %a = memref.alloc() : memref<256xbf16, 2>
        %x0 = arith.cmpi eq, %tx, %c0_h : index
        %y0 = arith.cmpi eq, %ty, %c0_h : index
        %both = arith.andi %x0, %y0 : i1
        scf.if %both {
          air.dma_memcpy_nd (%a[] [] [], %p0[] [] []) {id = 1 : i32} : (memref<256xbf16, 2>, memref<256xbf16, 1>)
        }
        %x1 = arith.cmpi eq, %tx, %c1_h : index
        %both1 = arith.andi %x1, %y0 : i1
        scf.if %both1 {
          air.dma_memcpy_nd (%a[] [] [], %p1[] [] []) {id = 2 : i32} : (memref<256xbf16, 2>, memref<256xbf16, 1>)
        }
        memref.dealloc %a : memref<256xbf16, 2>
      }
      memref.dealloc %b0 : memref<256xbf16, 1>
      memref.dealloc %b1 : memref<256xbf16, 1>
    }
  }
  return
}
