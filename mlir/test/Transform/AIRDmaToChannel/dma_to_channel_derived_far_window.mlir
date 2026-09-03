//===- dma_to_channel_derived_far_window.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel -split-input-file | FileCheck %s

// The far half's window and position are DERIVED, not declared.
//
// A channel is a FIFO, so the only correctness requirement is that the byte
// sequence put equals the byte sequence got. The near side's sequence is fixed
// by its own access pattern and its enclosing nest, and the far buffer is named
// on the same op -- so when that buffer holds N of the near window, the near
// side takes it N pieces at a time, in order, and there is exactly one far
// descriptor that produces that: N windows of the near size, ascending.
//
// Position follows too. A far half carrying a whole buffer's worth belongs once
// per FILL, not once per near execution, so it goes where the buffer is filled.
// Finding that by the BUFFER rather than by a channel symbol is what makes it
// work when the loop it must land in holds no channel endpoint of its own --
// which is the ordinary case for a feed whose inner loop exists only to step
// the window.
//
// This is what a front end would otherwise have to hand-write, and often
// CANNOT: the far offsets step with a loop that does not exist where the
// transfer is spelled. Note what the DMA below says -- a channel, and its own
// 256-word window. Nothing else.

// CHECK-LABEL: func.func @derived
// The far half lands inside the fill's loop, right after the fill...
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}@f
// ...as the buffer tiled by the near window: 2 x 256, stride 256.
// CHECK-NEXT: air.channel.put{{.*}}@x{{.*}}[0, 0] [2, 256] [256, 1]
// The near half keeps the window it asked for.
// CHECK: air.channel.get{{.*}}@x{{.*}}[] [] []
// The marker is internal and must not survive.
// CHECK-NOT: derived_far_window

air.channel @x [1]
air.channel @f [1]
func.func @derived(%arg0: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64xbf16> {
    air.segment @seg args(%sa=%la) : memref<64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %l2 = memref.alloc() : memref<512xbf16, 1>
      scf.for %r = %c0 to %c2 step %c1_s {
        air.channel.get @f[%c0] (%l2[] [] []) : (memref<512xbf16, 1>)
      }
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%b=%l2) : memref<512xbf16, 1> {
        %a = memref.alloc() : memref<256xbf16, 2>
        air.dma_memcpy_nd (%a[] [] [], %b[] [] []) {id = 1 : i32, channel = @x, channel_indices = array<i64: 0>} : (memref<256xbf16, 2>, memref<512xbf16, 1>)
        memref.dealloc %a : memref<256xbf16, 2>
      }
    }
  }
  return
}

// -----

// NEGATIVE CONTROL: equal windows are one transfer, not a tiling. Nothing to
// derive, and the far half keeps the position and pattern it always had --
// beside the hierarchy, moving the whole buffer once.

// CHECK-LABEL: func.func @equal_windows
// CHECK-NOT: [2, 256]
// CHECK: air.channel.put{{.*}}@x2
air.channel @x2 [1]
air.channel @f2 [1]
func.func @equal_windows(%arg0: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64xbf16> {
    air.segment @seg args(%sa=%la) : memref<64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %l2 = memref.alloc() : memref<256xbf16, 1>
      scf.for %r = %c0 to %c2 step %c1_s {
        air.channel.get @f2[%c0] (%l2[] [] []) : (memref<256xbf16, 1>)
      }
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%b=%l2) : memref<256xbf16, 1> {
        %a = memref.alloc() : memref<256xbf16, 2>
        air.dma_memcpy_nd (%a[] [] [], %b[] [] []) {id = 1 : i32, channel = @x2, channel_indices = array<i64: 0>} : (memref<256xbf16, 2>, memref<256xbf16, 1>)
        memref.dealloc %a : memref<256xbf16, 2>
      }
    }
  }
  return
}

// -----

// NEGATIVE CONTROL: a far buffer that is NOT a whole multiple of the near
// window has no unique tiling, so nothing is derived. Guessing one is how a
// silent misroute gets built.

// CHECK-LABEL: func.func @not_a_multiple
// CHECK-NOT: [2, 256]
// CHECK: air.channel.put{{.*}}@x3
air.channel @x3 [1]
air.channel @f3 [1]
func.func @not_a_multiple(%arg0: memref<64xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64xbf16> {
    air.segment @seg args(%sa=%la) : memref<64xbf16> {
      %c0 = arith.constant 0 : index
      %c1_s = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %l2 = memref.alloc() : memref<600xbf16, 1>
      scf.for %r = %c0 to %c2 step %c1_s {
        air.channel.get @f3[%c0] (%l2[] [] []) : (memref<600xbf16, 1>)
      }
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%b=%l2) : memref<600xbf16, 1> {
        %a = memref.alloc() : memref<256xbf16, 2>
        air.dma_memcpy_nd (%a[] [] [], %b[] [] []) {id = 1 : i32, channel = @x3, channel_indices = array<i64: 0>} : (memref<256xbf16, 2>, memref<600xbf16, 1>)
        memref.dealloc %a : memref<256xbf16, 2>
      }
    }
  }
  return
}

// -----

// NEGATIVE CONTROL: nothing REFILLS the far buffer, so there is no cycle to
// read. "The near side takes it N pieces at a time" is true of a buffer that is
// filled, then taken from, then filled again; a function argument read once is
// not that, and tiling it would send N times what was asked for -- silently,
// and it would compile. The window and the position rest on the same piece of
// evidence, so with no fill there is neither.

// CHECK-LABEL: func.func @far_never_filled
// CHECK-NOT: [2, 256]
// CHECK: air.channel.put{{.*}}@x4
air.channel @x4 [1]
func.func @far_never_filled(%arg0: memref<512xbf16>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<512xbf16> {
    air.segment @seg args(%sa=%la) : memref<512xbf16> {
      %c1_s = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%b=%sa) : memref<512xbf16> {
        %a = memref.alloc() : memref<256xbf16, 2>
        air.dma_memcpy_nd (%a[] [] [], %b[] [] []) {id = 1 : i32, channel = @x4, channel_indices = array<i64: 0>} : (memref<256xbf16, 2>, memref<512xbf16>)
        memref.dealloc %a : memref<256xbf16, 2>
      }
    }
  }
  return
}
