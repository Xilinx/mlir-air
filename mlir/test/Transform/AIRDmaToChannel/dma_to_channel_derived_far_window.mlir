//===- dma_to_channel_derived_far_window.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel -split-input-file | FileCheck %s

// The far half's window and position are DERIVED, not declared.
//
// A channel is a FIFO. The near side's sequence is fixed by its own access
// pattern and its enclosing nest, and the far buffer is named on the same op --
// so when that buffer holds N of the near window, the near side takes it N
// pieces at a time, in order, and the far half is what produces that: N windows
// of the near size, ascending.
//
// N SEPARATE puts, not one N-wide descriptor -- and NOT because a consumer that
// does N gets needs N puts. It does not: fused_decode's shipped memtile feeds
// two 256-word gets from one folded 512-word BD. The reason is that N pieces is
// what was actually derived, and it is the form air-opt-memtile-dma-bds can
// recognise as a tiled run and fold to one descriptor -- the same answer it
// gives the loop spelling. A 2-D [0, 0] [N, 256] [256, 1] is already a shaped
// access and that fold never fires, so the two spellings diverge at the
// hardware.
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
//
// Two independent ways to know the reading holds, and one must:
//   COUNT -- a static near trip count must equal N, checked directly;
//   REUSE -- with a non-static count, evidence that the buffer cycles, i.e.
//            something refills it.
// A static count that disagrees is a refutation and overrides reuse evidence.

// CHECK-LABEL: func.func @derived
// The far half lands inside the fill's loop, right after the fill...
// CHECK: scf.for
// CHECK: air.channel.get{{.*}}@f
// ...as the buffer tiled by the near window: TWO puts of 256, ascending, which
// air-opt-memtile-dma-bds later folds into the one descriptor they are.
// CHECK: air.channel.put{{.*}}@x{{.*}}[0] [256] [1]
// CHECK: air.channel.put{{.*}}@x{{.*}}[256] [256] [1]
// Never the wrapped single descriptor: same bytes, but already a shaped access,
// so the tiled-run fold has nothing to recognise.
// CHECK-NOT: [2, 256] [256, 1]
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
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2_h = arith.constant 2 : index
        // the near side takes the buffer TWICE, which is what it holds
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

// NEGATIVE CONTROL: equal windows are one transfer, not a tiling. Nothing to
// derive, and the far half keeps the position and pattern it always had --
// beside the hierarchy, moving the whole buffer once.

// CHECK-LABEL: func.func @equal_windows
// CHECK: air.channel.put{{.*}}@x2{{.*}}[] [] []
// One transfer, and only one: no tiling means no pieces to emit either.
// CHECK-NOT: air.channel.put
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
// CHECK: air.channel.put{{.*}}@x3{{.*}}[] [] []
// One transfer, and only one: no tiling means no pieces to emit either.
// CHECK-NOT: air.channel.put
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

// NEGATIVE CONTROL: the near side's trip count REFUTES the tiling. It takes 256
// once; the buffer holds two of that. N is 2 against a trip count of 1, so
// "takes it N pieces at a time" is simply false, and tiling would send twice
// what was asked for -- silently, and it would compile.
//
// Nothing refills this buffer either, so neither piece of evidence is available.
// Note the order the two are applied in: a static trip count that disagrees is a
// REFUTATION, and no amount of reuse evidence overrides it.

// CHECK-LABEL: func.func @far_never_filled
// CHECK: air.channel.put{{.*}}@x4{{.*}}[] [] []
// One transfer, and only one: no tiling means no pieces to emit either.
// CHECK-NOT: air.channel.put
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
