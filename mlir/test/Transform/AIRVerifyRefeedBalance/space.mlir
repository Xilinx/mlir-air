//===- space.mlir ----------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-verify-refeed-balance -split-input-file -verify-diagnostics

// Spatial structure must not be mistaken for a rate. These are the cases that
// distinguish "many destinations" from "many tokens".

// Broadcast fan-out is free: ONE put reaches every tile of the herd, so the
// four per-tile gets are not four times the demand. Counting them as such
// would report a 4x deficit on a balanced channel.
air.channel @bcast [1, 1] {broadcast_shape = [2 : index, 2 : index]}
func.func @broadcast_is_not_a_deficit() {
  %c2 = arith.constant 2 : index
  %src = memref.alloc() : memref<64xbf16, 1 : i32>
  air.channel.put @bcast[] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  air.herd tile (%x, %y) in (%sx = %c2, %sy = %c2) {
    %dst = memref.alloc() : memref<64xbf16, 2 : i32>
    air.channel.get @bcast[%x, %y] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
    memref.dealloc %dst : memref<64xbf16, 2 : i32>
    air.herd_terminator
  }
  return
}

// -----

// Without broadcast_shape each tile drives its own bundle edge, so the herd
// must be enumerated: the subscript is a function of the tile ids, and every
// one of the four edges gets its own producer. Treating the herd body as a
// single instance would leave three edges with no consumer.
air.channel @perTile [2, 2]
func.func @herd_tiles_drive_their_own_edge() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %src = memref.alloc() : memref<64xbf16, 1 : i32>
  air.channel.put @perTile[%c0, %c0] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  air.channel.put @perTile[%c0, %c1] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  air.channel.put @perTile[%c1, %c0] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  air.channel.put @perTile[%c1, %c1] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  air.herd tile (%x, %y) in (%sx = %c2, %sy = %c2) {
    %dst = memref.alloc() : memref<64xbf16, 2 : i32>
    air.channel.get @perTile[%x, %y] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
    memref.dealloc %dst : memref<64xbf16, 2 : i32>
    air.herd_terminator
  }
  return
}

// -----

// Per-tile specialization is written as scf.if on the tile ids. With the ids
// bound the condition folds, so a tile only consumes on the branch it takes.
// Visiting both branches would charge every tile with every tile's transfer.
air.channel @split [2, 2]
func.func @scf_if_on_tile_ids_folds() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %src = memref.alloc() : memref<64xbf16, 1 : i32>
  air.channel.put @split[%c0, %c0] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  air.channel.put @split[%c0, %c1] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  air.channel.put @split[%c1, %c0] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  air.channel.put @split[%c1, %c1] (%src[] [] []) : (memref<64xbf16, 1 : i32>)
  air.herd tile (%x, %y) in (%sx = %c2, %sy = %c2) {
    %c0_i = arith.constant 0 : index
    %dst = memref.alloc() : memref<64xbf16, 2 : i32>
    %isTop = arith.cmpi eq, %x, %c0_i : index
    scf.if %isTop {
      air.channel.get @split[%x, %y] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
    } else {
      air.channel.get @split[%x, %y] (%dst[] [] []) : (memref<64xbf16, 2 : i32>)
    }
    memref.dealloc %dst : memref<64xbf16, 2 : i32>
    air.herd_terminator
  }
  return
}
