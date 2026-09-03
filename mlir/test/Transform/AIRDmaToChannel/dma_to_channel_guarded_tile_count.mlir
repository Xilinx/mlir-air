//===- dma_to_channel_guarded_tile_count.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// A DMA carries ONE multiplicity, the consumer's: it sits inside the hierarchy,
// under whatever scf.if chain selects a tile. The producer's multiplicity has to
// be DERIVED -- it is the number of tiles satisfying that guard.
//
// Get it wrong and the shape is wrong, not just the spelling: wrapping a
// one-tile transfer in an scf.parallel over the whole iteration space issues it
// once per tile instead of once.
//
// This cannot be read off `broadcast_set`, which is the herd's BOUNDING BOX --
// for a 2x4 herd it is `0 <= s0 <= 1, 0 <= s1 <= 3` with no equalities, i.e.
// "all tiles" no matter what the guards say.

// RUN: air-opt %s -air-dependency -air-dma-to-channel -split-input-file | FileCheck %s

// tx == 0 and ty == 0 select exactly ONE tile of the 2x4 space, so the derived
// external half is a single put and needs no scf.parallel wrapper.
// CHECK-LABEL: func.func @single_tile_guard
// CHECK-NOT: scf.parallel
// CHECK: air.channel.put{{.*}}@c
air.channel @c [4]
func.func @single_tile_guard(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    air.segment @seg args(%sa=%la) : memref<64x64xi32> {
      %c0_s = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %l2 = memref.alloc() : memref<64x64xi32, 1>
      air.herd @h tile (%tx, %ty) in (%sx=%c2, %sy=%c4) args(%a=%l2) : memref<64x64xi32, 1> {
        %c0_h = arith.constant 0 : index
        %c32_h = arith.constant 32 : index
        %c64_h = arith.constant 64 : index
        %c1_h = arith.constant 1 : index
        %ex = arith.cmpi eq, %tx, %c0_h : index
        scf.if %ex {
          %ey = arith.cmpi eq, %ty, %c0_h : index
          scf.if %ey {
            %alloc = memref.alloc() : memref<32x32xi32, 2>
            air.dma_memcpy_nd (%alloc[] [] [], %a[%c0_h, %c0_h] [%c32_h, %c32_h] [%c64_h, %c1_h]) {id = 1 : i32, channel = @c, channel_indices = array<i64: 0>} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
            memref.dealloc %alloc : memref<32x32xi32, 2>
          }
        }
      }
      memref.dealloc %l2 : memref<64x64xi32, 1>
    }
  }
  return
}

// -----

// Control: an UNGUARDED DMA in the same 2x4 herd really does run on every tile,
// so the iteration space must still be hoisted as an scf.parallel. This is the
// behaviour the case above must not regress.
// CHECK-LABEL: func.func @all_tiles_no_guard
// CHECK: scf.parallel
// CHECK: air.channel.put{{.*}}@c
air.channel @c [4]
func.func @all_tiles_no_guard(%arg0: memref<64x64xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%ls=%c1) args(%la=%arg0) : memref<64x64xi32> {
    air.segment @seg args(%sa=%la) : memref<64x64xi32> {
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %l2 = memref.alloc() : memref<64x64xi32, 1>
      air.herd @h tile (%tx, %ty) in (%sx=%c2, %sy=%c4) args(%a=%l2) : memref<64x64xi32, 1> {
        %c0_h = arith.constant 0 : index
        %c32_h = arith.constant 32 : index
        %c64_h = arith.constant 64 : index
        %c1_h = arith.constant 1 : index
        %alloc = memref.alloc() : memref<32x32xi32, 2>
        air.dma_memcpy_nd (%alloc[] [] [], %a[%c0_h, %c0_h] [%c32_h, %c32_h] [%c64_h, %c1_h]) {id = 1 : i32, channel = @c, channel_indices = array<i64: 0>} : (memref<32x32xi32, 2>, memref<64x64xi32, 1>)
        memref.dealloc %alloc : memref<32x32xi32, 2>
      }
      memref.dealloc %l2 : memref<64x64xi32, 1>
    }
  }
  return
}
