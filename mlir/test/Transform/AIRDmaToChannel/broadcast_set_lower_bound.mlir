//===- broadcast_set_lower_bound.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel | FileCheck %s

// A broadcast to the tiles of a set that starts at column 1: the channel's
// broadcast shape is the set's extent, [1, 2], and a receiver is indexed by its
// position in the set, its tile less the set's lower bounds. Indexed by the tile
// itself, the receiver at column 2 would be outside the shape.

// CHECK: air.channel @[[C:.*]] [1, 1] {broadcast_shape = [1, 2]}
// CHECK-LABEL: func.func @set_from_column_one
// CHECK: air.herd
// CHECK: air.channel.get @[[C]][%c0{{.*}}, %c1{{.*}}]
#set = affine_set<()[s0, s1] : (s0 == 0, s1 - 1 >= 0, -s1 + 2 >= 0)>
func.func @set_from_column_one(%arg0: memref<32xi32>) {
  %c1 = arith.constant 1 : index
  air.launch (%lx) in (%sx = %c1) args(%in = %arg0) : memref<32xi32> {
    air.segment @seg args(%i = %in) : memref<32xi32> {
      %c1_0 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      air.herd @h tile (%tx, %ty) in (%nx = %c1_0, %ny = %c3) args(%s = %i) : memref<32xi32> {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %buf = memref.alloc() : memref<32xi32, 2>
        affine.if #set()[%c0, %c2] {
          air.dma_memcpy_nd (%buf[] [] [], %s[] [] []) {broadcast_set = #set} : (memref<32xi32, 2>, memref<32xi32>)
        }
        memref.dealloc %buf : memref<32xi32, 2>
      }
    }
  }
  return
}
