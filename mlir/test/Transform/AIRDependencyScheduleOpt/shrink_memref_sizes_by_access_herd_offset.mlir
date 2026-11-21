//===- shrink_memref_sizes_by_access_herd_offset.mlir ----------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-shrink-memref-sizes-by-access -split-input-file | FileCheck %s

// An offset computed from tile ids is rewritten for the shrunk buffer only
// where the rewrite knows how. Where the rewritten offset could still reach
// past the shrunk size, the buffer is not shrunk.

// Tile x gets 2 elements at 4x + 1. Shrunk to 2 elements, the offset would
// keep its + 1 and run past the end.

// CHECK-LABEL: func.func @iv_plus_const
// CHECK: memref.alloc() : memref<8xi32, 2 : i32>
air.channel @C [2, 1]
func.func @iv_plus_const() {
  %0 = air.segment @s async {
    %c2 = arith.constant 2 : index
    %cy1 = arith.constant 1 : index
    %t, %a = air.execute -> (memref<8xi32, 2 : i32>) {
      %al = memref.alloc() : memref<8xi32, 2 : i32>
      air.execute_terminator %al : memref<8xi32, 2 : i32>
    }
    %h = air.herd @h async [%t] tile (%x, %y) in (%sx=%c2, %sy=%cy1) args(%buf=%a) : memref<8xi32, 2 : i32> {
      %c1 = arith.constant 1 : index
      %c2_0 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %o0 = arith.muli %x, %c4 : index
      %o = arith.addi %o0, %c1 : index
      %g = air.channel.get async @C[%x, %y] (%buf[%o] [%c2_0] [%c1]) : (memref<8xi32, 2 : i32>)
    }
  }
  return
}

// -----

// The tile id reaches the offset two ops down, which the rewrite does not
// reset.

// CHECK-LABEL: func.func @iv_through_cast
// CHECK: memref.alloc() : memref<8xi32, 2 : i32>
air.channel @C [2, 1]
func.func @iv_through_cast() {
  %0 = air.segment @s async {
    %c2 = arith.constant 2 : index
    %cy1 = arith.constant 1 : index
    %t, %a = air.execute -> (memref<8xi32, 2 : i32>) {
      %al = memref.alloc() : memref<8xi32, 2 : i32>
      air.execute_terminator %al : memref<8xi32, 2 : i32>
    }
    %h = air.herd @h async [%t] tile (%x, %y) in (%sx=%c2, %sy=%cy1) args(%buf=%a) : memref<8xi32, 2 : i32> {
      %c1 = arith.constant 1 : index
      %c4_i32 = arith.constant 4 : i32
      %c4 = arith.constant 4 : index
      %xi = arith.index_cast %x : index to i32
      %o0 = arith.muli %xi, %c4_i32 : i32
      %o = arith.index_cast %o0 : i32 to index
      %g = air.channel.get async @C[%x, %y] (%buf[%o] [%c4] [%c1]) : (memref<8xi32, 2 : i32>)
    }
  }
  return
}

// -----

// Tile x gets 4 elements at 4x: each tile's part starts at 0 of its shrunk
// buffer.

// CHECK-LABEL: func.func @iv_only
// CHECK: memref.alloc() {air.shrinkage = true} : memref<4xi32, 2 : i32>
// CHECK: air.channel.get {{.*}}[0] [%{{.*}}] [1]
air.channel @C [2, 1]
func.func @iv_only() {
  %0 = air.segment @s async {
    %c2 = arith.constant 2 : index
    %cy1 = arith.constant 1 : index
    %t, %a = air.execute -> (memref<8xi32, 2 : i32>) {
      %al = memref.alloc() : memref<8xi32, 2 : i32>
      air.execute_terminator %al : memref<8xi32, 2 : i32>
    }
    %h = air.herd @h async [%t] tile (%x, %y) in (%sx=%c2, %sy=%cy1) args(%buf=%a) : memref<8xi32, 2 : i32> {
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %o = arith.muli %x, %c4 : index
      %g = air.channel.get async @C[%x, %y] (%buf[%o] [%c4] [%c1]) : (memref<8xi32, 2 : i32>)
    }
  }
  return
}
