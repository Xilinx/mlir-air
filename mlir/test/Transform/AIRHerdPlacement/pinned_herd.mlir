//===- pinned_herd.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -verify-diagnostics \
// RUN:   -air-place-herds='num-rows=4 num-cols=4 row-anchor=2 col-anchor=0' \
// RUN:   | FileCheck %s

// Herds carrying both x_loc and y_loc are pinned: their cells are reserved
// before automatic placement runs, so unpinned siblings route around them.

// CHECK-LABEL: @pin_honored
// CHECK: air.herd @pinned
// CHECK-SAME: x_loc = 2 : i64
// CHECK-SAME: y_loc = 4 : i64
// CHECK: air.herd @unpinned
// CHECK-SAME: x_loc = 0 : i64
// CHECK-SAME: y_loc = 2 : i64
func.func @pin_honored() {
  air.segment @seg {
    %c1 = arith.constant 1 : index
    // Pinned at physical (x=2, y=4). Segment anchor is (col=0, row=2)
    // and the segment is 4x4, so seg-relative (col=2, row=2) is legal.
    air.herd @pinned tile (%t0, %t1) in (%s0=%c1, %s1=%c1) attributes {x_loc = 2 : i64, y_loc = 4 : i64} {
    }
    air.herd @unpinned tile (%t0, %t1) in (%s0=%c1, %s1=%c1) {
    }
  }
  return
}

// -----

// Pinning outside the segment extent warns (naming the herd, the original
// physical coordinates, and the segment bounds) and falls back to the
// regular placer.

// CHECK-LABEL: @pin_out_of_bounds
// CHECK: air.herd @oob
// CHECK-SAME: x_loc = 0 : i64
// CHECK-SAME: y_loc = 2 : i64
func.func @pin_out_of_bounds() {
  air.segment @seg {
    %c1 = arith.constant 1 : index
    // expected-warning @below {{ignoring user-pinned x_loc/y_loc (10, 2) on air.herd 'oob': position is outside the segment (segment anchor (0, 2), extent 4x4)}}
    air.herd @oob tile (%t0, %t1) in (%s0=%c1, %s1=%c1) attributes {x_loc = 10 : i64, y_loc = 2 : i64} {
    }
  }
  return
}

// -----

// Two herds pinned to the same cell: first wins, second warns about the
// overlap and falls through to the placer (which picks the next free cell).

// CHECK-LABEL: @pin_overlap
// CHECK: air.herd @first
// CHECK-SAME: x_loc = 0 : i64
// CHECK-SAME: y_loc = 2 : i64
// CHECK: air.herd @second
// CHECK-SAME: x_loc = 1 : i64
// CHECK-SAME: y_loc = 2 : i64
func.func @pin_overlap() {
  air.segment @seg {
    %c1 = arith.constant 1 : index
    air.herd @first tile (%t0, %t1) in (%s0=%c1, %s1=%c1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
    }
    // expected-warning @below {{ignoring user-pinned x_loc/y_loc (0, 2) on air.herd 'second': position overlaps a previously-pinned herd}}
    air.herd @second tile (%t0, %t1) in (%s0=%c1, %s1=%c1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
    }
  }
  return
}
