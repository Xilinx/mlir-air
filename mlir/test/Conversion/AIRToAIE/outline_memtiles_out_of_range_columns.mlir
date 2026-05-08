//===- outline_memtiles_out_of_range_columns.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Regression test for RFC #1567 (PR #1570): outlineAIEMemtiles must filter
// memtile columns that fall outside the device's column range. Before the
// fix, getPhysTileOp would silently fabricate an aie.tile with an out-of-range
// column (e.g. aie.tile(4, 1) on npu1, which only has columns 0..3), producing
// invalid IR. After the fix, the column-bounds check (colHasMemTile) drops
// those columns up-front so the SequentialPlacer is only asked to place
// columns the device actually has.

// RUN: air-opt %s -air-to-aie='test-patterns=to-aie-mlir col-offset=3 row-offset=2 device=npu1' 2>&1 | FileCheck %s

// npu1 has 4 columns (0..3) with memtiles in row 1. With col-offset=3 and
// segment x_size=2, the segment requests memtile columns 3 (valid) and 4
// (out of range). Only the in-range memtile must be created.

// CHECK-LABEL: aie.device(npu1)
// CHECK: aie.tile(3, 1)
// CHECK-NOT: aie.tile(4,
// CHECK-NOT: aie.tile(5,

module {
  func.func @out_of_range_memtile_cols() {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) {
      air.segment @segment_0 attributes {x_size = 2 : i64} {
        %c1_0 = arith.constant 1 : index
        air.herd @herd_0 tile (%tx, %ty) in (%htx=%c1_0, %hty=%c1_0) {
        }
      }
    }
    return
  }
}
