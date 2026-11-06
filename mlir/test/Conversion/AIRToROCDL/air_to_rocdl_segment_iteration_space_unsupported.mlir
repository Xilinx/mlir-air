//===- air_to_rocdl_segment_iteration_space_unsupported.mlir ----*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl -verify-diagnostics

// -air-to-rocdl flattens air.segment away, which runs its body exactly once.
// gpu.launch has two coordinate levels and the pass spends both -- air.launch
// becomes the grid, air.herd the block -- so there is nowhere to map a wider
// segment iteration space onto. Running 1 of the 2 instances would silently
// compute half the work, so reject instead.

module {
  func.func @segment_iteration_space_too_wide(%arg0: memref<64xf32, #air.symmetric_heap>) {
    %c1 = arith.constant 1 : index
    air.launch (%bx, %by) in (%nbx=%c1, %nby=%c1) args(%in0=%arg0) : memref<64xf32, #air.symmetric_heap> {
      // air.launch is IsolatedFromAbove: sizes must come from inside it.
      %c2_l = arith.constant 2 : index
      %c1_l = arith.constant 1 : index
      // expected-error @+1 {{air.segment iteration space is not supported by -air-to-rocdl}}
      air.segment @seg unroll (%sx, %sy) in (%nsx=%c2_l, %nsy=%c1_l) args(%s0=%in0) : memref<64xf32, #air.symmetric_heap> {
        %c4_s = arith.constant 4 : index
        %c1_s = arith.constant 1 : index
        air.herd @herd tile (%tx, %ty) in (%ntx=%c4_s, %nty=%c1_s) args(%h0=%s0, %hid=%sx) : memref<64xf32, #air.symmetric_heap>, index {
          %val = memref.load %h0[%hid] : memref<64xf32, #air.symmetric_heap>
          %sum = arith.addf %val, %val : f32
          memref.store %sum, %h0[%hid] : memref<64xf32, #air.symmetric_heap>
        }
      }
    }
    return
  }
}
