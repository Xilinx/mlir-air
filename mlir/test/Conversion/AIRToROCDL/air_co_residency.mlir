//===- air_co_residency.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl='max-resident-workgroups=8' -verify-diagnostics

// air.launch is not "run these in some order": it promises its segments are
// all resident when the body begins (AIROpBase.td, `launch`;
// AIRComputeModel.md section 2.2). A grid the device has to serialize cannot
// keep that promise, so the lowering rejects it instead of quietly downgrading
// the guarantee to a scheduling hint.
//
// The failure is not academic. air.chiplet_dim_blocks waits until every
// workgroup has reported in; a workgroup that was never scheduled never
// reports, and the kernel hangs rather than returning a wrong answer.

module {
  func.func @too_many(%arg0: memref<8xf32>) {
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    // 4 x 3 = 12 workgroups, but this target holds 8.
    // expected-error @+1 {{air.launch asks for 12 co-resident workgroups, more than the 8 this target holds}}
    air.launch (%bx, %by) in (%nbx=%c4, %nby=%c3) args(%out=%arg0) : memref<8xf32> {
      air.segment @seg args(%s0=%out) : memref<8xf32> {
        %c1_s = arith.constant 1 : index
        air.herd @herd tile (%tx, %ty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
