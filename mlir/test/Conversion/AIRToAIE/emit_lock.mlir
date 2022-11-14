//===- emit_lock.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='emit-herd-lock=true' | FileCheck %s

// CHECK:  %[[VAL_0:.*]] = AIE.tile
// CHECK:  %[[VAL_2:.*]] = AIE.lock(%[[VAL_0]],
// CHECK:  %[[VAL_3:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:
// CHECK:    AIE.useLock(%[[VAL_2]], Acquire, 0)
// CHECK:    cf.br ^bb2
// CHECK:  ^bb2:
// CHECK:    AIE.useLock(%[[VAL_2]], Release, 0)
// CHECK:    AIE.end
func.func @func1() -> () {
  %herd_cols = arith.constant 1 : index
  %herd_rows = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) {
    air.herd_terminator
  }
  return
}
