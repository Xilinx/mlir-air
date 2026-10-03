//===- lock_race_fix_auto.mlir ----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// aircc runs air-to-aie with use-lock-race-condition-fix-auto unless told not
// to or given one of the other two lock race fixes.

// RUN: rm -rf %t && mkdir -p %t
// RUN: aircc %s --device=npu2 --tmpdir=%t/npu2 --output-format=none -v > %t/npu2.log 2>&1 || true
// RUN: FileCheck %s --input-file=%t/npu2.log --check-prefix=ON
// RUN: aircc %s --device=npu1 --tmpdir=%t/npu1 --output-format=none -v > %t/npu1.log 2>&1 || true
// RUN: FileCheck %s --input-file=%t/npu1.log --check-prefix=ON
// RUN: aircc %s --device=npu2 --tmpdir=%t/off --output-format=none -v --use-lock-race-condition-fix-auto=false > %t/off.log 2>&1 || true
// RUN: FileCheck %s --input-file=%t/off.log --check-prefix=OFF
// RUN: aircc %s --device=npu2 --tmpdir=%t/v1 --output-format=none -v --use-lock-race-condition-fix > %t/v1.log 2>&1 || true
// RUN: FileCheck %s --input-file=%t/v1.log --check-prefix=V1
// RUN: aircc %s --device=npu2 --tmpdir=%t/v2 --output-format=none -v --use-lock-race-condition-fix-v2 > %t/v2.log 2>&1 || true
// RUN: FileCheck %s --input-file=%t/v2.log --check-prefix=V2

// ON: air-to-aie{{.*}} use-lock-race-condition-fix=false use-lock-race-condition-fix-auto=true use-lock-race-condition-fix-v2=false
// OFF: air-to-aie{{.*}} use-lock-race-condition-fix=false use-lock-race-condition-fix-auto=false use-lock-race-condition-fix-v2=false
// V1: air-to-aie{{.*}} use-lock-race-condition-fix=true use-lock-race-condition-fix-auto=false use-lock-race-condition-fix-v2=false
// V2: air-to-aie{{.*}} use-lock-race-condition-fix=false use-lock-race-condition-fix-auto=false use-lock-race-condition-fix-v2=true

module {
  func.func @copy(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    air.herd @herd_0 tile (%x, %y) in (%sx = %c1, %sy = %c1) args(%a = %arg0, %b = %arg1) : memref<64xi32>, memref<64xi32> {
      %buf = memref.alloc() : memref<64xi32, 2 : i32>
      air.dma_memcpy_nd (%buf[] [] [], %a[] [] []) : (memref<64xi32, 2 : i32>, memref<64xi32>)
      air.dma_memcpy_nd (%b[] [] [], %buf[] [] []) : (memref<64xi32>, memref<64xi32, 2 : i32>)
      memref.dealloc %buf : memref<64xi32, 2 : i32>
    }
    return
  }
}
