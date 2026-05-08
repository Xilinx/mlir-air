//===- strict_option.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Inconclusive analysis: the IV appears in an offset, but the access size in
// that dim is not statically determinable. With strict=true (default) -> error;
// with strict=false -> warning, pass succeeds.

// RUN: not air-opt %s -air-verify-hierarchy-locality 2>&1 | FileCheck %s --check-prefix=STRICT
// RUN: air-opt %s -air-verify-hierarchy-locality=strict=false 2>&1 | FileCheck %s --check-prefix=LAX

// STRICT: error: 'air.herd' op kernel operand #0
// LAX:    warning: kernel operand #0

module {
  func.func @inconclusive(%dyn_size: index) {
    %c4 = arith.constant 4 : index
    %src = memref.alloc() : memref<256xi8, 1 : i32>
    %a = memref.alloc() : memref<4x4x16xi8, 2 : i32>
    air.herd tile (%x, %y) in (%sx = %c4, %sy = %c4)
        args(%arg = %a, %s = %src, %sz = %dyn_size) :
        memref<4x4x16xi8, 2 : i32>, memref<256xi8, 1 : i32>, index {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c256 = arith.constant 256 : index
      // %x and %y appear in offsets, but the size in the IV-indexed dim
      // is %sz (dynamic) — analysis cannot prove that distinct (%x, %y)
      // produce disjoint regions because it doesn't know the size.
      air.dma_memcpy_nd
          (%arg[%x, %y, %c0] [%sz, %sz, %c16] [%c1, %c1, %c1],
           %s[%c0] [%c256] [%c1])
          : (memref<4x4x16xi8, 2 : i32>, memref<256xi8, 1 : i32>)
      air.herd_terminator
    }
    return
  }
}
