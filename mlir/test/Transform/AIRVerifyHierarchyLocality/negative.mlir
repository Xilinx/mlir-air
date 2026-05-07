//===- negative.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-verify-hierarchy-locality -split-input-file -verify-diagnostics

// Cases the verifier must reject.

// -----
// Issue #1545 alloc_2 pattern: outside-herd L1 alloc with leading 1x1 dims
// for a 4x4 herd, accessed by every PE as the entire buffer.
module {
  func.func @neg_overlapping_l1() {
    %c4 = arith.constant 4 : index
    %src = memref.alloc() : memref<4096xi8, 1 : i32>
    %a = memref.alloc() : memref<1x1x4x32x4x8xi8, 2 : i32>
    // expected-error@+1 {{kernel operand #0}}
    air.herd tile (%x, %y) in (%sx = %c4, %sy = %c4)
        args(%arg = %a, %s = %src) :
        memref<1x1x4x32x4x8xi8, 2 : i32>, memref<4096xi8, 1 : i32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4096 = arith.constant 4096 : index
      // Whole-buffer access: %arg[][][] — every PE writes the same 4KB.
      // expected-note@+1 {{access here}}
      air.dma_memcpy_nd
          (%arg[] [] [], %s[%c0] [%c4096] [%c1])
          : (memref<1x1x4x32x4x8xi8, 2 : i32>, memref<4096xi8, 1 : i32>)
      air.herd_terminator
    }
    return
  }
}

// -----
// Outside-herd L1 alloc with shape 4x1 for a 4x4 herd, indexed only by %x.
// Distinct (%x, %y) pairs with the same %x produce overlapping accesses.
module {
  func.func @neg_partial_indexing() {
    %c4 = arith.constant 4 : index
    %src = memref.alloc() : memref<1024xi8, 1 : i32>
    %a = memref.alloc() : memref<4x1x32x8xi8, 2 : i32>
    // expected-error@+1 {{kernel operand #0}}
    air.herd tile (%x, %y) in (%sx = %c4, %sy = %c4)
        args(%arg = %a, %s = %src) :
        memref<4x1x32x8xi8, 2 : i32>, memref<1024xi8, 1 : i32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c32 = arith.constant 32 : index
      %c1024 = arith.constant 1024 : index
      // expected-note@+1 {{access here}}
      air.dma_memcpy_nd
          (%arg[%x, %c0, %c0, %c0] [%c1, %c1, %c32, %c8]
                [%c1, %c1, %c1, %c1],
           %s[%c0] [%c1024] [%c1])
          : (memref<4x1x32x8xi8, 2 : i32>, memref<1024xi8, 1 : i32>)
      air.herd_terminator
    }
    return
  }
}

// -----
// Outside-segment L2 alloc that all segment iterations overlap on. Confirms
// the rule applies symmetrically at the segment level.
module {
  func.func @neg_overlapping_l2() {
    %c2 = arith.constant 2 : index
    %src = memref.alloc() : memref<32x32xi8>
    %a = memref.alloc() : memref<32x32xi16, 1 : i32>
    // expected-error@+1 {{kernel operand #0}}
    air.segment @seg unroll(%i) in (%si = %c2)
        args(%arg = %a, %s = %src) :
        memref<32x32xi16, 1 : i32>, memref<32x32xi8> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      // Whole-buffer access from every segment iteration.
      // expected-note@+1 {{access here}}
      air.dma_memcpy_nd
          (%arg[] [] [], %s[%c0, %c0] [%c32, %c32] [%c32, %c1])
          : (memref<32x32xi16, 1 : i32>, memref<32x32xi8>)
      air.segment_terminator
    }
    return
  }
}
