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

// -----
// memref.store with constant index — every PE writes the same slot, race.
module {
  func.func @neg_memref_store_const_idx() {
    %c4 = arith.constant 4 : index
    %a = memref.alloc() : memref<2048xi8, 2 : i32>
    // expected-error@+1 {{kernel operand #0}}
    air.herd tile (%x, %y) in (%sx = %c4, %sy = %c4)
        args(%arg = %a) : memref<2048xi8, 2 : i32> {
      %c0 = arith.constant 0 : index
      %v = arith.constant 7 : i8
      // expected-note@+1 {{access here}}
      memref.store %v, %arg[%c0] : memref<2048xi8, 2 : i32>
      air.herd_terminator
    }
    return
  }
}

// -----
// vector.transfer_write with constant offsets — every PE writes the same
// region, race.
module {
  func.func @neg_vector_transfer_write_const() {
    %c4 = arith.constant 4 : index
    %a = memref.alloc() : memref<128x128xf32, 2 : i32>
    // expected-error@+1 {{kernel operand #0}}
    air.herd tile (%x, %y) in (%sx = %c4, %sy = %c4)
        args(%arg = %a) : memref<128x128xf32, 2 : i32> {
      %c0 = arith.constant 0 : index
      %v = arith.constant dense<1.0> : vector<16x16xf32>
      // expected-note@+1 {{access here}}
      vector.transfer_write %v, %arg[%c0, %c0] {in_bounds = [true, true]}
        : vector<16x16xf32>, memref<128x128xf32, 2 : i32>
      air.herd_terminator
    }
    return
  }
}

// -----
// func.call with NO iv-dependent operand — the per-PE replication signal
// does not fire and the call is opaque, so it bails as unrecognized.
module {
  func.func private @opaque_no_iv(memref<2048xi8, 2 : i32>, i32)
  func.func @neg_funccall_no_iv_dep() {
    %c2 = arith.constant 2 : index
    %a = memref.alloc() : memref<2048xi8, 2 : i32>
    // expected-error@+1 {{kernel operand #0}}
    air.herd tile (%x, %y) in (%sx = %c2, %sy = %c2)
        args(%arg = %a) : memref<2048xi8, 2 : i32> {
      %c1024 = arith.constant 1024 : i32
      // expected-note@+1 {{access here}}
      func.call @opaque_no_iv(%arg, %c1024)
        : (memref<2048xi8, 2 : i32>, i32) -> ()
      air.herd_terminator
    }
    return
  }
}

// -----
// Multi-IV joint partition where the smallest coefficient is below the
// access size — distinct iv tuples produce overlapping accesses.
//   lvx ext=2, coeff=4. lvy ext=4, coeff=8. static size=16.
//   sorted (4, 8): 4 >= 16? NO → reject.
module {
  func.func @neg_joint_partition_overlap(%arg0: memref<512xf32>) {
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    // expected-error@+1 {{kernel operand #0}}
    air.launch (%lvx, %lvy) in (%lsx=%c2, %lsy=%c4) args(%la=%arg0)
        : memref<512xf32> {
      %c4_l = arith.constant 4 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %0 = arith.muli %lvx, %c4_l : index
      %1 = arith.muli %lvy, %c8 : index
      %2 = arith.addi %0, %1 : index
      air.segment @seg args(%la_s=%la, %off_s=%2)
          : memref<512xf32>, index {
        %c1_s = arith.constant 1 : index
        air.herd @herd0 tile(%tx, %ty) in (%hsx=%c1_s, %hsy=%c1_s)
            args(%la_h=%la_s, %off_h=%off_s) : memref<512xf32>, index {
          %c1_h = arith.constant 1 : index
          %c16_h = arith.constant 16 : index
          %tile_out = memref.alloc() : memref<16xf32, 2>
          // expected-note@+1 {{access here}}
          air.dma_memcpy_nd
              (%la_h[%off_h] [%c16_h] [%c1_h], %tile_out[] [] [])
              : (memref<512xf32>, memref<16xf32, 2>)
          memref.dealloc %tile_out : memref<16xf32, 2>
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
