//===- positive.mlir -------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-verify-hierarchy-locality -split-input-file -verify-diagnostics

// Cases the verifier must accept silently.

// -----
// (R1) L1 alloc inside the herd body — implicit per-PE replication.
module {
  func.func @r1_inside_body() {
    %c4 = arith.constant 4 : index
    air.herd tile (%x, %y) in (%sx = %c4, %sy = %c4) {
      %a = memref.alloc() : memref<8x8xi8, 2 : i32>
      memref.dealloc %a : memref<8x8xi8, 2 : i32>
      air.herd_terminator
    }
    return
  }
}

// -----
// (R2) Outside-herd L1 alloc whose leading dims match the herd shape and
// access is indexed by (%x, %y). Each PE owns slice [%x, %y, ...].
module {
  func.func @r2_shape_match_leading() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c32 = arith.constant 32 : index
    %c128 = arith.constant 128 : index
    %c256 = arith.constant 256 : index
    %c2048 = arith.constant 2048 : index
    %c8192 = arith.constant 8192 : index
    %src = memref.alloc() : memref<32768xi16, 1 : i32>
    %a = memref.alloc() : memref<4x4x8x32x4x8xi16, 2 : i32>
    air.herd tile (%x, %y) in (%sx = %c4, %sy = %c4)
        args(%arg = %a, %s = %src) :
        memref<4x4x8x32x4x8xi16, 2 : i32>, memref<32768xi16, 1 : i32> {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c4_0 = arith.constant 4 : index
      %c8_0 = arith.constant 8 : index
      %c32_0 = arith.constant 32 : index
      %c8192_0 = arith.constant 8192 : index
      %c2048_0 = arith.constant 2048 : index
      %c256_0 = arith.constant 256 : index
      air.dma_memcpy_nd
          (%arg[%x, %y, %c0_0, %c0_0, %c0_0, %c0_0]
                [%c1_0, %c1_0, %c8_0, %c32_0, %c4_0, %c8_0]
                [%c8192_0, %c2048_0, %c256_0, %c8_0, %c32_0, %c1_0],
           %s[%c0_0] [%c8192_0] [%c1_0])
          : (memref<4x4x8x32x4x8xi16, 2 : i32>, memref<32768xi16, 1 : i32>)
      air.herd_terminator
    }
    return
  }
}

// -----
// (R2) Same as above but herd-indexed dims are not leading.
// Memref is <8x4x32x4x...>, access is [c, %x, c, %y, ...] with size 1 in
// dims 1 and 3.
module {
  func.func @r2_interleaved_dims() {
    %c4 = arith.constant 4 : index
    %src = memref.alloc() : memref<8192xi16, 1 : i32>
    %a = memref.alloc() : memref<8x4x32x4x16xi16, 2 : i32>
    air.herd tile (%x, %y) in (%sx = %c4, %sy = %c4)
        args(%arg = %a, %s = %src) :
        memref<8x4x32x4x16xi16, 2 : i32>, memref<8192xi16, 1 : i32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c8192 = arith.constant 8192 : index
      // Sizes: [8, 1, 32, 1, 16] — IVs are in dims 1 and 3, both size 1.
      air.dma_memcpy_nd
          (%arg[%c0, %x, %c0, %y, %c0]
                [%c8, %c1, %c32, %c1, %c16]
                [%c1, %c1, %c1, %c1, %c1],
           %s[%c0] [%c8192] [%c1])
          : (memref<8x4x32x4x16xi16, 2 : i32>, memref<8192xi16, 1 : i32>)
      air.herd_terminator
    }
    return
  }
}

// -----
// (R2) Outside-herd L1 alloc, accessed through affine.apply on herd IVs.
// %x is multiplied by 16, and the access size in that dim is 16, so
// distinct %x values yield disjoint ranges.
#map = affine_map<()[s0] -> (s0 * 16)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
module {
  func.func @r2_through_affine_apply() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %src = memref.alloc() : memref<4096xi16, 1 : i32>
    %a = memref.alloc() : memref<64x64xi16, 2 : i32>
    air.herd tile (%x, %y) in (%sx = %c4, %sy = %c4)
        args(%arg = %a, %s = %src) :
        memref<64x64xi16, 2 : i32>, memref<4096xi16, 1 : i32> {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c16_0 = arith.constant 16 : index
      %c4096_0 = arith.constant 4096 : index
      %ox = affine.apply #map()[%x]
      %oy = affine.apply #map1()[%y]
      air.dma_memcpy_nd
          (%arg[%ox, %oy] [%c16_0, %c16_0] [%c1_0, %c1_0],
           %s[%c0_0] [%c4096_0] [%c1_0])
          : (memref<64x64xi16, 2 : i32>, memref<4096xi16, 1 : i32>)
      air.herd_terminator
    }
    return
  }
}

// -----
// Cross-level passthrough: an L2 buffer (segment-shared) passed into a herd
// as a DMA source. Memory level (L2) does NOT match the herd's matching
// level (L1), so the rule does not apply — verifier silent even with a
// whole-buffer DMA from every PE.
module {
  func.func @cross_level_passthrough() {
    %c4 = arith.constant 4 : index
    %l2 = memref.alloc() : memref<32x32xi16, 1 : i32>
    %l1 = memref.alloc() : memref<32x32xi16, 2 : i32>
    air.herd tile (%x, %y) in (%sx = %c4, %sy = %c4)
        args(%dst = %l1, %src = %l2) :
        memref<32x32xi16, 2 : i32>, memref<32x32xi16, 1 : i32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      // The L1 dst happens to be referenced as a whole — but `%dst` is
      // disjoint by being defined inside the herd in real flows; here it
      // would fail R1/R2. Sink it inside to satisfy R1.
      %dst_inside = memref.alloc() : memref<32x32xi16, 2 : i32>
      // L2 source: rule does not apply — verifier silent regardless of
      // whether all PEs read the whole buffer.
      air.dma_memcpy_nd
          (%dst_inside[%c0, %c0] [%c32, %c32] [%c32, %c1],
           %src[%c0, %c0] [%c32, %c32] [%c32, %c1])
          : (memref<32x32xi16, 2 : i32>, memref<32x32xi16, 1 : i32>)
      memref.dealloc %dst_inside : memref<32x32xi16, 2 : i32>
      air.herd_terminator
    }
    return
  }
}

// -----
// Trivial iteration space (size 1): the IV exists but with only one
// iteration value, every access is vacuously disjoint across iterations.
// This is the relu/mul-style pattern where torch_mlir produces a 1×1
// air.launch wrapping a unary kernel; the launch arg is consumed by a
// channel.put with no IV in offsets, which is correct for a single
// iteration.
module {
  air.channel @ch [1]
  func.func @trivial_launch_size_one(%arg0: memref<10240xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%i) in (%si = %c1) args(%a = %arg0) : memref<10240xi32> {
      %c0 = arith.constant 0 : index
      %c10240 = arith.constant 10240 : index
      %c1_0 = arith.constant 1 : index
      air.channel.put @ch[%c0] (%a[%c0] [%c10240] [%c1_0])
          : (memref<10240xi32>)
      air.launch_terminator
    }
    return
  }
}
