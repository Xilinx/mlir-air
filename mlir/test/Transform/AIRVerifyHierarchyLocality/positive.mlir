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

// -----
// memref.store with iv-indexed offsets — each PE writes its own slot.
module {
  func.func @memref_store_iv_indexed() {
    %c4 = arith.constant 4 : index
    %a = memref.alloc() : memref<4x4xi8, 2 : i32>
    air.herd tile (%x, %y) in (%sx = %c4, %sy = %c4)
        args(%arg = %a) : memref<4x4xi8, 2 : i32> {
      %v = arith.constant 7 : i8
      memref.store %v, %arg[%x, %y] : memref<4x4xi8, 2 : i32>
      air.herd_terminator
    }
    return
  }
}

// -----
// memref.load is read-only — every PE may read the same slot, accepted
// regardless of iv-indexing.
module {
  func.func @memref_load_readonly() {
    %c4 = arith.constant 4 : index
    %a = memref.alloc() : memref<16xi8, 2 : i32>
    air.herd tile (%x, %y) in (%sx = %c4, %sy = %c4)
        args(%arg = %a) : memref<16xi8, 2 : i32> {
      %c0 = arith.constant 0 : index
      %0 = memref.load %arg[%c0] : memref<16xi8, 2 : i32>
      air.herd_terminator
    }
    return
  }
}

// -----
// vector.transfer_write with iv-indexed offset — accepted.
module {
  func.func @vector_transfer_write_iv_indexed() {
    %c4 = arith.constant 4 : index
    %a = memref.alloc() : memref<128x128xf32, 2 : i32>
    air.herd tile (%x, %y) in (%sx = %c4, %sy = %c4)
        args(%arg = %a) : memref<128x128xf32, 2 : i32> {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %ox = arith.muli %x, %c16 : index
      %oy = arith.muli %y, %c16 : index
      %v = arith.constant dense<1.0> : vector<16x16xf32>
      vector.transfer_write %v, %arg[%ox, %oy] {in_bounds = [true, true]}
        : vector<16x16xf32>, memref<128x128xf32, 2 : i32>
      air.herd_terminator
    }
    return
  }
}

// -----
// func.call with an iv-dependent scalar arg: pass-1 per-PE replication
// signal accepts (AIE external-kernel convention is the scalar is the
// per-PE offset/index).
module {
  func.func private @opaque_kernel(memref<2048xi8, 2 : i32>, i32)
  func.func @funccall_iv_dep_scalar() {
    %c2 = arith.constant 2 : index
    %a = memref.alloc() : memref<2048xi8, 2 : i32>
    air.herd tile (%x, %y) in (%sx = %c2, %sy = %c2)
        args(%arg = %a) : memref<2048xi8, 2 : i32> {
      %c1024_i32 = arith.constant 1024 : i32
      %xi32 = arith.index_cast %x : index to i32
      %off = arith.muli %xi32, %c1024_i32 : i32
      func.call @opaque_kernel(%arg, %off)
        : (memref<2048xi8, 2 : i32>, i32) -> ()
      air.herd_terminator
    }
    return
  }
}

// -----
// Multi-IV joint partition where both launch IVs co-occur in one offset
// dimension; lex-packing condition holds.
//   lvx ext=2, coeff=64.  lvy ext=4, coeff=128.  static size=16.
//   sorted (64, 128): 64 >= 16 ✓; 128 >= 16 + 64*1 = 80 ✓.
module {
  func.func @multi_iv_joint_partition(%arg0: memref<2048xf32>) {
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    air.launch (%lvx, %lvy) in (%lsx=%c2, %lsy=%c4) args(%la=%arg0) : memref<2048xf32> {
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %0 = arith.muli %lvx, %c64 : index
      %1 = arith.muli %lvy, %c128 : index
      %2 = arith.addi %0, %1 : index
      air.segment @seg args(%la_s=%la, %off_s=%2) : memref<2048xf32>, index {
        %c1_s = arith.constant 1 : index
        air.herd @herd0 tile(%tx, %ty) in (%hsx=%c1_s, %hsy=%c1_s)
            args(%la_h=%la_s, %off_h=%off_s) : memref<2048xf32>, index {
          %c1_h = arith.constant 1 : index
          %c16_h = arith.constant 16 : index
          %tile_out = memref.alloc() : memref<16xf32, 2>
          air.dma_memcpy_nd (%la_h[%off_h] [%c16_h] [%c1_h], %tile_out[] [] [])
            : (memref<2048xf32>, memref<16xf32, 2>)
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

// -----
// Inner-hierarchy IV span: launch operand whose terminal access is inside a
// nested herd, with offset = launch_iv*(tile*herd_extent) + herd_iv*tile.
// Both contribute and the launch_iv coefficient covers the herd_iv span.
//   launch ext=4, coeff=64. herd ext=4, herd_iv coeff=16. static size=16.
//   effective span = 16 + (4-1)*16 = 64. 64 >= 64 ✓.
module {
  func.func @inner_hierarchy_iv_span(%arg0: memref<512x512xf32>) {
    %c4 = arith.constant 4 : index
    air.launch (%lvx, %lvy) in (%lsx=%c4, %lsy=%c4) args(%la=%arg0) : memref<512x512xf32> {
      %c4_l = arith.constant 4 : index
      air.segment @seg args(%lvx_s=%lvx, %lvy_s=%lvy, %la_s=%la)
          : index, index, memref<512x512xf32> {
        %c4_s = arith.constant 4 : index
        %c64 = arith.constant 64 : index
        %loff_m = arith.muli %lvx_s, %c64 : index
        %loff_n = arith.muli %lvy_s, %c64 : index
        air.herd @herd0 tile(%tx, %ty) in (%hsx=%c4_s, %hsy=%c4_s)
            args(%loff_m_h=%loff_m, %loff_n_h=%loff_n, %la_h=%la_s)
            : index, index, memref<512x512xf32> {
          %c1_h = arith.constant 1 : index
          %c16_h = arith.constant 16 : index
          %c512_h = arith.constant 512 : index
          %tx_m = arith.muli %tx, %c16_h : index
          %ty_n = arith.muli %ty, %c16_h : index
          %off_m = arith.addi %loff_m_h, %tx_m : index
          %off_n = arith.addi %loff_n_h, %ty_n : index
          %tile_out = memref.alloc() : memref<16x16xf32, 2>
          air.dma_memcpy_nd
              (%la_h[%off_m, %off_n] [%c16_h, %c16_h] [%c512_h, %c1_h],
               %tile_out[] [] [])
              : (memref<512x512xf32>, memref<16x16xf32, 2>)
          memref.dealloc %tile_out : memref<16x16xf32, 2>
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----
// scf.parallel inner-iv span: launch operand written inside an scf.parallel
// loop nested in the launch body, with offset = launch_iv*128 + scf_iv*32.
//   launch ext=4, coeff=128. scf_iv ext=4, coeff=32. static size=32.
//   effective span = 32 + (4-1)*32 = 128. 128 >= 128 ✓.
#map = affine_map<()[s0, s1] -> (s0 + s1 * 32)>
module {
  air.channel @ch_scf [1, 4]
  func.func @scf_parallel_inner_iv_span(%arg0: memref<512x512xf32>) {
    %c4 = arith.constant 4 : index
    air.launch (%lvx, %lvy) in (%lsx=%c4, %lsy=%c4) args(%la=%arg0) : memref<512x512xf32> {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4_l = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c512 = arith.constant 512 : index
      %1 = arith.muli %lvx, %c64 : index
      %2 = arith.muli %lvy, %c128 : index
      scf.parallel (%arg8) = (%c0) to (%c4_l) step (%c1) {
        %3 = affine.apply #map()[%2, %arg8]
        air.channel.get @ch_scf[%c0, %arg8]
            (%la[%1, %3] [%c64, %c32] [%c512, %c1])
            : (memref<512x512xf32>)
        scf.reduce
      }
      air.launch_terminator
    }
    return
  }
}

// -----
// air.shrinkage marker on memref.alloc — verifier trusts the shrink pass's
// per-PE replication promise even when the access pattern is opaque.
module {
  func.func private @opaque(memref<32x32xbf16, 2 : i32>)
  func.func @shrinkage_on_alloc() {
    %c2 = arith.constant 2 : index
    %a = memref.alloc() {air.shrinkage = true} : memref<32x32xbf16, 2 : i32>
    air.herd tile (%x, %y) in (%sx = %c2, %sy = %c2)
        args(%arg = %a) : memref<32x32xbf16, 2 : i32> {
      func.call @opaque(%arg) : (memref<32x32xbf16, 2 : i32>) -> ()
      air.herd_terminator
    }
    return
  }
}

// -----
// air.shrinkage on the wrapping air.execute (more durable across folding).
module {
  func.func private @opaque2(memref<32x32xbf16, 2 : i32>)
  func.func @shrinkage_on_execute() {
    %c2 = arith.constant 2 : index
    %tok, %a = air.execute -> (memref<32x32xbf16, 2 : i32>) {
      %0 = memref.alloc() {air.shrinkage = true} : memref<32x32xbf16, 2 : i32>
      air.execute_terminator %0 : memref<32x32xbf16, 2 : i32>
    }
    %h = air.herd @h0 async [%tok] tile (%x, %y) in (%sx = %c2, %sy = %c2)
        args(%arg = %a) : memref<32x32xbf16, 2 : i32> {
      func.call @opaque2(%arg) : (memref<32x32xbf16, 2 : i32>) -> ()
      air.herd_terminator
    }
    return
  }
}

// -----
// Two-pass per-PE replication: a channel.get with iv-dependent channel
// index (pass-1 signal) trumps a sibling unrecognized terminal that would
// otherwise trip the per-operand check.
module {
  air.channel @ch_perPE [4]
  func.func private @opaque3(memref<32x32xbf16, 2 : i32>)
  func.func @two_pass_perPE_replication() {
    %c4 = arith.constant 4 : index
    %a = memref.alloc() : memref<32x32xbf16, 2 : i32>
    air.herd tile (%x, %y) in (%sx = %c4, %sy = %c4)
        args(%arg = %a) : memref<32x32xbf16, 2 : i32> {
      // Per-PE replication signal: channel index depends on %x.
      air.channel.get @ch_perPE[%x] (%arg[] [] [])
        : (memref<32x32xbf16, 2 : i32>)
      // Sibling terminal that would otherwise fail; pass-1 already accepted.
      func.call @opaque3(%arg) : (memref<32x32xbf16, 2 : i32>) -> ()
      air.herd_terminator
    }
    return
  }
}
