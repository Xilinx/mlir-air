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

// -----
// The access is indexed by (%x, %y) in the leading dims, but its strides do not
// follow the memref's layout: the dims of size 32 and 4 have strides 8 and 32,
// so a tile reaches element 2143 past its start, beyond the next tile's start
// at 2048.
module {
  func.func @neg_strides_spill_into_next_tile() {
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
    // expected-error@+1 {{the strides let one iteration's access reach}}
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
      // expected-note@+1 {{access here}}
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
// %x and %y index dims 1 and 3, but every stride is 1, so the tiles' accesses
// all overlap.
module {
  func.func @neg_unit_strides_overlap() {
    %c4 = arith.constant 4 : index
    %src = memref.alloc() : memref<8192xi16, 1 : i32>
    %a = memref.alloc() : memref<8x4x32x4x16xi16, 2 : i32>
    // expected-error@+1 {{the strides let one iteration's access reach}}
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
      // expected-note@+1 {{access here}}
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
// Tiles of 16x16 at (16 * %x, 16 * %y) with a row stride of 1 instead of 64:
// tile (1, 0) and tile (0, 1) both start at element 16.
#map = affine_map<()[s0] -> (s0 * 16)>
#map1 = affine_map<()[s0] -> (s0 * 16)>
module {
  func.func @neg_unit_row_stride_overlap() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %src = memref.alloc() : memref<4096xi16, 1 : i32>
    %a = memref.alloc() : memref<64x64xi16, 2 : i32>
    // expected-error@+1 {{the strides let one iteration's access reach}}
    air.herd tile (%x, %y) in (%sx = %c4, %sy = %c4)
        args(%arg = %a, %s = %src) :
        memref<64x64xi16, 2 : i32>, memref<4096xi16, 1 : i32> {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c16_0 = arith.constant 16 : index
      %c4096_0 = arith.constant 4096 : index
      %ox = affine.apply #map()[%x]
      %oy = affine.apply #map1()[%y]
      // expected-note@+1 {{access here}}
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
// The same column blocks with a block stride of 32: four blocks of 64 columns
// cover only 160, and the column tiles 128 apart overlap.
#map = affine_map<()[s0] -> (s0 * 256)>
#map1 = affine_map<()[s0] -> (s0 * 128)>
module {
  air.channel @channel_0 [1, 1]
  func.func @neg_column_blocks_overlap(%arg0: memref<2560x2304xbf16>) {
    %c10 = arith.constant 10 : index
    %c9 = arith.constant 9 : index
    // expected-error@+1 {{the strides let one iteration's access reach}}
    air.launch (%x, %y) in (%sx = %c10, %sy = %c9) args(%out = %arg0) : memref<2560x2304xbf16> {
      %row = affine.apply #map()[%x]
      %col = affine.apply #map1()[%y]
      // expected-note@+1 {{access here}}
      air.channel.get @channel_0[] (%out[0, %row, 0, %col] [1, 32, 4, 64] [73728, 2304, 32, 1]) : (memref<2560x2304xbf16>)
    }
    return
  }
}
