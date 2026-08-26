//===- split_launch_for_padding_affine_offsets.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-split-launch-for-padding | FileCheck %s

// The tile size is inferred from the launch block index's multiplier. The
// sibling tests spell that multiplier as its own arith.muli at segment scope;
// here the whole offset is one affine.apply per axis, and the block index
// reaches it two hops down -- through air.segment and then air.herd, both
// IsolatedFromAbove -- which is the shape a folded offset computation
// produces. The multiplier is then a coefficient inside the map rather than an
// op of its own.
//
// M=100, N=100 padded to 128x128; tileM = 32 (s0 coefficient of #map),
// tileN = 64 (s0 coefficient of #map1). mRem = 100 % 32 = 4,
// nRem = 100 % 64 = 36, so both axes are split and all four partitions exist.

// CHECK-LABEL: func.func @affine_offsets
// CHECK: air.segment @seg_interior
// CHECK: air.segment @seg_m_boundary
// CHECK: air.segment @seg_n_boundary
// CHECK: air.segment @seg_corner
// CHECK-NOT: air.actual_sizes

#map = affine_map<()[s0, s1] -> (s0 * 32 + s1 * 32)>
#map1 = affine_map<()[s0, s1] -> (s0 * 64 + s1 * 32)>
module {
  func.func @affine_offsets(%arg0: memref<128x128xf32>, %arg2: memref<128x128xf32>) {
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    air.launch (%arg3, %arg4) in (%arg5=%c4, %arg6=%c2) args(%arg7=%arg0, %arg9=%arg2) : memref<128x128xf32>, memref<128x128xf32> attributes {air.actual_sizes = array<i64: 100, 100, 1>} {
      air.segment @seg args(%arg10=%arg3, %arg11=%arg4, %arg12=%arg7, %arg14=%arg9) : index, index, memref<128x128xf32>, memref<128x128xf32> {
        %c1 = arith.constant 1 : index
        %c2_0 = arith.constant 2 : index
        air.herd @herd_0 tile (%arg15, %arg16) in (%arg17=%c1, %arg18=%c2_0) args(%arg19=%arg10, %arg20=%arg11, %arg21=%arg12, %arg23=%arg14) : index, index, memref<128x128xf32>, memref<128x128xf32> {
          %alloc = memref.alloc() : memref<32x32xf32, 2 : i32>
          %0 = affine.apply #map()[%arg19, %arg15]
          %1 = affine.apply #map1()[%arg20, %arg16]
          air.dma_memcpy_nd (%alloc[] [] [], %arg21[%0, %1] [32, 32] [128, 1]) : (memref<32x32xf32, 2 : i32>, memref<128x128xf32>)
          air.dma_memcpy_nd (%arg23[%0, %1] [32, 32] [128, 1], %alloc[] [] []) : (memref<128x128xf32>, memref<32x32xf32, 2 : i32>)
          memref.dealloc %alloc : memref<32x32xf32, 2 : i32>
        }
      }
    }
    return
  }
}
