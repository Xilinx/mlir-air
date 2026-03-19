//===- split_launch_for_padding_edge_cases.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-split-launch-for-padding | FileCheck %s

// Test 1: Tile-aligned dimensions — pass should be a no-op.
// M=256, N=256 with tileM=128, tileN=128 → mRem=0, nRem=0.

// CHECK-LABEL: func.func @tile_aligned
// CHECK: air.launch (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{[_0-9]*}}, %{{.*}}=%c2{{[_0-9]*}})
// CHECK-NOT: _m_boundary
// CHECK-NOT: _n_boundary
// CHECK-NOT: _corner

module {
  air.channel @channel_aligned_A [2, 1]

  func.func @tile_aligned(%arg0: memref<256x256xf32>, %arg1: memref<256x256xf32>) {
    %c2 = arith.constant 2 : index
    air.launch (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg0, %arg8=%arg1) : memref<256x256xf32>, memref<256x256xf32> attributes {air.actual_sizes = array<i64: 256, 256, 1>} {
      %c0_0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %offset_m = arith.muli %arg3, %c128 : index
      %offset_n = arith.muli %arg4, %c128 : index
      %t0 = air.channel.put async @channel_aligned_A[%arg3, %c0_0] (%arg7[%offset_m, %c0_0] [%c128, %c128] [%c128, %c0_0]) {id = 1 : i32} : (memref<256x256xf32>)
      air.segment @seg args(%s0=%arg7) : memref<256x256xf32> {
        %alloc = memref.alloc() : memref<128x128xf32, 1 : i32>
        memref.dealloc %alloc : memref<128x128xf32, 1 : i32>
      }
    }
    return
  }
}

// -----

// Test 2: Missing air.actual_sizes attribute — pass should be a no-op.

// CHECK-LABEL: func.func @no_actual_sizes
// CHECK: air.launch (%{{.*}}, %{{.*}}) in (%{{.*}}=%c2{{[_0-9]*}}, %{{.*}}=%c2{{[_0-9]*}})
// CHECK-NOT: _m_boundary

module {
  air.channel @channel_noattr_A [2, 1]

  func.func @no_actual_sizes(%arg0: memref<256x256xf32>, %arg1: memref<256x256xf32>) {
    %c2 = arith.constant 2 : index
    air.launch (%arg3, %arg4) in (%arg5=%c2, %arg6=%c2) args(%arg7=%arg0, %arg8=%arg1) : memref<256x256xf32>, memref<256x256xf32> {
      %c0_0 = arith.constant 0 : index
      %c128 = arith.constant 128 : index
      %offset_m = arith.muli %arg3, %c128 : index
      %offset_n = arith.muli %arg4, %c128 : index
      %t0 = air.channel.put async @channel_noattr_A[%arg3, %c0_0] (%arg7[%offset_m, %c0_0] [%c128, %c128] [%c128, %c0_0]) {id = 1 : i32} : (memref<256x256xf32>)
      air.segment @seg args(%s0=%arg7) : memref<256x256xf32> {
        %alloc = memref.alloc() : memref<128x128xf32, 1 : i32>
        memref.dealloc %alloc : memref<128x128xf32, 1 : i32>
      }
    }
    return
  }
}
