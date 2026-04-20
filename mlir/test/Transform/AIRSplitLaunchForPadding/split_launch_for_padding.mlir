//===- split_launch_for_padding.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-split-launch-for-padding | FileCheck %s

// M=300, M_TILE=128 → launchM=3, last M-block has 44 rows (300-2*128=44)
// N=300, N_TILE=256 → launchN=2, last N-block has 44 cols (300-256=44)
// L2→L1 A: sizes [8, 64, 8] strides [8, 64, 1] → padDimIdx=1 (M-rows), innerBlock=1
//   shim 0: 44 actual rows → pad 20 (≤31, hardware ok)
//   shim 1: 0 actual rows → skip (all garbage)
// L2→L1 B: sizes [4, 8, 64, 8] strides [0, 8, 64, 1] → padDimIdx=1 (N-blocks), innerBlock=8
//   shim 0-2: full → no pad
//   shim 3: 44-3*64=-148 → skip (out of range, handled by shimOffset)
//   Actually N=44 actual cols per N-block. 4 shims of 64 each: shim 0 has min(64,44)=44
//   Wait, N_actualLast=44 with 4 shims: shim 0: 44 cols, shims 1-3: 0 → skip
//   actualBlocks for shim 0 = ceil(44/8) = 6, padBlocks = 8-6 = 2

module {
  // A input channels: L3→L2 (2 shims) and L2→L1 (2 channels)
  air.channel @channel_A_l3 [2, 1]
  air.channel @channel_A_l2l1_0 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_A_l2l1_1 [1, 1] {broadcast_shape = [1, 4]}

  // B input channels: L3→L2 (4 shims) and L2→L1 (4 channels)
  air.channel @channel_B_l3 [4, 1]
  air.channel @channel_B_l2l1_0 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_B_l2l1_1 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_B_l2l1_2 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_B_l2l1_3 [1, 1] {broadcast_shape = [2, 1]}

  // C output channel
  air.channel @channel_C [2, 1]

  func.func @matmul_bf16(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index

    air.launch (%arg9, %arg10, %arg11) in (%arg12=%c3, %arg13=%c2, %arg14=%c1) args(%arg15=%arg0, %arg16=%arg1, %arg17=%arg2) : memref<*xbf16>, memref<*xbf16>, memref<*xbf16> attributes {air.actual_sizes = array<i64: 300, 300, 1>} {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %c512 = arith.constant 512 : index
      %c4 = arith.constant 4 : index
      %c3_0 = arith.constant 3 : index
      %c2_0 = arith.constant 2 : index

      %offset_a = arith.muli %arg9, %c128 : index
      %offset_b = arith.muli %arg10, %c256 : index

      // L3→L2 channel.put (launch level, shim DMAs)
      %a0 = air.channel.put async @channel_A_l3[%c0_0, %c0_0] (%arg15[%c0_0, %offset_a] [%c64, %c64] [%c256, %c1_0]) {id = 1 : i32} : (memref<*xbf16>)
      %a1 = air.channel.put async @channel_A_l3[%c1_0, %c0_0] (%arg15[%c64, %offset_a] [%c64, %c64] [%c256, %c1_0]) {id = 2 : i32} : (memref<*xbf16>)
      %b0 = air.channel.put async @channel_B_l3[%c0_0, %c0_0] (%arg16[%c0_0, %offset_b] [%c64, %c64] [%c512, %c1_0]) {id = 3 : i32} : (memref<*xbf16>)
      %b1 = air.channel.put async @channel_B_l3[%c1_0, %c0_0] (%arg16[%c0_0, %offset_b] [%c64, %c64] [%c512, %c1_0]) {id = 4 : i32} : (memref<*xbf16>)
      %b2 = air.channel.put async @channel_B_l3[%c2_0, %c0_0] (%arg16[%c0_0, %offset_b] [%c64, %c64] [%c512, %c1_0]) {id = 5 : i32} : (memref<*xbf16>)
      %b3 = air.channel.put async @channel_B_l3[%c3_0, %c0_0] (%arg16[%c0_0, %offset_b] [%c64, %c64] [%c512, %c1_0]) {id = 6 : i32} : (memref<*xbf16>)

      // L2→L3 output channel.get (launch level, shim DMA)
      %c_out0 = air.channel.get async @channel_C[%c0_0, %c0_0] (%arg17[%c0_0, %offset_b] [%c64, %c256] [%c512, %c1_0]) {id = 7 : i32} : (memref<*xbf16>)
      %c_out1 = air.channel.get async @channel_C[%c1_0, %c0_0] (%arg17[%c64, %offset_b] [%c64, %c256] [%c512, %c1_0]) {id = 8 : i32} : (memref<*xbf16>)

      air.segment @matmul_bf16_0 args(%seg_arg0=%arg9, %seg_arg1=%arg10, %seg_arg2=%arg15, %seg_arg3=%arg16, %seg_arg4=%arg17) : index, index, memref<*xbf16>, memref<*xbf16>, memref<*xbf16> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c3_s = arith.constant 3 : index
        %c4_s = arith.constant 4 : index
        %c8_s = arith.constant 8 : index
        %c64_s = arith.constant 64 : index
        %c256_s = arith.constant 256 : index

        // L2 buffers
        %alloc_a0 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_a1 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_b0 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_b1 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_b2 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_b3 = memref.alloc() : memref<64x64xbf16, 1 : i32>
        %alloc_c = memref.alloc() : memref<128x256xbf16, 1 : i32>

        // L3→L2 channel.get (segment level, memtile S2MM)
        %ga0 = air.channel.get async @channel_A_l3[%c0_s, %c0_s] (%alloc_a0[] [] []) {id = 9 : i32} : (memref<64x64xbf16, 1 : i32>)
        %ga1 = air.channel.get async @channel_A_l3[%c1_s, %c0_s] (%alloc_a1[] [] []) {id = 10 : i32} : (memref<64x64xbf16, 1 : i32>)
        %gb0 = air.channel.get async @channel_B_l3[%c0_s, %c0_s] (%alloc_b0[] [] []) {id = 11 : i32} : (memref<64x64xbf16, 1 : i32>)
        %gb1 = air.channel.get async @channel_B_l3[%c1_s, %c0_s] (%alloc_b1[] [] []) {id = 12 : i32} : (memref<64x64xbf16, 1 : i32>)
        %gb2 = air.channel.get async @channel_B_l3[%c2_s, %c0_s] (%alloc_b2[] [] []) {id = 13 : i32} : (memref<64x64xbf16, 1 : i32>)
        %gb3 = air.channel.get async @channel_B_l3[%c3_s, %c0_s] (%alloc_b3[] [] []) {id = 14 : i32} : (memref<64x64xbf16, 1 : i32>)

        // L2→L1 channel.put (segment level, memtile MM2S)
        // A: 3D tiled [8, 64, 8] strides [8, 64, 1]
        %pa0 = air.channel.put async [%ga0] @channel_A_l2l1_0[] (%alloc_a0[%c0_s, %c0_s, %c0_s] [%c8_s, %c64_s, %c8_s] [%c8_s, %c64_s, %c1_s]) {id = 15 : i32} : (memref<64x64xbf16, 1 : i32>)
        %pa1 = air.channel.put async [%ga1] @channel_A_l2l1_1[] (%alloc_a1[%c0_s, %c0_s, %c0_s] [%c8_s, %c64_s, %c8_s] [%c8_s, %c64_s, %c1_s]) {id = 16 : i32} : (memref<64x64xbf16, 1 : i32>)

        // B: 4D tiled [4, 8, 64, 8] strides [0, 8, 64, 1]
        %pb0 = air.channel.put async [%gb0] @channel_B_l2l1_0[] (%alloc_b0[%c0_s, %c0_s, %c0_s, %c0_s] [%c4_s, %c8_s, %c64_s, %c8_s] [%c0_s, %c8_s, %c64_s, %c1_s]) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
        %pb1 = air.channel.put async [%gb1] @channel_B_l2l1_1[] (%alloc_b1[%c0_s, %c0_s, %c0_s, %c0_s] [%c4_s, %c8_s, %c64_s, %c8_s] [%c0_s, %c8_s, %c64_s, %c1_s]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
        %pb2 = air.channel.put async [%gb2] @channel_B_l2l1_2[] (%alloc_b2[%c0_s, %c0_s, %c0_s, %c0_s] [%c4_s, %c8_s, %c64_s, %c8_s] [%c0_s, %c8_s, %c64_s, %c1_s]) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
        %pb3 = air.channel.put async [%gb3] @channel_B_l2l1_3[] (%alloc_b3[%c0_s, %c0_s, %c0_s, %c0_s] [%c4_s, %c8_s, %c64_s, %c8_s] [%c0_s, %c8_s, %c64_s, %c1_s]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)

        // L2→L3 output channel.put (segment level, memtile MM2S)
        %pc0 = air.channel.put async @channel_C[%c0_s, %c0_s] (%alloc_c[%c0_s, %c0_s] [%c64_s, %c256_s] [%c256_s, %c1_s]) {id = 21 : i32} : (memref<128x256xbf16, 1 : i32>)
        %pc1 = air.channel.put async @channel_C[%c1_s, %c0_s] (%alloc_c[%c64_s, %c0_s] [%c64_s, %c256_s] [%c256_s, %c1_s]) {id = 22 : i32} : (memref<128x256xbf16, 1 : i32>)

        memref.dealloc %alloc_a0 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_a1 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_b0 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_b1 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_b2 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_b3 : memref<64x64xbf16, 1 : i32>
        memref.dealloc %alloc_c : memref<128x256xbf16, 1 : i32>
      }
    }
    return
  }
}

// Channel declarations for 4 partitions (interior, m_boundary, n_boundary, corner)
// CHECK-DAG: air.channel @channel_A_l3_interior
// CHECK-DAG: air.channel @channel_A_l3_m_boundary
// CHECK-DAG: air.channel @channel_A_l3_n_boundary
// CHECK-DAG: air.channel @channel_A_l3_corner
// CHECK-DAG: air.channel @channel_B_l3_interior
// CHECK-DAG: air.channel @channel_B_l3_corner

// CHECK-LABEL: func.func @matmul_bf16

// 4 unique segment names (one per partition)
// CHECK-DAG: air.segment @matmul_bf16_0_interior
// CHECK-DAG: air.segment @matmul_bf16_0_m_boundary
// CHECK-DAG: air.segment @matmul_bf16_0_n_boundary
// CHECK-DAG: air.segment @matmul_bf16_0_corner

// Interior launch has multi-iteration grid (2x1)
// CHECK-DAG: air.launch ({{.*}}) in ({{.*}}=%c2{{[_0-9]*}}, {{.*}}=%c1{{[_0-9]*}}, {{.*}}=%c1{{[_0-9]*}})

// M-boundary: A shim 0 has 44 actual M-rows, pad 20
// CHECK-DAG: @channel_A_l2l1_0_m_boundary[] {{.*}} pad_after = array<i32: 0, 20, 0>

// N-boundary: B shim 0 has ceil(44/8)=6 blocks, pad 2
// CHECK-DAG: @channel_B_l2l1_0_n_boundary[] {{.*}} pad_after = array<i32: 0, 2, 0, 0>

// Corner: both A and B padding
// CHECK-DAG: @channel_A_l2l1_0_corner[] {{.*}} pad_after = array<i32: 0, 20, 0>
// CHECK-DAG: @channel_B_l2l1_0_corner[] {{.*}} pad_after = array<i32: 0, 2, 0, 0>

// L3→L2 shim DMA sizes reduced for boundary blocks:
// M-boundary: A shim 0 reads 64 K-rows x 44 M-cols (M reduced from 64 to 44)
// CHECK-DAG: channel.put {{.*}} @channel_A_l3_m_boundary[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] {{.*}} [%c64{{[_0-9]*}}, %c44
// N-boundary: B shim 0 reads 64 K-rows x 44 N-cols (N reduced from 64 to 44)
// CHECK-DAG: channel.put {{.*}} @channel_B_l3_n_boundary[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] {{.*}} [%c64{{[_0-9]*}}, %c44

// Non-boundary shims retain original sizes (no reduction):
// M-boundary: A shim 1 keeps full [64, 64] sizes
// CHECK-DAG: channel.put {{.*}} @channel_A_l3_m_boundary[%c1{{[_0-9]*}}, %c0{{[_0-9]*}}] {{.*}} [%c64{{[_0-9]*}}, %c64
// N-boundary: B shim 1 keeps full [64, 64] sizes
// CHECK-DAG: channel.put {{.*}} @channel_B_l3_n_boundary[%c1{{[_0-9]*}}, %c0{{[_0-9]*}}] {{.*}} [%c64{{[_0-9]*}}, %c64

// L3→L2 channel.get with explicit strides for boundary shims (memtile S2MM):
// A boundary: shim 0 gets sizes [44, 64] with strides [64, 1]
// CHECK-DAG: channel.get {{.*}} @channel_A_l3_m_boundary[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] (%alloc{{[^)]*}}[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] [%c44{{[_0-9]*}}, %c64{{[_0-9]*}}] [%c64{{[_0-9]*}}, %c1
// B boundary: shim 0 gets sizes [64, 44] with strides [64, 1]
// CHECK-DAG: channel.get {{.*}} @channel_B_l3_n_boundary[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] (%alloc{{[^)]*}}[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] [%c64{{[_0-9]*}}, %c44{{[_0-9]*}}] [%c64{{[_0-9]*}}, %c1
// Non-boundary shims: no explicit strides added (empty sizes)
// CHECK-DAG: channel.get {{.*}} @channel_A_l3_m_boundary[%c1{{[_0-9]*}}, %c0{{[_0-9]*}}] (%alloc{{[^)]*}}[] [] [])
