//===- split_launch_for_padding_n_only.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-split-launch-for-padding | FileCheck %s

// Test: only N dimension needs padding (M is tile-aligned).
// M=256, N=300, tileM=128, tileN=256.
// Launch grid: 2x2. mRem=256%128=0, nRem=300%256=44.
// Only 2 partitions: interior (2x1) and N-boundary (2x1).
// No M-boundary, no corner partition.

module {
  air.channel @channel_A_l3 [2, 1]
  air.channel @channel_A_l2l1_0 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_A_l2l1_1 [1, 1] {broadcast_shape = [1, 4]}

  air.channel @channel_B_l3 [4, 1]
  air.channel @channel_B_l2l1_0 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_B_l2l1_1 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_B_l2l1_2 [1, 1] {broadcast_shape = [2, 1]}
  air.channel @channel_B_l2l1_3 [1, 1] {broadcast_shape = [2, 1]}

  air.channel @channel_C [2, 1]

  func.func @matmul_n_only(%arg0: memref<*xbf16>, %arg1: memref<*xbf16>, %arg2: memref<*xbf16>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    air.launch (%arg9, %arg10, %arg11) in (%arg12=%c2, %arg13=%c2, %arg14=%c1) args(%arg15=%arg0, %arg16=%arg1, %arg17=%arg2) : memref<*xbf16>, memref<*xbf16>, memref<*xbf16> attributes {air.actual_sizes = array<i64: 256, 300, 1>} {
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

      // L3→L2 channel.put (launch level)
      %a0 = air.channel.put async @channel_A_l3[%c0_0, %c0_0] (%arg15[%c0_0, %offset_a] [%c64, %c64] [%c256, %c1_0]) {id = 1 : i32} : (memref<*xbf16>)
      %a1 = air.channel.put async @channel_A_l3[%c1_0, %c0_0] (%arg15[%c64, %offset_a] [%c64, %c64] [%c256, %c1_0]) {id = 2 : i32} : (memref<*xbf16>)
      %b0 = air.channel.put async @channel_B_l3[%c0_0, %c0_0] (%arg16[%c0_0, %offset_b] [%c64, %c64] [%c512, %c1_0]) {id = 3 : i32} : (memref<*xbf16>)
      %b1 = air.channel.put async @channel_B_l3[%c1_0, %c0_0] (%arg16[%c0_0, %offset_b] [%c64, %c64] [%c512, %c1_0]) {id = 4 : i32} : (memref<*xbf16>)
      %b2 = air.channel.put async @channel_B_l3[%c2_0, %c0_0] (%arg16[%c0_0, %offset_b] [%c64, %c64] [%c512, %c1_0]) {id = 5 : i32} : (memref<*xbf16>)
      %b3 = air.channel.put async @channel_B_l3[%c3_0, %c0_0] (%arg16[%c0_0, %offset_b] [%c64, %c64] [%c512, %c1_0]) {id = 6 : i32} : (memref<*xbf16>)

      %c_out0 = air.channel.get async @channel_C[%c0_0, %c0_0] (%arg17[%c0_0, %offset_b] [%c64, %c256] [%c512, %c1_0]) {id = 7 : i32} : (memref<*xbf16>)
      %c_out1 = air.channel.get async @channel_C[%c1_0, %c0_0] (%arg17[%c64, %offset_b] [%c64, %c256] [%c512, %c1_0]) {id = 8 : i32} : (memref<*xbf16>)

      air.segment @seg args(%seg_arg0=%arg9, %seg_arg1=%arg10, %seg_arg2=%arg15, %seg_arg3=%arg16, %seg_arg4=%arg17) : index, index, memref<*xbf16>, memref<*xbf16>, memref<*xbf16> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c3_s = arith.constant 3 : index
        %c4_s = arith.constant 4 : index
        %c8_s = arith.constant 8 : index
        %c64_s = arith.constant 64 : index
        %c256_s = arith.constant 256 : index

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

        // L2→L1 channel.put (segment level)
        %pa0 = air.channel.put async [%ga0] @channel_A_l2l1_0[] (%alloc_a0[%c0_s, %c0_s, %c0_s] [%c8_s, %c64_s, %c8_s] [%c8_s, %c64_s, %c1_s]) {id = 15 : i32} : (memref<64x64xbf16, 1 : i32>)
        %pa1 = air.channel.put async [%ga1] @channel_A_l2l1_1[] (%alloc_a1[%c0_s, %c0_s, %c0_s] [%c8_s, %c64_s, %c8_s] [%c8_s, %c64_s, %c1_s]) {id = 16 : i32} : (memref<64x64xbf16, 1 : i32>)

        %pb0 = air.channel.put async [%gb0] @channel_B_l2l1_0[] (%alloc_b0[%c0_s, %c0_s, %c0_s, %c0_s] [%c4_s, %c8_s, %c64_s, %c8_s] [%c0_s, %c8_s, %c64_s, %c1_s]) {id = 17 : i32} : (memref<64x64xbf16, 1 : i32>)
        %pb1 = air.channel.put async [%gb1] @channel_B_l2l1_1[] (%alloc_b1[%c0_s, %c0_s, %c0_s, %c0_s] [%c4_s, %c8_s, %c64_s, %c8_s] [%c0_s, %c8_s, %c64_s, %c1_s]) {id = 18 : i32} : (memref<64x64xbf16, 1 : i32>)
        %pb2 = air.channel.put async [%gb2] @channel_B_l2l1_2[] (%alloc_b2[%c0_s, %c0_s, %c0_s, %c0_s] [%c4_s, %c8_s, %c64_s, %c8_s] [%c0_s, %c8_s, %c64_s, %c1_s]) {id = 19 : i32} : (memref<64x64xbf16, 1 : i32>)
        %pb3 = air.channel.put async [%gb3] @channel_B_l2l1_3[] (%alloc_b3[%c0_s, %c0_s, %c0_s, %c0_s] [%c4_s, %c8_s, %c64_s, %c8_s] [%c0_s, %c8_s, %c64_s, %c1_s]) {id = 20 : i32} : (memref<64x64xbf16, 1 : i32>)

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

// Only 2 partitions created (no M-boundary, no corner):
// CHECK-DAG: air.channel @channel_A_l3_interior
// CHECK-DAG: air.channel @channel_A_l3_n_boundary
// CHECK-DAG: air.channel @channel_B_l3_interior
// CHECK-DAG: air.channel @channel_B_l3_n_boundary

// CHECK-LABEL: func.func @matmul_n_only

// 2 segment names (one per partition)
// CHECK-DAG: air.segment @seg_interior
// CHECK-DAG: air.segment @seg_n_boundary

// Interior launch grid: 2x1 (full M range, interior N only)
// CHECK-DAG: air.launch ({{.*}}) in ({{.*}}=%c2{{[_0-9]*}}, {{.*}}=%c1{{[_0-9]*}}, {{.*}}=%c1{{[_0-9]*}})

// N-boundary: B shim 0 has ceil(44/8)=6 blocks, pad 2
// CHECK-DAG: @channel_B_l2l1_0_n_boundary[] {{.*}} pad_after = array<i32: 0, 2, 0, 0>

// N-boundary: A gets NO padding (M is tile-aligned)
// CHECK-DAG: @channel_A_l2l1_0_n_boundary[] (%alloc{{[^)]*}}[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}, %c0{{[_0-9]*}}] [%c8{{[_0-9]*}}, %c64{{[_0-9]*}}, %c8

// L3→L2 B shim 0 reads reduced [64, 44] in N-boundary
// CHECK-DAG: channel.put {{.*}} @channel_B_l3_n_boundary[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] {{.*}} [%c64{{[_0-9]*}}, %c44

// L3→L2 A shim sizes are NOT reduced in N-boundary (M is not padded)
// CHECK-DAG: channel.put {{.*}} @channel_A_l3_n_boundary[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] {{.*}} [%c64{{[_0-9]*}}, %c64

// S2MM: B shim 0 gets explicit strides [64, 44] [64, 1]
// CHECK-DAG: channel.get {{.*}} @channel_B_l3_n_boundary[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] (%alloc{{[^)]*}}[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] [%c64{{[_0-9]*}}, %c44{{[_0-9]*}}] [%c64{{[_0-9]*}}, %c1

// S2MM: A shims are NOT modified in N-boundary (no strides added)
// CHECK-DAG: channel.get {{.*}} @channel_A_l3_n_boundary[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] (%alloc{{[^)]*}}[] [] [])

// No M-boundary or corner partitions should exist:
// CHECK-NOT: @seg_m_boundary
// CHECK-NOT: @seg_corner
