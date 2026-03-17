//===- split_launch_for_padding_4d.mlir -------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-split-launch-for-padding | FileCheck %s

// Test: @module_builder IR pattern with per-herd 4D L3→L2 channel puts,
// rank-4 L2 memrefs, and 3D L2→L1 channel puts.
//
// M=500, N=500 with tileM*herdM=256, tileN*herdN=128.
// Launch grid: 2x4. Per-herd tiles: tileM=64, tileN=32.
//
// M boundary: mRem = 500%256 = 244. 4 herds of 64:
//   Herd 0: 64 (full), Herd 1: 64 (full), Herd 2: 64 (full), Herd 3: 52 (pad 12)
// N boundary: nRem = 500%128 = 116. 4 herds of 32:
//   Herd 0: 32 (full), Herd 1: 32 (full), Herd 2: 32 (full), Herd 3: 20 (pad 12)

module {
  // A L3→L2 channel (4 per-herd shims)
  air.channel @channel_A_l3 [4, 1]
  // A L2→L1 channels (4 per-herd, broadcast to 4 N-herds)
  air.channel @channel_A_l2l1_0 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_A_l2l1_1 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_A_l2l1_2 [1, 1] {broadcast_shape = [1, 4]}
  air.channel @channel_A_l2l1_3 [1, 1] {broadcast_shape = [1, 4]}

  // B L3→L2 channel (4 per-herd shims)
  air.channel @channel_B_l3 [4, 1]
  // B L2→L1 channels (4 per-herd, broadcast to 4 M-herds)
  air.channel @channel_B_l2l1_0 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_B_l2l1_1 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_B_l2l1_2 [1, 1] {broadcast_shape = [4, 1]}
  air.channel @channel_B_l2l1_3 [1, 1] {broadcast_shape = [4, 1]}

  // C output channel
  air.channel @channel_C [4, 1]

  func.func @matmul_f32(%arg0: memref<784x504xf32>, %arg1: memref<784x504xf32>, %arg2: memref<512x512xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index

    air.launch (%arg3, %arg4) in (%arg5=%c2, %arg6=%c4) args(%arg7=%arg0, %arg8=%arg1, %arg9=%arg2) : memref<784x504xf32>, memref<784x504xf32>, memref<512x512xf32> attributes {air.actual_sizes = array<i64: 500, 500, 1>} {
      %c0_0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2_0 = arith.constant 2 : index
      %c3_0 = arith.constant 3 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c256 = arith.constant 256 : index
      %c504 = arith.constant 504 : index
      %c512 = arith.constant 512 : index
      %c8064 = arith.constant 8064 : index

      // Launch offsets: M uses %arg3*256, N uses %arg4*128
      %offset_m = arith.muli %arg3, %c256 : index
      %offset_n = arith.muli %arg4, %c128 : index

      // L3→L2 A channel.put: 4 per-herd, 4D sizes [1, 1, 64, 16]
      // offsets [shimIdx, 0, offset_m, 0] — M dimension at offset index 2
      %a0 = air.channel.put async @channel_A_l3[%c0_0, %c0_0] (%arg7[%c0_0, %c0_0, %offset_m, %c0_0] [%c1_0, %c1_0, %c64, %c16] [%c64, %c8064, %c1_0, %c504]) {id = 1 : i32} : (memref<784x504xf32>)
      %a1 = air.channel.put async @channel_A_l3[%c1_0, %c0_0] (%arg7[%c1_0, %c0_0, %offset_m, %c0_0] [%c1_0, %c1_0, %c64, %c16] [%c64, %c8064, %c1_0, %c504]) {id = 2 : i32} : (memref<784x504xf32>)
      %a2 = air.channel.put async @channel_A_l3[%c2_0, %c0_0] (%arg7[%c2_0, %c0_0, %offset_m, %c0_0] [%c1_0, %c1_0, %c64, %c16] [%c64, %c8064, %c1_0, %c504]) {id = 3 : i32} : (memref<784x504xf32>)
      %a3 = air.channel.put async @channel_A_l3[%c3_0, %c0_0] (%arg7[%c3_0, %c0_0, %offset_m, %c0_0] [%c1_0, %c1_0, %c64, %c16] [%c64, %c8064, %c1_0, %c504]) {id = 4 : i32} : (memref<784x504xf32>)

      // L3→L2 B channel.put: 4 per-herd, 4D sizes [1, 1, 16, 32]
      // offsets [0, shimIdx, 0, offset_n] — N dimension at offset index 3
      %b0 = air.channel.put async @channel_B_l3[%c0_0, %c0_0] (%arg8[%c0_0, %c0_0, %c0_0, %offset_n] [%c1_0, %c1_0, %c16, %c32] [%c8064, %c32, %c504, %c1_0]) {id = 5 : i32} : (memref<784x504xf32>)
      %b1 = air.channel.put async @channel_B_l3[%c1_0, %c0_0] (%arg8[%c0_0, %c1_0, %c0_0, %offset_n] [%c1_0, %c1_0, %c16, %c32] [%c8064, %c32, %c504, %c1_0]) {id = 6 : i32} : (memref<784x504xf32>)
      %b2 = air.channel.put async @channel_B_l3[%c2_0, %c0_0] (%arg8[%c0_0, %c2_0, %c0_0, %offset_n] [%c1_0, %c1_0, %c16, %c32] [%c8064, %c32, %c504, %c1_0]) {id = 7 : i32} : (memref<784x504xf32>)
      %b3 = air.channel.put async @channel_B_l3[%c3_0, %c0_0] (%arg8[%c0_0, %c3_0, %c0_0, %offset_n] [%c1_0, %c1_0, %c16, %c32] [%c8064, %c32, %c504, %c1_0]) {id = 8 : i32} : (memref<784x504xf32>)

      // L3 C output channel.get (4 per-herd)
      %c_out0 = air.channel.get async @channel_C[%c0_0, %c0_0] (%arg9[%c0_0, %offset_m, %c0_0, %offset_n] [%c1_0, %c64, %c1_0, %c32] [%c512, %c512, %c32, %c1_0]) {id = 9 : i32} : (memref<512x512xf32>)

      air.segment @matmul_seg args(%seg_arg0=%arg7, %seg_arg1=%arg8) : memref<784x504xf32>, memref<784x504xf32> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c3_s = arith.constant 3 : index
        %c4_s = arith.constant 4 : index
        %c8_s = arith.constant 8 : index
        %c16_s = arith.constant 16 : index
        %c32_s = arith.constant 32 : index
        %c64_s = arith.constant 64 : index

        // L2 buffers: rank-4 per-herd (NOT rank-2 like test 53)
        %alloc_a0 = memref.alloc() : memref<1x1x64x16xf32, 1 : i32>
        %alloc_a1 = memref.alloc() : memref<1x1x64x16xf32, 1 : i32>
        %alloc_a2 = memref.alloc() : memref<1x1x64x16xf32, 1 : i32>
        %alloc_a3 = memref.alloc() : memref<1x1x64x16xf32, 1 : i32>
        %alloc_b0 = memref.alloc() : memref<1x1x16x32xf32, 1 : i32>
        %alloc_b1 = memref.alloc() : memref<1x1x16x32xf32, 1 : i32>
        %alloc_b2 = memref.alloc() : memref<1x1x16x32xf32, 1 : i32>
        %alloc_b3 = memref.alloc() : memref<1x1x16x32xf32, 1 : i32>

        // L3→L2 channel.get (segment level, memtile S2MM)
        %ga0 = air.channel.get async @channel_A_l3[%c0_s, %c0_s] (%alloc_a0[] [] []) {id = 13 : i32} : (memref<1x1x64x16xf32, 1 : i32>)
        %ga1 = air.channel.get async @channel_A_l3[%c1_s, %c0_s] (%alloc_a1[] [] []) {id = 14 : i32} : (memref<1x1x64x16xf32, 1 : i32>)
        %ga2 = air.channel.get async @channel_A_l3[%c2_s, %c0_s] (%alloc_a2[] [] []) {id = 15 : i32} : (memref<1x1x64x16xf32, 1 : i32>)
        %ga3 = air.channel.get async @channel_A_l3[%c3_s, %c0_s] (%alloc_a3[] [] []) {id = 16 : i32} : (memref<1x1x64x16xf32, 1 : i32>)
        %gb0 = air.channel.get async @channel_B_l3[%c0_s, %c0_s] (%alloc_b0[] [] []) {id = 17 : i32} : (memref<1x1x16x32xf32, 1 : i32>)
        %gb1 = air.channel.get async @channel_B_l3[%c1_s, %c0_s] (%alloc_b1[] [] []) {id = 18 : i32} : (memref<1x1x16x32xf32, 1 : i32>)
        %gb2 = air.channel.get async @channel_B_l3[%c2_s, %c0_s] (%alloc_b2[] [] []) {id = 19 : i32} : (memref<1x1x16x32xf32, 1 : i32>)
        %gb3 = air.channel.get async @channel_B_l3[%c3_s, %c0_s] (%alloc_b3[] [] []) {id = 20 : i32} : (memref<1x1x16x32xf32, 1 : i32>)

        // L2→L1 channel.put (segment level, memtile MM2S)
        // A: 3D sizes [2, 64, 8] strides [8, 16, 1]
        %pa0 = air.channel.put async [%ga0] @channel_A_l2l1_0[] (%alloc_a0[%c0_s, %c0_s, %c0_s] [%c2_s, %c64_s, %c8_s] [%c8_s, %c16_s, %c1_s]) {id = 21 : i32} : (memref<1x1x64x16xf32, 1 : i32>)
        %pa1 = air.channel.put async [%ga1] @channel_A_l2l1_1[] (%alloc_a1[%c0_s, %c0_s, %c0_s] [%c2_s, %c64_s, %c8_s] [%c8_s, %c16_s, %c1_s]) {id = 22 : i32} : (memref<1x1x64x16xf32, 1 : i32>)
        %pa2 = air.channel.put async [%ga2] @channel_A_l2l1_2[] (%alloc_a2[%c0_s, %c0_s, %c0_s] [%c2_s, %c64_s, %c8_s] [%c8_s, %c16_s, %c1_s]) {id = 23 : i32} : (memref<1x1x64x16xf32, 1 : i32>)
        %pa3 = air.channel.put async [%ga3] @channel_A_l2l1_3[] (%alloc_a3[%c0_s, %c0_s, %c0_s] [%c2_s, %c64_s, %c8_s] [%c8_s, %c16_s, %c1_s]) {id = 24 : i32} : (memref<1x1x64x16xf32, 1 : i32>)

        // B: 3D sizes [4, 16, 8] strides [8, 32, 1]
        %pb0 = air.channel.put async [%gb0] @channel_B_l2l1_0[] (%alloc_b0[%c0_s, %c0_s, %c0_s] [%c4_s, %c16_s, %c8_s] [%c8_s, %c32_s, %c1_s]) {id = 25 : i32} : (memref<1x1x16x32xf32, 1 : i32>)
        %pb1 = air.channel.put async [%gb1] @channel_B_l2l1_1[] (%alloc_b1[%c0_s, %c0_s, %c0_s] [%c4_s, %c16_s, %c8_s] [%c8_s, %c32_s, %c1_s]) {id = 26 : i32} : (memref<1x1x16x32xf32, 1 : i32>)
        %pb2 = air.channel.put async [%gb2] @channel_B_l2l1_2[] (%alloc_b2[%c0_s, %c0_s, %c0_s] [%c4_s, %c16_s, %c8_s] [%c8_s, %c32_s, %c1_s]) {id = 27 : i32} : (memref<1x1x16x32xf32, 1 : i32>)
        %pb3 = air.channel.put async [%gb3] @channel_B_l2l1_3[] (%alloc_b3[%c0_s, %c0_s, %c0_s] [%c4_s, %c16_s, %c8_s] [%c8_s, %c32_s, %c1_s]) {id = 28 : i32} : (memref<1x1x16x32xf32, 1 : i32>)

        memref.dealloc %alloc_a0 : memref<1x1x64x16xf32, 1 : i32>
        memref.dealloc %alloc_a1 : memref<1x1x64x16xf32, 1 : i32>
        memref.dealloc %alloc_a2 : memref<1x1x64x16xf32, 1 : i32>
        memref.dealloc %alloc_a3 : memref<1x1x64x16xf32, 1 : i32>
        memref.dealloc %alloc_b0 : memref<1x1x16x32xf32, 1 : i32>
        memref.dealloc %alloc_b1 : memref<1x1x16x32xf32, 1 : i32>
        memref.dealloc %alloc_b2 : memref<1x1x16x32xf32, 1 : i32>
        memref.dealloc %alloc_b3 : memref<1x1x16x32xf32, 1 : i32>
      }
    }
    return
  }
}

// Channel declarations for 4 partitions
// CHECK-DAG: air.channel @channel_A_l3_interior
// CHECK-DAG: air.channel @channel_A_l3_m_boundary
// CHECK-DAG: air.channel @channel_A_l3_n_boundary
// CHECK-DAG: air.channel @channel_A_l3_corner

// CHECK-LABEL: func.func @matmul_f32

// 4 unique segment names (one per partition)
// CHECK-DAG: air.segment @matmul_seg_interior
// CHECK-DAG: air.segment @matmul_seg_m_boundary
// CHECK-DAG: air.segment @matmul_seg_n_boundary
// CHECK-DAG: air.segment @matmul_seg_corner

// Interior launch has multi-iteration grid (1x3)
// CHECK-DAG: air.launch ({{.*}}) in ({{.*}}=%c1{{[_0-9]*}}, {{.*}}=%c3{{[_0-9]*}})

// M-boundary: herd 3 (shimIdx=3) has 52 actual M-rows (244 - 3*64 = 52), pad 12
// L3→L2 A channel.put sizes reduced from [1,1,64,16] to [1,1,52,16]
// CHECK-DAG: @channel_A_l3_m_boundary[%c3{{[_0-9]*}}, %c0{{[_0-9]*}}] {{.*}} [%c1{{[_0-9]*}}, %c1{{[_0-9]*}}, %c52{{[_0-9]*}}, %c16
// L2→L1 A channel.put for herd 3: dim1 padded from 64 to 52, pad_after 12
// CHECK-DAG: @channel_A_l2l1_3_m_boundary[] {{.*}} pad_after = array<i32: 0, 12, 0>

// N-boundary: herd 3 (shimIdx=3) has 20 actual N-cols (116 - 3*32 = 20)
// L3→L2 B channel.put sizes reduced from [1,1,16,32] to [1,1,16,20]
// CHECK-DAG: @channel_B_l3_n_boundary[%c3{{[_0-9]*}}, %c0{{[_0-9]*}}] {{.*}} [%c1{{[_0-9]*}}, %c1{{[_0-9]*}}, %c16{{[_0-9]*}}, %c20
// L2→L1 B channel.put for herd 3: dim0 padded (N-blocks: 4→3, with innerBlock=8, pad=1)
// CHECK-DAG: @channel_B_l2l1_3_n_boundary[] {{.*}} pad_after = array<i32: 1, 0, 0>

// Non-boundary herds retain original sizes (no padding):
// M-boundary: A herd 0 keeps full [2, 64, 8] L2→L1 sizes (no pad_after)
// CHECK-DAG: @channel_A_l2l1_0_m_boundary[] (%alloc{{[^)]*}}[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}, %c0{{[_0-9]*}}] [%c2{{[_0-9]*}}, %c64{{[_0-9]*}}, %c8{{[_0-9]*}}]
// N-boundary: B herd 0 keeps full [4, 16, 8] L2→L1 sizes (no pad_after)
// CHECK-DAG: @channel_B_l2l1_0_n_boundary[] (%alloc{{[^)]*}}[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}, %c0{{[_0-9]*}}] [%c4{{[_0-9]*}}, %c16{{[_0-9]*}}, %c8{{[_0-9]*}}]

// S2MM explicit strides for boundary shims (rank-4 memrefs):
// M-boundary A herd 3: sizes [1,1,52,16] strides [1024,1024,16,1]
// CHECK-DAG: channel.get {{.*}} @channel_A_l3_m_boundary[%c3{{[_0-9]*}}, %c0{{[_0-9]*}}] (%alloc{{[^)]*}}[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}, %c0{{[_0-9]*}}, %c0{{[_0-9]*}}] [%c1{{[_0-9]*}}, %c1{{[_0-9]*}}, %c52{{[_0-9]*}}, %c16{{[_0-9]*}}] [%c1024{{[_0-9]*}}, %c1024{{[_0-9]*}}, %c16{{[_0-9]*}}, %c1
// N-boundary B herd 3: sizes [1,1,16,20] strides [512,512,32,1]
// CHECK-DAG: channel.get {{.*}} @channel_B_l3_n_boundary[%c3{{[_0-9]*}}, %c0{{[_0-9]*}}] (%alloc{{[^)]*}}[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}, %c0{{[_0-9]*}}, %c0{{[_0-9]*}}] [%c1{{[_0-9]*}}, %c1{{[_0-9]*}}, %c16{{[_0-9]*}}, %c20{{[_0-9]*}}] [%c512{{[_0-9]*}}, %c512{{[_0-9]*}}, %c32{{[_0-9]*}}, %c1
// Non-boundary shims: no strides added
// CHECK-DAG: channel.get {{.*}} @channel_A_l3_m_boundary[%c0{{[_0-9]*}}, %c0{{[_0-9]*}}] (%alloc{{[^)]*}}[] [] [])

// Corner partition: both A and B have padding
// CHECK-DAG: @channel_A_l2l1_3_corner[] {{.*}} pad_after = array<i32: 0, 12, 0>
// CHECK-DAG: @channel_B_l2l1_3_corner[] {{.*}} pad_after = array<i32: 1, 0, 0>
