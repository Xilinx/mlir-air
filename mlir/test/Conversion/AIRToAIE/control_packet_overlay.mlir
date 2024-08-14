//===- control_packet_overlay.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-to-aie='test-patterns=insert-control-packet-flow' --split-input-file | FileCheck %s

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK: %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK: %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK: %[[VAL_2:.*]] = aie.tile(0, 0)
// CHECK: aie.packet_flow(0) {
// CHECK-NEXT:   aie.packet_source<%[[VAL_2]], DMA : 0>
// CHECK-NEXT:   aie.packet_dest<%[[VAL_1]], Ctrl : 0>
// CHECK-NEXT: }
// CHECK: aie.packet_flow(0) {
// CHECK-NEXT:   aie.packet_source<%[[VAL_2]], DMA : 1>
// CHECK-NEXT:   aie.packet_dest<%[[VAL_0]], Ctrl : 0>
// CHECK-NEXT: }

#map = affine_map<(d0) -> (d0)>
module {
  aie.device(npu1_1col) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_0 = aie.tile(0, 0)
    %lock_0_1 = aie.lock(%tile_0_1, 3) {init = 1 : i32}
    %lock_0_1_0 = aie.lock(%tile_0_1, 2) {init = 0 : i32}
    %lock_0_1_1 = aie.lock(%tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_2 = aie.lock(%tile_0_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_5 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %buf3 = aie.buffer(%tile_0_1) {mem_bank = 0 : i32, sym_name = "buf3"} : memref<64xi32, 1> 
    %buf2 = aie.buffer(%tile_0_1) {mem_bank = 0 : i32, sym_name = "buf2"} : memref<64xi32, 1> 
    %buf1 = aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "buf1"} : memref<64xi32, 2> 
    %buf0 = aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "buf0"} : memref<64xi32, 2> 
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf1 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf0 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%lock_0_2_4, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%buf1 : memref<64xi32, 2>) outs(%buf0 : memref<64xi32, 2>) {
      ^bb0(%in: i32, %out: i32):
        %0 = arith.addi %in, %c1_i32 : i32
        linalg.yield %0 : i32
      }
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_5, Release, 1)
      aie.end
    } {elf_file = "segment0_core_0_2.elf"}
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<64xi32, 1>, 0, 64)
      aie.use_lock(%lock_0_1_2, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<64xi32, 1>, 0, 64)
      aie.use_lock(%lock_0_1_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb7
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3, repeat_count = 1)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<64xi32, 1>, 0, 64)
      aie.use_lock(%lock_0_1_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb0
      %3 = aie.dma_start(MM2S, 1, ^bb8, ^bb5, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<64xi32, 1>, 0, 64)
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb8
    }
    aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 0)
    memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
    aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 0)
    memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
  }
}

// -----

// Asynchronous version

// CHECK-LABEL:   aie.device(npu1_1col) {
// CHECK: %[[VAL_0:.*]] = aie.tile(0, 1)
// CHECK: %[[VAL_1:.*]] = aie.tile(0, 2)
// CHECK: %[[VAL_2:.*]] = aie.tile(0, 0)
// CHECK: aie.packet_flow(0) {
// CHECK-NEXT:   aie.packet_source<%[[VAL_2]], DMA : 0>
// CHECK-NEXT:   aie.packet_dest<%[[VAL_1]], Ctrl : 0>
// CHECK-NEXT: }
// CHECK: aie.packet_flow(0) {
// CHECK-NEXT:   aie.packet_source<%[[VAL_2]], DMA : 1>
// CHECK-NEXT:   aie.packet_dest<%[[VAL_0]], Ctrl : 0>
// CHECK-NEXT: }

#map = affine_map<(d0) -> (d0)>
module {
  aie.device(npu1_1col) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_0 = aie.tile(0, 0)
    %lock_0_1 = aie.lock(%tile_0_1, 3) {init = 1 : i32}
    %lock_0_1_0 = aie.lock(%tile_0_1, 2) {init = 0 : i32}
    %lock_0_1_1 = aie.lock(%tile_0_1, 1) {init = 1 : i32}
    %lock_0_1_2 = aie.lock(%tile_0_1, 0) {init = 0 : i32}
    %lock_0_2 = aie.lock(%tile_0_2, 3) {init = 1 : i32}
    %lock_0_2_3 = aie.lock(%tile_0_2, 2) {init = 0 : i32}
    %lock_0_2_4 = aie.lock(%tile_0_2, 1) {init = 1 : i32}
    %lock_0_2_5 = aie.lock(%tile_0_2, 0) {init = 0 : i32}
    %buf3 = aie.buffer(%tile_0_1) {mem_bank = 0 : i32, sym_name = "buf3"} : memref<64xi32, 1> 
    %buf2 = aie.buffer(%tile_0_1) {mem_bank = 0 : i32, sym_name = "buf2"} : memref<64xi32, 1> 
    %buf1 = aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "buf1"} : memref<64xi32, 2> 
    %buf0 = aie.buffer(%tile_0_2) {mem_bank = 0 : i32, sym_name = "buf0"} : memref<64xi32, 2> 
    %mem_0_2 = aie.mem(%tile_0_2) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb3, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf1 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%lock_0_2_3, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb0
      %1 = aie.dma_start(MM2S, 0, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_2_5, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf0 : memref<64xi32, 2>, 0, 64)
      aie.use_lock(%lock_0_2_4, Release, 1)
      aie.next_bd ^bb4
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %c1_i32 = arith.constant 1 : i32
      cf.br ^bb1
    ^bb1:  // pred: ^bb0
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      aie.use_lock(%lock_0_2_4, AcquireGreaterEqual, 1)
      aie.use_lock(%lock_0_2_3, AcquireGreaterEqual, 1)
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%buf1 : memref<64xi32, 2>) outs(%buf0 : memref<64xi32, 2>) {
      ^bb0(%in: i32, %out: i32):
        %0 = arith.addi %in, %c1_i32 : i32
        linalg.yield %0 : i32
      }
      aie.use_lock(%lock_0_2, Release, 1)
      aie.use_lock(%lock_0_2_5, Release, 1)
      aie.end
    } {elf_file = "segment0_core_0_2.elf"}
    air.channel @channel_0 [1, 1]
    air.channel @channel_1 [1, 1]
    air.channel @channel_2 [1, 1]
    air.channel @channel_3 [1, 1]
    aie.flow(%tile_0_0, DMA : 0, %tile_0_1, DMA : 0)
    aie.flow(%tile_0_1, DMA : 0, %tile_0_2, DMA : 0)
    aie.flow(%tile_0_2, DMA : 0, %tile_0_1, DMA : 1)
    aie.flow(%tile_0_1, DMA : 1, %tile_0_0, DMA : 0)
    %memtile_dma_0_1 = aie.memtile_dma(%tile_0_1) {
      %0 = aie.dma_start(S2MM, 0, ^bb1, ^bb7, repeat_count = 1)
    ^bb1:  // 2 preds: ^bb0, ^bb1
      aie.use_lock(%lock_0_1_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<64xi32, 1>, 0, 64)
      aie.use_lock(%lock_0_1_2, Release, 1)
      aie.next_bd ^bb1
    ^bb2:  // pred: ^bb3
      aie.end
    ^bb3:  // pred: ^bb5
      %1 = aie.dma_start(S2MM, 1, ^bb4, ^bb2, repeat_count = 1)
    ^bb4:  // 2 preds: ^bb3, ^bb4
      aie.use_lock(%lock_0_1, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<64xi32, 1>, 0, 64)
      aie.use_lock(%lock_0_1_0, Release, 1)
      aie.next_bd ^bb4
    ^bb5:  // pred: ^bb7
      %2 = aie.dma_start(MM2S, 0, ^bb6, ^bb3, repeat_count = 1)
    ^bb6:  // 2 preds: ^bb5, ^bb6
      aie.use_lock(%lock_0_1_2, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf3 : memref<64xi32, 1>, 0, 64)
      aie.use_lock(%lock_0_1_1, Release, 1)
      aie.next_bd ^bb6
    ^bb7:  // pred: ^bb0
      %3 = aie.dma_start(MM2S, 1, ^bb8, ^bb5, repeat_count = 1)
    ^bb8:  // 2 preds: ^bb7, ^bb8
      aie.use_lock(%lock_0_1_0, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf2 : memref<64xi32, 1>, 0, 64)
      aie.use_lock(%lock_0_1, Release, 1)
      aie.next_bd ^bb8
    }
    aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 0)
    memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
    aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 0)
    memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
  } {sym_name = "segment0"}
}
