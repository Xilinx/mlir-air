//===- rearm_tile_dma_order.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// rearm_tile_dma.mlir with the ^weights block written before ^s2mm. The tasks
// are pushed in the order the dma_start chain queues them (x, then the
// weights), not in the textual order of the blocks.

// RUN: air-opt -airrt-to-npu="rearm-tile-dma=true" %s | FileCheck %s

// CHECK-LABEL: aie.runtime_sequence @ctrl
// CHECK: %[[BD0:.*]] = arith.constant 0 : i32
// CHECK: aiex.npu.push_queue(0, 2, S2MM : 0) bd_id %[[BD0]]
// CHECK: %[[BD1:.*]] = arith.constant 1 : i32
// CHECK: aiex.npu.push_queue(0, 2, S2MM : 0) bd_id %[[BD1]]
// CHECK-NOT: aiex.npu.push_queue

module {
  aie.device(npu2) @seg {
    %shim = aie.tile(0, 0)
    aie.shim_dma_allocation @feedIn(%shim, MM2S, 0)
    %t = aie.tile(0, 2)
    %x = aie.buffer(%t) : memref<64xi8>
    %w0 = aie.buffer(%t) : memref<64xi8>
    %w1 = aie.buffer(%t) : memref<64xi8>
    %y = aie.buffer(%t) : memref<64xi8>
    %in_empty = aie.lock(%t, 0) {init = 3 : i32}
    %in_full = aie.lock(%t, 1) {init = 0 : i32}
    %out_empty = aie.lock(%t, 2) {init = 1 : i32}
    %out_full = aie.lock(%t, 3) {init = 0 : i32}
    %mem = aie.mem(%t) {
      %c1 = arith.constant 1 : i32
      %0 = aie.dma_start(MM2S, 0, ^out, ^s2mm)
    ^out:
      aie.use_lock(%out_full, AcquireGreaterEqual, %c1)
      aie.dma_bd(%y : memref<64xi8> offset = 0 len = 64) {bd_id = 8 : i32}
      aie.use_lock(%out_empty, Release, %c1)
      aie.next_bd ^out
    ^weights:
      %2 = aie.dma_start(S2MM, 0, ^w0, ^end, repeat_count = 127)
    ^s2mm:
      %1 = aie.dma_start(S2MM, 0, ^x, ^weights)
    ^x:
      aie.use_lock(%in_empty, AcquireGreaterEqual, %c1)
      aie.dma_bd(%x : memref<64xi8> offset = 0 len = 64) {bd_id = 0 : i32}
      aie.use_lock(%in_full, Release, %c1)
      aie.next_bd ^end
    ^w0:
      aie.use_lock(%in_empty, AcquireGreaterEqual, %c1)
      aie.dma_bd(%w0 : memref<64xi8> offset = 0 len = 64) {bd_id = 1 : i32}
      aie.use_lock(%in_full, Release, %c1)
      aie.next_bd ^w1
    ^w1:
      aie.use_lock(%in_empty, AcquireGreaterEqual, %c1)
      aie.dma_bd(%w1 : memref<64xi8> offset = 0 len = 64) {bd_id = 2 : i32}
      aie.use_lock(%in_full, Release, %c1)
      aie.next_bd ^end
    ^end:
      aie.end
    }
    %core = aie.core(%t) {
      aie.end
    } {link_with = "kernel.o"}
  } {sym_name = "seg"}

  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "seg"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }

  func.func @ctrl(%arg0: memref<64xi8>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %feed = arith.constant 4 : i32
    %f = airrt.dma_memcpy_nd(%feed, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<64xi8>) : !airrt.event
    airrt.wait_all %f {"air.launch_end"}
    %p = airrt.segment_load "seg" : i64
    return
  }
}
