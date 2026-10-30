//===- shim_read_back_producer_skew.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -split-input-file -airrt-to-npu %s | FileCheck %s

// Each job reads the previous job's output back after an explicit wait, and the
// jobs feed unequal amounts of input. Past the shim task queue depth, the wait
// that frees a drain's queue slot must come after every feed of the job that
// produces that drain, however few feeds the later jobs have.

// Five jobs feeding [14, 1, 1, 1, 1] transfers: the wait on the first drain
// follows the first job's last feed, at offset 832.
// CHECK-LABEL: aie.runtime_sequence @test
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @out
// CHECK-NOT: aiex.dma_await_task(%[[D0]])
// CHECK: aie.dma_bd(%arg0 : memref<65536xi32> offset = 832
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_await_task(%[[D0]])

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @a(%t, MM2S, 0)
    aie.shim_dma_allocation @b(%t1, MM2S, 0)
    aie.shim_dma_allocation @out(%t, S2MM, 0)
  } {sym_name = "seg"}
  airrt.module_metadata{}
  func.func @test(%arg0: memref<65536xi32>, %arg1: memref<65536xi32>) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %len = arith.constant 64 : i64
    %id = arith.constant 1 : i32
    %seg = airrt.segment_load "seg" : i64
    %o0 = arith.constant 0 : i64
    %o64 = arith.constant 64 : i64
    %o128 = arith.constant 128 : i64
    %o192 = arith.constant 192 : i64
    %o256 = arith.constant 256 : i64
    %o320 = arith.constant 320 : i64
    %o384 = arith.constant 384 : i64
    %o448 = arith.constant 448 : i64
    %o512 = arith.constant 512 : i64
    %o576 = arith.constant 576 : i64
    %o640 = arith.constant 640 : i64
    %o704 = arith.constant 704 : i64
    %o768 = arith.constant 768 : i64
    %o832 = arith.constant 832 : i64
    %o896 = arith.constant 896 : i64
    %o960 = arith.constant 960 : i64
    %o1024 = arith.constant 1024 : i64
    %o1088 = arith.constant 1088 : i64
    %o1152 = arith.constant 1152 : i64
    %o1216 = arith.constant 1216 : i64
    %o1280 = arith.constant 1280 : i64
    %o1344 = arith.constant 1344 : i64
    %o1408 = arith.constant 1408 : i64
    %o1472 = arith.constant 1472 : i64
    %d0 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o0], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @out} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %d1 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o64], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @out} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %d2 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o128], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @out} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %d3 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o192], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @out} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %d4 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o256], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @out} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f0 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o0], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @a} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f1 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o64], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f2 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o128], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f3 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o192], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f4 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o256], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f5 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o320], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f6 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o384], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f7 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o448], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f8 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o512], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f9 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o576], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f10 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o640], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f11 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o704], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f12 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o768], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f13 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o832], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    airrt.wait_all %d0
    %f14 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o0], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @a} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    airrt.wait_all %d1
    %f15 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o64], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @a} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    airrt.wait_all %d2
    %f16 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o128], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @a} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    airrt.wait_all %d3
    %f17 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o192], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @a} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    airrt.wait_all %d0, %d1, %d2, %d3, %d4
    return
  }
}

// -----

// Six jobs feeding [10, 1, 1, 1, 1, 1] transfers: the wait on the first drain
// follows the first job's last feed, at offset 576.
// CHECK-LABEL: aie.runtime_sequence @test
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @out
// CHECK-NOT: aiex.dma_await_task(%[[D0]])
// CHECK: aie.dma_bd(%arg0 : memref<65536xi32> offset = 576
// CHECK: aiex.dma_start_task
// CHECK: aiex.dma_await_task(%[[D0]])

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @a(%t, MM2S, 0)
    aie.shim_dma_allocation @b(%t1, MM2S, 0)
    aie.shim_dma_allocation @out(%t, S2MM, 0)
  } {sym_name = "seg"}
  airrt.module_metadata{}
  func.func @test(%arg0: memref<65536xi32>, %arg1: memref<65536xi32>) {
    %zero = arith.constant 0 : i64
    %one = arith.constant 1 : i64
    %len = arith.constant 64 : i64
    %id = arith.constant 1 : i32
    %seg = airrt.segment_load "seg" : i64
    %o0 = arith.constant 0 : i64
    %o64 = arith.constant 64 : i64
    %o128 = arith.constant 128 : i64
    %o192 = arith.constant 192 : i64
    %o256 = arith.constant 256 : i64
    %o320 = arith.constant 320 : i64
    %o384 = arith.constant 384 : i64
    %o448 = arith.constant 448 : i64
    %o512 = arith.constant 512 : i64
    %o576 = arith.constant 576 : i64
    %o640 = arith.constant 640 : i64
    %o704 = arith.constant 704 : i64
    %o768 = arith.constant 768 : i64
    %o832 = arith.constant 832 : i64
    %o896 = arith.constant 896 : i64
    %o960 = arith.constant 960 : i64
    %o1024 = arith.constant 1024 : i64
    %o1088 = arith.constant 1088 : i64
    %o1152 = arith.constant 1152 : i64
    %o1216 = arith.constant 1216 : i64
    %o1280 = arith.constant 1280 : i64
    %d0 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o0], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @out} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %d1 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o64], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @out} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %d2 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o128], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @out} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %d3 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o192], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @out} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %d4 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o256], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @out} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %d5 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o320], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @out} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f0 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o0], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @a} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f1 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o64], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f2 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o128], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f3 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o192], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f4 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o256], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f5 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o320], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f6 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o384], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f7 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o448], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f8 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o512], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    %f9 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg0[%zero, %zero, %zero, %o576], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @b} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    airrt.wait_all %d0
    %f10 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o0], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @a} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    airrt.wait_all %d1
    %f11 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o64], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @a} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    airrt.wait_all %d2
    %f12 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o128], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @a} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    airrt.wait_all %d3
    %f13 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o192], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @a} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    airrt.wait_all %d4
    %f14 = airrt.dma_memcpy_nd(%id, %zero, %zero, %arg1[%zero, %zero, %zero, %o256], [%one, %one, %one, %len], [%zero, %zero, %zero, %one]) {metadata = @a} : (i32, i64, i64, memref<65536xi32>) : !airrt.event
    airrt.wait_all %d0, %d1, %d2, %d3, %d4, %d5
    return
  }
}
