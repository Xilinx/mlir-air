//===- shim_wait_names_task.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file %s | FileCheck %s

// A wait on a transfer's event awaits that transfer's task, whatever other
// waits and tasks the channel has.

// Two drains on one channel; the input reads back the second. The await in
// front of it is for the second drain -- with the first awaited ahead of it,
// since the channel retires in order -- not for whichever drain comes first.
// CHECK-LABEL: aie.runtime_sequence @second_of_two
// CHECK: %[[D0:.*]] = aiex.dma_configure_task_for @stt_out
// CHECK: %[[D1:.*]] = aiex.dma_configure_task_for @stt_out
// CHECK: aiex.dma_await_task(%[[D0]])
// CHECK-NEXT: aiex.dma_await_task(%[[D1]])
// CHECK-NEXT: aiex.dma_configure_task_for @stt_in
// CHECK-NOT: aiex.dma_await_task(%[[D0]])
// CHECK-NOT: aiex.dma_await_task(%[[D1]])
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma_allocation @stt_out(%t0, S2MM, 0)
    aie.shim_dma_allocation @stt_in(%t0, MM2S, 0)
  } {sym_name = "stt_seg"}
  airrt.module_metadata{}
  func.func @second_of_two(%arg0: memref<128xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %p = airrt.segment_load "stt_seg" : i64
    %0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @stt_out} : (i32, i64, i64, memref<128xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @stt_out} : (i32, i64, i64, memref<128xi32>) : !airrt.event
    airrt.wait_all %1
    %2 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @stt_in} : (i32, i64, i64, memref<128xi32>) : !airrt.event
    airrt.wait_all %0, %2
    return
  }
}

// -----

// Between the launches of a multi-launch sequence every shim channel the
// launch used is drained. The wait names the launch's transfers on the
// channel, so a channel used twice is awaited after its second transfer, and
// one the launch did not use is not waited on at all. A transfer with no event
// of its own is drained too.
// CHECK-LABEL: aie.runtime_sequence @launch_drain
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @ld_out
// CHECK: %[[A1:.*]] = aiex.dma_configure_task_for @ld_out
// CHECK: aiex.dma_await_task(%[[A0]])
// CHECK-NEXT: aiex.dma_await_task(%[[A1]])
// CHECK: %[[B0:.*]] = aiex.dma_configure_task_for @ld_out
// CHECK: aiex.dma_await_task(%[[B0]])
// CHECK-NOT: aiex.dma_await_task
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @ld_out(%t0, S2MM, 0)
    aie.shim_dma_allocation @ld_idle(%t1, S2MM, 0)
  } {sym_name = "ld_seg"}
  airrt.module_metadata{}
  func.func @launch_drain(%arg0: memref<128xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c3_i32 = arith.constant 3 : i32
    %p = airrt.segment_load "ld_seg" : i64
    %0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ld_out} : (i32, i64, i64, memref<128xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ld_out} : (i32, i64, i64, memref<128xi32>) : !airrt.event
    %e0 = airrt.wait_all {air.launch_end} : !airrt.event
    %p1 = airrt.segment_load "ld_seg" : i64
    airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ld_out} : (i32, i64, i64, memref<128xi32>)
    %e1 = airrt.wait_all {air.launch_end} : !airrt.event
    return
  }
}

// -----

// A launch end that waits on one transfer of a channel leaves the channel's
// other transfers to the drain: the second transfer, issued after the one it
// names, is awaited too.
// CHECK-LABEL: aie.runtime_sequence @launch_end_names_one
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @ln_out
// CHECK: %[[A1:.*]] = aiex.dma_configure_task_for @ln_out
// CHECK: aiex.dma_await_task(%[[A0]])
// CHECK-NEXT: aiex.dma_await_task(%[[A1]])
// CHECK: aiex.dma_configure_task_for @ln_out
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma_allocation @ln_out(%t0, S2MM, 0)
  } {sym_name = "ln_seg"}
  airrt.module_metadata{}
  func.func @launch_end_names_one(%arg0: memref<128xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c3_i32 = arith.constant 3 : i32
    %p = airrt.segment_load "ln_seg" : i64
    %0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ln_out} : (i32, i64, i64, memref<128xi32>) : !airrt.event
    airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ln_out} : (i32, i64, i64, memref<128xi32>)
    %e0 = airrt.wait_all %0 {air.launch_end} : !airrt.event
    %p1 = airrt.segment_load "ln_seg" : i64
    airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @ln_out} : (i32, i64, i64, memref<128xi32>)
    %e1 = airrt.wait_all {air.launch_end} : !airrt.event
    return
  }
}

// -----

// A channel named by an object FIFO rather than a shim allocation is drained at
// the launch boundary too.
// CHECK-LABEL: aie.runtime_sequence @launch_drain_objectfifo
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @of_out
// CHECK: aiex.dma_start_task(%[[A0]])
// CHECK-NEXT: aiex.dma_await_task(%[[A0]])
// CHECK-NEXT: aiex.dma_configure_task_for @of_out
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t3 = aie.tile(0, 3)
    aie.objectfifo @of_out(%t3, {%t0}, 1 : i32) : !aie.objectfifo<memref<64xi32>>
  } {sym_name = "of_seg"}
  airrt.module_metadata{}
  func.func @launch_drain_objectfifo(%arg0: memref<128xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c3_i32 = arith.constant 3 : i32
    %p = airrt.segment_load "of_seg" : i64
    airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @of_out} : (i32, i64, i64, memref<128xi32>)
    %e0 = airrt.wait_all {air.launch_end} : !airrt.event
    %p1 = airrt.segment_load "of_seg" : i64
    airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @of_out} : (i32, i64, i64, memref<128xi32>)
    %e1 = airrt.wait_all {air.launch_end} : !airrt.event
    return
  }
}

// -----

// A transfer issued as several tasks (here four) is waited on as all of them:
// each wait frees the pieces of its own transfer, not the channel's next tasks.
// CHECK-LABEL: aie.runtime_sequence @split_pieces
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @sp_in
// CHECK: %[[A1:.*]] = aiex.dma_configure_task_for @sp_in
// CHECK: %[[A2:.*]] = aiex.dma_configure_task_for @sp_in
// CHECK: %[[A3:.*]] = aiex.dma_configure_task_for @sp_in
// CHECK: aiex.dma_free_task(%[[A0]])
// CHECK-NEXT: aiex.dma_free_task(%[[A1]])
// CHECK-NEXT: aiex.dma_free_task(%[[A2]])
// CHECK-NEXT: aiex.dma_free_task(%[[A3]])
// CHECK: %[[B0:.*]] = aiex.dma_configure_task_for @sp_in
// CHECK: %[[B1:.*]] = aiex.dma_configure_task_for @sp_in
// CHECK: %[[B2:.*]] = aiex.dma_configure_task_for @sp_in
// CHECK: %[[B3:.*]] = aiex.dma_configure_task_for @sp_in
// CHECK: aiex.dma_free_task(%[[B0]])
// CHECK-NEXT: aiex.dma_free_task(%[[B1]])
// CHECK-NEXT: aiex.dma_free_task(%[[B2]])
// CHECK-NEXT: aiex.dma_free_task(%[[B3]])
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma_allocation @sp_in(%t0, MM2S, 0)
  } {sym_name = "sp_seg"}
  airrt.module_metadata{}
  func.func @split_pieces(%arg0: memref<128x8x8x64xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c8_i64 = arith.constant 8 : i64
    %c16_i64 = arith.constant 16 : i64
    %c64_i64 = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %c512_i64 = arith.constant 512 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c2_i32 = arith.constant 2 : i32
    %p = airrt.segment_load "sp_seg" : i64
    %0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64, %c1_i64]) {metadata = @sp_in} : (i32, i64, i64, memref<128x8x8x64xbf16>) : !airrt.event
    airrt.wait_all %0
    %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c16_i64], [%c128_i64, %c8_i64, %c8_i64, %c16_i64], [%c4096_i64, %c64_i64, %c512_i64, %c1_i64]) {metadata = @sp_in} : (i32, i64, i64, memref<128x8x8x64xbf16>) : !airrt.event
    airrt.wait_all %1
    return
  }
}

// -----

// A wait inside a rolled loop body waits on the transfer issued in the same
// iteration.
// CHECK-LABEL: aie.runtime_sequence @rolled_loop
// CHECK: scf.for
// CHECK: %[[T:.*]] = aiex.dma_configure_task_for @rl_out
// CHECK: aiex.dma_start_task(%[[T]])
// CHECK-NEXT: aiex.dma_await_task(%[[T]])
// CHECK-NEXT: }
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma_allocation @rl_out(%t0, S2MM, 0)
  } {sym_name = "rl_seg"}
  airrt.module_metadata{}
  func.func @rolled_loop(%arg0: memref<256xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c3_i32 = arith.constant 3 : i32
    %p = airrt.segment_load "rl_seg" : i64
    scf.for %i = %c0 to %c4 step %c1 {
      %off = arith.index_cast %i : index to i64
      %0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %off, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c64_i64, %c1_i64]) {metadata = @rl_out} : (i32, i64, i64, memref<256xi32>) : !airrt.event
      airrt.wait_all %0
    }
    return
  }
}

// -----

// A launch's transfer inside an arm of a select that stays rolled is drained at
// the end of that arm, the last point its event is visible.
// CHECK-LABEL: aie.runtime_sequence @select_arm
// CHECK: scf.index_switch
// CHECK: case 0 {
// CHECK: %[[T:.*]] = aiex.dma_configure_task_for @sa_out
// CHECK: aiex.dma_await_task(%[[T]])
// CHECK-NEXT: scf.yield
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    aie.shim_dma_allocation @sa_out(%t0, S2MM, 0)
  } {sym_name = "sa_seg"}
  airrt.module_metadata{}
  func.func @select_arm(%arg0: memref<256xi32>, %sel: index) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c3_i32 = arith.constant 3 : i32
    %p = airrt.segment_load "sa_seg" : i64
    scf.index_switch %sel
    case 0 {
      %0 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @sa_out} : (i32, i64, i64, memref<256xi32>) : !airrt.event
      scf.yield
    }
    default {
    }
    %e0 = airrt.wait_all {air.launch_end} : !airrt.event
    %p1 = airrt.segment_load "sa_seg" : i64
    %e1 = airrt.wait_all {air.launch_end} : !airrt.event
    return
  }
}

// -----

// A wait that names a channel's transfers out of order releases them in the
// order the channel issued them; across channels it keeps its own order.
// CHECK-LABEL: aie.runtime_sequence @issue_order
// CHECK: %[[A0:.*]] = aiex.dma_configure_task_for @io_a
// CHECK: %[[A1:.*]] = aiex.dma_configure_task_for @io_a
// CHECK: %[[B0:.*]] = aiex.dma_configure_task_for @io_b
// CHECK: aiex.dma_free_task(%[[B0]])
// CHECK-NEXT: aiex.dma_free_task(%[[A0]])
// CHECK-NEXT: aiex.dma_free_task(%[[A1]])
module {
  aie.device(npu2) {
    %t0 = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @io_a(%t0, MM2S, 0)
    aie.shim_dma_allocation @io_b(%t1, MM2S, 0)
  } {sym_name = "io_seg"}
  airrt.module_metadata{}
  func.func @issue_order(%arg0: memref<256xi32>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "io_seg" : i64
    %0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @io_a} : (i32, i64, i64, memref<256xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c64_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @io_a} : (i32, i64, i64, memref<256xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @io_b} : (i32, i64, i64, memref<256xi32>) : !airrt.event
    airrt.wait_all %2, %1, %0
    return
  }
}
