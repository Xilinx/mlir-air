//===- runtime_repeat_iteration.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s --split-input-file | FileCheck %s --check-prefix=TXN
// RUN: air-opt -airrt-to-npu="output-elf=true" %s --split-input-file | FileCheck %s --check-prefix=ELF

// A runtime number of 16x128 tiles, 8192 elements apart: the task repeats a
// runtime number of times and the BD's iteration steps each run, wrapping at
// the most runs the buffer holds (8). A full ELF takes the count from the
// scratchpad, one less than the argument, bounded by the task's own count.
// Either way the host keeps the count within the runs the iteration holds.

// TXN: module attributes {air.ctrl_bounds = array<i64: 1, 1, 0, 8>}
// TXN-LABEL: aie.runtime_sequence @steps
// TXN: aiex.dma_configure_task_for @airMemcpyId2 repeat
// TXN: aie.dma_bd(%{{.*}} : memref<8x64x128xbf16> offset = 0 len = 2048 sizes = [8, 1, 16, 128] strides = [8192, 8192, 128, 1])

// ELF: module attributes {air.ctrl_bounds = array<i64: 1, 1, 0, 8>}
// ELF: aiex.scratchpad_parameter @__air_param_argrep_1_x1_m1 : i32
// ELF: aie.runtime_sequence @segment0_sequence
// ELF: aie.dma_bd(%{{.*}} : memref<8x64x128xbf16> offset = 0 len = 2048 sizes = [8, 1, 16, 128] strides = [8192, 8192, 128, 1])
// ELF: } {repeat_count = 7 : i32}
// ELF: aiex.dma_start_task(%{{.*}}) {repeat_parameter = @__air_param_argrep_1_x1_m1}

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%t, MM2S, 0)
  } {sym_name = "segment0"}
  func.func @steps(%arg0: memref<8x64x128xbf16>, %n: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %p = airrt.segment_load "segment0" : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%n, %c1_i64, %c16_i64, %c128_i64], [%c8192_i64, %c8192_i64, %c128_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<8x64x128xbf16>)
    return
  }
}

// -----

// A runtime-length run of 2048-element blocks in each of 8 rows 8192 apart:
// one linear BD whose iteration steps the rows, repeated 8 times. A full ELF
// keeps one block of the length static (a BD that iterates takes no zero
// length), so its parameter holds the argument less one. The host keeps the
// last row's run inside the buffer.

// TXN: module attributes {air.ctrl_bounds = array<i64: 1, 2048, 0, 8192>}
// TXN-LABEL: aie.runtime_sequence @rows
// TXN: aie.dma_bd(%{{.*}} : memref<8x64x128xbf16> offset = 0 len = %{{.*}}) {iteration = #aie.bd_iteration<size = 8, stride = 8192, current = 0>}
// TXN: } {repeat_count = 7 : i32}

// ELF: module attributes {air.ctrl_bounds = array<i64: 1, 2048, 0, 8192>}
// ELF: aiex.scratchpad_parameter @__air_param_arglen_1_x1_m1 : i32
// ELF: aie.runtime_sequence @segment0_sequence
// ELF: aie.dma_bd(%{{.*}} : memref<8x64x128xbf16> offset = 0 len = 2048) {iteration = #aie.bd_iteration<size = 8, stride = 8192, current = 0>, length_parameter = @__air_param_arglen_1_x1_m1, length_unit = 2048 : i32}
// ELF: } {repeat_count = 7 : i32}

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%t, MM2S, 0)
  } {sym_name = "segment0"}
  func.func @rows(%arg0: memref<8x64x128xbf16>, %n: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c8_i64 = arith.constant 8 : i64
    %c2048_i64 = arith.constant 2048 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %p = airrt.segment_load "segment0" : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c8_i64, %n, %c2048_i64], [%c0_i64, %c8192_i64, %c2048_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<8x64x128xbf16>)
    return
  }
}

// -----

// The buffer holds 128 runs, more than a shim BD iterates: the iteration's
// limit bounds the runtime count.

// TXN: module attributes {air.ctrl_bounds = array<i64: 1, 1, 0, 63>}
// TXN-LABEL: aie.runtime_sequence @many
// TXN: aie.dma_bd(%{{.*}} : memref<128x64x128xbf16> offset = 0 len = 2048 sizes = [63, 1, 16, 128] strides = [8192, 8192, 128, 1])

// ELF: module attributes {air.ctrl_bounds = array<i64: 1, 1, 0, 63>}
// ELF: aie.runtime_sequence @segment0_sequence
// ELF: aie.dma_bd(%{{.*}} : memref<128x64x128xbf16> offset = 0 len = 2048 sizes = [63, 1, 16, 128] strides = [8192, 8192, 128, 1])
// ELF: } {repeat_count = 62 : i32}

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%t, MM2S, 0)
  } {sym_name = "segment0"}
  func.func @many(%arg0: memref<128x64x128xbf16>, %n: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c16_i64 = arith.constant 16 : i64
    %c128_i64 = arith.constant 128 : i64
    %c8192_i64 = arith.constant 8192 : i64
    %p = airrt.segment_load "segment0" : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%n, %c1_i64, %c16_i64, %c128_i64], [%c8192_i64, %c8192_i64, %c128_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<128x64x128xbf16>)
    return
  }
}

// -----

// A pattern of fewer than four dimensions comes padded with an outer unit
// dimension: the runtime size inside it is still the outermost.

// TXN: module attributes {air.ctrl_bounds = array<i64: 1, 1, 0, 8>}
// TXN-LABEL: aie.runtime_sequence @padded
// TXN: aiex.dma_configure_task_for @airMemcpyId2 repeat
// TXN: aie.dma_bd(%{{.*}} : memref<8x1x128xi32> offset = 0 len = 64 sizes = [8, 1, 1, 64] strides = [128, 0, 128, 1])

// ELF: aiex.scratchpad_parameter @__air_param_argrep_1_x1_m1 : i32
// ELF: aie.runtime_sequence @segment0_sequence
// ELF: } {repeat_count = 7 : i32}
// ELF: aiex.dma_start_task(%{{.*}}) {repeat_parameter = @__air_param_argrep_1_x1_m1}

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%t, MM2S, 0)
  } {sym_name = "segment0"}
  func.func @padded(%arg0: memref<8x1x128xi32>, %n: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c64_i64 = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %p = airrt.segment_load "segment0" : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %n, %c1_i64, %c64_i64], [%c0_i64, %c128_i64, %c128_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<8x1x128xi32>)
    return
  }
}

// -----

// A static offset of one run plus a runtime one: the runs start no earlier
// than the static part, so one fewer fits.

// TXN: module attributes {air.ctrl_bounds = array<i64: 1, 1, 0, 7>}
// TXN-LABEL: aie.runtime_sequence @offset
// TXN: aie.dma_bd(%{{.*}} : memref<1024xi32> offset = %{{.*}} len = 64 sizes = [7, 1, 1, 64] strides = [128, 128, 128, 1])

// ELF: module attributes {air.ctrl_bounds = array<i64: 1, 1, 0, 7>}
// ELF: } {repeat_count = 6 : i32}

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%t, MM2S, 0)
  } {sym_name = "segment0"}
  func.func @offset(%arg0: memref<1024xi32>, %n: i64, %o: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c64_i64 = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %p = airrt.segment_load "segment0" : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c1_i64, %c0_i64, %o], [%n, %c1_i64, %c1_i64, %c64_i64], [%c128_i64, %c128_i64, %c128_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<1024xi32>)
    return
  }
}

// -----

// In a rolled loop, a runtime-length run of each of 4 rows that picks between
// two arguments by the iteration: one task per choice, each BD keeping one
// block of its length static, and the host bounding both arguments.

// TXN: module attributes {air.ctrl_bounds = array<i64: 1, 1024, 0, 262144, 2, 1024, 0, 262144>}
// TXN-LABEL: aie.runtime_sequence @seli

// ELF: module attributes {air.ctrl_bounds = array<i64: 1, 1024, 0, 262144, 2, 1024, 0, 262144>}
// ELF: scf.if
// ELF: aie.dma_bd({{.*}} offset = 0 len = 1024) {iteration = #aie.bd_iteration<size = 4, stride = 262144, current = 0>, length_parameter = @__air_param_arglen_1_x1_m1, length_unit = 1024 : i32}
// ELF: } else {
// ELF: aie.dma_bd({{.*}} offset = 0 len = 1024) {iteration = #aie.bd_iteration<size = 4, stride = 262144, current = 0>, length_parameter = @__air_param_arglen_2_x1_m1, length_unit = 1024 : i32}

module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "seg_seli"}
  func.func @seli(%arg0: memref<1048576xbf16>, %a: i64, %b: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c4_i64 = arith.constant 4 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %c262144_i64 = arith.constant 262144 : i64
    %p = airrt.segment_load "seg_seli" : i64
    scf.for %i = %c0 to %c4 step %c1 {
      %is2 = arith.cmpi eq, %i, %c2 : index
      %n = arith.select %is2, %a, %b : i64
      %e = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c4_i64, %n, %c1024_i64], [%c0_i64, %c262144_i64, %c1024_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<1048576xbf16>) : !airrt.event
      airrt.wait_all %e
    }
    return
  }
}
