//===- dma_length_scratchpad.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu="output-elf=true" -canonicalize --split-input-file %s | FileCheck %s --check-prefix=ELF
// RUN: air-opt -airrt-to-npu -canonicalize --split-input-file %s | FileCheck %s --check-prefix=SSA

// A transfer length only known at dispatch is a runtime BD word, which only
// the C++ TXN target can encode. For a full ELF, a length `unit * arg + base`
// in the sequence's own argument becomes a BD of static length `base` whose
// length_parameter adds `arg * unit` elements at dispatch. The host writes
// the argument as it is.

// -----

// The length is the argument's count of contiguous [16, 64] blocks, the shape
// of a KV cache readback.

// ELF:       aiex.scratchpad_parameter @__air_param_arglen_1 : i32
// ELF-LABEL: aie.runtime_sequence @seg_rows_sequence
// ELF:         aie.dma_bd(%{{.*}} offset = 0 len = 0) {length_parameter = @__air_param_arglen_1, length_unit = 1024 : i32}

// SSA-LABEL: aie.runtime_sequence @rows
// SSA-NOT:     length_parameter
// SSA:         aie.dma_bd(%{{.*}} offset = 0 len = %{{.*}})
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "seg_rows"}
  func.func @rows(%arg0: memref<1048576xbf16>, %nrows: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c16_i64 = arith.constant 16 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %p = airrt.segment_load "seg_rows" : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %nrows, %c16_i64, %c64_i64], [%c0_i64, %c1024_i64, %c64_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<1048576xbf16>)
    return
  }
}

// -----

// One block more than the argument: the extra block is the BD's static
// length.

// ELF-LABEL: aie.runtime_sequence @seg_plus_sequence
// ELF:         aie.dma_bd(%{{.*}} offset = 0 len = 1024) {length_parameter = @__air_param_arglen_1, length_unit = 1024 : i32}
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "seg_plus"}
  func.func @plus(%arg0: memref<1048576xbf16>, %nrows: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c16_i64 = arith.constant 16 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %n = arith.addi %nrows, %c1_i64 : i64
    %p = airrt.segment_load "seg_plus" : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %n, %c16_i64, %c64_i64], [%c0_i64, %c1024_i64, %c64_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<1048576xbf16>)
    return
  }
}

// -----

// A runtime offset as well, from another argument: the BD takes an offset
// parameter beside the length parameter.

// ELF:       aiex.scratchpad_parameter @__air_param_argoff_2 : i32
// ELF-LABEL: aie.runtime_sequence @seg_both_sequence
// ELF:         aie.dma_bd(%{{.*}} offset = 0 len = 0) {length_parameter = @__air_param_arglen_1, length_unit = 1024 : i32, offset_parameter = @__air_param_argoff_2}
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "seg_both"}
  func.func @both(%arg0: memref<1048576xbf16>, %nrows: i64, %off: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c16_i64 = arith.constant 16 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %p = airrt.segment_load "seg_both" : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %off], [%c1_i64, %nrows, %c16_i64, %c64_i64], [%c0_i64, %c1024_i64, %c64_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<1048576xbf16>)
    return
  }
}

// -----

// A narrowing cast can change the argument's value, which the host writes
// unchanged: the length stays a runtime BD word.

// ELF-LABEL: aie.runtime_sequence @seg_narrow_sequence
// ELF-NOT:     length_parameter
// ELF:         aie.dma_bd(%{{.*}} offset = 0 len = %{{.*}})
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "seg_narrow"}
  func.func @narrow(%arg0: memref<1048576xbf16>, %nrows: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c16_i64 = arith.constant 16 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %t = arith.trunci %nrows : i64 to i8
    %n = arith.extui %t : i8 to i64
    %p = airrt.segment_load "seg_narrow" : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %n, %c16_i64, %c64_i64], [%c0_i64, %c1024_i64, %c64_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<1048576xbf16>)
    return
  }
}

// -----

// shim_drain_weave's @mid, with the feed that reads the drain back sized by a
// length parameter. Its static length of 0 is not its footprint, so it still
// counts as reading the drain: channel B is woven in, and the drain's await
// travels with that feed.

// ELF-LABEL: aie.runtime_sequence @midp_seg_sequence
// ELF: %[[A0:.*]] = aiex.dma_configure_task_for @midp_a
// ELF: %[[B0:.*]] = aiex.dma_configure_task_for @midp_b
// ELF: %[[A1:.*]] = aiex.dma_configure_task_for @midp_a
// ELF: %[[B1:.*]] = aiex.dma_configure_task_for @midp_b
// ELF: %[[D:.*]] = aiex.dma_configure_task_for @midp_out
// ELF: %[[A2:.*]] = aiex.dma_configure_task_for @midp_a
// ELF: %[[B2:.*]] = aiex.dma_configure_task_for @midp_b
// ELF: %[[A3:.*]] = aiex.dma_configure_task_for @midp_a
// ELF: %[[B3:.*]] = aiex.dma_configure_task_for @midp_b
// ELF: %[[B4:.*]] = aiex.dma_configure_task_for @midp_b
// ELF: aiex.dma_await_task(%[[D]])
// ELF: %[[A4:.*]] = aiex.dma_configure_task_for @midp_a
// ELF: length_parameter = @__air_param_arglen_3
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    aie.shim_dma_allocation @midp_out(%shim_noc_tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @midp_a(%shim_noc_tile_0_0, MM2S, 0)
    aie.shim_dma_allocation @midp_b(%shim_noc_tile_1_0, MM2S, 0)
  } {sym_name = "midp_seg"}
  airrt.module_metadata{}
  func.func @midp(%arg0: memref<64xi32>, %arg1: memref<1024xi32>, %arg2: memref<64xi32>, %nrows: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c64_i64 = arith.constant 64 : i64
    %c512_i64 = arith.constant 512 : i64
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %p = airrt.segment_load "midp_seg" : i64
    %0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @midp_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @midp_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %2 = airrt.dma_memcpy_nd(%c3_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c512_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @midp_out} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %3 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @midp_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %4 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @midp_a} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %2
    %5 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %nrows, %c64_i64], [%c0_i64, %c0_i64, %c64_i64, %c1_i64]) {metadata = @midp_a} : (i32, i64, i64, memref<1024xi32>) : !airrt.event
    %6 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @midp_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %7 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @midp_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %8 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @midp_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %9 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @midp_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    %10 = airrt.dma_memcpy_nd(%c4_i32, %c0_i64, %c0_i64, %arg2[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @midp_b} : (i32, i64, i64, memref<64xi32>) : !airrt.event
    airrt.wait_all %0, %1, %3, %4, %5, %6, %7, %8, %9, %10
    return
  }
}

// -----

// In a rolled loop, a length that picks between two arguments by the
// iteration: one configure task per choice, under an scf.if on it, each with
// its own length parameter.

// ELF-LABEL: aie.runtime_sequence @seg_sel_sequence
// ELF:         scf.for
// ELF:           %[[T:.*]] = scf.if
// ELF:             aiex.dma_configure_task_for @airMemcpyId2
// ELF:               aie.dma_bd({{.*}} offset = 0 len = 0) {length_parameter = @__air_param_arglen_1, length_unit = 1024 : i32}
// ELF:           } else {
// ELF:             aiex.dma_configure_task_for @airMemcpyId2
// ELF:               aie.dma_bd({{.*}} offset = 0 len = 0) {length_parameter = @__air_param_arglen_2, length_unit = 1024 : i32}
// ELF:           aiex.dma_start_task(%[[T]])
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "seg_sel"}
  func.func @sel(%arg0: memref<1048576xbf16>, %a: i64, %b: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c16_i64 = arith.constant 16 : i64
    %c64_i64 = arith.constant 64 : i64
    %c1024_i64 = arith.constant 1024 : i64
    %p = airrt.segment_load "seg_sel" : i64
    scf.for %i = %c0 to %c4 step %c1 {
      %is2 = arith.cmpi eq, %i, %c2 : index
      %n = arith.select %is2, %a, %b : i64
      %e = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %n, %c16_i64, %c64_i64], [%c0_i64, %c1024_i64, %c64_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<1048576xbf16>) : !airrt.event
      airrt.wait_all %e
    }
    return
  }
}

// -----

// The same choice for an offset alone: each task's BD takes its own offset
// parameter.

// ELF-LABEL: aie.runtime_sequence @seg_selo_sequence
// ELF:         scf.if
// ELF:           aie.dma_bd({{.*}} offset = 0 len = 64 {{.*}}) {offset_parameter = @__air_param_argoff_1}
// ELF:         } else {
// ELF:           aie.dma_bd({{.*}} offset = 0 len = 64 {{.*}}) {offset_parameter = @__air_param_argoff_2}
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "seg_selo"}
  func.func @selo(%arg0: memref<1048576xbf16>, %a: i64, %b: i64) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c2 = arith.constant 2 : index
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c64_i64 = arith.constant 64 : i64
    %p = airrt.segment_load "seg_selo" : i64
    scf.for %i = %c0 to %c4 step %c1 {
      %is2 = arith.cmpi eq, %i, %c2 : index
      %o = arith.select %is2, %a, %b : i64
      %e = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %o], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<1048576xbf16>) : !airrt.event
      airrt.wait_all %e
    }
    return
  }
}
