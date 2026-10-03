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

// A runtime offset as well: the length stays a runtime BD word.

// ELF-LABEL: aie.runtime_sequence @seg_both_sequence
// ELF-NOT:     length_parameter
// ELF:         aie.dma_bd(%{{.*}} offset = %{{.*}} len = %{{.*}})
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
