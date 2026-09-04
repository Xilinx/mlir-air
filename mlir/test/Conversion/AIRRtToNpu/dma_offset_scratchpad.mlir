//===- dma_offset_scratchpad.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu="output-elf=true" -canonicalize --split-input-file %s | FileCheck %s --check-prefix=ELF
// RUN: air-opt -airrt-to-npu -canonicalize --split-input-file %s | FileCheck %s --check-prefix=SSA

// A BD offset that is only known at dispatch becomes a runtime BD word, which
// only the C++ TXN target can encode. For a static TXN (full ELF) the offset
// moves to an mlir-aie scratchpad parameter instead: firmware adds
// StateTable[idx] * element_size to the BD address register at dispatch, so
// the instruction stream stays constant.
//
// Everywhere else the SSA offset is a perfectly good BD operand, and rerouting
// it would silently change the host contract -- an unwritten parameter reads
// as zero, so the transfer would quietly move to the wrong address instead of
// failing. Hence the gate.

// -----

// The plain case: the offset IS the sequence argument. No static part, so the
// BD keeps offset 0 and the parameter carries everything.

// ELF-LABEL: aie.runtime_sequence @seg_plain_sequence
// ELF:         aie.dma_bd(%{{.*}} offset = 0 {{.*}}) {offset_parameter = @__air_param_argoff_1}

// SSA-LABEL: aie.runtime_sequence @plain
// SSA-NOT:     offset_parameter
// SSA:         aie.dma_bd(%{{.*}} offset = %{{.*}})
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "seg_plain"}
  func.func @plain(%arg0: memref<1048576xbf16>, %off: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c256_i64 = arith.constant 256 : i64
    %p = airrt.segment_load "seg_plain" : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %off], [%c1_i64, %c1_i64, %c1_i64, %c256_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<1048576xbf16>)
    return
  }
}

// -----

// The KV append's real shape, and the bug that reached hardware. K writes at
// `base + f(L)` and V at `base + KV_HALF + f(L)`. Following only the varying
// side of the add and DROPPING the constant put both BDs on the same slot, so
// half the cache was never written -- and it still built, still ran, and only
// showed up as diverging logits.
//
// Every addend that does not reach the sequence argument is static and belongs
// on the BD, where it folds once the enclosing loop unrolls. The two BDs share
// one parameter because they need the same number written; they differ in
// their own offsets.

// ELF-LABEL: aie.runtime_sequence @seg_kv_sequence
// ELF:         aie.dma_bd(%{{.*}} offset = 4096 {{.*}}) {offset_parameter = @__air_param_argoff_1_x256_m256}
// ELF:         aie.dma_bd(%{{.*}} offset = 20480 {{.*}}) {offset_parameter = @__air_param_argoff_1_x256_m256}
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @appendK(%shim_noc_tile_0_0, S2MM, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    aie.shim_dma_allocation @appendV(%shim_noc_tile_1_0, S2MM, 0)
  } {sym_name = "seg_kv"}
  func.func @kv_append(%arg0: memref<1048576xbf16>, %L: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c256_i64 = arith.constant 256 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c16384_i64 = arith.constant 16384 : i64
    %p = airrt.segment_load "seg_kv" : i64
    // f(L) = (L - 1) * 256, the slot this token appends at.
    %lm1 = arith.subi %L, %c1_i64 : i64
    %delta = arith.muli %lm1, %c256_i64 : i64
    // K goes at base + delta, V one KV_HALF further along.
    %koff = arith.addi %c4096_i64, %delta : i64
    %vhalf = arith.addi %c4096_i64, %c16384_i64 : i64
    %voff = arith.addi %vhalf, %delta : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %koff], [%c1_i64, %c1_i64, %c1_i64, %c256_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @appendK} : (i32, i64, i64, memref<1048576xbf16>)
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %voff], [%c1_i64, %c1_i64, %c1_i64, %c256_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @appendV} : (i32, i64, i64, memref<1048576xbf16>)
    return
  }
}

// -----

// Two BDs needing DIFFERENT affine functions of the same argument get
// DIFFERENT parameters. The host writes one number per parameter, so sharing
// an entry here would be right for one BD and wrong for the other. The
// coefficients are part of the name, which is also how a host can derive the
// value from params.txt without knowing the design.

// ELF-LABEL: aie.runtime_sequence @seg_two_sequence
// ELF-DAG:     {offset_parameter = @__air_param_argoff_1_x256_m256}
// ELF-DAG:     {offset_parameter = @__air_param_argoff_1_x512_p0}
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @outA(%shim_noc_tile_0_0, S2MM, 0)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    aie.shim_dma_allocation @outB(%shim_noc_tile_1_0, S2MM, 0)
  } {sym_name = "seg_two"}
  func.func @two_formulas(%arg0: memref<1048576xbf16>, %L: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c256_i64 = arith.constant 256 : i64
    %c512_i64 = arith.constant 512 : i64
    %p = airrt.segment_load "seg_two" : i64
    %lm1 = arith.subi %L, %c1_i64 : i64
    %offA = arith.muli %lm1, %c256_i64 : i64
    %offB = arith.muli %L, %c512_i64 : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %offA], [%c1_i64, %c1_i64, %c1_i64, %c256_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @outA} : (i32, i64, i64, memref<1048576xbf16>)
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %offB], [%c1_i64, %c1_i64, %c1_i64, %c256_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @outB} : (i32, i64, i64, memref<1048576xbf16>)
    return
  }
}

// -----

// A constant offset is not a parameter, even under output-elf. State-table
// entries are a module-wide resource of 32, and a value the compiler already
// knows costs nothing to encode.

// ELF-LABEL: aie.runtime_sequence @seg_const_sequence
// ELF-NOT:     offset_parameter
// ELF:         aie.dma_bd(%{{.*}} offset = 4096
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @airMemcpyId2(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "seg_const"}
  func.func @constant_offset(%arg0: memref<1048576xbf16>) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c256_i64 = arith.constant 256 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %p = airrt.segment_load "seg_const" : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c4096_i64], [%c1_i64, %c1_i64, %c1_i64, %c256_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<1048576xbf16>)
    return
  }
}
