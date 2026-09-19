// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// RUN: air-opt -airrt-to-npu="output-elf=true" -canonicalize -cse --split-input-file %s | FileCheck %s --enable-var-scope

// Split groups share the same host parameter but have different static BD bases.

// CHECK: aiex.scratchpad_parameter @__air_param_argoff_1 : i32
// CHECK-LABEL: aie.runtime_sequence @segment_sequence
// CHECK: aie.dma_bd(%{{.*}} offset = 4194304 {{.*}}) {offset_parameter = @__air_param_argoff_1}
// CHECK: aie.dma_bd(%{{.*}} offset = 6291456 {{.*}}) {offset_parameter = @__air_param_argoff_1}
// CHECK-NOT: offset = %
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @transfer(%tile, S2MM, 0)
  } {sym_name = "segment"}
  func.func @test(%buf: memref<8388608xbf16>, %base: i64) {
    %c0 = arith.constant 0 : i64
    %c2 = arith.constant 2 : i64
    %c1 = arith.constant 1 : i64
    %c256 = arith.constant 256 : i64
    %c2097152 = arith.constant 2097152 : i64
    %id = arith.constant 2 : i32
    %seg = airrt.segment_load "segment" : i64
    airrt.dma_memcpy_nd(%id, %c0, %c0, %buf[%c0, %c0, %c2, %base], [%c1, %c1, %c2, %c256], [%c0, %c0, %c2097152, %c1]) {metadata = @transfer} : (i32, i64, i64, memref<8388608xbf16>)
    return
  }
}

// -----

// A casted runtime expression keeps its static residual plus the other dimensions.

// CHECK: aiex.scratchpad_parameter @__air_param_argoff_1 : i32
// CHECK-LABEL: aie.runtime_sequence @segment_sequence
// CHECK: aie.dma_bd(%{{.*}} offset = 519 {{.*}}) {offset_parameter = @__air_param_argoff_1}
// CHECK-NOT: offset = %
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @transfer(%tile, S2MM, 0)
  } {sym_name = "segment"}
  func.func @test(%buf: memref<4096xbf16>, %base: i64) {
    %c0 = arith.constant 0 : i64
    %c2 = arith.constant 2 : i64
    %c1 = arith.constant 1 : i64
    %c256 = arith.constant 256 : i64
    %idx = arith.index_cast %base : i64 to index
    %ci = arith.constant 7 : index
    %plus = arith.addi %idx, %ci : index
    %back = arith.index_cast %plus : index to i64
    %id = arith.constant 2 : i32
    %seg = airrt.segment_load "segment" : i64
    airrt.dma_memcpy_nd(%id, %c0, %c0, %buf[%c0, %c0, %c2, %back], [%c1, %c1, %c2, %c256], [%c0, %c0, %c256, %c1]) {metadata = @transfer} : (i32, i64, i64, memref<4096xbf16>)
    return
  }
}
