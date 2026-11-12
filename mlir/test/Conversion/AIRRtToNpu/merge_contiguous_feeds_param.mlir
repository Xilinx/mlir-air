//===- merge_contiguous_feeds_param.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu="output-elf=true merge-contiguous-feeds=true" -canonicalize %s | FileCheck %s

// Under output-elf a run-time offset is a scratchpad parameter added to the
// BD's static offset at dispatch. Two feeds whose static ranges continue one
// another (0 and 256) are not contiguous when their parameters differ, so
// merge-contiguous-feeds leaves feeds with a run-time offset apart.

// CHECK-LABEL: aie.runtime_sequence @seg_sequence
// CHECK: aie.dma_bd(%{{.*}} offset = 0 len = 256{{.*}}) {offset_parameter = @__air_param_argoff_1}
// CHECK: aie.dma_bd(%{{.*}} offset = 256 len = 256{{.*}}) {offset_parameter = @__air_param_argoff_2}
module {
  aie.device(npu2) {
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.shim_dma_allocation @feedIn(%shim_noc_tile_0_0, MM2S, 0)
  } {sym_name = "seg"}
  func.func @two_params(%arg0: memref<1048576xbf16>, %a: i64, %b: i64) {
    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c256_i64 = arith.constant 256 : i64
    %p = airrt.segment_load "seg" : i64
    %ob = arith.addi %b, %c256_i64 : i64
    %f0 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %a], [%c1_i64, %c1_i64, %c1_i64, %c256_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<1048576xbf16>) : !airrt.event
    %f1 = airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %ob], [%c1_i64, %c1_i64, %c1_i64, %c256_i64], [%c0_i64, %c0_i64, %c0_i64, %c1_i64]) {metadata = @feedIn} : (i32, i64, i64, memref<1048576xbf16>) : !airrt.event
    %e = airrt.wait_all %f0, %f1 {"air.launch_end"} : !airrt.event
    return
  }
}
