//===- herd_load_to_npu.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu -split-input-file %s | FileCheck %s

// CHECK-LABEL: func1
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_10_20, 0, 11)
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_10_20, 1, 22)
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_10_21, 0, 11)
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_10_21, 1, 22)
airrt.module_metadata{
    airrt.segment_metadata attributes {sym_name = "segment"} {
        airrt.herd_metadata {size_x = 1 : i64, size_y = 2 : i64, loc_x = 10 : i64, loc_y = 20 : i64, sym_name = "herd"}
    }
}
func.func @func1(%arg0: i32, %arg1: i32) {
    %c11_i32 = arith.constant 11 : i32
    %c22_i32 = arith.constant 22 : i32
    %h = airrt.herd_load "herd" (%c11_i32, %c22_i32) : (i32, i32) -> i64
    %c1 = arith.constant 1 : index
    return
}

// -----

// CHECK-LABEL: module
// CHECK: aie.device(npu1) @segment_0
// CHECK: aie.runtime_sequence @func2
// CHECK: aiex.npu.rtp_write(@__air_herd_rtp_0_2, 0, 5)
// CHECK: aiex.set_lock(%__air_herd_lock_0_2, 1)

module {
  aie.device(npu1) @segment_0 {
    %tile_0_2 = aie.tile(0, 2)
    %__air_herd_lock_0_2 = aie.lock(%tile_0_2, 0) {init = 0 : i32, sym_name = "__air_herd_lock_0_2"}
    %__air_herd_rtp_0_2 = aie.buffer(%tile_0_2) {sym_name = "__air_herd_rtp_0_2"} : memref<1xi32>
  }
  airrt.module_metadata {
    airrt.segment_metadata attributes {sym_name = "segment_0"} {
      airrt.herd_metadata {size_x = 1 : i64, size_y = 1 : i64, loc_x = 0 : i64, loc_y = 2 : i64, sym_name = "herd_0"}
    }
  }
  func.func @func2() {
    %p = airrt.segment_load "segment_0" : i64
    %c5_i32 = arith.constant 5 : i32
    %h = airrt.herd_load "herd_0" (%c5_i32) {segment_name = "segment_0"} : (i32) -> i64
    return
  }
}
