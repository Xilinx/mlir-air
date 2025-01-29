//===- herd_load_to_npu.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu %s | FileCheck %s
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