// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT

// RUN: air-opt -airrt-to-npu -canonicalize -cse --split-input-file %s | FileCheck %s --enable-var-scope

// Even a small MM2S must retain static contributions beside a runtime offset.

// CHECK-LABEL: aie.runtime_sequence @small_offset
// CHECK-SAME: %[[BASE:[a-zA-Z0-9_]+]]: i64
// CHECK-DAG: %[[C:.*]] = arith.constant 512 : i64
// CHECK: %[[SUM:.*]] = arith.addi %[[BASE]], %[[C]] : i64
// CHECK: %[[OFF:.*]] = arith.trunci %[[SUM]] : i64 to i32
// CHECK: aie.dma_bd(%{{.*}} : memref<4096xi8> offset = %[[OFF]] len = 512 sizes = [2, 256] strides = [256, 1])
module {
  aie.device(npu1) {
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @transfer(%tile, MM2S, 0)
  } {sym_name = "segment"}
  func.func @small_offset(%buf: memref<4096xi8>, %base: i64) {
    %c0 = arith.constant 0 : i64
    %c2 = arith.constant 2 : i64
    %c1 = arith.constant 1 : i64
    %c256 = arith.constant 256 : i64
    %id = arith.constant 2 : i32
    %seg = airrt.segment_load "segment" : i64
    airrt.dma_memcpy_nd(%id, %c0, %c0, %buf[%c0, %c0, %c2, %base], [%c1, %c1, %c2, %c256], [%c0, %c0, %c256, %c1]) {metadata = @transfer} : (i32, i64, i64, memref<4096xi8>)
    return
  }
}

// -----

// Scale the runtime offset and include both static dimensions: 16384 + 3*2048.

// CHECK-LABEL: aie.runtime_sequence @scaled_offset
// CHECK-SAME: %[[BASE:[a-zA-Z0-9_]+]]: i64
// CHECK-DAG: %[[C:.*]] = arith.constant 22528 : i64
// CHECK-DAG: %[[S:.*]] = arith.constant 4 : i64
// CHECK: %[[SCALED:.*]] = arith.muli %[[BASE]], %[[S]] : i64
// CHECK: %[[SUM:.*]] = arith.addi %[[SCALED]], %[[C]] : i64
// CHECK: %[[OFF:.*]] = arith.trunci %[[SUM]] : i64 to i32
// CHECK: aie.dma_bd(%{{.*}} offset = %[[OFF]] len = 512 sizes = [2, 256] strides = [2048, 4])
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @transfer(%tile, S2MM, 0)
  } {sym_name = "segment"}
  func.func @scaled_offset(%buf: memref<65536xbf16>, %base: i64) {
    %c0 = arith.constant 0 : i64
    %c3 = arith.constant 3 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    %c256 = arith.constant 256 : i64
    %c4 = arith.constant 4 : i64
    %c2048 = arith.constant 2048 : i64
    %c16384 = arith.constant 16384 : i64
    %id = arith.constant 2 : i32
    %seg = airrt.segment_load "segment" : i64
    airrt.dma_memcpy_nd(%id, %c0, %c0, %buf[%c0, %c1, %c3, %base], [%c1, %c1, %c2, %c256], [%c0, %c16384, %c2048, %c4]) {metadata = @transfer} : (i32, i64, i64, memref<65536xbf16>)
    return
  }
}

// -----

// Runtime length uses a separate BD path and must retain the base as well.

// CHECK-LABEL: aie.runtime_sequence @dynamic_length
// CHECK-SAME: %[[BASE:[a-zA-Z0-9_]+]]: i64, %[[N:[a-zA-Z0-9_]+]]: i64
// CHECK-DAG: %[[C:.*]] = arith.constant 512 : i64
// CHECK-DAG: %[[SUM:.*]] = arith.addi %[[BASE]], %[[C]] : i64
// CHECK-DAG: %[[OFF:.*]] = arith.trunci %[[SUM]] : i64 to i32
// CHECK-DAG: %[[LEN:.*]] = arith.trunci %[[N]] : i64 to i32
// CHECK: aie.dma_bd(%{{.*}} offset = %[[OFF]] len = %[[LEN]])
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @transfer(%tile, S2MM, 0)
  } {sym_name = "segment"}
  func.func @dynamic_length(%buf: memref<4096xbf16>, %base: i64, %n: i64) {
    %c0 = arith.constant 0 : i64
    %c2 = arith.constant 2 : i64
    %c1 = arith.constant 1 : i64
    %c256 = arith.constant 256 : i64
    %id = arith.constant 2 : i32
    %seg = airrt.segment_load "segment" : i64
    airrt.dma_memcpy_nd(%id, %c0, %c0, %buf[%c0, %c0, %c2, %base], [%c1, %c1, %c1, %n], [%c0, %c0, %c256, %c1]) {metadata = @transfer} : (i32, i64, i64, memref<4096xbf16>)
    return
  }
}

// -----

// A split oversized stride must give each group a distinct runtime address.

// CHECK-LABEL: aie.runtime_sequence @split_stride
// CHECK-SAME: %[[BASE:[a-zA-Z0-9_]+]]: i64
// CHECK-DAG: %[[C:.*]] = arith.constant 2097152 : i64
// CHECK-DAG: %[[SUM:.*]] = arith.addi %[[BASE]], %[[C]] : i64
// CHECK-DAG: %[[SECOND:.*]] = arith.trunci %[[SUM]] : i64 to i32
// CHECK-DAG: %[[FIRST:.*]] = arith.trunci %[[BASE]] : i64 to i32
// CHECK: aie.dma_bd(%{{.*}} offset = %[[FIRST]] len = 256 sizes = [256] strides = [1])
// CHECK: aie.dma_bd(%{{.*}} offset = %[[SECOND]] len = 256 sizes = [256] strides = [1])
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 0)
    aie.shim_dma_allocation @transfer(%tile, S2MM, 0)
  } {sym_name = "segment"}
  func.func @split_stride(%buf: memref<4194304xbf16>, %base: i64) {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %c2 = arith.constant 2 : i64
    %c256 = arith.constant 256 : i64
    %c2097152 = arith.constant 2097152 : i64
    %id = arith.constant 2 : i32
    %seg = airrt.segment_load "segment" : i64
    airrt.dma_memcpy_nd(%id, %c0, %c0, %buf[%c0, %c0, %c0, %base], [%c1, %c1, %c2, %c256], [%c0, %c0, %c2097152, %c1]) {metadata = @transfer} : (i32, i64, i64, memref<4194304xbf16>)
    return
  }
}
