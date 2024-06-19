//===- insert_trace_packet_flow.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='test-patterns=to-aie-mlir insert-trace-packet-flow=true' | FileCheck %s
// CHECK:    aie.packet_flow(0) {
// CEHCK:      aie.packet_source<%tile_1_1, Trace : 0>
// CHECK:      aie.packet_dest<%tile_1_0, DMA : 1>
// CHECK:    } {keep_pkt_header = true}
module {

func.func @foo(%arg0: i32) {
  %cst1 = arith.constant 1 : index
  air.herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) {
    %src0 = memref.alloc() : memref<1xi32, 2>
    %src1 = memref.alloc() : memref<1xi32, 2>
    %zero = arith.constant 0 : index
    %0 = memref.load %src0[%zero] : memref<1xi32, 2>
    %1 = memref.load %src1[%zero] : memref<1xi32, 2>
    %2 = arith.addi %0, %1 :  i32
    %dst0 = memref.alloc() : memref<1xi32, 2>
    memref.store %2, %dst0[%zero] : memref<1xi32, 2>
  }
  return
}

}
