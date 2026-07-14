//===- generate_trace_write32.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -airrt-to-npu='trace-offset=65536 trace-size=65536' | FileCheck %s
module {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    aie.packet_flow(0) {
      aie.packet_source<%tile_0_1, Trace : 0>
      aie.packet_dest<%tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.packet_flow(1) {
      aie.packet_source<%tile_0_2, Trace : 0>
      aie.packet_dest<%tile_0_0, DMA : 1>
    } {keep_pkt_header = true}
    aie.shim_dma_allocation @airMemcpyId7(%tile_0_0, S2MM, 0)
    aie.shim_dma_allocation @airMemcpyId2(%tile_0_0, MM2S, 0)
  } {sym_name = "segment0"}
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_3 [1, 1]
  func.func @func0(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
// CHECK:      %[[A0:.*]] = arith.constant 606208 : i32
// CHECK:      %[[V0:.*]] = arith.constant 40192 : i32
// CHECK:      aiex.npu.write32(%[[A0]], %[[V0]]) {column = 0 : i32, row = 1 : i32}
// CHECK:      %[[A1:.*]] = arith.constant 606416 : i32
// CHECK:      %[[V1:.*]] = arith.constant 10289152 : i32
// CHECK:      aiex.npu.write32(%[[A1]], %[[V1]]) {column = 0 : i32, row = 1 : i32}
// CHECK:      %[[A2:.*]] = arith.constant 606420 : i32
// CHECK:      %[[V2:.*]] = arith.constant 12288 : i32
// CHECK:      aiex.npu.write32(%[[A2]], %[[V2]]) {column = 0 : i32, row = 1 : i32}
// CHECK:      %[[A3:.*]] = arith.constant 606432 : i32
// CHECK:      %[[V3:.*]] = arith.constant 22041688 : i32
// CHECK:      aiex.npu.write32(%[[A3]], %[[V3]]) {column = 0 : i32, row = 1 : i32}
// CHECK:      %[[A4:.*]] = arith.constant 606436 : i32
// CHECK:      %[[V4:.*]] = arith.constant 1549821032 : i32
// CHECK:      aiex.npu.write32(%[[A4]], %[[V4]]) {column = 0 : i32, row = 1 : i32}
// CHECK:      %[[A5:.*]] = arith.constant 724736 : i32
// CHECK:      %[[V5:.*]] = arith.constant 2236704 : i32
// CHECK:      aiex.npu.write32(%[[A5]], %[[V5]]) {column = 0 : i32, row = 1 : i32}
// CHECK:      %[[A6:.*]] = arith.constant 724740 : i32
// CHECK:      %[[V6:.*]] = arith.constant 197121 : i32
// CHECK:      aiex.npu.write32(%[[A6]], %[[V6]]) {column = 0 : i32, row = 1 : i32}
// CHECK:      aiex.npu.writebd {bd_id = 15 : i32, buffer_length = 16384 : i32, buffer_offset = 65536 : i32, column = 0 : i32
// CHECK-SAME: enable_packet = 1 : i32
// CHECK-SAME: packet_id = 0 : i32, packet_type = 3 : i32
// CHECK:      aiex.npu.address_patch({{.*}}) {addr = 119268 : ui32, arg_idx = 2 : i32}
// CHECK:      %[[A7:.*]] = arith.constant 119308 : i32
// CHECK:      %[[V7:.*]] = arith.constant 15 : i32
// CHECK:      aiex.npu.write32(%[[A7]], %[[V7]]) {column = 0 : i32, row = 0 : i32}
// CHECK:      %[[A8:.*]] = arith.constant 212992 : i32
// CHECK:      %[[V8:.*]] = arith.constant 31232 : i32
// CHECK:      aiex.npu.write32(%[[A8]], %[[V8]]) {column = 0 : i32, row = 2 : i32}
// CHECK:      %[[A9:.*]] = arith.constant 213200 : i32
// CHECK:      %[[V9:.*]] = arith.constant 7995392 : i32
// CHECK:      aiex.npu.write32(%[[A9]], %[[V9]]) {column = 0 : i32, row = 2 : i32}
// CHECK:      %[[A10:.*]] = arith.constant 213204 : i32
// CHECK:      %[[V10:.*]] = arith.constant 1 : i32
// CHECK:      aiex.npu.write32(%[[A10]], %[[V10]]) {column = 0 : i32, row = 2 : i32}
// CHECK:      %[[A11:.*]] = arith.constant 213216 : i32
// CHECK:      %[[V11:.*]] = arith.constant 18948645 : i32
// CHECK:      aiex.npu.write32(%[[A11]], %[[V11]]) {column = 0 : i32, row = 2 : i32}
// CHECK:      %[[A12:.*]] = arith.constant 213220 : i32
// CHECK:      %[[V12:.*]] = arith.constant 741165903 : i32
// CHECK:      aiex.npu.write32(%[[A12]], %[[V12]]) {column = 0 : i32, row = 2 : i32}
// CHECK:      %[[A13:.*]] = arith.constant 261888 : i32
// CHECK:      %[[V13:.*]] = arith.constant 289 : i32
// CHECK:      aiex.npu.write32(%[[A13]], %[[V13]]) {column = 0 : i32, row = 2 : i32}
// CHECK:      aiex.npu.writebd {bd_id = 14 : i32, buffer_length = 16384 : i32, buffer_offset = 65536 : i32, column = 0 : i32
// CHECK-SAME: enable_packet = 1 : i32
// CHECK-SAME: packet_id = 1 : i32, packet_type = 0 : i32
// CHECK:      aiex.npu.address_patch({{.*}}) {addr = 119236 : ui32, arg_idx = 2 : i32}
// CHECK:      %[[A14:.*]] = arith.constant 119308 : i32
// CHECK:      %[[V14:.*]] = arith.constant 14 : i32
// CHECK:      aiex.npu.write32(%[[A14]], %[[V14]]) {column = 0 : i32, row = 0 : i32}
// CHECK:      %[[A15:.*]] = arith.constant 212992 : i32
// CHECK:      %[[V15:.*]] = arith.constant 32512 : i32
// CHECK:      aiex.npu.write32(%[[A15]], %[[V15]]) {column = 0 : i32, row = 0 : i32}
// CHECK:      %[[A16:.*]] = arith.constant 213068 : i32
// CHECK:      %[[V16:.*]] = arith.constant 127 : i32
// CHECK:      aiex.npu.write32(%[[A16]], %[[V16]]) {column = 0 : i32, row = 0 : i32}
// CHECK:      %[[A17:.*]] = arith.constant 213000 : i32
// CHECK:      %[[V17:.*]] = arith.constant 127 : i32
// CHECK:      aiex.npu.write32(%[[A17]], %[[V17]]) {column = 0 : i32, row = 0 : i32}

    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c64_i64 = arith.constant 64 : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
    %p = airrt.segment_load "segment0" : i64
    %c7_i32 = arith.constant 7 : i32
    airrt.dma_memcpy_nd(%c7_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64, i64])
    return
  }
}
