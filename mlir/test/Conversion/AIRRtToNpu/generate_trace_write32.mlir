//===- generate_trace_write32.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -airrt-to-npu='trace-offset=65536 trace-size=32768'  | FileCheck %s
module {
  aie.device(npu1_4col) {
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
    aie.shim_dma_allocation @airMemcpyId7(S2MM, 0, 0)
    memref.global "public" @airMemcpyId7 : memref<64xi32, 1>
    aie.shim_dma_allocation @airMemcpyId2(MM2S, 0, 0)
    memref.global "public" @airMemcpyId2 : memref<64xi32, 1>
  } {sym_name = "segment0"}
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  air.channel @channel_3 [1, 1]
  func.func @func0(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
// CHECK:      aiex.npu.write32 {address = 606208 : ui32, column = 0 : i32, row = 1 : i32, value = 40192 : ui32}
// CHECK:      aiex.npu.write32 {address = 606416 : ui32, column = 0 : i32, row = 1 : i32, value = 10289152 : ui32}
// CHECK:      aiex.npu.write32 {address = 606420 : ui32, column = 0 : i32, row = 1 : i32, value = 12288 : ui32}
// CHECK:      aiex.npu.write32 {address = 606432 : ui32, column = 0 : i32, row = 1 : i32, value = 22041688 : ui32}
// CHECK:      aiex.npu.write32 {address = 606436 : ui32, column = 0 : i32, row = 1 : i32, value = 1549821032 : ui32}
// CHECK:      aiex.npu.write32 {address = 724736 : ui32, column = 0 : i32, row = 1 : i32, value = 2236704 : ui32}
// CHECK:      aiex.npu.write32 {address = 724740 : ui32, column = 0 : i32, row = 1 : i32, value = 197121 : ui32}
// CHECK:      aiex.npu.writebd_shimtile {bd_id = 15 : i32, buffer_length = 8192 : i32, buffer_offset = 65536 : i32, column = 0 : i32, column_num = 1 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d2_stride = 0 : i32, ddr_id = 2 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 3 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
// CHECK:      aiex.npu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 15 : ui32}
// CHECK:      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 2 : i32, value = 31232 : ui32}
// CHECK:      aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = 7995392 : ui32}
// CHECK:      aiex.npu.write32 {address = 213204 : ui32, column = 0 : i32, row = 2 : i32, value = 1 : ui32}
// CHECK:      aiex.npu.write32 {address = 213216 : ui32, column = 0 : i32, row = 2 : i32, value = 18948645 : ui32}
// CHECK:      aiex.npu.write32 {address = 213220 : ui32, column = 0 : i32, row = 2 : i32, value = 741165903 : ui32}
// CHECK:      aiex.npu.write32 {address = 261888 : ui32, column = 0 : i32, row = 2 : i32, value = 289 : ui32}
// CHECK:      aiex.npu.writebd_shimtile {bd_id = 15 : i32, buffer_length = 8192 : i32, buffer_offset = 65536 : i32, column = 0 : i32, column_num = 1 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d2_stride = 0 : i32, ddr_id = 2 : i32, enable_packet = 1 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 1 : i32, packet_type = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}
// CHECK:      aiex.npu.write32 {address = 119308 : ui32, column = 0 : i32, row = 0 : i32, value = 15 : ui32}
// CHECK:      aiex.npu.write32 {address = 212992 : ui32, column = 0 : i32, row = 0 : i32, value = 32512 : ui32}
// CHECK:      aiex.npu.write32 {address = 213068 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}
// CHECK:      aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 127 : ui32}

    %c0_i64 = arith.constant 0 : i64
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %c64_i64 = arith.constant 64 : i64
    airrt.dma_memcpy_nd(%c2_i32, %c0_i64, %c0_i64, %arg0[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId2} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
    %p = airrt.segment_load "segment0" : i64
    %c7_i32 = arith.constant 7 : i32
    airrt.dma_memcpy_nd(%c7_i32, %c0_i64, %c0_i64, %arg1[%c0_i64, %c0_i64, %c0_i64, %c0_i64], [%c1_i64, %c1_i64, %c1_i64, %c64_i64], [%c0_i64, %c0_i64, %c0_i64]) {metadata = @airMemcpyId7} : (i32, i64, i64, memref<64xi32>, [i64, i64, i64, i64], [i64, i64, i64, i64], [i64, i64, i64])
    return
  }
}