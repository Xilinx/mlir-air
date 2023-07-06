//===- air_ping_pong_to_objectFifo_buffer_resources.mlir -------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-to-aie='test-patterns=lower-air-ping-pong' --air-to-aie='test-patterns=lower-air-channels' | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:   %[[VAL_0:.*]] = AIE.tile(1, 1)
// CHECK:   %[[VAL_1:.*]] = AIE.tile(2, 0)
// CHECK:   %[[VAL_2:.*]] = AIE.objectFifo.createObjectFifo(%[[VAL_0]], {%[[VAL_1]]}, 2) {sym_name = "air_channel_1"} : !AIE.objectFifo<memref<32xi32, 2>>
// CHECK:   %[[VAL_3:.*]] = AIE.objectFifo.createObjectFifo(%[[VAL_1]], {%[[VAL_0]]}, 2) {sym_name = "air_channel_0"} : !AIE.objectFifo<memref<32xi32, 2>>
// CHECK:   %[[VAL_4:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:     affine.for %[[VAL_5:.*]] = 0 to 4096 step 32 {
// CHECK:       %[[VAL_6:.*]] = AIE.objectFifo.acquire<Consume> (%[[VAL_3]] : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
// CHECK:       %[[VAL_7:.*]] = AIE.objectFifo.subview.access %[[VAL_6]][0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
// CHECK:       %[[VAL_8:.*]] = AIE.objectFifo.acquire<Produce> (%[[VAL_2]] : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
// CHECK:       %[[VAL_9:.*]] = AIE.objectFifo.subview.access %[[VAL_8]][0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
// CHECK:       AIE.objectFifo.release<Produce> (%[[VAL_2]] : !AIE.objectFifo<memref<32xi32, 2>>, 1)
// CHECK:     }
// CHECK:     AIE.end
// CHECK:   } {elf_file = "segment_0_core_1_1.elf"}
// CHECK: }

AIE.device(xcvc1902) {
  %0 = AIE.tile(1, 1)
  %1 = AIE.core(%0) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4096 = arith.constant 4096 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
    %async_token_0 = air.wait_all async
    %2 = scf.for %arg0 = %c0 to %c4096 step %c32 iter_args(%arg11 = %async_token_0) -> (!air.async.token) {
      %3 = air.channel.get async [%arg11]  @channel_0[] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
      %4 = air.channel.put async [%3]  @channel_1[] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
      scf.yield %4 : !air.async.token
    } {isolated = true, unroll = 2 : i64}
    memref.dealloc %alloc : memref<32xi32, 2>
    AIE.end
  } {elf_file = "segment_0_core_1_1.elf"}
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
}
