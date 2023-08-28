//===- air_channel_to_objectFifo_broadcast.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-to-aie='test-patterns=lower-air-channels' | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:   %[[VAL_0:.*]] = AIE.tile(1, 1)
// CHECK:   %[[VAL_1:.*]] = AIE.tile(2, 1)
// CHECK:   %[[VAL_2:.*]] = AIE.tile(1, 2)
// CHECK:   %[[VAL_3:.*]] = AIE.tile(2, 2)
// CHECK:   AIE.objectFifo @[[VAL_4:.*]](%[[VAL_0]], {%[[VAL_3]], %[[VAL_2]], %[[VAL_1]]}, 1 : i32) : !AIE.objectFifo<memref<32xi32>>
// CHECK:   %[[VAL_5:.*]] = AIE.core(%[[VAL_3]]) {
// CHECK:     %[[VAL_6:.*]] = AIE.objectFifo.acquire @[[VAL_4]](Consume, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %[[VAL_7:.*]] = AIE.objectFifo.subview.access %[[VAL_6]][0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     AIE.objectFifo.release @[[VAL_4]](Consume, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_2_2.elf"}
// CHECK:   %[[VAL_8:.*]] = AIE.core(%[[VAL_2]]) {
// CHECK:     %[[VAL_9:.*]] = AIE.objectFifo.acquire @[[VAL_4]](Consume, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %[[VAL_10:.*]] = AIE.objectFifo.subview.access %[[VAL_9]][0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     AIE.objectFifo.release @[[VAL_4]](Consume, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_1_2.elf"}
// CHECK:   %[[VAL_11:.*]] = AIE.core(%[[VAL_1]]) {
// CHECK:     %[[VAL_12:.*]] = AIE.objectFifo.acquire @[[VAL_4]](Consume, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %[[VAL_13:.*]] = AIE.objectFifo.subview.access %[[VAL_12]][0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     AIE.objectFifo.release @[[VAL_4]](Consume, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_2_1.elf"}
// CHECK:   %[[VAL_14:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:     %[[VAL_15:.*]] = AIE.objectFifo.acquire @[[VAL_4]](Produce, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %[[VAL_16:.*]] = AIE.objectFifo.subview.access %[[VAL_15]][0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     AIE.objectFifo.release @[[VAL_4]](Produce, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_1_1.elf"}
// CHECK: }

AIE.device(xcvc1902) {
  %0 = AIE.tile(1, 1)
  %1 = AIE.tile(2, 1)
  %2 = AIE.tile(1, 2)
  %3 = AIE.tile(2, 2)
  air.channel @channel_0 [1, 1] {broadcast_shape = [1, 3]}
  %4 = AIE.core(%3) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %alloc = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
    air.channel.get  @channel_0[%c0, %c2] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    AIE.end
  } {elf_file = "partition_0_core_2_2.elf"}
  %5 = AIE.core(%2) {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
    air.channel.get  @channel_0[%c0, %c1] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    AIE.end
  } {elf_file = "partition_0_core_1_2.elf"}
  %6 = AIE.core(%1) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
    air.channel.get  @channel_0[%c0, %c0] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    AIE.end
  } {elf_file = "partition_0_core_2_1.elf"}
  %7 = AIE.core(%0) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
    air.channel.put  @channel_0[%c0, %c0] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    AIE.end
  } {elf_file = "partition_0_core_1_1.elf"}
}
