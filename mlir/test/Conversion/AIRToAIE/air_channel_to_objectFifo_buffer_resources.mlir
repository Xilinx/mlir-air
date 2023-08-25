//===- air_channel_to_objectFifo_buffer_resources.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-to-aie='test-patterns=lower-air-channels' | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:   %[[VAL_0:.*]] = AIE.tile(1, 1)
// CHECK:   %[[VAL_1:.*]] = AIE.tile(1, 2)
// CHECK:   AIE.objectFifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !AIE.objectFifo<memref<32xi32>>
// CHECK:   AIE.objectFifo @[[VAL_3:.*]](%[[VAL_0]], {%[[VAL_1]]}, 1 : i32) : !AIE.objectFifo<memref<32xi32>>
// CHECK:   %[[VAL_4:.*]] = AIE.core(%[[VAL_1]]) {
// CHECK:     %[[VAL_5:.*]] = AIE.objectFifo.acquire @[[VAL_3]](Consume, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %[[VAL_6:.*]] = AIE.objectFifo.subview.access %[[VAL_5]][0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     %[[VAL_7:.*]] = AIE.objectFifo.acquire @[[VAL_2]](Consume, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %[[VAL_8:.*]] = AIE.objectFifo.subview.access %[[VAL_7]][0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     AIE.objectFifo.release @[[VAL_2]](Consume, 1)
// CHECK:     AIE.objectFifo.release @[[VAL_3]](Consume, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_1_2.elf"}
// CHECK:   %[[VAL_9:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:     %[[VAL_10:.*]] = AIE.objectFifo.acquire @[[VAL_3]](Produce, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %[[VAL_11:.*]] = AIE.objectFifo.subview.access %[[VAL_10]][0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     %[[VAL_12:.*]] = AIE.objectFifo.acquire @[[VAL_2]](Produce, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %[[VAL_13:.*]] = AIE.objectFifo.subview.access %[[VAL_12]][0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     AIE.objectFifo.release @[[VAL_2]](Produce, 1)
// CHECK:     AIE.objectFifo.release @[[VAL_3]](Produce, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_1_1.elf"}
// CHECK: }

AIE.device(xcvc1902) {
  %0 = AIE.tile(1, 1)
  %1 = AIE.tile(1, 2)
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1] {buffer_resources = 2}
  %2 = AIE.core(%1) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
    %alloc2 = memref.alloc() {sym_name = "scratch_copy2"} : memref<32xi32, 2>
    air.channel.get @channel_0[] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    air.channel.get @channel_1[] (%alloc2[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    memref.dealloc %alloc2 : memref<32xi32, 2>
    AIE.end
  } {elf_file = "partition_0_core_1_2.elf"}
  %3 = AIE.core(%0) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
    %alloc2 = memref.alloc() {sym_name = "scratch2"} : memref<32xi32, 2>
    air.channel.put @channel_0[] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    air.channel.put @channel_1[] (%alloc2[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    memref.dealloc %alloc2 : memref<32xi32, 2>
    AIE.end
  } {elf_file = "partition_0_core_1_1.elf"}
}

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:   %[[VAL_0:.*]] = AIE.tile(1, 1)
// CHECK:   %[[VAL_1:.*]] = AIE.tile(1, 2)
// CHECK:   AIE.objectFifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 2 : i32) : !AIE.objectFifo<memref<32xi32>>
// CHECK:   AIE.objectFifo @[[VAL_3:.*]](%[[VAL_0]], {%[[VAL_1]]}, 1 : i32) : !AIE.objectFifo<memref<32xi32>>
// CHECK:   %[[VAL_4:.*]] = AIE.core(%[[VAL_1]]) {
// CHECK:     %[[VAL_5:.*]] = AIE.objectFifo.acquire @[[VAL_3]](Consume, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %[[VAL_6:.*]] = AIE.objectFifo.subview.access %[[VAL_5]][0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     %[[VAL_7:.*]] = AIE.objectFifo.acquire @[[VAL_2]](Consume, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %[[VAL_8:.*]] = AIE.objectFifo.subview.access %[[VAL_7]][0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     AIE.objectFifo.release @[[VAL_3]](Consume, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_1_2.elf"}
// CHECK:   %[[VAL_9:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:     %[[VAL_10:.*]] = AIE.objectFifo.acquire @[[VAL_3]](Produce, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %[[VAL_11:.*]] = AIE.objectFifo.subview.access %[[VAL_10]][0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     %[[VAL_12:.*]] = AIE.objectFifo.acquire @[[VAL_2]](Produce, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %[[VAL_13:.*]] = AIE.objectFifo.subview.access %[[VAL_12]][0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     AIE.objectFifo.release @[[VAL_2]](Produce, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_1_1.elf"}
// CHECK: }

AIE.device(xcvc1902) {
  %0 = AIE.tile(1, 1)
  %1 = AIE.tile(1, 2)
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1] {buffer_resources = 2}
  %2 = AIE.core(%1) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
    %alloc2 = memref.alloc() {sym_name = "scratch_copy2"} : memref<32xi32, 2>
    %4 = air.channel.get async  @channel_0[] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    %5 = air.channel.get async [%4] @channel_1[] (%alloc2[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    memref.dealloc %alloc2 : memref<32xi32, 2>
    AIE.end
  } {elf_file = "partition_0_core_1_2.elf"}
  %3 = AIE.core(%0) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
    %alloc2 = memref.alloc() {sym_name = "scratch2"} : memref<32xi32, 2>
    %4 = air.channel.put async @channel_0[] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    %5 = air.channel.put async @channel_1[] (%alloc2[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    memref.dealloc %alloc2 : memref<32xi32, 2>
    AIE.end
  } {elf_file = "partition_0_core_1_1.elf"}
}
