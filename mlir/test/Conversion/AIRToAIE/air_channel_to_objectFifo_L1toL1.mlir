//===- air_channel_to_objectFifo_L1toL1.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-to-aie='test-patterns=lower-air-channels' | FileCheck %s

// CHECK: module @aie.segment_0 {
// CHECK:   %0 = AIE.tile(1, 1)
// CHECK:   %1 = AIE.tile(1, 2)
// CHECK:   %2 = AIE.objectFifo.createObjectFifo(%0, {%1}, 1) {sym_name = "air_channel_0"} : !AIE.objectFifo<memref<32xi32, 2>>
// CHECK:   %3 = AIE.core(%1) {
// CHECK:     %5 = AIE.objectFifo.acquire<Consume> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
// CHECK:     %6 = AIE.objectFifo.subview.access %5[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
// CHECK:     AIE.objectFifo.release<Consume> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "segment_0_core_1_2.elf"}
// CHECK:   %4 = AIE.core(%0) {
// CHECK:     %5 = AIE.objectFifo.acquire<Produce> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
// CHECK:     %6 = AIE.objectFifo.subview.access %5[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
// CHECK:     AIE.objectFifo.release<Produce> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "segment_0_core_1_1.elf"}
// CHECK: }

module @aie.segment_0 {
  %0 = AIE.tile(1, 1)
  %1 = AIE.tile(1, 2)
  air.channel @channel_0 [1, 1]
  %2 = AIE.core(%1) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
    air.channel.get  @channel_0[] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    AIE.end
  } {elf_file = "segment_0_core_1_2.elf"}
  %3 = AIE.core(%0) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
    air.channel.put  @channel_0[] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    AIE.end
  } {elf_file = "segment_0_core_1_1.elf"}
}
