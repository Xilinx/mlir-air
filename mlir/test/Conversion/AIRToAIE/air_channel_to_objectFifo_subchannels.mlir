//===- air_channel_to_objectFifo_subchannels.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-to-aie='test-patterns=lower-air-channels' | FileCheck %s

// CHECK: module @aie.partition_0 {
// CHECK:   %0 = AIE.tile(1, 1)
// CHECK:   %1 = AIE.tile(2, 1)
// CHECK:   %2 = AIE.tile(1, 2)
// CHECK:   %3 = AIE.tile(2, 2)
// CHECK:   %4 = AIE.objectFifo.createObjectFifo(%0, {%1}, 1) : !AIE.objectFifo<memref<32xi32>>
// CHECK:   %5 = AIE.objectFifo.createObjectFifo(%2, {%3}, 1) : !AIE.objectFifo<memref<32xi32>>
// CHECK:   %6 = AIE.core(%3) {
// CHECK:     %10 = AIE.objectFifo.acquire<Consume> (%5 : !AIE.objectFifo<memref<32xi32>>, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %11 = AIE.objectFifo.subview.access %10[0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     AIE.objectFifo.release<Consume> (%5 : !AIE.objectFifo<memref<32xi32>>, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_2_2.elf"}
// CHECK:   %7 = AIE.core(%2) {
// CHECK:     %10 = AIE.objectFifo.acquire<Produce> (%5 : !AIE.objectFifo<memref<32xi32>>, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %11 = AIE.objectFifo.subview.access %10[0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     AIE.objectFifo.release<Produce> (%5 : !AIE.objectFifo<memref<32xi32>>, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_1_2.elf"}
// CHECK:   %8 = AIE.core(%1) {
// CHECK:     %10 = AIE.objectFifo.acquire<Consume> (%4 : !AIE.objectFifo<memref<32xi32>>, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %11 = AIE.objectFifo.subview.access %10[0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     AIE.objectFifo.release<Consume> (%4 : !AIE.objectFifo<memref<32xi32>>, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_2_1.elf"}
// CHECK:   %9 = AIE.core(%0) {
// CHECK:     %10 = AIE.objectFifo.acquire<Produce> (%4 : !AIE.objectFifo<memref<32xi32>>, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:     %11 = AIE.objectFifo.subview.access %10[0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     AIE.objectFifo.release<Produce> (%4 : !AIE.objectFifo<memref<32xi32>>, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_1_1.elf"}
// CHECK: }

module @aie.partition_0 {
  %0 = AIE.tile(1, 1)
  %1 = AIE.tile(2, 1)
  %2 = AIE.tile(1, 2)
  %3 = AIE.tile(2, 2)
  air.channel @channel_0 [1, 2]
  %4 = AIE.core(%3) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
    air.channel.get  @channel_0[%c0, %c1] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    AIE.end
  } {elf_file = "partition_0_core_2_2.elf"}
  %5 = AIE.core(%2) {
    %c32 = arith.constant 32 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
    air.channel.put  @channel_0[%c0, %c1] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
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
