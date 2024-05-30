//===- air_channel_to_objectfifo_subchannels.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-to-aie='test-patterns=specialize-channel-bundle' | air-opt --air-to-aie='test-patterns=lower-air-channels' | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:   %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:   %[[VAL_1:.*]] = aie.tile(2, 1)
// CHECK:   %[[VAL_2:.*]] = aie.tile(1, 2)
// CHECK:   %[[VAL_3:.*]] = aie.tile(2, 2)
// CHECK:   aie.objectfifo @[[VAL_4:.*]](%[[VAL_2]], {%[[VAL_3]]}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
// CHECK:   aie.objectfifo @[[VAL_5:.*]](%[[VAL_0]], {%[[VAL_1]]}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
// CHECK:   %[[VAL_6:.*]] = aie.core(%[[VAL_3]]) {
// CHECK:     %[[VAL_7:.*]] = aie.objectfifo.acquire @[[VAL_4]](Consume, 1) : !aie.objectfifosubview<memref<32xi32>>
// CHECK:     %[[VAL_8:.*]] = aie.objectfifo.subview.access %[[VAL_7]][0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     aie.objectfifo.release @[[VAL_4]](Consume, 1)
// CHECK:     aie.end
// CHECK:   } {elf_file = "partition_0_core_2_2.elf"}
// CHECK:   %[[VAL_9:.*]] = aie.core(%[[VAL_2]]) {
// CHECK:     %[[VAL_10:.*]] = aie.objectfifo.acquire @[[VAL_4]](Produce, 1) : !aie.objectfifosubview<memref<32xi32>>
// CHECK:     %[[VAL_11:.*]] = aie.objectfifo.subview.access %[[VAL_10]][0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     aie.objectfifo.release @[[VAL_4]](Produce, 1)
// CHECK:     aie.end
// CHECK:   } {elf_file = "partition_0_core_1_2.elf"}
// CHECK:   %[[VAL_12:.*]] = aie.core(%[[VAL_1]]) {
// CHECK:     %[[VAL_13:.*]] = aie.objectfifo.acquire @[[VAL_5]](Consume, 1) : !aie.objectfifosubview<memref<32xi32>>
// CHECK:     %[[VAL_14:.*]] = aie.objectfifo.subview.access %[[VAL_13]][0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     aie.objectfifo.release @[[VAL_5]](Consume, 1)
// CHECK:     aie.end
// CHECK:   } {elf_file = "partition_0_core_2_1.elf"}
// CHECK:   %[[VAL_15:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:     %[[VAL_16:.*]] = aie.objectfifo.acquire @[[VAL_5]](Produce, 1) : !aie.objectfifosubview<memref<32xi32>>
// CHECK:     %[[VAL_17:.*]] = aie.objectfifo.subview.access %[[VAL_16]][0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
// CHECK:     aie.objectfifo.release @[[VAL_5]](Produce, 1)
// CHECK:     aie.end
// CHECK:   } {elf_file = "partition_0_core_1_1.elf"}
// CHECK: }
   
aie.device(xcvc1902) {
    %0 = aie.tile(1, 1)
    %1 = aie.tile(2, 1)
    %2 = aie.tile(1, 2)
    %3 = aie.tile(2, 2)
    air.channel @channel_0 [1, 2]
    %4 = aie.core(%3) {
        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %alloc = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
        air.channel.get  @channel_0[%c0, %c1] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
        memref.dealloc %alloc : memref<32xi32, 2>
        aie.end
    } {elf_file = "partition_0_core_2_2.elf"}
    %5 = aie.core(%2) {
        %c32 = arith.constant 32 : index
        %c1 = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
        air.channel.put  @channel_0[%c0, %c1] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
        memref.dealloc %alloc : memref<32xi32, 2>
        aie.end
    } {elf_file = "partition_0_core_1_2.elf"}
    %6 = aie.core(%1) {
        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %alloc = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
        air.channel.get  @channel_0[%c0, %c0] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
        memref.dealloc %alloc : memref<32xi32, 2>
        aie.end
    } {elf_file = "partition_0_core_2_1.elf"}
    %7 = aie.core(%0) {
        %c32 = arith.constant 32 : index
        %c0 = arith.constant 0 : index
        %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
        air.channel.put  @channel_0[%c0, %c0] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
        memref.dealloc %alloc : memref<32xi32, 2>
        aie.end
    } {elf_file = "partition_0_core_1_1.elf"}
}
