//===- specialize_channel_bundle.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-to-aie='test-patterns=specialize-channel-bundle' | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:   %[[VAL_0:.*]] = AIE.tile(1, 1)
// CHECK:   %[[VAL_1:.*]] = AIE.tile(1, 2)
// CHECK-COUNT-8:    air.channel @{{.*}}[1, 1]
// CHECK:   %[[VAL_2:.*]] = AIE.core(%[[VAL_1]]) {
// CHECK:     air.channel.get @channel{{.*}}[]
// CHECK:     air.channel.get @channel{{.*}}[]
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_1_2.elf"}
// CHECK:   %[[VAL_3:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:     air.channel.put @channel{{.*}}[]
// CHECK:     air.channel.put @channel{{.*}}[]
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_1_1.elf"}
// CHECK: }

AIE.device(xcvc1902) {
  %0 = AIE.tile(1, 1)
  %1 = AIE.tile(1, 2)
  air.channel @channel_0 [2, 2]
  air.channel @channel_1 [2, 2]
  %2 = AIE.core(%1) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
    %alloc2 = memref.alloc() {sym_name = "scratch_copy2"} : memref<32xi32, 2>
    air.channel.get @channel_0[%c0, %c0] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    air.channel.get @channel_1[%c0, %c1] (%alloc2[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    memref.dealloc %alloc2 : memref<32xi32, 2>
    AIE.end
  } {elf_file = "partition_0_core_1_2.elf"}
  %3 = AIE.core(%0) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
    %alloc2 = memref.alloc() {sym_name = "scratch2"} : memref<32xi32, 2>
    air.channel.put @channel_0[%c0, %c0] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    air.channel.put @channel_1[%c0, %c1] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    memref.dealloc %alloc2 : memref<32xi32, 2>
    AIE.end
  } {elf_file = "partition_0_core_1_1.elf"}
}

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:   %[[VAL_0:.*]] = AIE.tile(1, 1)
// CHECK:   %[[VAL_1:.*]] = AIE.tile(1, 2)
// CHECK-COUNT-8:    air.channel @{{.*}}[1, 1]
// CHECK:   %[[VAL_2:.*]] = AIE.core(%[[VAL_1]]) {
// CHECK:     %[[VAL_3:.*]] = air.channel.get async @channel{{.*}}[]
// CHECK:     %[[VAL_4:.*]] = air.channel.get async @channel{{.*}}[]
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_1_2.elf"}
// CHECK:   %[[VAL_5:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:     %[[VAL_6:.*]] = air.channel.put async @channel{{.*}}[]
// CHECK:     %[[VAL_7:.*]] = air.channel.put async @channel{{.*}}[]
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_1_1.elf"}
// CHECK: }

AIE.device(xcvc1902) {
  %0 = AIE.tile(1, 1)
  %1 = AIE.tile(1, 2)
  air.channel @channel_0 [2, 2]
  air.channel @channel_1 [2, 2]
  %2 = AIE.core(%1) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
    %alloc2 = memref.alloc() {sym_name = "scratch_copy2"} : memref<32xi32, 2>
    %4 = air.channel.get async @channel_0[%c0, %c0] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    %5 = air.channel.get async @channel_1[%c0, %c1] (%alloc2[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    memref.dealloc %alloc2 : memref<32xi32, 2>
    AIE.end
  } {elf_file = "partition_0_core_1_2.elf"}
  %3 = AIE.core(%0) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
    %alloc2 = memref.alloc() {sym_name = "scratch2"} : memref<32xi32, 2>
    %4 = air.channel.put async @channel_0[%c0, %c0] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    %5 = air.channel.put async @channel_1[%c0, %c1] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    memref.dealloc %alloc : memref<32xi32, 2>
    memref.dealloc %alloc2 : memref<32xi32, 2>
    AIE.end
  } {elf_file = "partition_0_core_1_1.elf"}
}
