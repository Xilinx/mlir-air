//===- air_channel_to_objectFifo_L1toL1.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='test-patterns=specialize-affine-if,lower-air-channels' | air-opt --canonicalize | FileCheck %s
// CHECK: module @aie.partition_0 {
// CHECK:   %0 = AIE.tile(1, 1)
// CHECK:   %1 = AIE.tile(1, 2)
// CHECK:   %2 = AIE.objectFifo.createObjectFifo(%0, {%1}, 1) : !AIE.objectFifo<memref<32xi32, 2>>
// CHECK:   %3 = AIE.core(%1) {
// CHECK:     %5 = AIE.objectFifo.acquire<Consume> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
// CHECK:     %6 = AIE.objectFifo.subview.access %5[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
// CHECK:     AIE.objectFifo.release<Consume> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_1_2.elf"}
// CHECK:   %4 = AIE.core(%0) {
// CHECK:     %5 = AIE.objectFifo.acquire<Produce> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1) : !AIE.objectFifoSubview<memref<32xi32, 2>>
// CHECK:     %6 = AIE.objectFifo.subview.access %5[0] : !AIE.objectFifoSubview<memref<32xi32, 2>> -> memref<32xi32, 2>
// CHECK:     AIE.objectFifo.release<Produce> (%2 : !AIE.objectFifo<memref<32xi32, 2>>, 1)
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_1_1.elf"}
// CHECK: }

#set = affine_set<()[s0, s1] : (s0 >= 0, s1 == 0)>
#set1 = affine_set<()[s0, s1] : (s0 >= 0, s1 - 1 == 0)>
module @aie.partition_0 {
  %0 = AIE.tile(1, 1)
  %1 = AIE.tile(1, 2)
  air.channel @channel_0 [1, 1] // Has to be added back in after 'test-patterns=to-aie-mlir'.
  %2 = AIE.core(%1) {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1_0 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    %c0_1 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    affine.if #set()[%c0, %c1] {
      %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
      air.channel.put  @channel_0[] (%alloc[%c0_1] [%c32] [%c0_1]) : (memref<32xi32, 2>)
      memref.dealloc %alloc : memref<32xi32, 2>
    }
    affine.if #set1()[%c0, %c1] {
      %alloc = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
      air.channel.get  @channel_0[] (%alloc[%c0_1] [%c32] [%c0_1]) : (memref<32xi32, 2>)
      memref.dealloc %alloc : memref<32xi32, 2>
    }
    AIE.end
  } {elf_file = "partition_0_core_1_2.elf"}
  %3 = AIE.core(%0) {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    cf.br ^bb2
  ^bb2:  // pred: ^bb1
    %c0_1 = arith.constant 0 : index
    %c32 = arith.constant 32 : index
    affine.if #set()[%c0, %c0_0] {
      %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
      air.channel.put  @channel_0[] (%alloc[%c0_1] [%c32] [%c0_1]) : (memref<32xi32, 2>)
      memref.dealloc %alloc : memref<32xi32, 2>
    }
    affine.if #set1()[%c0, %c0_0] {
      %alloc = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
      air.channel.get  @channel_0[] (%alloc[%c0_1] [%c32] [%c0_1]) : (memref<32xi32, 2>)
      memref.dealloc %alloc : memref<32xi32, 2>
    }
    AIE.end
  } {elf_file = "partition_0_core_1_1.elf"}
}
