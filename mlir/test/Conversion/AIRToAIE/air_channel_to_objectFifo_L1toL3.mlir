//===- air_channel_to_objectFifo_L1toL3.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-to-aie='test-patterns=lower-air-channels' | FileCheck %s

// CHECK: module @aie.partition_0 {
// CHECK:   %0 = AIE.tile(1, 1)
// CHECK:   %1 = AIE.tile(2, 0)
// CHECK:   %2 = AIE.objectFifo.createObjectFifo(%0, {%1}, 1) : !AIE.objectFifo<memref<32xi32>>
// CHECK:   %3 = AIE.objectFifo.createObjectFifo(%1, {%0}, 1) : !AIE.objectFifo<memref<32xi32>>
// CHECK:   %4 = AIE.core(%0) {
// CHECK:     affine.for %arg0 = 0 to 4096 step 32 {
// CHECK:       %5 = AIE.objectFifo.acquire<Consume> (%3 : !AIE.objectFifo<memref<32xi32>>, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:       %6 = AIE.objectFifo.subview.access %5[0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:       %7 = AIE.objectFifo.acquire<Produce> (%2 : !AIE.objectFifo<memref<32xi32>>, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:       %8 = AIE.objectFifo.subview.access %7[0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:       affine.for %arg1 = 0 to 32 {
// CHECK:         %9 = affine.load %6[%arg1] : memref<32xi32>
// CHECK:         affine.store %9, %8[%arg1] : memref<32xi32>
// CHECK:       }
// CHECK:       AIE.objectFifo.release<Produce> (%2 : !AIE.objectFifo<memref<32xi32>>, 1)
// CHECK:       AIE.objectFifo.release<Consume> (%3 : !AIE.objectFifo<memref<32xi32>>, 1)
// CHECK:     }
// CHECK:     AIE.end
// CHECK:   } {elf_file = "partition_0_core_1_1.elf"}
// CHECK: }

module @aie.partition_0 {
  %0 = AIE.tile(1, 1)
  %1 = AIE.core(%0) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
    %alloc_0 = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
    affine.for %arg0 = 0 to 4096 step 32 {
      air.channel.get  @channel_0[] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
      affine.for %arg1 = 0 to 32 {
        %2 = affine.load %alloc[%arg1] : memref<32xi32, 2>
        affine.store %2, %alloc_0[%arg1] : memref<32xi32, 2>
      }
      air.channel.put  @channel_1[] (%alloc_0[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    }
    memref.dealloc %alloc_0 : memref<32xi32, 2>
    memref.dealloc %alloc : memref<32xi32, 2>
    AIE.end
  } {elf_file = "partition_0_core_1_1.elf"}
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
}
