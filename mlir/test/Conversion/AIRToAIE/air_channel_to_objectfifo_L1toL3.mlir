//===- air_channel_to_objectfifo_L1toL3.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-to-aie='test-patterns=lower-air-channels'  | FileCheck %s

// CHECK-LABEL:   aie.device(xcvc1902) {
// CHECK:   %[[VAL_0:.*]] = aie.tile(1, 1)
// CHECK:   %[[VAL_1:.*]] = aie.tile(2, 0)
// CHECK:   aie.objectfifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
// CHECK:   aie.objectfifo @[[VAL_3:.*]](%[[VAL_1]], {%[[VAL_0]]}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
// CHECK:   %[[VAL_4:.*]] = aie.core(%[[VAL_0]]) {
// CHECK:     affine.for %[[VAL_5:.*]] = 0 to 4096 step 32 {
// CHECK:       %[[VAL_6:.*]] = aie.objectfifo.acquire @[[VAL_3]](Consume, 1) : !aie.objectfifosubview<memref<32xi32>>
// CHECK:       %[[VAL_7:.*]] = aie.objectfifo.subview.access %[[VAL_6]][0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
// CHECK:       %[[VAL_8:.*]] = aie.objectfifo.acquire @[[VAL_2]](Produce, 1) : !aie.objectfifosubview<memref<32xi32>>
// CHECK:       %[[VAL_9:.*]] = aie.objectfifo.subview.access %[[VAL_8]][0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
// CHECK:       affine.for %[[VAL_10:.*]] = 0 to 32 {
// CHECK:         %[[VAL_11:.*]] = affine.load %[[VAL_7]]{{\[}}%[[VAL_10]]] : memref<32xi32>
// CHECK:         affine.store %[[VAL_11]], %[[VAL_9]]{{\[}}%[[VAL_10]]] : memref<32xi32>
// CHECK:       }
// CHECK:       aie.objectfifo.release @[[VAL_2]](Produce, 1)
// CHECK:       aie.objectfifo.release @[[VAL_3]](Consume, 1)
// CHECK:     }
// CHECK:     aie.end
// CHECK:   } {elf_file = "segment_0_core_1_1.elf"}
// CHECK: }

aie.device(xcvc1902) {
  %0 = aie.tile(1, 1)
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  %1 = aie.core(%0) {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
    %alloc_0 = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
    affine.for %arg0 = 0 to 4096 step 32 {
      %3 = air.channel.get async @channel_0[] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
      affine.for %arg1 = 0 to 32 {
        %2 = affine.load %alloc[%arg1] : memref<32xi32, 2>
        affine.store %2, %alloc_0[%arg1] : memref<32xi32, 2>
      }
      air.channel.put  @channel_1[] (%alloc_0[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
    }
    memref.dealloc %alloc_0 : memref<32xi32, 2>
    memref.dealloc %alloc : memref<32xi32, 2>
    aie.end
  } {elf_file = "segment_0_core_1_1.elf"}
}
