//===- air_channel_to_objectFifo_L1toL2.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --air-to-aie='test-patterns=lower-air-channels device=xcve2302' | FileCheck %s

// CHECK-LABEL:   AIE.device(xcvc1902) {
// CHECK:    %[[VAL_0:.*]] = AIE.tile(1, 1)
// CHECK:    %[[VAL_1:.*]] = AIE.tile(1, 2)
// CHECK:    AIE.objectFifo @[[VAL_2:.*]](%[[VAL_0]], {%[[VAL_1]]}, 1 : i32) : !AIE.objectFifo<memref<32xi32>>
// CHECK:    %[[VAL_3:.*]] = AIE.core(%[[VAL_1]]) {
// CHECK:      %[[VAL_4:.*]] = AIE.objectFifo.acquire @[[VAL_2]](Consume, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:      %[[VAL_5:.*]] = AIE.objectFifo.subview.access %[[VAL_4]][0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:      AIE.objectFifo.release @[[VAL_2]](Consume, 1)
// CHECK:      AIE.end
// CHECK:    } {elf_file = "segment_0_core_1_2.elf"}
// CHECK:    %[[VAL_6:.*]] = AIE.core(%[[VAL_0]]) {
// CHECK:      %[[VAL_7:.*]] = AIE.objectFifo.acquire @[[VAL_2]](Produce, 1) : !AIE.objectFifoSubview<memref<32xi32>>
// CHECK:      %[[VAL_8:.*]] = AIE.objectFifo.subview.access %[[VAL_7]][0] : !AIE.objectFifoSubview<memref<32xi32>> -> memref<32xi32>
// CHECK:      AIE.objectFifo.release @[[VAL_2]](Produce, 1)
// CHECK:      AIE.end
// CHECK:    } {elf_file = "segment_0_core_1_1.elf"}
// CHECK:  }

// AIE.device(xcve2302) {
//   %0 = AIE.tile(1, 1)
//   %1 = AIE.tile(1, 2)
//   air.channel @channel_0 [1, 1]
//   %2 = AIE.core(%1) {
//     %c32 = arith.constant 32 : index
//     %c0 = arith.constant 0 : index
//     %alloc = memref.alloc() {sym_name = "scratch_copy"} : memref<32xi32, 2>
//     air.channel.get  @channel_0[] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
//     memref.dealloc %alloc : memref<32xi32, 2>
//     AIE.end
//   } {elf_file = "segment_0_core_1_2.elf"}
//   %3 = AIE.core(%0) {
//     %c32 = arith.constant 32 : index
//     %c0 = arith.constant 0 : index
//     %alloc = memref.alloc() {sym_name = "scratch"} : memref<32xi32, 2>
//     air.channel.put  @channel_0[] (%alloc[%c0] [%c32] [%c0]) : (memref<32xi32, 2>)
//     memref.dealloc %alloc : memref<32xi32, 2>
//     AIE.end
//   } {elf_file = "segment_0_core_1_1.elf"}
// }

module {
  air.channel @channel_0 [1, 1]
  func.func @L2toL1(%arg0: memref<32xi32>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index

    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c2) args(%arg7=%arg0) : memref<32xi32> attributes {id = 1 : i32} {
      air.segment args(%arg10=%arg3, %arg11=%arg4) : index, index attributes {id = 1 : i32} {
        %alloc = memref.alloc() : memref<32xi32, 1>
        air.channel.put @channel_0[] (%alloc[] [] []) {id = 2 : i32} : (memref<32xi32, 1>)
        %c1_2 = arith.constant 1 : index
        air.herd @herd_0 tile (%arg14, %arg15) in (%arg16=%c1_2, %arg17=%c1_2) attributes {id = 3 : i32} {
            %alloc2 = memref.alloc() : memref<32xi32, 2>
            air.channel.get @channel_0[] (%alloc2[] [] []) {id = 4 : i32} : (memref<32xi32, 2>)
            air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}