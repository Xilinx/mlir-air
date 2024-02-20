//===- air_channel_to_objectfifo_join.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds='num-rows=2 num-cols=2 row-anchor=3 col-anchor=5' --air-to-aie='use-objectfifo=true device=xcve2802' --canonicalize | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2802) {
// CHECK:    %[[VAL_0:.*]] = aie.tile(2, 0)
// CHECK:    %[[VAL_1:.*]] = aie.tile(1, 1)
// CHECK:    %[[VAL_2:.*]] = aie.tile(5, 3)
// CHECK:    %[[VAL_3:.*]] = aie.tile(5, 4)
// CHECK:    aie.objectfifo @air_channel_0(%[[VAL_1]], {%[[VAL_0]]}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
// CHECK:    aie.objectfifo @air_channel_3(%[[VAL_3]], {%[[VAL_1]]}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
// CHECK:    aie.objectfifo @air_channel_2(%[[VAL_2]], {%[[VAL_1]]}, 1 : i32) : !aie.objectfifo<memref<16xi32>>
// CHECK:    aie.objectfifo.link [@air_channel_2, @air_channel_3] -> [@air_channel_0]()
// CHECK:    %[[VAL_4:.*]] = aie.core(%[[VAL_3]]) {
// CHECK:      %[[VAL_5:.*]] = aie.objectfifo.acquire @air_channel_3(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
// CHECK:      %[[VAL_6:.*]] = aie.objectfifo.subview.access %[[VAL_5]][0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:      aie.objectfifo.release @air_channel_3(Produce, 1)
// CHECK:      aie.end
// CHECK:    } {elf_file = "segment_0_core_5_4.elf"}
// CHECK:    %[[VAL_5:.*]] = aie.core(%[[VAL_2]]) {
// CHECK:      %[[VAL_6:.*]] = aie.objectfifo.acquire @air_channel_2(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
// CHECK:      %[[VAL_7:.*]] = aie.objectfifo.subview.access %[[VAL_6]][0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
// CHECK:      aie.objectfifo.release @air_channel_2(Produce, 1)
// CHECK:      aie.end
// CHECK:    } {elf_file = "segment_0_core_5_3.elf"}
// CHECK:  }

module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 2]
  func.func @L1toL2(%arg0: memref<32xi32>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index

    air.launch (%arg1, %arg2) in (%arg3=%c1, %arg4=%c2) args(%arg5=%arg0) : memref<32xi32> attributes {id = 1 : i32} {
      %async_token, %results = air.execute -> (memref<32xi32>) {
        %alloc = memref.alloc() {alignment = 32 : i64} : memref<32xi32>
        air.execute_terminator %alloc : memref<32xi32>
      }
      %0 = air.wait_all async
      %1 = air.channel.get async [%async_token, %0]  @channel_0[] (%results[] [] []) {id = 2 : i32} : (memref<32xi32>)
      
      %2 = air.segment async args(%arg6=%arg1, %arg7=%arg2) : index, index attributes {id = 3 : i32} {
        %c0_2 = arith.constant 0 : index
        %c1_2 = arith.constant 1 : index
        %c2_2 = arith.constant 2 : index
        %c16 = arith.constant 16 : index
        %c32 = arith.constant 32 : index
        
        %3 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_2, %arg11=%c2_2) attributes {id = 4 : i32} {
            %9 = air.wait_all async 
            %async_token_2, %results_2 = air.execute -> (memref<16xi32, 2>) {
              %alloc2 = memref.alloc() : memref<16xi32, 2>
              air.execute_terminator %alloc2 : memref<16xi32, 2>
            }
            %10 = air.channel.put async [%async_token_2, %9] @channel_1[%arg8, %arg9] (%results_2[] [] []) {id = 5 : i32} : (memref<16xi32, 2>)
            %async_token_3 = air.execute [%10] {
              memref.dealloc %results_2 : memref<16xi32, 2>
            }
            air.herd_terminator
        }
        %4 = air.wait_all async 
        %async_token_1, %results_1 = air.execute -> (memref<32xi32, 1>) {
          %alloc1 = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %alloc1 : memref<32xi32, 1>
        }
        %5 = air.channel.get async [%4, %async_token_1] @channel_1[%c0_2, %c0_2] (%results_1[%c0_2] [%c16] []) {id = 6 : i32} : (memref<32xi32, 1>)
        %6 = air.channel.get async [%4, %async_token_1] @channel_1[%c0_2, %c1_2] (%results_1[%c16] [%c16] []) {id = 7 : i32} : (memref<32xi32, 1>)
        %7 = air.wait_all async
        %8 = air.channel.put async [%7] @channel_0[] (%results_1[] [] []) {id = 8 : i32} : (memref<32xi32, 1>)
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
