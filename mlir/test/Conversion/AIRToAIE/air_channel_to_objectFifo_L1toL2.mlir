//===- air_channel_to_objectfifo_L1toL2.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds='num-rows=2 num-cols=2 row-anchor=3 col-anchor=5' --air-to-aie='emit-while-loop=false row-offset=3 col-offset=5 use-objectfifo=true device=xcve2802' | air-opt --canonicalize | FileCheck %s

// CHECK-LABEL:   aie.device(xcve2802) {
// CHECK:    %[[VAL_0:.*]] = aie.tile(5, 1)
// CHECK:    %[[VAL_1:.*]] = aie.tile(6, 1)
// CHECK:    %[[VAL_2:.*]] = aie.tile(5, 3)
// CHECK:    %[[VAL_3:.*]] = aie.tile(2, 0)
// CHECK:    %buf0 = aie.buffer(%tile_5_1) {sym_name = "buf0"} : memref<32xi32, 1> 
// CHECK:    aie.objectfifo @air_channel_1(%[[VAL_0]], {%[[VAL_2]]}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
// CHECK:    aie.objectfifo @air_channel_0(%[[VAL_3]], {%[[VAL_0]]}, 1 : i32) : !aie.objectfifo<memref<32xi32>>
// CHECK:    aie.objectfifo.link [@air_channel_0] -> [@air_channel_1]()
// CHECK:    %[[VAL_4:.*]] = aie.core(%[[VAL_2]]) {
// CHECK:      %[[VAL_5:.*]] = aie.objectfifo.acquire @air_channel_1(Consume, 1) : !aie.objectfifosubview<memref<32xi32>>
// CHECK:      %[[VAL_6:.*]] = aie.objectfifo.subview.access %[[VAL_5]][0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
// CHECK:      aie.objectfifo.release @air_channel_1(Consume, 1)
// CHECK:      aie.end
// CHECK:    } {elf_file = "segment_0_core_5_3.elf"}
// CHECK:  }

module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  func.func @L2toL1(%arg0: memref<32xi32>) {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index

    air.launch (%arg1, %arg2) in (%arg3=%c1, %arg4=%c2) args(%arg5=%arg0) : memref<32xi32> attributes {id = 1 : i32} {
      %async_token, %results = air.execute -> (memref<32xi32>) {
        %alloc = memref.alloc() {alignment = 32 : i64} : memref<32xi32>
        air.execute_terminator %alloc : memref<32xi32>
      }
      %async_token_0 = air.execute [%async_token] {
        memref.copy %arg5, %results : memref<32xi32> to memref<32xi32>
      }
      %0 = air.wait_all async
      %1 = air.channel.put async [%async_token_0, %0]  @channel_0[] (%results[] [] []) {id = 2 : i32} : (memref<32xi32>)
      
      %2 = air.segment async args(%arg6=%arg1, %arg7=%arg2) : index, index attributes {id = 3 : i32} {
        %3 = air.wait_all async 
        %async_token_1, %results_1 = air.execute -> (memref<32xi32, 1>) {
          %alloc1 = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %alloc1 : memref<32xi32, 1>
        }
        %4 = air.channel.get async [%async_token_1, %3] @channel_0[] (%results_1[] [] []) {id = 4 : i32} : (memref<32xi32, 1>)
        %5 = air.wait_all async [%4] 
        %6 = air.channel.put async [%5] @channel_1[] (%results_1[] [] []) {id = 5 : i32} : (memref<32xi32, 1>)
        %c1_2 = arith.constant 1 : index
        
        %7 = air.herd @herd_0 async [%4] tile (%arg8, %arg9) in (%arg10=%c1_2, %arg11=%c1_2) attributes {id = 6 : i32} {
            %8 = air.wait_all async 
            %async_token_2, %results_2 = air.execute -> (memref<32xi32, 2>) {
              %alloc2 = memref.alloc() : memref<32xi32, 2>
              air.execute_terminator %alloc2 : memref<32xi32, 2>
            }
            %9 = air.channel.get async [%async_token_2, %8] @channel_1[] (%results_2[] [] []) {id = 7 : i32} : (memref<32xi32, 2>)
            %async_token_3 = air.execute [%9] {
              memref.dealloc %results_2 : memref<32xi32, 2>
            }
            air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
