//===- insert_trace_packet_flow_tiles_unroll.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2 insert-trace-packet-flow=true trace-tiles=0:2" | FileCheck %s --check-prefix=FIRST
// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2 insert-trace-packet-flow=true trace-tiles=4:2" | FileCheck %s --check-prefix=SECOND
// RUN: not air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2 insert-trace-packet-flow=true trace-tiles=0:1" 2>&1 | FileCheck %s --check-prefix=MEMTILE

// A 2x1 segment unroll makes two npu2_4col devices, each with its core at
// tile(0, 2), merged later at columns 0 and 4. trace-tiles gives columns of
// the merged device, so 0:2 picks the first core and 4:2 the second.

// FIRST-LABEL: aie.device(npu2_4col) @segment_2x1_0_0
// FIRST: aie.packet_source<%{{.*}}tile_0_2, Trace : 0>
// FIRST-LABEL: aie.device(npu2_4col) @segment_2x1_1_0
// FIRST-NOT: Trace

// SECOND-LABEL: aie.device(npu2_4col) @segment_2x1_0_0
// SECOND-NOT: Trace
// SECOND-LABEL: aie.device(npu2_4col) @segment_2x1_1_0
// SECOND: aie.packet_source<%{{.*}}tile_0_2, Trace : 0>

// Only core tiles can be selected.
// MEMTILE: error: trace-tiles: 0:1 is not a core tile

module {
  // Channel with [2, 1] dimensions to match the 2x1 segment unroll factor
  air.channel @channel_2x1 [2, 1]

  func.func @segment_unroll_2x1(%arg0: memref<64xi32>) {
    %0 = air.launch async () in () args(%input=%arg0) : memref<64xi32> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      
      // Host puts to channel subchannel [0,0] and [1,0]
      %put0 = air.channel.put async @channel_2x1[%c0, %c0] (%input[%c0] [%c32] [%c1]) {id = 1 : i32} : (memref<64xi32>)
      %put1 = air.channel.put async @channel_2x1[%c1, %c0] (%input[%c32] [%c32] [%c1]) {id = 2 : i32} : (memref<64xi32>)
      
      // 2x1 unroll: x dimension has 2 iterations, y has 1
      %segment = air.segment @segment_2x1 async unroll(%ux, %uy) in (%sx=%c2, %sy=%c1) 
          attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 4 : i64} {
        %c1_seg = arith.constant 1 : index
        %c0_seg = arith.constant 0 : index
        
        // Pass segment unroll indices to herd through args
        %herd = air.herd @herd_2x1 async tile (%tx, %ty) in (%htx=%c1_seg, %hty=%c1_seg) 
            args(%hux=%ux, %huy=%uy) : index, index
            attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %async_token, %buf = air.execute -> (memref<32xi32, 2>) {
            %alloc = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %alloc : memref<32xi32, 2>
          }
          // Each unrolled segment instance gets from its own subchannel [hux, huy]
          %get = air.channel.get async [%async_token] @channel_2x1[%hux, %huy] (%buf[] [] []) {id = 3 : i32} : (memref<32xi32, 2>)
          %dealloc = air.execute [%get] {
            memref.dealloc %buf : memref<32xi32, 2>
          }
        }
      }
    }
    return
  }
}

