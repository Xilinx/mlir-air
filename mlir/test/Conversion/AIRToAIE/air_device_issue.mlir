//===- air_device_issue.mlir -----------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2" | FileCheck %s

// The router core starts the put the host arms (air.device_issue, from
// air-dma-to-channel): it reads the BD's address back, adds 2 bytes per
// element of its offset, writes the address and queues the BD with a token,
// through control packets to the shim.

// CHECK-LABEL: aie.device(npu2) @seg
// CHECK: %[[SHIM:.*]] = aie.logical_tile<ShimNOCTile>(?, ?) {controller_id = #aie.packet_info<pkt_type = 0, pkt_id = 15>}
// CHECK: %[[T:.*]] = aie.tile(0, 2)
// CHECK-DAG: %[[OUT:.*]] = aie.buffer(%[[T]]) {sym_name = "__air_issue_ctrl_out_0_2"} : memref<4xi32, 2 : i32>
// CHECK-DAG: %[[IN:.*]] = aie.buffer(%[[T]]) {sym_name = "__air_issue_ctrl_in_0_2"} : memref<2xi32, 2 : i32>
// CHECK-DAG: %[[OC:.*]] = aie.lock(%[[T]], {{.*}}) {init = 0 : i32, sym_name = "__air_issue_out_cons_0_2"}
// CHECK-DAG: %[[IP:.*]] = aie.lock(%[[T]], {{.*}}) {init = 1 : i32, sym_name = "__air_issue_in_prod_0_2"}
// CHECK-DAG: %[[IC:.*]] = aie.lock(%[[T]], {{.*}}) {init = 0 : i32, sym_name = "__air_issue_in_cons_0_2"}
// CHECK-DAG: %[[ARM:.*]] = aie.lock(%[[T]], {{.*}}) {init = 0 : i32, sym_name = "__air_device_issue_lock_0_2"}
//
// Control packets: read (2 words), write address (4), queue (3), in a cycle.
// CHECK: aie.mem(%[[T]])
// CHECK: aie.dma_start(MM2S, 0, ^[[RD:.*]], ^{{.*}})
// CHECK: ^[[RD]]:
// CHECK: aie.dma_bd(%[[OUT]] : memref<4xi32, 2 : i32> offset = 0 len = 2)
// CHECK: aie.dma_bd(%[[OUT]] : memref<4xi32, 2 : i32> offset = 0 len = 4)
// CHECK: aie.dma_bd(%[[OUT]] : memref<4xi32, 2 : i32> offset = 0 len = 3)
// CHECK: aie.next_bd ^[[RD]]
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.dma_bd(%[[IN]] : memref<2xi32, 2 : i32> offset = 0 len = 2)
//
// CHECK: aie.packet_flow(1) {
// CHECK-NEXT: aie.packet_source<%[[T]], DMA : 0>
// CHECK-NEXT: aie.packet_dest<%[[SHIM]], TileControl : 0>
// CHECK: aie.packet_flow(2) {
// CHECK-NEXT: aie.packet_source<%[[SHIM]], TileControl : 0>
// CHECK-NEXT: aie.packet_dest<%[[T]], DMA : 0>
//
// Header 0x80020001: core (0, 2), packet 1. Read 0x8251D1E4: reply as
// packet 2, two words from BD 15 word 1. Write 0x0111D1E4, queue 0x8101D214
// with 0x8000000F: BD 15, token.
// CHECK: aie.core(%[[T]])
// CHECK-DAG: arith.constant -2147352575 : i32
// CHECK-DAG: arith.constant -2108567068 : i32
// CHECK-DAG: arith.constant 17945060 : i32
// CHECK-DAG: arith.constant -2130587116 : i32
// CHECK-DAG: arith.constant -2147483633 : i32
// CHECK: aie.use_lock(%[[ARM]], AcquireGreaterEqual, %{{.*}})
// CHECK: aie.use_lock(%[[OC]], Release, %{{.*}})
// CHECK: aie.use_lock(%[[IC]], AcquireGreaterEqual, %{{.*}})
// CHECK: memref.load %[[IN]]
// CHECK: memref.load %[[IN]]
// CHECK: aie.use_lock(%[[IP]], Release, %{{.*}})
// CHECK: arith.muli %{{.*}}, %{{.*}} : i64
// The task-queue word is tagged for airrt-to-npu to add the repeat count.
// CHECK: memref.store %{{.*}}, %[[OUT]][%{{.*}}] {air.device_issue = 0 : i64}
// CHECK-NOT: air.channel.issue
// CHECK: } {air.herd_local_id
//
// CHECK: air.channel.put {{.*}}air.device_issue = 0 : i64, air.device_issue_bd = 15 : i32, air.device_issue_lock = "__air_device_issue_lock_0_2"
module {
  air.channel @Wexp [1]
  func.func @moe(%arg0: memref<32x256xbf16>, %arg1: memref<256xbf16>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0) : memref<32x256xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %1 = air.channel.put async  @Wexp[%c0] (%arg6[0, 0] [1, 256] [256, 1]) {air.device_issue = 0 : i64, id = 1 : i32} : (memref<32x256xbf16>)
      %2 = air.segment @seg async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c1_0 = arith.constant 1 : index
        %3 = air.herd @router async  tile (%arg7, %arg8) in (%arg9=%c1_0, %arg10=%c1_0) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %c256 = arith.constant 256 : index
          %c0_1 = arith.constant 0 : index
          %c2 = arith.constant 2 : index
          %async_token, %results = air.execute -> (memref<4xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<4xi32, 2 : i32>
            air.execute_terminator %alloc : memref<4xi32, 2 : i32>
          }
          %async_token_2, %results_3 = air.execute [%async_token] -> (i32) {
            %7 = memref.load %results[%c2] : memref<4xi32, 2 : i32>
            air.execute_terminator %7 : i32
          }
          %async_token_4 = air.execute [%async_token_2] {
            memref.dealloc %results : memref<4xi32, 2 : i32>
          }
          %5 = arith.index_cast %results_3 : i32 to index
          %6 = arith.muli %5, %c256 : index
          %async_token_5 = air.execute {
            air.channel.issue @Wexp[%c0_1] offset(%6 : index) {issue_id = 0 : i64}
          }
        }
        %4 = air.herd @mvm async  tile (%arg7, %arg8) in (%arg9=%c1_0, %arg10=%c1_0) attributes {id = 4 : i32, x_loc = 1 : i64, y_loc = 2 : i64} {
          %c0_1 = arith.constant 0 : index
          %async_token, %results = air.execute -> (memref<256xbf16, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xbf16, 2 : i32>
            air.execute_terminator %alloc : memref<256xbf16, 2 : i32>
          }
          %5 = air.channel.get async [%async_token]  @Wexp[%c0_1] (%results[] [] []) {id = 2 : i32} : (memref<256xbf16, 2 : i32>)
          %async_token_2 = air.execute [%5] {
            memref.dealloc %results : memref<256xbf16, 2 : i32>
          }
        }
        air.wait_all [%3, %4]  {air.segment_end}
      }
    }
    return
  }
}
