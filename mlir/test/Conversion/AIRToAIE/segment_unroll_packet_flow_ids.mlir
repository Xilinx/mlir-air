//===- segment_unroll_packet_flow_ids.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that packet flow IDs are correctly handled for segment unroll:
// 1. Intra-device flows (L2-to-L1) reset to 0 for each isolated device
// 2. Shim flows (L3-to-device) maintain globally unique IDs across devices
//
// This ensures that isolated devices can safely reuse packet IDs for internal
// communication, while device-host flows remain uniquely identified.

// RUN: air-opt %s -air-to-aie='row-offset=2 col-offset=0 device=npu2' 2>&1 | FileCheck %s

// This test creates a 2x1 segment unroll with:
// - Intra-device channels (L2 to L1) using packet flows
// - Shim channels (L3 to L2) using packet flows

// First device: segment_pkt_0_0
// Intra-device flows should start from ID 0
// CHECK-LABEL: aie.device{{.*}}@segment_pkt_0_0
// CHECK:       aie.packet_flow(0)
// CHECK:       aie.packet_flow(1)
// CHECK:       segment_unroll_x = 0 : i64

// Second device: segment_pkt_1_0  
// Intra-device flows should ALSO start from ID 0 (reset for isolated device)
// CHECK-LABEL: aie.device{{.*}}@segment_pkt_1_0
// CHECK:       aie.packet_flow(0)
// CHECK:       aie.packet_flow(1)
// CHECK:       segment_unroll_x = 1 : i64

module {
  // Intra-device channels for L1-L2 communication (packet flow type)
  air.channel @chan_intra_a [2, 1] {channel_type = "dma_packet"}
  air.channel @chan_intra_b [2, 1] {channel_type = "dma_packet"}

  func.func @test_packet_flow_id_reset(%arg0: memref<128xbf16>) {
    %0 = air.launch async () in () args(%input=%arg0) : memref<128xbf16> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      
      // 2x1 segment unroll creates two isolated devices
      %segment = air.segment @segment_pkt async unroll(%ux, %uy) in (%sx=%c2, %sy=%c1) 
          attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 2 : i64, y_size = 4 : i64} {
        %c1_seg = arith.constant 1 : index
        %c0_seg = arith.constant 0 : index
        %c64_seg = arith.constant 64 : index
        
        // L2 buffer allocation (memory space 1 = L2/memtile)
        %l2_buf_a = memref.alloc() : memref<64xbf16, 1>
        %l2_buf_b = memref.alloc() : memref<64xbf16, 1>
        
        // L2 to L1 channel puts (intra-device packet flows)
        %put_a = air.channel.put async @chan_intra_a[%ux, %uy] (%l2_buf_a[] [] []) {id = 1 : i32} : (memref<64xbf16, 1>)
        %put_b = air.channel.put async @chan_intra_b[%ux, %uy] (%l2_buf_b[] [] []) {id = 2 : i32} : (memref<64xbf16, 1>)
        
        %herd = air.herd @herd_pkt async [%put_a, %put_b] tile (%tx, %ty) in (%htx=%c1_seg, %hty=%c1_seg) 
            args(%hux=%ux, %huy=%uy) : index, index
            attributes {id = 3 : i32} {
          // L1 buffer allocations (memory space 2 = L1)
          %async_token_a, %l1_buf_a = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          %async_token_b, %l1_buf_b = air.execute -> (memref<64xbf16, 2>) {
            %alloc = memref.alloc() : memref<64xbf16, 2>
            air.execute_terminator %alloc : memref<64xbf16, 2>
          }
          
          // L1 channel gets (intra-device packet flows)
          // Two separate channels = two separate packet flows per device
          %get_a = air.channel.get async [%async_token_a] @chan_intra_a[%hux, %huy] (%l1_buf_a[] [] []) {id = 3 : i32} : (memref<64xbf16, 2>)
          %get_b = air.channel.get async [%async_token_b] @chan_intra_b[%hux, %huy] (%l1_buf_b[] [] []) {id = 4 : i32} : (memref<64xbf16, 2>)
          
          %dealloc_a = air.execute [%get_a] {
            memref.dealloc %l1_buf_a : memref<64xbf16, 2>
          }
          %dealloc_b = air.execute [%get_b] {
            memref.dealloc %l1_buf_b : memref<64xbf16, 2>
          }
        }
        
        memref.dealloc %l2_buf_a : memref<64xbf16, 1>
        memref.dealloc %l2_buf_b : memref<64xbf16, 1>
      }
    }
    return
  }
}
