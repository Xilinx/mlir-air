//===- air_segment_unroll_to_multi_device.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that air.segment with unroll creates multiple aie.device ops,
// one for each unroll iteration, with correct device type and attributes.

// RUN: air-opt %s -split-input-file -air-to-aie='row-offset=2 col-offset=0 device=npu2 test-patterns=to-aie-mlir' 2>&1 | FileCheck %s --check-prefix=CHECK
// RUN: air-opt %s -split-input-file -air-to-aie='row-offset=2 col-offset=0 device=npu2' 2>&1 | FileCheck %s --check-prefix=FULL

// Test creates 2 aie.device ops for a 2x1 segment unroll:
// - segment_2x1_0_0 with segment_unroll_x = 0, contains aie.core at tile(0, 2)
// - segment_2x1_1_0 with segment_unroll_x = 1, contains aie.core at tile(0, 2)

// First unrolled device: segment_2x1_0_0 with unroll_x = 0
// CHECK-LABEL: aie.device{{.*}}@segment_2x1_0_0
// CHECK:         aie.tile(0, 2)
// CHECK:         aie.core
// CHECK:       segment_unroll_x = 0 : i64
// FULL-LABEL: aie.device{{.*}}@segment_2x1_0_0

// Second unrolled device: segment_2x1_1_0 with unroll_x = 1
// CHECK-LABEL: aie.device{{.*}}@segment_2x1_1_0
// CHECK:         aie.tile(0, 2)
// CHECK:         aie.core
// CHECK:       segment_unroll_x = 1 : i64
// FULL-LABEL: aie.device{{.*}}@segment_2x1_1_0

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

// -----

// Test that orphaned channels are correctly removed during segment unroll.
// After channel bundle specialization, each device should only contain the
// channel that matches its segment iteration index. Orphaned channels
// (puts without matching gets, or gets without matching puts) are removed.

// Test for orphaned channel removal (using full pass via FULL prefix)
// FULL-LABEL: aie.device{{.*}}@segment_orphan_0_0
// Verify channel_0 is present and channel_1 is NOT present (orphaned)
// FULL:         air.channel @channel_0
// FULL-NOT:     air.channel @channel_1
// Verify unique shim DMA allocation name with unroll indices
// FULL:         aie.shim_dma_allocation @air_channel_orphan_0_0
// FULL:       segment_unroll_x = 0 : i64

// FULL-LABEL: aie.device{{.*}}@segment_orphan_1_0
// Verify channel_1 is present and channel_0 is NOT present (orphaned)
// FULL:         air.channel @channel_1
// FULL-NOT:     air.channel @channel_0
// Verify unique shim DMA allocation name with unroll indices
// FULL:         aie.shim_dma_allocation @air_channel_orphan_1_0
// FULL:       segment_unroll_x = 1 : i64

module {
  // Channel bundle [2, 1] will be specialized to @channel_0 and @channel_1
  air.channel @channel_orphan [2, 1]

  func.func @test_orphaned_channel_removal(%arg0: memref<64xi32>) {
    %0 = air.launch async () in () args(%input=%arg0) : memref<64xi32> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c32 = arith.constant 32 : index
      
      // Two puts at L3 level: @channel_orphan[0,0] and @channel_orphan[1,0]
      // After specialization: @channel_0 and @channel_1
      // Both are cloned to both devices, but each device only has a matching get for one.
      %put0 = air.channel.put async @channel_orphan[%c0, %c0] (%input[%c0] [%c32] [%c1]) {id = 1 : i32} : (memref<64xi32>)
      %put1 = air.channel.put async @channel_orphan[%c1, %c0] (%input[%c32] [%c32] [%c1]) {id = 2 : i32} : (memref<64xi32>)
      
      // 2x1 segment unroll creates two devices
      %segment = air.segment @segment_orphan async unroll(%ux, %uy) in (%sx=%c2, %sy=%c1) 
          attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 2 : i64, y_size = 2 : i64} {
        %c1_seg = arith.constant 1 : index
        
        %herd = air.herd @herd_orphan async tile (%tx, %ty) in (%htx=%c1_seg, %hty=%c1_seg) 
            args(%hux=%ux, %huy=%uy) : index, index
            attributes {id = 3 : i32} {
          %async_token, %buf = air.execute -> (memref<32xi32, 2>) {
            %alloc = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %alloc : memref<32xi32, 2>
          }
          // Get uses segment index [hux, huy]:
          // - Device 0 (unroll_x=0): gets from @channel_0, @channel_1 is orphaned
          // - Device 1 (unroll_x=1): gets from @channel_1, @channel_0 is orphaned
          %get = air.channel.get async [%async_token] @channel_orphan[%hux, %huy] (%buf[] [] []) {id = 3 : i32} : (memref<32xi32, 2>)
          %dealloc = air.execute [%get] {
            memref.dealloc %buf : memref<32xi32, 2>
          }
        }
      }
    }
    return
  }
}
