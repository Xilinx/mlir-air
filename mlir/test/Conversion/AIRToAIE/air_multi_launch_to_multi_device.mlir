//===- air_multi_launch_to_multi_device.mlir -------------------*- MLIR -*-===//
//
// Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// Test that multiple air.launch ops in a single function produce
// multiple aie.device ops, each with correct channel isolation.
// This is the pattern needed for reconfigurable designs where different
// kernels run on the same physical tiles at different times.

// RUN: air-opt %s -air-to-aie='device=npu2' | FileCheck %s

// CHECK: aie.device(npu2) @add_three
// CHECK:   %[[SHIM3:.*]] = aie.tile(0, 0)
// CHECK:   %[[TILE3:.*]] = aie.tile(0, 2)
// CHECK:   aie.lock(%[[TILE3]]
// CHECK:   aie.buffer(%[[TILE3]])
// CHECK:   aie.mem(%[[TILE3]])
// CHECK:   aie.core(%[[TILE3]])
// CHECK:     %[[C3:.*]] = arith.constant 3 : i32
// CHECK:     arith.addi %{{.*}}, %[[C3]]
// CHECK:   air.channel @channel_in_add_three
// CHECK:   air.channel @channel_out_add_three
// CHECK:   aie.flow(%[[SHIM3]], DMA : 0, %[[TILE3]], DMA : 0)
// CHECK:   aie.flow(%[[TILE3]], DMA : 0, %[[SHIM3]], DMA : 0)
// CHECK:   aie.shim_dma_allocation @air_channel_out_add_three
// CHECK:   aie.shim_dma_allocation @air_channel_in_add_three
// CHECK: }

// CHECK: aie.device(npu2) @add_two
// CHECK:   %[[SHIM2:.*]] = aie.tile(0, 0)
// CHECK:   %[[TILE2:.*]] = aie.tile(0, 2)
// CHECK:   aie.lock(%[[TILE2]]
// CHECK:   aie.buffer(%[[TILE2]])
// CHECK:   aie.mem(%[[TILE2]])
// CHECK:   aie.core(%[[TILE2]])
// CHECK:     %[[C2:.*]] = arith.constant 2 : i32
// CHECK:     arith.addi %{{.*}}, %[[C2]]
// CHECK:   air.channel @channel_in_add_two
// CHECK:   air.channel @channel_out_add_two
// CHECK:   aie.flow(%[[SHIM2]], DMA : 0, %[[TILE2]], DMA : 0)
// CHECK:   aie.flow(%[[TILE2]], DMA : 0, %[[SHIM2]], DMA : 0)
// CHECK:   aie.shim_dma_allocation @air_channel_out_add_two
// CHECK:   aie.shim_dma_allocation @air_channel_in_add_two
// CHECK: }

// CHECK: airrt.module_metadata
// CHECK:   airrt.segment_metadata{{.*}}sym_name = "add_two"
// CHECK:     airrt.herd_metadata{{.*}}sym_name = "herd_add_two"
// CHECK:   airrt.segment_metadata{{.*}}sym_name = "add_three"
// CHECK:     airrt.herd_metadata{{.*}}sym_name = "herd_add_three"

// CHECK: func.func @multi_launch_example
// CHECK:   air.launch
// CHECK:     air.segment @add_two
// CHECK:       air.herd @herd_add_two
// CHECK:   air.launch
// CHECK:     air.segment @add_three
// CHECK:       air.herd @herd_add_three

module {
  air.channel @channel_in_add_two [1, 1]
  air.channel @channel_out_add_two [1, 1]
  air.channel @channel_in_add_three [1, 1]
  air.channel @channel_out_add_three [1, 1]

  func.func @multi_launch_example(%arg0: memref<512xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    // Launch 1: @add_two design
    %launch_add_two = air.launch async () in () args(%input=%arg0) : memref<512xi32> attributes {id = 1 : i32} {
      %c0_0 = arith.constant 0 : index
      %c4_0 = arith.constant 4 : index
      %c1_0 = arith.constant 1 : index

      %put_ext = air.channel.put async @channel_in_add_two[] (%input[%c0_0] [%c4_0] [%c1_0]) : (memref<512xi32>)

      %segment = air.segment @add_two async attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index

        %herd = air.herd @herd_add_two async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %async_token_in, %l1_in = air.execute -> (memref<4xi32, 2>) {
            %alloc = memref.alloc() : memref<4xi32, 2>
            air.execute_terminator %alloc : memref<4xi32, 2>
          }
          %async_token_out, %l1_out = air.execute -> (memref<4xi32, 2>) {
            %alloc = memref.alloc() : memref<4xi32, 2>
            air.execute_terminator %alloc : memref<4xi32, 2>
          }

          %get = air.channel.get async [%async_token_in, %async_token_out] @channel_in_add_two[] (%l1_in[] [] []) : (memref<4xi32, 2>)

          %compute = air.execute [%get] {
            %c0_e = arith.constant 0 : index
            %c1_e = arith.constant 1 : index
            %c4_e = arith.constant 4 : index
            %c2_e = arith.constant 2 : i32
            scf.for %i = %c0_e to %c4_e step %c1_e {
              %val = memref.load %l1_in[%i] : memref<4xi32, 2>
              %result = arith.addi %val, %c2_e : i32
              memref.store %result, %l1_out[%i] : memref<4xi32, 2>
            }
          }

          %put = air.channel.put async [%compute] @channel_out_add_two[] (%l1_out[] [] []) : (memref<4xi32, 2>)

          %dealloc_in = air.execute [%put] {
            memref.dealloc %l1_in : memref<4xi32, 2>
          }
          %dealloc_out = air.execute [%put] {
            memref.dealloc %l1_out : memref<4xi32, 2>
          }
        }
      }

      %get_ext = air.channel.get async [%segment] @channel_out_add_two[] (%input[%c0_0] [%c4_0] [%c1_0]) : (memref<512xi32>)
    }

    // Launch 2: @add_three design (reconfiguration)
    %launch_add_three = air.launch async [%launch_add_two] () in () args(%input=%arg0) : memref<512xi32> attributes {id = 4 : i32} {
      %c4_0 = arith.constant 4 : index
      %c1_0 = arith.constant 1 : index

      %put_ext = air.channel.put async @channel_in_add_three[] (%input[%c4_0] [%c4_0] [%c1_0]) : (memref<512xi32>)

      %segment = air.segment @add_three async attributes {id = 5 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 2 : i64, y_size = 1 : i64} {
        %c1_1 = arith.constant 1 : index

        %herd = air.herd @herd_add_three async tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) attributes {id = 6 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %async_token_in, %l1_in = air.execute -> (memref<4xi32, 2>) {
            %alloc = memref.alloc() : memref<4xi32, 2>
            air.execute_terminator %alloc : memref<4xi32, 2>
          }
          %async_token_out, %l1_out = air.execute -> (memref<4xi32, 2>) {
            %alloc = memref.alloc() : memref<4xi32, 2>
            air.execute_terminator %alloc : memref<4xi32, 2>
          }

          %get = air.channel.get async [%async_token_in, %async_token_out] @channel_in_add_three[] (%l1_in[] [] []) : (memref<4xi32, 2>)

          %compute = air.execute [%get] {
            %c0_e = arith.constant 0 : index
            %c1_e = arith.constant 1 : index
            %c4_e = arith.constant 4 : index
            %c3_e = arith.constant 3 : i32
            scf.for %i = %c0_e to %c4_e step %c1_e {
              %val = memref.load %l1_in[%i] : memref<4xi32, 2>
              %result = arith.addi %val, %c3_e : i32
              memref.store %result, %l1_out[%i] : memref<4xi32, 2>
            }
          }

          %put = air.channel.put async [%compute] @channel_out_add_three[] (%l1_out[] [] []) : (memref<4xi32, 2>)

          %dealloc_in = air.execute [%put] {
            memref.dealloc %l1_in : memref<4xi32, 2>
          }
          %dealloc_out = air.execute [%put] {
            memref.dealloc %l1_out : memref<4xi32, 2>
          }
        }
      }

      %get_ext = air.channel.get async [%segment] @channel_out_add_three[] (%input[%c4_0] [%c4_0] [%c1_0]) : (memref<512xi32>)
    }

    air.wait_all [%launch_add_three]
    return
  }
}
