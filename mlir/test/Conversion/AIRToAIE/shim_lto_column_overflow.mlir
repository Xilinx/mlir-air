//===- shim_lto_column_overflow.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -pass-pipeline='builtin.module(air-to-aie{row-offset=2 col-offset=0 device=npu1})' --split-input-file | FileCheck %s
// RUN: air-opt %s -pass-pipeline='builtin.module(air-to-aie{row-offset=2 col-offset=0 device=npu1},aie.device(aie-place-tiles{merge-logical-tiles=false}))' --split-input-file | FileCheck %s --check-prefix=PLACED

// Column 0 carries herd a's flow and both of herd b's: three per direction,
// one more than a shim tile has. Its second shim tile would be the device's
// fifth, so the third flow shares another column's shim tile instead.

// CHECK-COUNT-4: aie.logical_tile<ShimNOCTile>
// CHECK-NOT: aie.logical_tile<ShimNOCTile>
// CHECK-DAG: aie.shim_dma_allocation @air_InA_1(%[[S1:.*]], MM2S, 0)
// CHECK-DAG: aie.shim_dma_allocation @air_InB_1(%[[S1]], MM2S, 1)
// CHECK-DAG: aie.shim_dma_allocation @air_OutA_1(%[[S1]], S2MM, 0)
// CHECK-DAG: aie.shim_dma_allocation @air_OutB_1(%[[S1]], S2MM, 1)

// PLACED-COUNT-4: aie.tile({{[0-3]}}, 0)
// PLACED-NOT: aie.tile({{[0-9]+}}, 0)

module {
  air.channel @InA [4, 1]
  air.channel @OutA [4, 1]
  air.channel @InB [1, 2]
  air.channel @OutB [1, 2]
  func.func @cp(%arg0: memref<4x256xi32>, %arg1: memref<2x256xi32>, %arg2: memref<4x256xi32>, %arg3: memref<2x256xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<4x256xi32>, memref<2x256xi32>, memref<4x256xi32>, memref<2x256xi32> attributes {id = 1 : i32} {
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %1 = air.channel.put async  @InA[%c0, %c0] (%arg8[0, 0] [1, 256] [256, 1]) {id = 1 : i32} : (memref<4x256xi32>)
      %2 = air.channel.put async  @InA[%c1_0, %c0] (%arg8[1, 0] [1, 256] [256, 1]) {id = 2 : i32} : (memref<4x256xi32>)
      %3 = air.channel.put async  @InA[%c2, %c0] (%arg8[2, 0] [1, 256] [256, 1]) {id = 3 : i32} : (memref<4x256xi32>)
      %4 = air.channel.put async  @InA[%c3, %c0] (%arg8[3, 0] [1, 256] [256, 1]) {id = 4 : i32} : (memref<4x256xi32>)
      %5 = air.channel.put async  @InB[%c0, %c0] (%arg9[0, 0] [1, 256] [256, 1]) {id = 5 : i32} : (memref<2x256xi32>)
      %6 = air.channel.put async  @InB[%c0, %c1_0] (%arg9[1, 0] [1, 256] [256, 1]) {id = 6 : i32} : (memref<2x256xi32>)
      %7 = air.channel.get async  @OutA[%c0, %c0] (%arg10[0, 0] [1, 256] [256, 1]) {id = 7 : i32} : (memref<4x256xi32>)
      %8 = air.channel.get async  @OutA[%c1_0, %c0] (%arg10[1, 0] [1, 256] [256, 1]) {id = 8 : i32} : (memref<4x256xi32>)
      %9 = air.channel.get async  @OutA[%c2, %c0] (%arg10[2, 0] [1, 256] [256, 1]) {id = 9 : i32} : (memref<4x256xi32>)
      %10 = air.channel.get async  @OutA[%c3, %c0] (%arg10[3, 0] [1, 256] [256, 1]) {id = 10 : i32} : (memref<4x256xi32>)
      %11 = air.channel.get async  @OutB[%c0, %c0] (%arg11[0, 0] [1, 256] [256, 1]) {id = 11 : i32} : (memref<2x256xi32>)
      %12 = air.channel.get async  @OutB[%c0, %c1_0] (%arg11[1, 0] [1, 256] [256, 1]) {id = 12 : i32} : (memref<2x256xi32>)
      %13 = air.segment @seg async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c4 = arith.constant 4 : index
        %c2_1 = arith.constant 2 : index
        %c1_2 = arith.constant 1 : index
        %14 = air.herd @a async  tile (%arg12, %arg13) in (%arg14=%c4, %arg15=%c1_2) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %async_token, %results = air.execute -> (memref<256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<256xi32, 2 : i32>
          }
          %16 = air.channel.get async [%async_token]  @InA[%arg12, %arg13] (%results[] [] []) {id = 13 : i32} : (memref<256xi32, 2 : i32>)
          %17 = air.channel.put async [%16]  @OutA[%arg12, %arg13] (%results[] [] []) {id = 14 : i32} : (memref<256xi32, 2 : i32>)
          %async_token_3 = air.execute [%17] {
            memref.dealloc %results : memref<256xi32, 2 : i32>
          }
        }
        %15 = air.herd @b async  tile (%arg12, %arg13) in (%arg14=%c1_2, %arg15=%c2_1) attributes {id = 4 : i32, x_loc = 0 : i64, y_loc = 3 : i64} {
          %async_token, %results = air.execute -> (memref<256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<256xi32, 2 : i32>
          }
          %16 = air.channel.get async [%async_token]  @InB[%arg12, %arg13] (%results[] [] []) {id = 15 : i32} : (memref<256xi32, 2 : i32>)
          %17 = air.channel.put async [%16]  @OutB[%arg12, %arg13] (%results[] [] []) {id = 16 : i32} : (memref<256xi32, 2 : i32>)
          %async_token_3 = air.execute [%17] {
            memref.dealloc %results : memref<256xi32, 2 : i32>
          }
        }
        air.wait_all [%14, %15]  {air.segment_end}
      }
    }
    return
  }
}

// -----

// As above, with herd a's outputs packet readbacks allocated first, so every
// shim tile carries one. A circuit flow may still share those tiles.

// CHECK-COUNT-4: aie.logical_tile<ShimNOCTile>
// CHECK-NOT: aie.logical_tile<ShimNOCTile>
// CHECK-DAG: aie.shim_dma_allocation @air_OutA_1(%[[P1:.*]], S2MM, 0)
// CHECK-DAG: aie.shim_dma_allocation @air_OutB_1(%[[P1]], S2MM, 1)
// CHECK-DAG: aie.shim_dma_allocation @air_InB_1(%[[P1]], MM2S, 1)

// PLACED-COUNT-4: aie.tile({{[0-3]}}, 0)
// PLACED-NOT: aie.tile({{[0-9]+}}, 0)

module {
  air.channel @InA [4, 1]
  air.channel @OutA [4, 1] {channel_type = "npu_dma_packet"}
  air.channel @InB [1, 2]
  air.channel @OutB [1, 2]
  func.func @cp(%arg0: memref<4x256xi32>, %arg1: memref<2x256xi32>, %arg2: memref<4x256xi32>, %arg3: memref<2x256xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) args(%arg8=%arg0, %arg9=%arg1, %arg10=%arg2, %arg11=%arg3) : memref<4x256xi32>, memref<2x256xi32>, memref<4x256xi32>, memref<2x256xi32> attributes {id = 1 : i32} {
      %c3 = arith.constant 3 : index
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %1 = air.channel.get async  @OutA[%c0, %c0] (%arg10[0, 0] [1, 256] [256, 1]) {id = 1 : i32} : (memref<4x256xi32>)
      %2 = air.channel.get async  @OutA[%c1_0, %c0] (%arg10[1, 0] [1, 256] [256, 1]) {id = 2 : i32} : (memref<4x256xi32>)
      %3 = air.channel.get async  @OutA[%c2, %c0] (%arg10[2, 0] [1, 256] [256, 1]) {id = 3 : i32} : (memref<4x256xi32>)
      %4 = air.channel.get async  @OutA[%c3, %c0] (%arg10[3, 0] [1, 256] [256, 1]) {id = 4 : i32} : (memref<4x256xi32>)
      %5 = air.channel.put async  @InA[%c0, %c0] (%arg8[0, 0] [1, 256] [256, 1]) {id = 5 : i32} : (memref<4x256xi32>)
      %6 = air.channel.put async  @InA[%c1_0, %c0] (%arg8[1, 0] [1, 256] [256, 1]) {id = 6 : i32} : (memref<4x256xi32>)
      %7 = air.channel.put async  @InA[%c2, %c0] (%arg8[2, 0] [1, 256] [256, 1]) {id = 7 : i32} : (memref<4x256xi32>)
      %8 = air.channel.put async  @InA[%c3, %c0] (%arg8[3, 0] [1, 256] [256, 1]) {id = 8 : i32} : (memref<4x256xi32>)
      %9 = air.channel.put async  @InB[%c0, %c0] (%arg9[0, 0] [1, 256] [256, 1]) {id = 9 : i32} : (memref<2x256xi32>)
      %10 = air.channel.put async  @InB[%c0, %c1_0] (%arg9[1, 0] [1, 256] [256, 1]) {id = 10 : i32} : (memref<2x256xi32>)
      %11 = air.channel.get async  @OutB[%c0, %c0] (%arg11[0, 0] [1, 256] [256, 1]) {id = 11 : i32} : (memref<2x256xi32>)
      %12 = air.channel.get async  @OutB[%c0, %c1_0] (%arg11[1, 0] [1, 256] [256, 1]) {id = 12 : i32} : (memref<2x256xi32>)
      %13 = air.segment @seg async  attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 4 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c4 = arith.constant 4 : index
        %c2_1 = arith.constant 2 : index
        %c1_2 = arith.constant 1 : index
        %14 = air.herd @a async  tile (%arg12, %arg13) in (%arg14=%c4, %arg15=%c1_2) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 2 : i64} {
          %async_token, %results = air.execute -> (memref<256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<256xi32, 2 : i32>
          }
          %16 = air.channel.get async [%async_token]  @InA[%arg12, %arg13] (%results[] [] []) {id = 13 : i32} : (memref<256xi32, 2 : i32>)
          %17 = air.channel.put async [%16]  @OutA[%arg12, %arg13] (%results[] [] []) {id = 14 : i32} : (memref<256xi32, 2 : i32>)
          %async_token_3 = air.execute [%17] {
            memref.dealloc %results : memref<256xi32, 2 : i32>
          }
        }
        %15 = air.herd @b async  tile (%arg12, %arg13) in (%arg14=%c1_2, %arg15=%c2_1) attributes {id = 4 : i32, x_loc = 0 : i64, y_loc = 3 : i64} {
          %async_token, %results = air.execute -> (memref<256xi32, 2 : i32>) {
            %alloc = memref.alloc() : memref<256xi32, 2 : i32>
            air.execute_terminator %alloc : memref<256xi32, 2 : i32>
          }
          %16 = air.channel.get async [%async_token]  @InB[%arg12, %arg13] (%results[] [] []) {id = 15 : i32} : (memref<256xi32, 2 : i32>)
          %17 = air.channel.put async [%16]  @OutB[%arg12, %arg13] (%results[] [] []) {id = 16 : i32} : (memref<256xi32, 2 : i32>)
          %async_token_3 = air.execute [%17] {
            memref.dealloc %results : memref<256xi32, 2 : i32>
          }
        }
        air.wait_all [%14, %15]  {air.segment_end}
      }
    }
    return
  }
}
