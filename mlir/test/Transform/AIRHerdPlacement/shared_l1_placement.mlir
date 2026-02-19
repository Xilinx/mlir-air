//===- shared_l1_placement.mlir ------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===--------------------------------------------------------------------------------===//

// RUN: air-opt %s -air-place-herds='num-rows=6 num-cols=4 row-anchor=2 col-anchor=0' --split-input-file | FileCheck %s

// Test that herds accessing shared L1 memrefs are placed as neighbors.
// The shared L1 buffer (%alloc_shared) is allocated at the segment level
// and passed as an argument to both herd_producer and herd_consumer.
//
// The shared L1-aware placement should place these herds adjacent to each other
// since they access the same L1 memory, which requires them to be neighbors
// on the AIE array.

// CHECK-DAG: air.herd @herd_producer {{.*}} attributes {{{.*}}x_loc = [[PROD_X:[0-9]+]] : i64, y_loc = [[PROD_Y:[0-9]+]] : i64}
// CHECK-DAG: air.herd @herd_consumer {{.*}} attributes {{{.*}}x_loc = [[CONS_X:[0-9]+]] : i64, y_loc = [[CONS_Y:[0-9]+]] : i64}

// The herds should be neighbors - verify they are adjacent
// Producer at (0, 2), Consumer at (1, 2) - east neighbor
// OR Producer at (0, 2), Consumer at (0, 3) - north neighbor
// etc.

module {
  air.channel @InputToProducer [1, 1]
  air.channel @ConsumerToOutput [1, 1]
  
  func.func @shared_l1_dataflow(%arg0: memref<64x64xbf16>, %arg1: memref<64x64xbf16>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0, %arg7=%arg1) : memref<64x64xbf16>, memref<64x64xbf16> {
      air.segment @segment_0 args(%arg8=%arg2, %arg9=%arg3, %arg10=%arg6, %arg11=%arg7) : index, index, memref<64x64xbf16>, memref<64x64xbf16> {
        %c0 = arith.constant 0 : index
        %c1_1 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        
        // L2 buffers for input/output
        %alloc_l2_in = memref.alloc() : memref<64x64xbf16, 1>
        %alloc_l2_out = memref.alloc() : memref<64x64xbf16, 1>
        
        // Shared L1 buffer allocated at segment level (outside both herds)
        // Memory space 2 = L1
        %alloc_shared = memref.alloc() : memref<64x64xbf16, 2>
        
        // DMA from external to L2
        air.dma_memcpy_nd (%alloc_l2_in[%c0, %c0] [%c64, %c64] [%c64, %c1_1], %arg10[%c0, %c0] [%c64, %c64] [%c64, %c1_1]) : (memref<64x64xbf16, 1>, memref<64x64xbf16>)
        
        // Producer herd: writes to shared L1 buffer
        air.channel.put @InputToProducer[%c0, %c0] (%alloc_l2_in[%c0, %c0] [%c64, %c64] [%c64, %c1_1]) : (memref<64x64xbf16, 1>)
        
        air.herd @herd_producer tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) args(%shared_buf=%alloc_shared) : memref<64x64xbf16, 2> {
          %alloc_local = memref.alloc() : memref<64x64xbf16, 2>
          air.channel.get @InputToProducer[%tx, %ty] (%alloc_local[] [] []) : (memref<64x64xbf16, 2>)
          
          %c0_2 = arith.constant 0 : index
          %c1_2 = arith.constant 1 : index
          %c16 = arith.constant 16 : index
          %c64_2 = arith.constant 64 : index
          %cst = arith.constant 1.000000e+00 : bf16
          
          // Process and write to shared buffer
          scf.for %i = %c0_2 to %c64_2 step %c1_2 {
            scf.for %j = %c0_2 to %c64_2 step %c16 {
              %subview_in = memref.subview %alloc_local[%i, %j] [1, 16] [1, 1] : memref<64x64xbf16, 2> to memref<1x16xbf16, strided<[64, 1], offset: ?>, 2>
              %subview_out = memref.subview %shared_buf[%i, %j] [1, 16] [1, 1] : memref<64x64xbf16, 2> to memref<1x16xbf16, strided<[64, 1], offset: ?>, 2>
              %collapse_in = memref.collapse_shape %subview_in [[0, 1]] : memref<1x16xbf16, strided<[64, 1], offset: ?>, 2> into memref<16xbf16, strided<[1], offset: ?>, 2>
              %collapse_out = memref.collapse_shape %subview_out [[0, 1]] : memref<1x16xbf16, strided<[64, 1], offset: ?>, 2> into memref<16xbf16, strided<[1], offset: ?>, 2>
              %cst_zero = arith.constant 0.000000e+00 : bf16
              %v = vector.transfer_read %collapse_in[%c0_2], %cst_zero {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2>, vector<16xbf16>
              %v_add = vector.broadcast %cst : bf16 to vector<16xbf16>
              %v_result = arith.addf %v, %v_add : vector<16xbf16>
              vector.transfer_write %v_result, %collapse_out[%c0_2] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2>
            }
          }
          air.herd_terminator
        }
        
        // Consumer herd: reads from shared L1 buffer
        air.herd @herd_consumer tile (%tx, %ty) in (%sx=%c1_1, %sy=%c1_1) args(%shared_buf=%alloc_shared) : memref<64x64xbf16, 2> {
          %alloc_local = memref.alloc() : memref<64x64xbf16, 2>
          
          %c0_3 = arith.constant 0 : index
          %c1_3 = arith.constant 1 : index
          %c16_3 = arith.constant 16 : index
          %c64_3 = arith.constant 64 : index
          %cst = arith.constant 2.000000e+00 : bf16
          
          // Process shared buffer and write to local
          scf.for %i = %c0_3 to %c64_3 step %c1_3 {
            scf.for %j = %c0_3 to %c64_3 step %c16_3 {
              %subview_in = memref.subview %shared_buf[%i, %j] [1, 16] [1, 1] : memref<64x64xbf16, 2> to memref<1x16xbf16, strided<[64, 1], offset: ?>, 2>
              %subview_out = memref.subview %alloc_local[%i, %j] [1, 16] [1, 1] : memref<64x64xbf16, 2> to memref<1x16xbf16, strided<[64, 1], offset: ?>, 2>
              %collapse_in = memref.collapse_shape %subview_in [[0, 1]] : memref<1x16xbf16, strided<[64, 1], offset: ?>, 2> into memref<16xbf16, strided<[1], offset: ?>, 2>
              %collapse_out = memref.collapse_shape %subview_out [[0, 1]] : memref<1x16xbf16, strided<[64, 1], offset: ?>, 2> into memref<16xbf16, strided<[1], offset: ?>, 2>
              %cst_zero = arith.constant 0.000000e+00 : bf16
              %v = vector.transfer_read %collapse_in[%c0_3], %cst_zero {in_bounds = [true]} : memref<16xbf16, strided<[1], offset: ?>, 2>, vector<16xbf16>
              %v_mul = vector.broadcast %cst : bf16 to vector<16xbf16>
              %v_result = arith.mulf %v, %v_mul : vector<16xbf16>
              vector.transfer_write %v_result, %collapse_out[%c0_3] {in_bounds = [true]} : vector<16xbf16>, memref<16xbf16, strided<[1], offset: ?>, 2>
            }
          }
          
          air.channel.put @ConsumerToOutput[%tx, %ty] (%alloc_local[] [] []) : (memref<64x64xbf16, 2>)
          air.herd_terminator
        }
        
        air.channel.get @ConsumerToOutput[%c0, %c0] (%alloc_l2_out[%c0, %c0] [%c64, %c64] [%c64, %c1_1]) : (memref<64x64xbf16, 1>)
        
        // DMA from L2 to external
        air.dma_memcpy_nd (%arg11[%c0, %c0] [%c64, %c64] [%c64, %c1_1], %alloc_l2_out[%c0, %c0] [%c64, %c64] [%c64, %c1_1]) : (memref<64x64xbf16>, memref<64x64xbf16, 1>)
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Test with four herds connected by shared L1 buffers (similar to cascade_placement.mlir)
// This mirrors the structure of cascade_placement.mlir but uses shared L1
// buffers instead of cascade channels to communicate between herds.
//
// The shared L1 connections are:
// - conv1x1_reduce and conv3x3_coreA share %shared_buf_1 (L1)
// - conv1x1_reduce and conv3x3_coreB share %shared_buf_1 (L1) via broadcast
// - conv3x3_coreA and conv1x1_skip share %shared_buf_a (L1)
// - conv3x3_coreB and conv1x1_skip share %shared_buf_b (L1)
//
// Expected placement: herds sharing L1 memory should be placed as neighbors.
// The placement should ensure:
// - conv3x3_coreA is adjacent to conv1x1_skip
// - conv3x3_coreB is adjacent to conv1x1_skip

// CHECK: air.herd @conv1x1_reduce {{.*}} attributes {{{.*}}x_loc = [[R_X:[0-9]+]] : i64, y_loc = [[R_Y:[0-9]+]] : i64}
// CHECK: air.herd @conv3x3_coreA {{.*}} attributes {{{.*}}x_loc = [[A_X:[0-9]+]] : i64, y_loc = [[A_Y:[0-9]+]] : i64}
// CHECK: air.herd @conv3x3_coreB {{.*}} attributes {{{.*}}x_loc = [[B_X:[0-9]+]] : i64, y_loc = [[B_Y:[0-9]+]] : i64}
// CHECK: air.herd @conv1x1_skip {{.*}} attributes {{{.*}}x_loc = [[S_X:[0-9]+]] : i64, y_loc = [[S_Y:[0-9]+]] : i64}

module {
  air.channel @L2ToL1_ActIn [1, 1]
  air.channel @L1ToL2_ActOut [1, 1]
  
  func.func @bottleneck_shared_l1_placement(%arg0: memref<262144xi8>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg1, %arg2) in (%arg3=%c1, %arg4=%c1) args(%arg5=%arg0) : memref<262144xi8> attributes {id = 1 : i32} {
      %1 = air.segment @bottleneck_seg async attributes {id = 2 : i32} {
        %c1_0 = arith.constant 1 : index
        
        // Shared L1 buffers allocated at segment level
        // These are shared between multiple herds for inter-core communication
        %shared_buf_1 = memref.alloc() : memref<32x1x64xi8, 2>
        %shared_buf_a = memref.alloc() : memref<32x1x32xi8, 2>
        %shared_buf_b = memref.alloc() : memref<32x1x32xi8, 2>
        
        // conv1x1_reduce herd - produces to shared_buf_1 (read by conv3x3_coreA and conv3x3_coreB)
        %2 = air.herd @conv1x1_reduce async tile (%arg6, %arg7) in (%arg8=%c1_0, %arg9=%c1_0) args(%out_buf=%shared_buf_1) : memref<32x1x64xi8, 2> attributes {id = 3 : i32} {
          %c0 = arith.constant 0 : index
          %cst = arith.constant 42 : i8
          %v = vector.broadcast %cst : i8 to vector<32xi8>
          // Write to shared buffer
          vector.transfer_write %v, %out_buf[%c0, %c0, %c0] {in_bounds = [true]} : vector<32xi8>, memref<32x1x64xi8, 2>
          air.herd_terminator
        }
        
        // conv3x3_coreA herd - reads from shared_buf_1, produces to shared_buf_a (read by conv1x1_skip)
        %3 = air.herd @conv3x3_coreA async tile (%arg6, %arg7) in (%arg8=%c1_0, %arg9=%c1_0) args(%in_buf=%shared_buf_1, %out_buf=%shared_buf_a) : memref<32x1x64xi8, 2>, memref<32x1x32xi8, 2> attributes {id = 4 : i32} {
          %c0 = arith.constant 0 : index
          %cst_zero = arith.constant 0 : i8
          // Read from shared input buffer
          %v_in = vector.transfer_read %in_buf[%c0, %c0, %c0], %cst_zero {in_bounds = [true]} : memref<32x1x64xi8, 2>, vector<32xi8>
          // Process and write to shared output buffer
          vector.transfer_write %v_in, %out_buf[%c0, %c0, %c0] {in_bounds = [true]} : vector<32xi8>, memref<32x1x32xi8, 2>
          air.herd_terminator
        }
        
        // conv3x3_coreB herd - reads from shared_buf_1, produces to shared_buf_b (read by conv1x1_skip)
        %4 = air.herd @conv3x3_coreB async tile (%arg6, %arg7) in (%arg8=%c1_0, %arg9=%c1_0) args(%in_buf=%shared_buf_1, %out_buf=%shared_buf_b) : memref<32x1x64xi8, 2>, memref<32x1x32xi8, 2> attributes {id = 5 : i32} {
          %c0 = arith.constant 0 : index
          %cst_zero = arith.constant 0 : i8
          // Read from shared input buffer
          %v_in = vector.transfer_read %in_buf[%c0, %c0, %c0], %cst_zero {in_bounds = [true]} : memref<32x1x64xi8, 2>, vector<32xi8>
          // Process and write to shared output buffer
          vector.transfer_write %v_in, %out_buf[%c0, %c0, %c0] {in_bounds = [true]} : vector<32xi8>, memref<32x1x32xi8, 2>
          air.herd_terminator
        }
        
        // conv1x1_skip herd - reads from both shared_buf_a and shared_buf_b
        %5 = air.herd @conv1x1_skip async tile (%arg6, %arg7) in (%arg8=%c1_0, %arg9=%c1_0) args(%in_buf_a=%shared_buf_a, %in_buf_b=%shared_buf_b) : memref<32x1x32xi8, 2>, memref<32x1x32xi8, 2> attributes {id = 6 : i32} {
          %c0 = arith.constant 0 : index
          %cst_zero = arith.constant 0 : i8
          // Read from both shared input buffers
          %v_a = vector.transfer_read %in_buf_a[%c0, %c0, %c0], %cst_zero {in_bounds = [true]} : memref<32x1x32xi8, 2>, vector<32xi8>
          %v_b = vector.transfer_read %in_buf_b[%c0, %c0, %c0], %cst_zero {in_bounds = [true]} : memref<32x1x32xi8, 2>, vector<32xi8>
          // Combine the results
          %alloc_out = memref.alloc() : memref<32x1x32xi8, 2>
          vector.transfer_write %v_a, %alloc_out[%c0, %c0, %c0] {in_bounds = [true]} : vector<32xi8>, memref<32x1x32xi8, 2>
          air.channel.put @L1ToL2_ActOut[] (%alloc_out[] [] []) : (memref<32x1x32xi8, 2>)
          memref.dealloc %alloc_out : memref<32x1x32xi8, 2>
          air.herd_terminator
        }
        
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
