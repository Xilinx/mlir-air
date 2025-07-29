//===- air_channel.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK: air.channel @channel_1 [2, 2]
// CHECK: func.func @channel
// CHECK: %[[V1:.*]] = air.channel.put async [{{.*}}] @channel_1[{{.*}}, {{.*}}]
// CHECK: %[[V2:.*]] = air.channel.get async [{{.*}}] @channel_1[{{.*}}, {{.*}}]
air.channel @channel_1 [2,2] {channel_type = "dma_stream"}
func.func @channel() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %0 = memref.alloc() : memref<64x64xbf16, 1>
  %1 = memref.alloc() : memref<32x32xbf16, 2>
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = air.wait_all async
    %3 = air.channel.put async [%2] @channel_1[%arg0, %arg1] (%0[%c0, %c0] [%c32, %c32] [%c64, %c1]) : (memref<64x64xbf16, 1>)
  }
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %4 = air.wait_all async
    %5 = air.channel.get async [%4] @channel_1[%arg0, %arg1] (%1[] [] []) : (memref<32x32xbf16, 2>)
  }
  return 
} 

// CHECK: func.func @fork
// CHECK: %[[V1:.*]] = air.channel.put async [{{.*}}] @bcast[] ({{.*}}[{{.*}},{{.*}}]
// CHECK: air.channel.get @bcast[{{.*}}, {{.*}}] ({{.*}}[] [] []) 
air.channel @bcast [2,1] {channel_type = "dma_stream"}
func.func @fork() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %0 = memref.alloc() : memref<64x64xbf16, 1>
  %1 = memref.alloc() : memref<32x32xbf16, 2>
  %2 = air.wait_all async
  %3 = air.channel.put async [%2] @bcast[] (%0[%c0, %c0] [%c32, %c32] [%c64, %c1]) : (memref<64x64xbf16, 1>)
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    air.channel.get @bcast[%arg0, %arg1] (%1[] [] []) : (memref<32x32xbf16, 2>)
  }
  return 
} 

// CHECK: func.func @distribute
// CHECK: air.channel.put @merge[{{.*}}, {{.*}}] ({{.*}}[
// CHECK: %[[V2:.*]] = air.channel.get async [{{.*}}] @merge[]
air.channel @merge[2,2] {channel_type = "dma_stream"}
func.func @distribute() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %0 = memref.alloc() : memref<64x64xbf16, 1>
  %1 = memref.alloc() : memref<32x32xbf16, 2>
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    air.channel.put @merge[%arg0, %arg1] (%0[%c0, %c0] [%c32, %c32] [%c64, %c1]) : (memref<64x64xbf16, 1>)
  }
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %4 = air.wait_all async
    %5 = air.channel.get async [%4] @merge[] (%1[] [] []) : (memref<32x32xbf16, 2>)
  }
  return 
} 

// CHECK: air.channel @packet_flow [2, 2] {channel_type = "dma_packet"}
// CHECK: func.func @packet_flow_func
// CHECK: %[[V1:.*]] = air.channel.put async [{{.*}}] @packet_flow[{{.*}}, {{.*}}]
// CHECK: %[[V2:.*]] = air.channel.get async [{{.*}}] @packet_flow[{{.*}}, {{.*}}]
air.channel @packet_flow[2,2] {channel_type = "dma_packet"}
func.func @packet_flow_func() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  %0 = memref.alloc() : memref<64x64xbf16, 1>
  %1 = memref.alloc() : memref<32x32xbf16, 2>
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = air.wait_all async
    %3 = air.channel.put async [%2] @packet_flow[%arg0, %arg1] (%0[%c0, %c0] [%c32, %c32] [%c64, %c1]) : (memref<64x64xbf16, 1>)
  }
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %4 = air.wait_all async
    %5 = air.channel.get async [%4] @packet_flow[%arg0, %arg1] (%1[] [] []) : (memref<32x32xbf16, 2>)
  }
  return 
} 

// CHECK: air.channel @cascade [3] {channel_type = "cascade"}
// CHECK: func.func @cascade_func
// CHECK: affine.if
// CHECK: air.channel.put  @cascade[%{{.*}}]
// CHECK: else
// CHECK: affine.if
// CHECK: air.channel.get  @cascade[%{{.*}}]
// CHECK: air.channel.put  @cascade[%{{.*}}]
// CHECK: else
// CHECK: air.channel.get  @cascade[%{{.*}}]
#set = affine_set<()[s0] : (s0 == 0)>
#set1 = affine_set<()[s0] : (s0 - 1 >= 0, -s0 + 2 >= 0)>
air.channel @cascade [3] {channel_type = "cascade"}
func.func @cascade_func() {
  %c4 = arith.constant 4 : index
  %c1_0 = arith.constant 1 : index
  air.herd @herd_0  tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c4) {
    %c1_i32 = arith.constant 1 : i32
    %alloc = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
    affine.if #set()[%arg9] {
      %alloc_1 = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
      air.channel.put  @cascade[%arg9] (%alloc[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
    } else {
      affine.if #set1()[%arg9] {
        %alloc_1 = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
        %c1_1 = arith.constant 1 : index
        %iv_sub1 = arith.subi %arg9, %c1_1 : index
        air.channel.get  @cascade[%iv_sub1] (%alloc_1[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
        air.channel.put  @cascade[%arg9] (%alloc[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
      } else {
        %alloc_1 = memref.alloc() : memref<1x1x2048xi32, 2 : i32>
        %c1_1 = arith.constant 1 : index
        %iv_sub1 = arith.subi %arg9, %c1_1 : index
        air.channel.get  @cascade[%iv_sub1] (%alloc_1[] [] []) : (memref<1x1x2048xi32, 2 : i32>)
      }
    }
  }
  return
}
