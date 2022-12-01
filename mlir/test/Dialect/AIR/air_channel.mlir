//===- air_channel.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc. All rights reserved.
// Copyright (C) 2022, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: func.func @channel
// CHECK: air.channel @channel_1 [2, 2]
// CHECK: %[[V1:.*]] = air.channel.put async [{{.*}}] @channel_1[{{.*}}, {{.*}}]
// CHECK: %[[V2:.*]] = air.channel.get async [{{.*}}] @channel_1[{{.*}}, {{.*}}]
func.func @channel() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  air.channel @channel_1 [2,2]
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

// CHECK-LABEL: func.func @fork
// CHECK: %[[V1:.*]] = air.channel.put async [{{.*}}] @bcast[] ({{.*}}[{{.*}},{{.*}}]
// CHECK: air.channel.get @bcast[{{.*}}, {{.*}}] ({{.*}}[] [] []) 
func.func @fork() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  air.channel @bcast [2,1]
  %0 = memref.alloc() : memref<64x64xbf16, 1>
  %1 = memref.alloc() : memref<32x32xbf16, 2>
  %2 = air.wait_all async
  %3 = air.channel.put async [%2] @bcast[] (%0[%c0, %c0] [%c32, %c32] [%c64, %c1]) : (memref<64x64xbf16, 1>)
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    air.channel.get @bcast[%arg0, %arg1] (%1[] [] []) : (memref<32x32xbf16, 2>)
  }
  return 
} 

// CHECK-LABEL: func.func @distribute
// CHECK: air.channel.put @merge[{{.*}}, {{.*}}] ({{.*}}[
// CHECK: %[[V2:.*]] = air.channel.get async [{{.*}}] @merge[]
func.func @distribute() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  air.channel @merge[2,2]
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
