//===- air_channel.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2022, Xilinx Inc.
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
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
