// (c) Copyright 2022 Xilinx Inc.

// RUN: air-opt %s | FileCheck %s

// CHECK-LABEL: func.func @channel
// CHECK: air.channel @channel_1{count_x = 2, count_y = 2}
// CHECK: %[[V1:.*]] = air.push @channel_1({{.*}}, {{.*}}) async [{{.*}}] 
// CHECK: %[[V2:.*]] = air.pop @channel_1({{.*}}, {{.*}}) async [{{.*}}] 
func.func @channel() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  air.channel @channel_1 {count_x = 2, count_y = 2}
  %0 = memref.alloc() : memref<64x64xbf16, 1>
  %1 = memref.alloc() : memref<32x32xbf16, 2>
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = air.wait_all async
    %3 = air.push @channel_1(%arg0, %arg1) async [%2] (%0[%c0, %c0] [%c32, %c32] [%c64, %c1]) : (memref<64x64xbf16, 1>)
  }
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %4 = air.wait_all async
    %5 = air.pop @channel_1(%arg0, %arg1) async [%4] (%1[] [] []) : (memref<32x32xbf16, 2>)
  }
  return 
} 

// CHECK-LABEL: func.func @fork
// CHECK: air.fork @bcast_in @bcast_out
// CHECK: %[[V1:.*]] = air.push @bcast_in({{.*}}, {{.*}}) async [{{.*}}] 
// CHECK: %[[V2:.*]] = air.pop @bcast_out({{.*}}, {{.*}}) async [{{.*}}] 
func.func @fork() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  air.channel @bcast_in {count_x = 2, count_y = 1}
  air.channel @bcast_out {count_x = 2, count_y = 2}
  air.fork @bcast_in @bcast_out
  %0 = memref.alloc() : memref<64x64xbf16, 1>
  %1 = memref.alloc() : memref<32x32xbf16, 2>
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c1) step (%c1, %c1) {
    %2 = air.wait_all async
    %3 = air.push @bcast_in(%arg0, %arg1) async [%2] (%0[%c0, %c0] [%c32, %c32] [%c64, %c1]) : (memref<64x64xbf16, 1>)
  }
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %4 = air.wait_all async
    %5 = air.pop @bcast_out(%arg0, %arg1) async [%4] (%1[] [] []) : (memref<32x32xbf16, 2>)
  }
  return 
} 

// CHECK-LABEL: func.func @distribute
// CHECK: air.distribute @merge_in @merge_out
// CHECK: %[[V1:.*]] = air.push @merge_in({{.*}}, {{.*}}) async [{{.*}}] 
// CHECK: %[[V2:.*]] = air.pop @merge_out({{.*}}, {{.*}}) async [{{.*}}] 
func.func @distribute() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c32 = arith.constant 32 : index
  %c64 = arith.constant 64 : index
  air.channel @merge_in {count_x = 2, count_y = 2}
  air.channel @merge_out {count_x = 2, count_y = 1}
  air.distribute @merge_in @merge_out
  %0 = memref.alloc() : memref<64x64xbf16, 1>
  %1 = memref.alloc() : memref<32x32xbf16, 2>
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c2) step (%c1, %c1) {
    %2 = air.wait_all async
    %3 = air.push @merge_in(%arg0, %arg1) async [%2] (%0[%c0, %c0] [%c32, %c32] [%c64, %c1]) : (memref<64x64xbf16, 1>)
  }
  scf.parallel (%arg0, %arg1) = (%c0, %c0) to (%c2, %c1) step (%c1, %c1) {
    %4 = air.wait_all async
    %5 = air.pop @merge_out(%arg0, %arg1) async [%4] (%1[] [] []) : (memref<32x32xbf16, 2>)
  }
  return 
} 
