//===- dma_to_channel_same_space.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dma-to-channel --split-input-file -verify-diagnostics | FileCheck %s

// A copy between two L2 buffers of a segment becomes a put and a get of a new
// channel, both where the copy was, both after the copy's dependencies; the
// copy's users wait for the two.

// CHECK: air.channel @channel_0
// CHECK-LABEL: func.func @l2_to_l2
// CHECK: air.launch
// CHECK-NOT: air.channel
// CHECK: air.segment
// CHECK: %[[T0:.*]], %[[SRC:.*]] = air.execute
// CHECK: %[[T1:.*]], %[[DST:.*]] = air.execute
// CHECK-NOT: air.dma_memcpy_nd
// CHECK-DAG: %[[GET:.*]] = air.channel.get async [%[[T0]]{{.*}}%[[T1]]{{.*}}] @channel_0[] (%[[DST]][%{{.*}}] [%{{.*}}] [%{{.*}}]) {{.*}}: (memref<128xbf16, 1>)
// CHECK-DAG: %[[PUT:.*]] = air.channel.put async [%[[T0]]{{.*}}%[[T1]]{{.*}}] @channel_0[] (%[[SRC]][] [] []) {{.*}}: (memref<64xbf16, 1>)
// CHECK: %[[BOTH:.*]] = air.wait_all async [%[[GET]], %[[PUT]]]
// CHECK: air.wait_all async [%[[BOTH]]]
module {
  func.func @l2_to_l2() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%x) in (%sx=%c1) {
      %1 = air.segment @seg async {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c64 = arith.constant 64 : index
        %t0, %src = air.execute -> (memref<64xbf16, 1>) {
          %a = memref.alloc() : memref<64xbf16, 1>
          air.execute_terminator %a : memref<64xbf16, 1>
        }
        %t1, %dst = air.execute -> (memref<128xbf16, 1>) {
          %a = memref.alloc() : memref<128xbf16, 1>
          air.execute_terminator %a : memref<128xbf16, 1>
        }
        %2 = air.dma_memcpy_nd async [%t0, %t1] (%dst[%c64] [%c64] [%c1_0], %src[] [] []) {id = 1 : i32} : (memref<128xbf16, 1>, memref<64xbf16, 1>)
        %3 = air.wait_all async [%2]
      }
    }
    return
  }
}

// -----

// Without async tokens, the put and the get simply replace the copy.

// CHECK: air.channel @channel_0
// CHECK-LABEL: func.func @l2_to_l2_sync
// CHECK: air.segment
// CHECK: %[[SRC:.*]] = memref.alloc() : memref<64xbf16, 1>
// CHECK: %[[DST:.*]] = memref.alloc() : memref<64xbf16, 1>
// CHECK-NOT: air.dma_memcpy_nd
// CHECK-DAG: air.channel.get  @channel_0[] (%[[DST]][] [] [])
// CHECK-DAG: air.channel.put  @channel_0[] (%[[SRC]][] [] [])
module {
  func.func @l2_to_l2_sync() {
    %c1 = arith.constant 1 : index
    air.launch (%x) in (%sx=%c1) {
      air.segment @seg {
        %src = memref.alloc() : memref<64xbf16, 1>
        %dst = memref.alloc() : memref<64xbf16, 1>
        air.dma_memcpy_nd (%dst[] [] [], %src[] [] []) {id = 1 : i32} : (memref<64xbf16, 1>, memref<64xbf16, 1>)
      }
    }
    return
  }
}

// -----

// A core's DMA cannot copy between two L1 buffers.

module {
  func.func @l1_to_l1() {
    %c1 = arith.constant 1 : index
    air.launch (%x) in (%sx=%c1) {
      air.segment @seg {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) {
          %src = memref.alloc() : memref<64xbf16, 2>
          %dst = memref.alloc() : memref<64xbf16, 2>
          // expected-error@+1 {{copies between two L1 buffers, which a core's DMA cannot do}}
          air.dma_memcpy_nd (%dst[] [] [], %src[] [] []) {id = 1 : i32} : (memref<64xbf16, 2>, memref<64xbf16, 2>)
        }
      }
    }
    return
  }
}
