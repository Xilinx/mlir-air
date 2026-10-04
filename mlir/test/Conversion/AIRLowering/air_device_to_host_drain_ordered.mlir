//===- air_device_to_host_drain_ordered.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-std | FileCheck %s

// A launch that reads back through host memory what its own drains wrote (a
// chain of jobs sharing one activation buffer) cannot defer every drain wait to
// the terminator: an input issued after a drain may read the drained rows
// before they land. When its builder marks the drains air.order_drains, they
// stay in program order and are marked air.ordered_drain. Unmarked drains (the
// second function) keep the default: every wait deferred to the terminator. Per shim channel, consecutive drains form a group, and
// group g is awaited (a result-less, blocking wait_all) right after group g + 1
// is armed. Drains of different channels are not chained to each other. The
// last group of each channel is awaited at the launch terminator.

// CHECK-LABEL: func.func @drain_ordered
// CHECK: %[[D0:.*]] = airrt.dma_memcpy_nd({{.*}}air.ordered_drain{{.*}}metadata = @drainAlloc0
// CHECK: %[[D1:.*]] = airrt.dma_memcpy_nd({{.*}}air.ordered_drain{{.*}}metadata = @drainAlloc1
// CHECK-NOT: airrt.wait_all %[[D0]]
// CHECK-NOT: airrt.wait_all %[[D1]]
// CHECK: airrt.dma_memcpy_nd({{.*}}metadata = @inAlloc
// CHECK: %[[D2:.*]] = airrt.dma_memcpy_nd({{.*}}air.ordered_drain{{.*}}metadata = @drainAlloc0
// CHECK-NEXT: airrt.wait_all %[[D0]]{{$}}
// CHECK: %[[D3:.*]] = airrt.dma_memcpy_nd({{.*}}air.ordered_drain{{.*}}metadata = @drainAlloc1
// CHECK-NEXT: airrt.wait_all %[[D1]]{{$}}
// CHECK: airrt.dma_memcpy_nd({{.*}}metadata = @inAlloc
// CHECK: airrt.wait_all {{.*}}%[[D2]], %[[D3]] {air.launch_end}

// CHECK-LABEL: func.func @drain_unmarked
// CHECK-NOT: air.ordered_drain
// CHECK: airrt.wait_all {{.*}}air.launch_end

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @drainAlloc0(%t, S2MM, 0)
    aie.shim_dma_allocation @drainAlloc1(%t, S2MM, 1)
    aie.shim_dma_allocation @inAlloc(%t, MM2S, 0)
  } {sym_name = "seg0"}
  air.channel @drain0 [1, 1]
  air.channel @drain1 [1, 1]
  air.channel @win [1, 1]
  func.func @drain_ordered(%buf: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf) : memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c48 = arith.constant 48 : index
      %c1_1 = arith.constant 1 : index
      // First job's outputs, one group per channel.
      %d0 = air.channel.get async  @drain0[] (%b[%c0] [%c16] [%c1_1]) {air.order_drains, id = 1 : i32, metadata = @drainAlloc0} : (memref<64xi32>)
      %d1 = air.channel.get async  @drain1[] (%b[%c32] [%c16] [%c1_1]) {air.order_drains, id = 2 : i32, metadata = @drainAlloc1} : (memref<64xi32>)
      %w0 = air.channel.put async  @win[] (%b[%c16] [%c16] [%c1_1]) {id = 3 : i32, metadata = @inAlloc} : (memref<64xi32>)
      // Second job's outputs; the second job reads the first job's.
      %d2 = air.channel.get async [%w0]  @drain0[] (%b[%c16] [%c16] [%c1_1]) {air.order_drains, id = 4 : i32, metadata = @drainAlloc0} : (memref<64xi32>)
      %d3 = air.channel.get async [%w0]  @drain1[] (%b[%c48] [%c16] [%c1_1]) {air.order_drains, id = 5 : i32, metadata = @drainAlloc1} : (memref<64xi32>)
      %w1 = air.channel.put async [%d2, %d3]  @win[] (%b[%c0] [%c16] [%c1_1]) {id = 6 : i32, metadata = @inAlloc} : (memref<64xi32>)
      %e = air.wait_all async [%d0, %d1, %d2, %d3, %w1] {air.launch_end}
      %s = air.segment @seg0 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<16xi32, 2>) {
            %alloc = memref.alloc() : memref<16xi32, 2>
            air.execute_terminator %alloc : memref<16xi32, 2>
          }
          %p0 = air.channel.put async [%tok]  @drain0[] (%a[] [] []) {id = 7 : i32} : (memref<16xi32, 2>)
          %p1 = air.channel.put async [%tok]  @drain1[] (%a[] [] []) {id = 8 : i32} : (memref<16xi32, 2>)
          %g0 = air.channel.get async [%p0, %p1]  @win[] (%a[] [] []) {id = 9 : i32} : (memref<16xi32, 2>)
          %p2 = air.channel.put async [%g0]  @drain0[] (%a[] [] []) {id = 10 : i32} : (memref<16xi32, 2>)
          %p3 = air.channel.put async [%g0]  @drain1[] (%a[] [] []) {id = 11 : i32} : (memref<16xi32, 2>)
          %g1 = air.channel.get async [%p2, %p3]  @win[] (%a[] [] []) {id = 12 : i32} : (memref<16xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
  func.func @drain_unmarked(%buf: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf) : memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c32 = arith.constant 32 : index
      %c48 = arith.constant 48 : index
      %c1_1 = arith.constant 1 : index
      // First job's outputs, one group per channel.
      %d0 = air.channel.get async  @drain0[] (%b[%c0] [%c16] [%c1_1]) {id = 1 : i32, metadata = @drainAlloc0} : (memref<64xi32>)
      %d1 = air.channel.get async  @drain1[] (%b[%c32] [%c16] [%c1_1]) {id = 2 : i32, metadata = @drainAlloc1} : (memref<64xi32>)
      %w0 = air.channel.put async  @win[] (%b[%c16] [%c16] [%c1_1]) {id = 3 : i32, metadata = @inAlloc} : (memref<64xi32>)
      // Second job's outputs; the second job reads the first job's.
      %d2 = air.channel.get async [%w0]  @drain0[] (%b[%c16] [%c16] [%c1_1]) {id = 4 : i32, metadata = @drainAlloc0} : (memref<64xi32>)
      %d3 = air.channel.get async [%w0]  @drain1[] (%b[%c48] [%c16] [%c1_1]) {id = 5 : i32, metadata = @drainAlloc1} : (memref<64xi32>)
      %w1 = air.channel.put async [%d2, %d3]  @win[] (%b[%c0] [%c16] [%c1_1]) {id = 6 : i32, metadata = @inAlloc} : (memref<64xi32>)
      %e = air.wait_all async [%d0, %d1, %d2, %d3, %w1] {air.launch_end}
      %s = air.segment @seg0 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<16xi32, 2>) {
            %alloc = memref.alloc() : memref<16xi32, 2>
            air.execute_terminator %alloc : memref<16xi32, 2>
          }
          %p0 = air.channel.put async [%tok]  @drain0[] (%a[] [] []) {id = 7 : i32} : (memref<16xi32, 2>)
          %p1 = air.channel.put async [%tok]  @drain1[] (%a[] [] []) {id = 8 : i32} : (memref<16xi32, 2>)
          %g0 = air.channel.get async [%p0, %p1]  @win[] (%a[] [] []) {id = 9 : i32} : (memref<16xi32, 2>)
          %p2 = air.channel.put async [%g0]  @drain0[] (%a[] [] []) {id = 10 : i32} : (memref<16xi32, 2>)
          %p3 = air.channel.put async [%g0]  @drain1[] (%a[] [] []) {id = 11 : i32} : (memref<16xi32, 2>)
          %g1 = air.channel.get async [%p2, %p3]  @win[] (%a[] [] []) {id = 12 : i32} : (memref<16xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}
