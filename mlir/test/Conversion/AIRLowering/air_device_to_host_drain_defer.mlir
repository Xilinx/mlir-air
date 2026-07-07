//===- air_device_to_host_drain_defer.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-std -split-input-file | FileCheck %s

// A launch-scope air.channel.get that drains an on-device producer to host DDR
// lowers to a device->host (S2MM) airrt.dma_memcpy. Its data is produced on the
// device (the matching air.channel.put lives in a herd, writing a disjoint L1
// buffer), so air-dependency cannot express the implicit @channel put->get
// backpressure dependency and schedules the drain's wait early -- before the
// input DMAs that drive the compute. air-to-std must DEFER such a drain's wait
// to the launch terminator (air.launch_end) so the host issues all inputs before
// blocking on the drain (issue-early / wait-late). Regression test for that.

// CHECK-LABEL: func.func @drain_defer
// The device->host drain memcpy is issued...
// CHECK: %[[DRAIN:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @drainAlloc{{.*}} : !airrt.event
// ...and its own (early) wait_all must NOT wait on it -- the wait is deferred:
// CHECK-NEXT: airrt.wait_all : !airrt.event
// A host->device input keeps its normal in-place wait:
// CHECK: %[[IN:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @inAlloc{{.*}} : !airrt.event
// CHECK-NEXT: airrt.wait_all %[[IN]] : !airrt.event
// The drain token is gathered by the launch terminator wait_all instead:
// CHECK: airrt.wait_all {{.*}}%[[DRAIN]] {air.launch_end}

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @drainAlloc(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc(%t, MM2S, 0)
  } {sym_name = "seg0"}
  air.channel @drain [1, 1]
  air.channel @win [1, 1]
  func.func @drain_defer(%out: memref<64xi32>, %in: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%aout=%out, %ain=%in) : memref<64xi32>, memref<64xi32> {
      // device->host drain (data produced on-device): issued early, no deps.
      %d = air.channel.get async  @drain[] (%aout[] [] []) {id = 1 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      // host->device input.
      %w = air.channel.put async  @win[] (%ain[] [] []) {id = 2 : i32, metadata = @inAlloc} : (memref<64xi32>)
      %e = air.wait_all async [%d, %w] {air.launch_end}
      %s = air.segment @seg0 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          // the drain's on-device producer (disjoint L1 memref).
          %pk = air.channel.put async [%tok]  @drain[] (%a[] [] []) {id = 3 : i32} : (memref<64xi32, 2>)
          %gk = air.channel.get async [%tok]  @win[] (%a[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Deferring the WAIT is not enough: airrt-to-npu emits each drain's
// dma_start_task in program order, so a drain whose launch-scope channel.get is
// emitted AFTER the input DMAs is armed only after every (blocking) input has
// been issued. A multi-round on-chip producer then backpressures with no host
// receiver -> the inputs never complete -> deadlock. air-to-std must also HOIST
// the drain's issue (its airrt.dma_memcpy and the slice computing its
// offsets/sizes) ahead of the first input DMA -- arm the S2MM receiver early.
// Here the drain channel.get is written AFTER the input put; the lowered drain
// memcpy must still appear BEFORE the input memcpy.

// CHECK-LABEL: func.func @drain_hoist
// The drain issue is hoisted ahead of the input DMA:
// CHECK: airrt.dma_memcpy_nd({{.*}}metadata = @drainAlloc2{{.*}} : !airrt.event
// CHECK: airrt.dma_memcpy_nd({{.*}}metadata = @inAlloc2{{.*}} : !airrt.event
// ...and the drain's wait is still deferred to the launch terminator:
// CHECK: airrt.wait_all {{.*}}{air.launch_end}

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @drainAlloc2(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc2(%t, MM2S, 0)
  } {sym_name = "seg1"}
  air.channel @drain2 [1, 1]
  air.channel @win2 [1, 1]
  func.func @drain_hoist(%out: memref<64xi32>, %in: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%aout=%out, %ain=%in) : memref<64xi32>, memref<64xi32> {
      // host->device input is emitted FIRST...
      %w = air.channel.put async  @win2[] (%ain[] [] []) {id = 2 : i32, metadata = @inAlloc2} : (memref<64xi32>)
      // ...the device->host drain is emitted AFTER (must be hoisted ahead of it).
      %d = air.channel.get async  @drain2[] (%aout[] [] []) {id = 1 : i32, metadata = @drainAlloc2} : (memref<64xi32>)
      %e = air.wait_all async [%d, %w] {air.launch_end}
      %s = air.segment @seg1 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %pk = air.channel.put async [%tok]  @drain2[] (%a[] [] []) {id = 3 : i32} : (memref<64xi32, 2>)
          %gk = air.channel.get async [%tok]  @win2[] (%a[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Blast-radius guard: a launch with NO device->host drain (only host->device
// inputs) must be left untouched -- no reordering, and each input keeps its own
// in-place wait. Confirms the rewrite is a no-op for ordinary launches.

// CHECK-LABEL: func.func @no_drain
// CHECK: %[[IN0:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @inA{{.*}} : !airrt.event
// CHECK-NEXT: airrt.wait_all %[[IN0]] : !airrt.event
// CHECK: %[[IN1:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @inB{{.*}} : !airrt.event
// CHECK-NEXT: airrt.wait_all %[[IN1]] : !airrt.event

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @inA(%t, MM2S, 0)
    aie.shim_dma_allocation @inB(%t, MM2S, 1)
  } {sym_name = "seg2"}
  air.channel @winA [1, 1]
  air.channel @winB [1, 1]
  func.func @no_drain(%a: memref<64xi32>, %b: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%aa=%a, %ab=%b) : memref<64xi32>, memref<64xi32> {
      %p0 = air.channel.put async  @winA[] (%aa[] [] []) {id = 1 : i32, metadata = @inA} : (memref<64xi32>)
      %p1 = air.channel.put async  @winB[] (%ab[] [] []) {id = 2 : i32, metadata = @inB} : (memref<64xi32>)
      %e = air.wait_all async [%p0, %p1] {air.launch_end}
      %s = air.segment @seg2 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %m = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %g0 = air.channel.get async [%tok]  @winA[] (%m[] [] []) {id = 3 : i32} : (memref<64xi32, 2>)
          %g1 = air.channel.get async [%tok]  @winB[] (%m[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}
