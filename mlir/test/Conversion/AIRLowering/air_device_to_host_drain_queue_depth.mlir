//===- air_device_to_host_drain_queue_depth.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -air-to-std -verify-diagnostics

// air-to-std arms every device->host drain of a launch up front and retires
// them at the terminator, so one shim channel holds all of its drains in its
// 4-deep task queue at once. A fifth drain on the same channel hangs the launch;
// warn on it. Four drains on a channel, or drains spread over channels, are fine.

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @drainAlloc(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc(%t, MM2S, 0)
  } {sym_name = "seg0"}
  air.channel @drain [1, 1]
  air.channel @win [1, 1]
  func.func @five_drains(%o0: memref<64xi32>, %o1: memref<64xi32>, %o2: memref<64xi32>, %o3: memref<64xi32>, %o4: memref<64xi32>, %in: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%a0=%o0, %a1=%o1, %a2=%o2, %a3=%o3, %a4=%o4, %ain=%in) : memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32> {
      %w = air.channel.put async  @win[] (%ain[] [] []) {id = 1 : i32, metadata = @inAlloc} : (memref<64xi32>)
      %d0 = air.channel.get async  @drain[] (%a0[] [] []) {id = 2 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d1 = air.channel.get async  @drain[] (%a1[] [] []) {id = 3 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d2 = air.channel.get async  @drain[] (%a2[] [] []) {id = 4 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d3 = air.channel.get async  @drain[] (%a3[] [] []) {id = 5 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      // expected-warning@+1 {{more than 4 device-to-host drains on shim channel @drainAlloc}}
      %d4 = air.channel.get async  @drain[] (%a4[] [] []) {id = 6 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %e = air.wait_all async [%w, %d0, %d1, %d2, %d3, %d4] {air.launch_end}
      %s = air.segment @seg0 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %gk = air.channel.get async [%tok]  @win[] (%a[] [] []) {id = 7 : i32} : (memref<64xi32, 2>)
          %p0 = air.channel.put async [%gk]  @drain[] (%a[] [] []) {id = 8 : i32} : (memref<64xi32, 2>)
          %p1 = air.channel.put async [%p0]  @drain[] (%a[] [] []) {id = 9 : i32} : (memref<64xi32, 2>)
          %p2 = air.channel.put async [%p1]  @drain[] (%a[] [] []) {id = 10 : i32} : (memref<64xi32, 2>)
          %p3 = air.channel.put async [%p2]  @drain[] (%a[] [] []) {id = 11 : i32} : (memref<64xi32, 2>)
          %p4 = air.channel.put async [%p3]  @drain[] (%a[] [] []) {id = 12 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Exactly the queue depth: no warning.

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @drainAlloc(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc(%t, MM2S, 0)
  } {sym_name = "seg0"}
  air.channel @drain [1, 1]
  air.channel @win [1, 1]
  func.func @four_drains(%o0: memref<64xi32>, %o1: memref<64xi32>, %o2: memref<64xi32>, %o3: memref<64xi32>, %in: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%a0=%o0, %a1=%o1, %a2=%o2, %a3=%o3, %ain=%in) : memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32> {
      %w = air.channel.put async  @win[] (%ain[] [] []) {id = 1 : i32, metadata = @inAlloc} : (memref<64xi32>)
      %d0 = air.channel.get async  @drain[] (%a0[] [] []) {id = 2 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d1 = air.channel.get async  @drain[] (%a1[] [] []) {id = 3 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d2 = air.channel.get async  @drain[] (%a2[] [] []) {id = 4 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d3 = air.channel.get async  @drain[] (%a3[] [] []) {id = 5 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %e = air.wait_all async [%w, %d0, %d1, %d2, %d3] {air.launch_end}
      %s = air.segment @seg0 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %gk = air.channel.get async [%tok]  @win[] (%a[] [] []) {id = 7 : i32} : (memref<64xi32, 2>)
          %p0 = air.channel.put async [%gk]  @drain[] (%a[] [] []) {id = 8 : i32} : (memref<64xi32, 2>)
          %p1 = air.channel.put async [%p0]  @drain[] (%a[] [] []) {id = 9 : i32} : (memref<64xi32, 2>)
          %p2 = air.channel.put async [%p1]  @drain[] (%a[] [] []) {id = 10 : i32} : (memref<64xi32, 2>)
          %p3 = air.channel.put async [%p2]  @drain[] (%a[] [] []) {id = 11 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}
