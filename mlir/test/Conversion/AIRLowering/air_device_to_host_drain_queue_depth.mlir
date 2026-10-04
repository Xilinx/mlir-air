//===- air_device_to_host_drain_queue_depth.mlir --------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -air-to-std -verify-diagnostics

// air-to-std starts every device->host drain of a launch up front and awaits
// them at the terminator, so one shim channel can have all of its drains
// outstanding at once. More drains than the channel's task queue depth on one
// channel may hang the launch; warn once, on the first drain past the depth.
// Drains within the depth, drains spread over channels, and drains of different
// launches are fine.

// Five drains on one channel: warn on the fifth.

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
      // expected-warning@+2 {{device-to-host drains on shim channel @drainAlloc exceed its task queue depth (4); the launch may hang}}
      // expected-note@+1 {{drain more of the output per channel.get, or split the launch}}
      %d4 = air.channel.get async  @drain[] (%a4[] [] []) {id = 6 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %e = air.wait_all async [%w, %d0, %d1, %d2, %d3, %d4] {air.launch_end}
      %s = air.segment @seg0 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %gk = air.channel.get async [%tok]  @win[] (%a[] [] []) {id = 90 : i32} : (memref<64xi32, 2>)
          %p0 = air.channel.put async [%gk]  @drain[] (%a[] [] []) {id = 100 : i32} : (memref<64xi32, 2>)
          %p1 = air.channel.put async [%p0]  @drain[] (%a[] [] []) {id = 101 : i32} : (memref<64xi32, 2>)
          %p2 = air.channel.put async [%p1]  @drain[] (%a[] [] []) {id = 102 : i32} : (memref<64xi32, 2>)
          %p3 = air.channel.put async [%p2]  @drain[] (%a[] [] []) {id = 103 : i32} : (memref<64xi32, 2>)
          %p4 = air.channel.put async [%p3]  @drain[] (%a[] [] []) {id = 104 : i32} : (memref<64xi32, 2>)
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
          %gk = air.channel.get async [%tok]  @win[] (%a[] [] []) {id = 90 : i32} : (memref<64xi32, 2>)
          %p0 = air.channel.put async [%gk]  @drain[] (%a[] [] []) {id = 100 : i32} : (memref<64xi32, 2>)
          %p1 = air.channel.put async [%p0]  @drain[] (%a[] [] []) {id = 101 : i32} : (memref<64xi32, 2>)
          %p2 = air.channel.put async [%p1]  @drain[] (%a[] [] []) {id = 102 : i32} : (memref<64xi32, 2>)
          %p3 = air.channel.put async [%p2]  @drain[] (%a[] [] []) {id = 103 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Six drains on one channel: one warning, on the fifth.

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @drainAlloc(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc(%t, MM2S, 0)
  } {sym_name = "seg0"}
  air.channel @drain [1, 1]
  air.channel @win [1, 1]
  func.func @six_drains(%o0: memref<64xi32>, %o1: memref<64xi32>, %o2: memref<64xi32>, %o3: memref<64xi32>, %o4: memref<64xi32>, %o5: memref<64xi32>, %in: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%a0=%o0, %a1=%o1, %a2=%o2, %a3=%o3, %a4=%o4, %a5=%o5, %ain=%in) : memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32> {
      %w = air.channel.put async  @win[] (%ain[] [] []) {id = 1 : i32, metadata = @inAlloc} : (memref<64xi32>)
      %d0 = air.channel.get async  @drain[] (%a0[] [] []) {id = 2 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d1 = air.channel.get async  @drain[] (%a1[] [] []) {id = 3 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d2 = air.channel.get async  @drain[] (%a2[] [] []) {id = 4 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d3 = air.channel.get async  @drain[] (%a3[] [] []) {id = 5 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      // expected-warning@+2 {{device-to-host drains on shim channel @drainAlloc exceed its task queue depth (4); the launch may hang}}
      // expected-note@+1 {{drain more of the output per channel.get, or split the launch}}
      %d4 = air.channel.get async  @drain[] (%a4[] [] []) {id = 6 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d5 = air.channel.get async  @drain[] (%a5[] [] []) {id = 7 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %e = air.wait_all async [%w, %d0, %d1, %d2, %d3, %d4, %d5] {air.launch_end}
      %s = air.segment @seg0 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %gk = air.channel.get async [%tok]  @win[] (%a[] [] []) {id = 90 : i32} : (memref<64xi32, 2>)
          %p0 = air.channel.put async [%gk]  @drain[] (%a[] [] []) {id = 100 : i32} : (memref<64xi32, 2>)
          %p1 = air.channel.put async [%p0]  @drain[] (%a[] [] []) {id = 101 : i32} : (memref<64xi32, 2>)
          %p2 = air.channel.put async [%p1]  @drain[] (%a[] [] []) {id = 102 : i32} : (memref<64xi32, 2>)
          %p3 = air.channel.put async [%p2]  @drain[] (%a[] [] []) {id = 103 : i32} : (memref<64xi32, 2>)
          %p4 = air.channel.put async [%p3]  @drain[] (%a[] [] []) {id = 104 : i32} : (memref<64xi32, 2>)
          %p5 = air.channel.put async [%p4]  @drain[] (%a[] [] []) {id = 105 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Drains split over two channels, three each: no warning.

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @drainAAlloc(%t, S2MM, 0)
    aie.shim_dma_allocation @drainBAlloc(%t, S2MM, 1)
    aie.shim_dma_allocation @inAlloc(%t, MM2S, 0)
  } {sym_name = "seg0"}
  air.channel @drainA [1, 1]
  air.channel @drainB [1, 1]
  air.channel @win [1, 1]
  func.func @two_channels(%o0: memref<64xi32>, %o1: memref<64xi32>, %o2: memref<64xi32>, %o3: memref<64xi32>, %o4: memref<64xi32>, %o5: memref<64xi32>, %in: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%a0=%o0, %a1=%o1, %a2=%o2, %a3=%o3, %a4=%o4, %a5=%o5, %ain=%in) : memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32> {
      %w = air.channel.put async  @win[] (%ain[] [] []) {id = 1 : i32, metadata = @inAlloc} : (memref<64xi32>)
      %d0 = air.channel.get async  @drainA[] (%a0[] [] []) {id = 2 : i32, metadata = @drainAAlloc} : (memref<64xi32>)
      %d1 = air.channel.get async  @drainB[] (%a1[] [] []) {id = 3 : i32, metadata = @drainBAlloc} : (memref<64xi32>)
      %d2 = air.channel.get async  @drainA[] (%a2[] [] []) {id = 4 : i32, metadata = @drainAAlloc} : (memref<64xi32>)
      %d3 = air.channel.get async  @drainB[] (%a3[] [] []) {id = 5 : i32, metadata = @drainBAlloc} : (memref<64xi32>)
      %d4 = air.channel.get async  @drainA[] (%a4[] [] []) {id = 6 : i32, metadata = @drainAAlloc} : (memref<64xi32>)
      %d5 = air.channel.get async  @drainB[] (%a5[] [] []) {id = 7 : i32, metadata = @drainBAlloc} : (memref<64xi32>)
      %e = air.wait_all async [%w, %d0, %d1, %d2, %d3, %d4, %d5] {air.launch_end}
      %s = air.segment @seg0 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %gk = air.channel.get async [%tok]  @win[] (%a[] [] []) {id = 90 : i32} : (memref<64xi32, 2>)
          %p0 = air.channel.put async [%gk]  @drainA[] (%a[] [] []) {id = 100 : i32} : (memref<64xi32, 2>)
          %p1 = air.channel.put async [%p0]  @drainB[] (%a[] [] []) {id = 101 : i32} : (memref<64xi32, 2>)
          %p2 = air.channel.put async [%p1]  @drainA[] (%a[] [] []) {id = 102 : i32} : (memref<64xi32, 2>)
          %p3 = air.channel.put async [%p2]  @drainB[] (%a[] [] []) {id = 103 : i32} : (memref<64xi32, 2>)
          %p4 = air.channel.put async [%p3]  @drainA[] (%a[] [] []) {id = 104 : i32} : (memref<64xi32, 2>)
          %p5 = air.channel.put async [%p4]  @drainB[] (%a[] [] []) {id = 105 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Two launches in one function, three drains each on the same channel: no
// warning, counting stays within a launch.

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @drainAlloc(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc(%t, MM2S, 0)
  } {sym_name = "seg0"}
  air.channel @drain [1, 1]
  air.channel @win [1, 1]
  func.func @two_launches(%o0: memref<64xi32>, %o1: memref<64xi32>, %o2: memref<64xi32>, %in: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    %la = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%a0=%o0, %a1=%o1, %a2=%o2, %ain=%in) : memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32> {
      %w = air.channel.put async  @win[] (%ain[] [] []) {id = 1 : i32, metadata = @inAlloc} : (memref<64xi32>)
      %d0 = air.channel.get async  @drain[] (%a0[] [] []) {id = 2 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d1 = air.channel.get async  @drain[] (%a1[] [] []) {id = 3 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d2 = air.channel.get async  @drain[] (%a2[] [] []) {id = 4 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %e = air.wait_all async [%w, %d0, %d1, %d2] {air.launch_end}
      %s = air.segment @seg0 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %gk = air.channel.get async [%tok]  @win[] (%a[] [] []) {id = 51 : i32} : (memref<64xi32, 2>)
          %p0 = air.channel.put async [%gk]  @drain[] (%a[] [] []) {id = 52 : i32} : (memref<64xi32, 2>)
          %p1 = air.channel.put async [%p0]  @drain[] (%a[] [] []) {id = 53 : i32} : (memref<64xi32, 2>)
          %p2 = air.channel.put async [%p1]  @drain[] (%a[] [] []) {id = 54 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    %lb = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%a0=%o0, %a1=%o1, %a2=%o2, %ain=%in) : memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32> {
      %w = air.channel.put async  @win[] (%ain[] [] []) {id = 101 : i32, metadata = @inAlloc} : (memref<64xi32>)
      %d0 = air.channel.get async  @drain[] (%a0[] [] []) {id = 102 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d1 = air.channel.get async  @drain[] (%a1[] [] []) {id = 103 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d2 = air.channel.get async  @drain[] (%a2[] [] []) {id = 104 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %e = air.wait_all async [%w, %d0, %d1, %d2] {air.launch_end}
      %s = air.segment @seg0 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %gk = air.channel.get async [%tok]  @win[] (%a[] [] []) {id = 151 : i32} : (memref<64xi32, 2>)
          %p0 = air.channel.put async [%gk]  @drain[] (%a[] [] []) {id = 152 : i32} : (memref<64xi32, 2>)
          %p1 = air.channel.put async [%p0]  @drain[] (%a[] [] []) {id = 153 : i32} : (memref<64xi32, 2>)
          %p2 = air.channel.put async [%p1]  @drain[] (%a[] [] []) {id = 154 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// The depth was measured on NPU2 only: AIE1 is skipped, no warning.

module {
  aie.device(xcvc1902) {
    %t = aie.tile(6, 0)
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
      %d4 = air.channel.get async  @drain[] (%a4[] [] []) {id = 6 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %e = air.wait_all async [%w, %d0, %d1, %d2, %d3, %d4] {air.launch_end}
      %s = air.segment @seg0 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %gk = air.channel.get async [%tok]  @win[] (%a[] [] []) {id = 90 : i32} : (memref<64xi32, 2>)
          %p0 = air.channel.put async [%gk]  @drain[] (%a[] [] []) {id = 100 : i32} : (memref<64xi32, 2>)
          %p1 = air.channel.put async [%p0]  @drain[] (%a[] [] []) {id = 101 : i32} : (memref<64xi32, 2>)
          %p2 = air.channel.put async [%p1]  @drain[] (%a[] [] []) {id = 102 : i32} : (memref<64xi32, 2>)
          %p3 = air.channel.put async [%p2]  @drain[] (%a[] [] []) {id = 103 : i32} : (memref<64xi32, 2>)
          %p4 = air.channel.put async [%p3]  @drain[] (%a[] [] []) {id = 104 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// No input DMA and no compute dispatch in the launch (the channel's other side
// is in another function), so there is nothing to move the drains ahead of and
// they stay where they are. They are still all started, and only awaited at the
// terminator, so they are in flight together: warn.

module {
  aie.device(npu1_1col) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @drainAlloc(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc(%t, MM2S, 0)
  } {sym_name = "seg0"}
  air.channel @drain [1, 1]
  func.func @only_drains(%o0: memref<64xi32>, %o1: memref<64xi32>, %o2: memref<64xi32>, %o3: memref<64xi32>, %o4: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%a0=%o0, %a1=%o1, %a2=%o2, %a3=%o3, %a4=%o4) : memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32>, memref<64xi32> {
      %d0 = air.channel.get async  @drain[] (%a0[] [] []) {id = 1 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d1 = air.channel.get async  @drain[] (%a1[] [] []) {id = 2 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d2 = air.channel.get async  @drain[] (%a2[] [] []) {id = 3 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %d3 = air.channel.get async  @drain[] (%a3[] [] []) {id = 4 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      // expected-warning@+2 {{device-to-host drains on shim channel @drainAlloc exceed its task queue depth (4); the launch may hang}}
      // expected-note@+1 {{drain more of the output per channel.get, or split the launch}}
      %d4 = air.channel.get async  @drain[] (%a4[] [] []) {id = 5 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %e = air.wait_all async [%d0, %d1, %d2, %d3, %d4] {air.launch_end}
      air.launch_terminator
    }
    return
  }
  func.func @producer(%x: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%ax=%x) : memref<64xi32> {
      %p = air.channel.put async  @drain[] (%ax[] [] []) {id = 6 : i32, metadata = @inAlloc} : (memref<64xi32>)
      air.launch_terminator
    }
    return
  }
}

// -----

// The target the depth was measured on (NPU2): five drains warn on the fifth.

module {
  aie.device(npu2) {
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
      // expected-warning@+2 {{device-to-host drains on shim channel @drainAlloc exceed its task queue depth (4); the launch may hang}}
      // expected-note@+1 {{drain more of the output per channel.get, or split the launch}}
      %d4 = air.channel.get async  @drain[] (%a4[] [] []) {id = 6 : i32, metadata = @drainAlloc} : (memref<64xi32>)
      %e = air.wait_all async [%w, %d0, %d1, %d2, %d3, %d4] {air.launch_end}
      %s = air.segment @seg0 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %gk = air.channel.get async [%tok]  @win[] (%a[] [] []) {id = 90 : i32} : (memref<64xi32, 2>)
          %p0 = air.channel.put async [%gk]  @drain[] (%a[] [] []) {id = 100 : i32} : (memref<64xi32, 2>)
          %p1 = air.channel.put async [%p0]  @drain[] (%a[] [] []) {id = 101 : i32} : (memref<64xi32, 2>)
          %p2 = air.channel.put async [%p1]  @drain[] (%a[] [] []) {id = 102 : i32} : (memref<64xi32, 2>)
          %p3 = air.channel.put async [%p2]  @drain[] (%a[] [] []) {id = 103 : i32} : (memref<64xi32, 2>)
          %p4 = air.channel.put async [%p3]  @drain[] (%a[] [] []) {id = 104 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}
