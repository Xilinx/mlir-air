//===- l2_slot_buffer_lock_init.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=2 col-offset=0 device=npu2" -split-input-file | FileCheck %s

// A memtile buffer of four slots, each filled by its own transfer and drained
// a half at a time: all four slots can fill before any drain, so the write
// lock starts at the eight drains, each fill acquiring its two.

// CHECK-LABEL: aie.device(npu2)
// CHECK: %[[W:.*]] = aie.lock(%{{.*}}, 1) {init = 8 : i32}
// CHECK: aie.dma_start(S2MM
// CHECK: aie.use_lock(%[[W]], AcquireGreaterEqual, %c2_i32)
// CHECK-NEXT: aie.dma_bd(%{{.*}} : memref<4x128xi32, 1> offset = 0 len = 128)
module {
  air.channel @in [1, 1]
  air.channel @to_core [1, 1]
  func.func @slots(%src: memref<512xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) args(%a=%src) : memref<512xi32> {
      air.channel.put @in[] (%a[0] [128] [1]) {id = 21 : i32} : (memref<512xi32>)
      air.channel.put @in[] (%a[128] [128] [1]) {id = 22 : i32} : (memref<512xi32>)
      air.channel.put @in[] (%a[256] [128] [1]) {id = 23 : i32} : (memref<512xi32>)
      air.channel.put @in[] (%a[384] [128] [1]) {id = 24 : i32} : (memref<512xi32>)
      air.segment @seg0 {
        %c0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %l2 = memref.alloc() : memref<4x128xi32, 1>
        air.channel.get @in[] (%l2[0, 0] [1, 128] [128, 1]) {id = 1 : i32} : (memref<4x128xi32, 1>)
        air.channel.get @in[] (%l2[1, 0] [1, 128] [128, 1]) {id = 2 : i32} : (memref<4x128xi32, 1>)
        air.channel.get @in[] (%l2[2, 0] [1, 128] [128, 1]) {id = 3 : i32} : (memref<4x128xi32, 1>)
        air.channel.get @in[] (%l2[3, 0] [1, 128] [128, 1]) {id = 4 : i32} : (memref<4x128xi32, 1>)
        air.channel.put @to_core[] (%l2[0, 0] [1, 64] [128, 1]) {id = 5 : i32} : (memref<4x128xi32, 1>)
        air.channel.put @to_core[] (%l2[0, 64] [1, 64] [128, 1]) {id = 6 : i32} : (memref<4x128xi32, 1>)
        air.channel.put @to_core[] (%l2[1, 0] [1, 64] [128, 1]) {id = 7 : i32} : (memref<4x128xi32, 1>)
        air.channel.put @to_core[] (%l2[1, 64] [1, 64] [128, 1]) {id = 8 : i32} : (memref<4x128xi32, 1>)
        air.channel.put @to_core[] (%l2[2, 0] [1, 64] [128, 1]) {id = 9 : i32} : (memref<4x128xi32, 1>)
        air.channel.put @to_core[] (%l2[2, 64] [1, 64] [128, 1]) {id = 10 : i32} : (memref<4x128xi32, 1>)
        air.channel.put @to_core[] (%l2[3, 0] [1, 64] [128, 1]) {id = 11 : i32} : (memref<4x128xi32, 1>)
        air.channel.put @to_core[] (%l2[3, 64] [1, 64] [128, 1]) {id = 12 : i32} : (memref<4x128xi32, 1>)
        air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
          %buf = memref.alloc() : memref<64xi32, 2>
          air.channel.get @to_core[] (%buf[] [] []) {id = 13 : i32} : (memref<64xi32, 2>)
          air.channel.get @to_core[] (%buf[] [] []) {id = 14 : i32} : (memref<64xi32, 2>)
          air.channel.get @to_core[] (%buf[] [] []) {id = 15 : i32} : (memref<64xi32, 2>)
          air.channel.get @to_core[] (%buf[] [] []) {id = 16 : i32} : (memref<64xi32, 2>)
          air.channel.get @to_core[] (%buf[] [] []) {id = 17 : i32} : (memref<64xi32, 2>)
          air.channel.get @to_core[] (%buf[] [] []) {id = 18 : i32} : (memref<64xi32, 2>)
          air.channel.get @to_core[] (%buf[] [] []) {id = 19 : i32} : (memref<64xi32, 2>)
          air.channel.get @to_core[] (%buf[] [] []) {id = 20 : i32} : (memref<64xi32, 2>)
          memref.dealloc %buf : memref<64xi32, 2>
        }
        memref.dealloc %l2 : memref<4x128xi32, 1>
      }
    }
    return
  }
}

// -----

// Two slots, but three reads of the first and one of the second: once both
// slots are full, two reads of slot 0 would let it refill before its third
// read. The write lock keeps its usual start.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.lock(%{{.*}}, 1) {init = 2 : i32}
module {
  air.channel @in [1, 1]
  air.channel @to_core [1, 1]
  func.func @uneven_reads(%src: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) args(%a=%src) : memref<256xi32> {
      air.channel.put @in[] (%a[0] [128] [1]) {id = 21 : i32} : (memref<256xi32>)
      air.channel.put @in[] (%a[128] [128] [1]) {id = 22 : i32} : (memref<256xi32>)
      air.segment @seg0 {
        %c1_0 = arith.constant 1 : index
        %l2 = memref.alloc() : memref<2x128xi32, 1>
        air.channel.get @in[] (%l2[0, 0] [1, 128] [128, 1]) {id = 1 : i32} : (memref<2x128xi32, 1>)
        air.channel.get @in[] (%l2[1, 0] [1, 128] [128, 1]) {id = 2 : i32} : (memref<2x128xi32, 1>)
        air.channel.put @to_core[] (%l2[0, 0] [1, 64] [128, 1]) {id = 5 : i32} : (memref<2x128xi32, 1>)
        air.channel.put @to_core[] (%l2[0, 64] [1, 64] [128, 1]) {id = 6 : i32} : (memref<2x128xi32, 1>)
        air.channel.put @to_core[] (%l2[0, 32] [1, 64] [128, 1]) {id = 7 : i32} : (memref<2x128xi32, 1>)
        air.channel.put @to_core[] (%l2[1, 0] [1, 64] [128, 1]) {id = 8 : i32} : (memref<2x128xi32, 1>)
        air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
          %buf = memref.alloc() : memref<64xi32, 2>
          air.channel.get @to_core[] (%buf[] [] []) {id = 13 : i32} : (memref<64xi32, 2>)
          air.channel.get @to_core[] (%buf[] [] []) {id = 14 : i32} : (memref<64xi32, 2>)
          air.channel.get @to_core[] (%buf[] [] []) {id = 15 : i32} : (memref<64xi32, 2>)
          air.channel.get @to_core[] (%buf[] [] []) {id = 16 : i32} : (memref<64xi32, 2>)
          memref.dealloc %buf : memref<64xi32, 2>
        }
        memref.dealloc %l2 : memref<2x128xi32, 1>
      }
    }
    return
  }
}

// -----

// Two slots, each drained on its own channel: the drains run independently,
// so one could free a credit that refills the other slot before its drain.
// The write lock keeps its usual start.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.lock(%{{.*}}, 1) {init = 1 : i32}
module {
  air.channel @in [1, 1]
  air.channel @to_core0 [1, 1]
  air.channel @to_core1 [1, 1]
  func.func @two_drain_channels(%src: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) args(%a=%src) : memref<256xi32> {
      air.channel.put @in[] (%a[0] [128] [1]) {id = 21 : i32} : (memref<256xi32>)
      air.channel.put @in[] (%a[128] [128] [1]) {id = 22 : i32} : (memref<256xi32>)
      air.segment @seg0 {
        %c1_0 = arith.constant 1 : index
        %l2 = memref.alloc() : memref<2x128xi32, 1>
        air.channel.get @in[] (%l2[0, 0] [1, 128] [128, 1]) {id = 1 : i32} : (memref<2x128xi32, 1>)
        air.channel.get @in[] (%l2[1, 0] [1, 128] [128, 1]) {id = 2 : i32} : (memref<2x128xi32, 1>)
        air.channel.put @to_core0[] (%l2[0, 0] [1, 128] [128, 1]) {id = 5 : i32} : (memref<2x128xi32, 1>)
        air.channel.put @to_core1[] (%l2[1, 0] [1, 128] [128, 1]) {id = 6 : i32} : (memref<2x128xi32, 1>)
        air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
          %buf = memref.alloc() : memref<128xi32, 2>
          %buf1 = memref.alloc() : memref<128xi32, 2>
          air.channel.get @to_core0[] (%buf[] [] []) {id = 13 : i32} : (memref<128xi32, 2>)
          air.channel.get @to_core1[] (%buf1[] [] []) {id = 14 : i32} : (memref<128xi32, 2>)
          memref.dealloc %buf : memref<128xi32, 2>
          memref.dealloc %buf1 : memref<128xi32, 2>
        }
        memref.dealloc %l2 : memref<2x128xi32, 1>
      }
    }
    return
  }
}


// -----

// Two slots filled in one loop and drained a half at a time in another, as
// air-isolate-async-dma-loop-nests leaves them: the BD chains take each
// loop's transfers in order, so the write lock starts at the four drains.

// CHECK-LABEL: aie.device(npu2)
// CHECK: %[[W:.*]] = aie.lock(%{{.*}}, 1) {init = 4 : i32}
// CHECK: aie.dma_start(S2MM
// CHECK: aie.use_lock(%[[W]], AcquireGreaterEqual, %c2_i32)
module {
  air.channel @in [1, 1]
  air.channel @to_core [1, 1]
  func.func @slots_in_loops(%src: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) args(%a=%src) : memref<256xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1_l {
        air.channel.put @in[] (%a[0] [128] [1]) {id = 21 : i32} : (memref<256xi32>)
        air.channel.put @in[] (%a[128] [128] [1]) {id = 22 : i32} : (memref<256xi32>)
      }
      air.segment @seg0 {
        %c0_0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %c4_0 = arith.constant 4 : index
        %l2 = memref.alloc() : memref<2x128xi32, 1>
        scf.for %i = %c0_0 to %c4_0 step %c1_0 {
          air.channel.get @in[] (%l2[0, 0] [1, 128] [128, 1]) {id = 1 : i32} : (memref<2x128xi32, 1>)
          air.channel.get @in[] (%l2[1, 0] [1, 128] [128, 1]) {id = 2 : i32} : (memref<2x128xi32, 1>)
        }
        scf.for %i = %c0_0 to %c4_0 step %c1_0 {
          air.channel.put @to_core[] (%l2[0, 0] [1, 64] [128, 1]) {id = 5 : i32} : (memref<2x128xi32, 1>)
          air.channel.put @to_core[] (%l2[0, 64] [1, 64] [128, 1]) {id = 6 : i32} : (memref<2x128xi32, 1>)
          air.channel.put @to_core[] (%l2[1, 0] [1, 64] [128, 1]) {id = 7 : i32} : (memref<2x128xi32, 1>)
          air.channel.put @to_core[] (%l2[1, 64] [1, 64] [128, 1]) {id = 8 : i32} : (memref<2x128xi32, 1>)
        }
        air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c16_h = arith.constant 16 : index
          %buf = memref.alloc() : memref<64xi32, 2>
          scf.for %j = %c0_h to %c16_h step %c1_h {
            air.channel.get @to_core[] (%buf[] [] []) {id = 13 : i32} : (memref<64xi32, 2>)
          }
          memref.dealloc %buf : memref<64xi32, 2>
        }
        memref.dealloc %l2 : memref<2x128xi32, 1>
      }
    }
    return
  }
}

// -----

// Two slots filled in one loop, then drained by two loops that take slot 1
// before slot 0: with both slots full, the drain of slot 1 would let slot 0
// refill before its own drain. The write lock keeps its usual start.

// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.lock(%{{.*}}, 1) {init = 1 : i32}
module {
  air.channel @in [1, 1]
  air.channel @to_core [1, 1]
  func.func @slots_out_of_order(%src: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg0) in (%arg1=%c1) args(%a=%src) : memref<256xi32> {
      air.channel.put @in[] (%a[0] [128] [1]) {id = 21 : i32} : (memref<256xi32>)
      air.channel.put @in[] (%a[128] [128] [1]) {id = 22 : i32} : (memref<256xi32>)
      air.segment @seg0 {
        %c0_0 = arith.constant 0 : index
        %c1_0 = arith.constant 1 : index
        %l2 = memref.alloc() : memref<2x128xi32, 1>
        scf.for %i = %c0_0 to %c1_0 step %c1_0 {
          air.channel.get @in[] (%l2[0, 0] [1, 128] [128, 1]) {id = 1 : i32} : (memref<2x128xi32, 1>)
          air.channel.get @in[] (%l2[1, 0] [1, 128] [128, 1]) {id = 2 : i32} : (memref<2x128xi32, 1>)
        }
        scf.for %i = %c0_0 to %c1_0 step %c1_0 {
          air.channel.put @to_core[] (%l2[1, 0] [1, 128] [128, 1]) {id = 5 : i32} : (memref<2x128xi32, 1>)
        }
        scf.for %i = %c0_0 to %c1_0 step %c1_0 {
          air.channel.put @to_core[] (%l2[0, 0] [1, 128] [128, 1]) {id = 6 : i32} : (memref<2x128xi32, 1>)
        }
        air.herd @herd_0 tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0) {
          %buf = memref.alloc() : memref<128xi32, 2>
          air.channel.get @to_core[] (%buf[] [] []) {id = 13 : i32} : (memref<128xi32, 2>)
          air.channel.get @to_core[] (%buf[] [] []) {id = 14 : i32} : (memref<128xi32, 2>)
          memref.dealloc %buf : memref<128xi32, 2>
        }
        memref.dealloc %l2 : memref<2x128xi32, 1>
      }
    }
    return
  }
}
