//===- air_device_to_host_drain_readback.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-std -split-input-file | FileCheck %s

// A launch that reads back through host memory what its own drains wrote: job r
// reads row r - 1, which job r - 1 drained, and depends on that drain (the edge
// air-dependency adds; the second one through a wait_all). air-to-std awaits
// each drain right before the first input that depends on it, and moves a drain
// up past the inputs ahead of it only as far as such an input.

// CHECK-LABEL: func.func @chain
// CHECK: %[[D1:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc
// CHECK: %[[D2:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc
// CHECK: airrt.dma_memcpy_nd({{.*}}[0, 0, 0, 0], [1, 1, 1, 64]{{.*}}metadata = @inAlloc
// CHECK: airrt.wait_all %[[D1]]{{$}}
// CHECK-NEXT: airrt.dma_memcpy_nd({{.*}}[0, 0, 0, 64], [1, 1, 1, 64]{{.*}}metadata = @inAlloc
// CHECK: %[[D3:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc
// CHECK: airrt.wait_all %[[D2]]{{$}}
// CHECK-NEXT: airrt.dma_memcpy_nd({{.*}}[0, 0, 0, 128], [1, 1, 1, 64]{{.*}}metadata = @inAlloc
// CHECK: airrt.wait_all {{.*}}%[[D3]] {air.launch_end}

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @outAlloc(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc(%t, MM2S, 0)
  } {sym_name = "seg0"}
  air.channel @out [1, 1]
  air.channel @in [1, 1]
  func.func @chain(%buf: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf) : memref<256xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c192 = arith.constant 192 : index
      %d1 = air.channel.get async  @out[] (%b[%c64] [%c64] [%c1_l]) {id = 1 : i32, metadata = @outAlloc} : (memref<256xi32>)
      %r0 = air.channel.put async  @in[] (%b[%c0] [%c64] [%c1_l]) {id = 2 : i32, metadata = @inAlloc} : (memref<256xi32>)
      %d2 = air.channel.get async  @out[] (%b[%c128] [%c64] [%c1_l]) {id = 3 : i32, metadata = @outAlloc} : (memref<256xi32>)
      %r1 = air.channel.put async [%d1]  @in[] (%b[%c64] [%c64] [%c1_l]) {id = 4 : i32, metadata = @inAlloc} : (memref<256xi32>)
      %d3 = air.channel.get async  @out[] (%b[%c192] [%c64] [%c1_l]) {id = 5 : i32, metadata = @outAlloc} : (memref<256xi32>)
      %w2 = air.wait_all async [%d2]
      %r2 = air.channel.put async [%w2]  @in[] (%b[%c128] [%c64] [%c1_l]) {id = 6 : i32, metadata = @inAlloc} : (memref<256xi32>)
      %e = air.wait_all async [%d1, %r0, %d2, %r1, %d3, %r2] {air.launch_end}
      %s = air.segment @seg0 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %g = air.channel.get async [%tok]  @in[] (%a[] [] []) {id = 7 : i32} : (memref<64xi32, 2>)
          %p = air.channel.put async [%g]  @out[] (%a[] [] []) {id = 8 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// The same buffer, but the input reads rows the drain does not write, so it
// does not depend on the drain: the drain is armed ahead of the input, nothing
// waits on it before the launch end. (Awaiting the drain before that input
// would make the job wait for its own output.)

// CHECK-LABEL: func.func @disjoint
// CHECK: %[[D:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc2
// CHECK-NOT: airrt.wait_all %[[D]]{{$}}
// CHECK: airrt.dma_memcpy_nd({{.*}}metadata = @inAlloc2
// CHECK: airrt.wait_all {{.*}}%[[D]]{{.*}} {air.launch_end}

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @outAlloc2(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc2(%t, MM2S, 0)
  } {sym_name = "seg1"}
  air.channel @out2 [1, 1]
  air.channel @in2 [1, 1]
  func.func @disjoint(%buf: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf) : memref<256xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %w = air.channel.put async  @in2[] (%b[%c0] [%c64] [%c1_l]) {id = 2 : i32, metadata = @inAlloc2} : (memref<256xi32>)
      %d = air.channel.get async  @out2[] (%b[%c128] [%c64] [%c1_l]) {id = 1 : i32, metadata = @outAlloc2} : (memref<256xi32>)
      %e = air.wait_all async [%d, %w] {air.launch_end}
      %s = air.segment @seg1 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h2 async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %g = air.channel.get async [%tok]  @in2[] (%a[] [] []) {id = 3 : i32} : (memref<64xi32, 2>)
          %p = air.channel.put async [%g]  @out2[] (%a[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// In place: the input reads the rows the drain later overwrites (C = A * B + C).
// The drain is armed ahead of the input, but the input reads what was there
// before, so nothing waits on the drain before the launch end. (Awaiting it
// there would wait for an output that needs that input.)

// CHECK-LABEL: func.func @in_place
// CHECK: %[[D:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc3
// CHECK-NOT: airrt.wait_all %[[D]]{{$}}
// CHECK: airrt.dma_memcpy_nd({{.*}}metadata = @inAlloc3
// CHECK: airrt.wait_all {{.*}}%[[D]]{{.*}} {air.launch_end}

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @outAlloc3(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc3(%t, MM2S, 0)
  } {sym_name = "seg2"}
  air.channel @out3 [1, 1]
  air.channel @in3 [1, 1]
  func.func @in_place(%buf: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf) : memref<256xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %r = air.channel.put async  @in3[] (%b[%c0] [%c64] [%c1_l]) {id = 2 : i32, metadata = @inAlloc3} : (memref<256xi32>)
      %d = air.channel.get async [%r]  @out3[] (%b[%c0] [%c64] [%c1_l]) {id = 1 : i32, metadata = @outAlloc3} : (memref<256xi32>)
      %e = air.wait_all async [%d, %r] {air.launch_end}
      %s = air.segment @seg2 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h3 async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %g = air.channel.get async [%tok]  @in3[] (%a[] [] []) {id = 3 : i32} : (memref<64xi32, 2>)
          %p = air.channel.put async [%g]  @out3[] (%a[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// An input that depends only on the second of two drains on one channel. An
// await takes the channel's oldest outstanding task, so the first drain is
// awaited with it.

// CHECK-LABEL: func.func @fifo
// CHECK: %[[D1:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc4
// CHECK: %[[D2:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc4
// CHECK: airrt.wait_all %[[D1]], %[[D2]]{{$}}
// CHECK-NEXT: airrt.dma_memcpy_nd({{.*}}metadata = @inAlloc4

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @outAlloc4(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc4(%t, MM2S, 0)
  } {sym_name = "seg3"}
  air.channel @out4 [1, 1]
  air.channel @in4 [1, 1]
  func.func @fifo(%buf: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf) : memref<256xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %d1 = air.channel.get async  @out4[] (%b[%c0] [%c64] [%c1_l]) {id = 1 : i32, metadata = @outAlloc4} : (memref<256xi32>)
      %d2 = air.channel.get async  @out4[] (%b[%c64] [%c64] [%c1_l]) {id = 2 : i32, metadata = @outAlloc4} : (memref<256xi32>)
      %r = air.channel.put async [%d2]  @in4[] (%b[%c64] [%c64] [%c1_l]) {id = 3 : i32, metadata = @inAlloc4} : (memref<256xi32>)
      %e = air.wait_all async [%d1, %d2, %r] {air.launch_end}
      %s = air.segment @seg3 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h4 async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %g = air.channel.get async [%tok]  @in4[] (%a[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
          %p = air.channel.put async [%g]  @out4[] (%a[] [] []) {id = 5 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}
