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

// An input that depends only on the second of two drains on one channel waits
// for that drain alone. (The channel retires in order, so on the device the
// first drain is complete by then too; airrt-to-npu awaits it ahead.)

// CHECK-LABEL: func.func @fifo
// CHECK: %[[D1:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc4
// CHECK: %[[D2:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc4
// CHECK: airrt.wait_all %[[D2]]{{$}}
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

// -----

// Two drains interleave the rows of one buffer, and the input depends on both
// but reads only rows of the second. Only the drain it reads is awaited: the
// first drain's linear range spans the input, but none of its elements do.

// CHECK-LABEL: func.func @interleaved
// CHECK: %[[D0:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc5a
// CHECK: %[[D1:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc5b
// CHECK-NOT: airrt.wait_all %[[D0]]
// CHECK: airrt.wait_all %[[D1]]{{$}}
// CHECK-NEXT: airrt.dma_memcpy_nd({{.*}}metadata = @inAlloc5
// CHECK: airrt.wait_all {{.*}}%[[D0]]{{.*}} {air.launch_end}

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @outAlloc5a(%t, S2MM, 0)
    aie.shim_dma_allocation @outAlloc5b(%t, S2MM, 1)
    aie.shim_dma_allocation @inAlloc5(%t, MM2S, 0)
  } {sym_name = "seg4"}
  air.channel @out5 [2, 1]
  air.channel @in5 [1, 1]
  func.func @interleaved(%buf: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf) : memref<256xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %d0 = air.channel.get async  @out5[%c0, %c0] (%b[%c0, %c0] [%c2, %c64] [%c128, %c1_l]) {id = 1 : i32, metadata = @outAlloc5a} : (memref<256xi32>)
      %d1 = air.channel.get async  @out5[%c1_l, %c0] (%b[%c0, %c64] [%c2, %c64] [%c128, %c1_l]) {id = 2 : i32, metadata = @outAlloc5b} : (memref<256xi32>)
      %r = air.channel.put async [%d0, %d1]  @in5[] (%b[%c64] [%c64] [%c1_l]) {id = 3 : i32, metadata = @inAlloc5} : (memref<256xi32>)
      %e = air.wait_all async [%d0, %d1, %r] {air.launch_end}
      %s = air.segment @seg4 async {
        %c1_0 = arith.constant 1 : index
        %c2_0 = arith.constant 2 : index
        %h = air.herd @h5 async  tile (%x, %y) in (%sx=%c2_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<128xi32, 2>) {
            %alloc = memref.alloc() : memref<128xi32, 2>
            air.execute_terminator %alloc : memref<128xi32, 2>
          }
          %g = air.channel.get async [%tok]  @in5[] (%a[] [] []) {id = 4 : i32} : (memref<128xi32, 2>)
          %p = air.channel.put async [%g]  @out5[%x, %y] (%a[] [] []) {id = 5 : i32} : (memref<128xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// An input whose offset is known only at runtime may read what the drain wrote,
// so it waits for the drain it depends on.

// CHECK-LABEL: func.func @runtime_offset
// CHECK: %[[D:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc6
// CHECK: airrt.wait_all %[[D]]{{$}}
// CHECK-NEXT: airrt.dma_memcpy_nd({{.*}}metadata = @inAlloc6

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @outAlloc6(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc6(%t, MM2S, 0)
  } {sym_name = "seg5"}
  air.channel @out6 [1, 1]
  air.channel @in6 [1, 1]
  func.func @runtime_offset(%buf: memref<256xi32>, %off: index) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf, %o=%off) : memref<256xi32>, index {
      %c1_l = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %d = air.channel.get async  @out6[] (%b[%c64] [%c64] [%c1_l]) {id = 1 : i32, metadata = @outAlloc6} : (memref<256xi32>)
      %r = air.channel.put async [%d]  @in6[] (%b[%o] [%c64] [%c1_l]) {id = 2 : i32, metadata = @inAlloc6} : (memref<256xi32>)
      %e = air.wait_all async [%d, %r] {air.launch_end}
      %s = air.segment @seg5 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h6 async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %g = air.channel.get async [%tok]  @in6[] (%a[] [] []) {id = 3 : i32} : (memref<64xi32, 2>)
          %p = air.channel.put async [%g]  @out6[] (%a[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Two inputs on different channels read back one drain. Each waits for it:
// airrt-to-npu may issue either first.

// CHECK-LABEL: func.func @two_readers
// CHECK: %[[D:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc7
// CHECK: airrt.wait_all %[[D]]{{$}}
// CHECK-NEXT: airrt.dma_memcpy_nd({{.*}}metadata = @inAlloc7a
// CHECK: airrt.wait_all %[[D]]{{$}}
// CHECK-NEXT: airrt.dma_memcpy_nd({{.*}}metadata = @inAlloc7b

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    %t1 = aie.tile(1, 0)
    aie.shim_dma_allocation @outAlloc7(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc7a(%t, MM2S, 0)
    aie.shim_dma_allocation @inAlloc7b(%t1, MM2S, 0)
  } {sym_name = "seg6"}
  air.channel @out7 [1, 1]
  air.channel @in7 [2, 1]
  func.func @two_readers(%buf: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf) : memref<256xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %d = air.channel.get async  @out7[] (%b[%c0] [%c64] [%c1_l]) {id = 1 : i32, metadata = @outAlloc7} : (memref<256xi32>)
      %ra = air.channel.put async [%d]  @in7[%c0, %c0] (%b[%c0] [%c32] [%c1_l]) {id = 2 : i32, metadata = @inAlloc7a} : (memref<256xi32>)
      %rb = air.channel.put async [%d]  @in7[%c1_l, %c0] (%b[%c32] [%c32] [%c1_l]) {id = 3 : i32, metadata = @inAlloc7b} : (memref<256xi32>)
      %e = air.wait_all async [%d, %ra, %rb] {air.launch_end}
      %s = air.segment @seg6 async {
        %c1_0 = arith.constant 1 : index
        %c2_0 = arith.constant 2 : index
        %h = air.herd @h7 async  tile (%x, %y) in (%sx=%c2_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<32xi32, 2>) {
            %alloc = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %alloc : memref<32xi32, 2>
          }
          %g = air.channel.get async [%tok]  @in7[%x, %y] (%a[] [] []) {id = 4 : i32} : (memref<32xi32, 2>)
          %p = air.channel.put async [%g]  @out7[] (%a[] [] []) {id = 5 : i32} : (memref<32xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// A drain too finely strided to compare element by element (more contiguous
// runs than the exact comparison takes) is compared by the range it spans: the
// input reads an element between two it writes, and still waits for it.

// CHECK-LABEL: func.func @too_many_runs
// CHECK: %[[D:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc8
// CHECK: airrt.wait_all %[[D]]{{$}}
// CHECK-NEXT: airrt.dma_memcpy_nd({{.*}}metadata = @inAlloc8

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @outAlloc8(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc8(%t, MM2S, 0)
  } {sym_name = "seg7"}
  air.channel @out8 [1, 1]
  air.channel @in8 [1, 1]
  func.func @too_many_runs(%buf: memref<16384xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf) : memref<16384xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4100 = arith.constant 4100 : index
      %d = air.channel.get async  @out8[] (%b[%c0, %c0] [%c4100, %c1_l] [%c2, %c1_l]) {id = 1 : i32, metadata = @outAlloc8} : (memref<16384xi32>)
      %r = air.channel.put async [%d]  @in8[] (%b[%c1_l] [%c1_l] [%c1_l]) {id = 2 : i32, metadata = @inAlloc8} : (memref<16384xi32>)
      %e = air.wait_all async [%d, %r] {air.launch_end}
      %s = air.segment @seg7 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h8 async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %g = air.channel.get async [%tok]  @in8[] (%a[] [] []) {id = 3 : i32} : (memref<64xi32, 2>)
          %p = air.channel.put async [%g]  @out8[] (%a[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// The input reads a subview of the buffer the drain writes. The two DMAs name
// different memrefs over one buffer, so the input still waits for the drain.

// CHECK-LABEL: func.func @subview_read
// CHECK: %[[D:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc9
// CHECK: airrt.wait_all %[[D]]{{$}}
// CHECK-NEXT: airrt.dma_memcpy_nd({{.*}}metadata = @inAlloc9

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @outAlloc9(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc9(%t, MM2S, 0)
  } {sym_name = "seg9"}
  air.channel @out9 [1, 1]
  air.channel @in9 [1, 1]
  func.func @subview_read(%buf: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf) : memref<256xi32> {
      %c1_l = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      %d = air.channel.get async  @out9[] (%b[%c64] [%c64] [%c1_l]) {id = 1 : i32, metadata = @outAlloc9} : (memref<256xi32>)
      %v = memref.subview %b[64] [64] [1] : memref<256xi32> to memref<64xi32, strided<[1], offset: 64>>
      %r = air.channel.put async [%d]  @in9[] (%v[] [] []) {id = 2 : i32, metadata = @inAlloc9} : (memref<64xi32, strided<[1], offset: 64>>)
      %e = air.wait_all async [%d, %r] {air.launch_end}
      %s = air.segment @seg9 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h9 async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %g = air.channel.get async [%tok]  @in9[] (%a[] [] []) {id = 3 : i32} : (memref<64xi32, 2>)
          %p = air.channel.put async [%g]  @out9[] (%a[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// Equal strides [2, 1], disjoint in the inner dimension (offset 2 against 0),
// but the inner offset steps a whole outer row: element 2 * i + 2 of the drain
// is element 2 * (i + 1) of the read. Too many runs to compare element by
// element, and the dimensions do not nest, so the input waits.

// CHECK-LABEL: func.func @aliasing_strides
// CHECK: %[[D:.*]] = airrt.dma_memcpy_nd({{.*}}metadata = @outAlloc10
// CHECK: airrt.wait_all %[[D]]{{$}}
// CHECK-NEXT: airrt.dma_memcpy_nd({{.*}}metadata = @inAlloc10

module {
  aie.device(npu2) {
    %t = aie.tile(0, 0)
    aie.shim_dma_allocation @outAlloc10(%t, S2MM, 0)
    aie.shim_dma_allocation @inAlloc10(%t, MM2S, 0)
  } {sym_name = "seg10"}
  air.channel @out10 [1, 1]
  air.channel @in10 [1, 1]
  func.func @aliasing_strides(%buf: memref<16384xi32>) {
    %c1 = arith.constant 1 : index
    %l = air.launch async (%i, %j) in (%si=%c1, %sj=%c1) args(%b=%buf) : memref<16384xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c5000 = arith.constant 5000 : index
      %d = air.channel.get async  @out10[] (%b[%c0, %c2] [%c5000, %c1_l] [%c2, %c1_l]) {id = 1 : i32, metadata = @outAlloc10} : (memref<16384xi32>)
      %r = air.channel.put async [%d]  @in10[] (%b[%c0, %c0] [%c5000, %c1_l] [%c2, %c1_l]) {id = 2 : i32, metadata = @inAlloc10} : (memref<16384xi32>)
      %e = air.wait_all async [%d, %r] {air.launch_end}
      %s = air.segment @seg10 async {
        %c1_0 = arith.constant 1 : index
        %h = air.herd @h10 async  tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %tok, %a = air.execute -> (memref<64xi32, 2>) {
            %alloc = memref.alloc() : memref<64xi32, 2>
            air.execute_terminator %alloc : memref<64xi32, 2>
          }
          %g = air.channel.get async [%tok]  @in10[] (%a[] [] []) {id = 3 : i32} : (memref<64xi32, 2>)
          %p = air.channel.put async [%g]  @out10[] (%a[] [] []) {id = 4 : i32} : (memref<64xi32, 2>)
        }
      }
      air.launch_terminator
    }
    return
  }
}
