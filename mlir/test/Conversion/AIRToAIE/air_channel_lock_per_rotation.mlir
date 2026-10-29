//===- air_channel_lock_per_rotation.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" -split-input-file | FileCheck %s

// One channel fills buffer A and then buffer B, which has another type. The
// two buffers are not slots of one rotation, so each gets its own lock pair
// with one credit, not a shared pair with two credits that either fill could
// take.

// CHECK-COUNT-4: aie.lock(%tile_2_4, {{[0-9]}}) {init = {{[01]}} : i32}
// CHECK-NOT: init = 2
// CHECK: aie.mem(%tile_2_4)
// CHECK: aie.use_lock(%[[AE:.*]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd({{.*}} : memref<16x64xbf16, 2>
// CHECK-NEXT: aie.use_lock(%[[AF:.*]], Release, %{{.*}})
// CHECK: aie.use_lock(%[[BE:.*]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd({{.*}} : memref<16x32xbf16, 2>
// CHECK-NEXT: aie.use_lock(%[[BF:.*]], Release, %{{.*}})
// CHECK: aie.core(%tile_2_4)
// CHECK: aie.use_lock(%[[AF]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: func.call @use_a
// CHECK-NEXT: aie.use_lock(%[[AE]], Release, %{{.*}})
// CHECK-NEXT: aie.use_lock(%[[BF]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: func.call @use_b
// CHECK-NEXT: aie.use_lock(%[[BE]], Release, %{{.*}})

air.channel @ab [1, 1]
func.func private @use_a(memref<16x64xbf16, 2>)
func.func private @use_b(memref<16x32xbf16, 2>)
func.func private @other()
func.func @a_then_b() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) attributes {link_with = "w.o"} {
        %c0_inner = arith.constant 0 : index
        %cmp = arith.cmpi eq, %arg9, %c0_inner : index
        %3 = scf.if %cmp -> (!air.async.token) {
          %ta, %a = air.execute -> (memref<16x64xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x64xbf16, 2>
            air.execute_terminator %alloc : memref<16x64xbf16, 2>
          }
          %tb, %b = air.execute -> (memref<16x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x32xbf16, 2>
            air.execute_terminator %alloc : memref<16x32xbf16, 2>
          }
          %4 = air.channel.put async [%ta, %tb]  @ab[] (%a[] [] []) : (memref<16x64xbf16, 2>)
          %5 = air.channel.put async [%4]  @ab[] (%b[] [] []) : (memref<16x32xbf16, 2>)
          %da = air.execute [%5] {
            memref.dealloc %a : memref<16x64xbf16, 2>
          }
          %db = air.execute [%da] {
            memref.dealloc %b : memref<16x32xbf16, 2>
          }
          scf.yield %db : !air.async.token
        } else {
          %ta, %a = air.execute -> (memref<16x64xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x64xbf16, 2>
            air.execute_terminator %alloc : memref<16x64xbf16, 2>
          }
          %tb, %b = air.execute -> (memref<16x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x32xbf16, 2>
            air.execute_terminator %alloc : memref<16x32xbf16, 2>
          }
          %4 = air.channel.get async [%ta, %tb]  @ab[] (%a[] [] []) : (memref<16x64xbf16, 2>)
          %ua = air.execute [%4] {
            func.call @use_a(%a) : (memref<16x64xbf16, 2>) -> ()
          }
          %5 = air.channel.get async [%ua]  @ab[] (%b[] [] []) : (memref<16x32xbf16, 2>)
          %ub = air.execute [%5] {
            func.call @use_b(%b) : (memref<16x32xbf16, 2>) -> ()
          }
          %o = air.execute [%ub] {
            func.call @other() : () -> ()
          }
          %da = air.execute [%o] {
            memref.dealloc %a : memref<16x64xbf16, 2>
          }
          %db = air.execute [%da] {
            memref.dealloc %b : memref<16x32xbf16, 2>
          }
          scf.yield %db : !air.async.token
        }
      }
    }
  }
  return
}

// -----

// The channel fills two buffers of A's type in turn, then B. The two A
// buffers rotate, so they share one lock pair with two credits; B keeps a
// pair of its own with one, and its descriptor takes only B's locks.

// CHECK: aie.device
// CHECK-DAG: %[[BE:.*]] = aie.lock(%tile_2_4, 3) {init = 1 : i32}
// CHECK-DAG: %[[BF:.*]] = aie.lock(%tile_2_4, 2) {init = 0 : i32}
// CHECK-DAG: %[[AE:.*]] = aie.lock(%tile_2_4, 1) {init = 2 : i32}
// CHECK-DAG: %[[AF:.*]] = aie.lock(%tile_2_4, 0) {init = 0 : i32}
// CHECK: aie.mem(%tile_2_4)
// CHECK: aie.use_lock(%[[AE]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd({{.*}} : memref<16x64xbf16, 2>
// CHECK-NEXT: aie.use_lock(%[[AF]], Release, %{{.*}})
// CHECK: aie.use_lock(%[[AE]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd({{.*}} : memref<16x64xbf16, 2>
// CHECK-NEXT: aie.use_lock(%[[AF]], Release, %{{.*}})
// CHECK: aie.use_lock(%[[BE]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd({{.*}} : memref<16x32xbf16, 2>
// CHECK-NEXT: aie.use_lock(%[[BF]], Release, %{{.*}})

air.channel @ab [1, 1]
func.func private @use_a(memref<16x64xbf16, 2>)
func.func private @use_b(memref<16x32xbf16, 2>)
func.func private @other()
func.func @two_a_then_b() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) attributes {link_with = "w.o"} {
        %c0_inner = arith.constant 0 : index
        %cmp = arith.cmpi eq, %arg9, %c0_inner : index
        %3 = scf.if %cmp -> (!air.async.token) {
          %ta, %a = air.execute -> (memref<16x64xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x64xbf16, 2>
            air.execute_terminator %alloc : memref<16x64xbf16, 2>
          }
          %tb, %b = air.execute -> (memref<16x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x32xbf16, 2>
            air.execute_terminator %alloc : memref<16x32xbf16, 2>
          }
          %ta1, %a1 = air.execute -> (memref<16x64xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x64xbf16, 2>
            air.execute_terminator %alloc : memref<16x64xbf16, 2>
          }
          %4 = air.channel.put async [%ta, %tb]  @ab[] (%a[] [] []) : (memref<16x64xbf16, 2>)
          %41 = air.channel.put async [%4, %ta1]  @ab[] (%a1[] [] []) : (memref<16x64xbf16, 2>)
          %5 = air.channel.put async [%41]  @ab[] (%b[] [] []) : (memref<16x32xbf16, 2>)
          %da = air.execute [%5] {
            memref.dealloc %a : memref<16x64xbf16, 2>
          }
          %db = air.execute [%da] {
            memref.dealloc %b : memref<16x32xbf16, 2>
          }
          scf.yield %db : !air.async.token
        } else {
          %ta, %a = air.execute -> (memref<16x64xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x64xbf16, 2>
            air.execute_terminator %alloc : memref<16x64xbf16, 2>
          }
          %tb, %b = air.execute -> (memref<16x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x32xbf16, 2>
            air.execute_terminator %alloc : memref<16x32xbf16, 2>
          }
          %4 = air.channel.get async [%ta, %tb]  @ab[] (%a[] [] []) : (memref<16x64xbf16, 2>)
          %ua = air.execute [%4] {
            func.call @use_a(%a) : (memref<16x64xbf16, 2>) -> ()
          }
          %ta1, %a1 = air.execute -> (memref<16x64xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x64xbf16, 2>
            air.execute_terminator %alloc : memref<16x64xbf16, 2>
          }
          %41 = air.channel.get async [%ua, %ta1]  @ab[] (%a1[] [] []) : (memref<16x64xbf16, 2>)
          %ua1 = air.execute [%41] {
            func.call @use_a(%a1) : (memref<16x64xbf16, 2>) -> ()
          }
          %5 = air.channel.get async [%ua1]  @ab[] (%b[] [] []) : (memref<16x32xbf16, 2>)
          %ub = air.execute [%5] {
            func.call @use_b(%b) : (memref<16x32xbf16, 2>) -> ()
          }
          %o = air.execute [%ub] {
            func.call @other() : () -> ()
          }
          %da = air.execute [%o] {
            memref.dealloc %a : memref<16x64xbf16, 2>
          }
          %db = air.execute [%da] {
            memref.dealloc %b : memref<16x32xbf16, 2>
          }
          scf.yield %db : !air.async.token
        }
      }
    }
  }
  return
}

// -----

// As above, but the core takes the two A transfers into 16x64 views of 32x64
// buffers. The views' type is not the buffers'; the buffers still rotate and
// share a pair with two credits.

// CHECK: aie.device
// CHECK-DAG: aie.lock(%tile_2_4, 3) {init = 1 : i32}
// CHECK-DAG: aie.lock(%tile_2_4, 1) {init = 2 : i32}

air.channel @ab [1, 1]
func.func private @use_a(memref<16x64xbf16, 2>)
func.func private @use_a_view(memref<16x64xbf16, strided<[64, 1]>, 2>)
func.func private @use_b(memref<16x32xbf16, 2>)
func.func private @other()
func.func @two_a_views_then_b() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) attributes {link_with = "w.o"} {
        %c0_inner = arith.constant 0 : index
        %cmp = arith.cmpi eq, %arg9, %c0_inner : index
        %3 = scf.if %cmp -> (!air.async.token) {
          %ta, %a = air.execute -> (memref<16x64xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x64xbf16, 2>
            air.execute_terminator %alloc : memref<16x64xbf16, 2>
          }
          %tb, %b = air.execute -> (memref<16x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x32xbf16, 2>
            air.execute_terminator %alloc : memref<16x32xbf16, 2>
          }
          %ta1, %a1 = air.execute -> (memref<16x64xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x64xbf16, 2>
            air.execute_terminator %alloc : memref<16x64xbf16, 2>
          }
          %4 = air.channel.put async [%ta, %tb]  @ab[] (%a[] [] []) : (memref<16x64xbf16, 2>)
          %41 = air.channel.put async [%4, %ta1]  @ab[] (%a1[] [] []) : (memref<16x64xbf16, 2>)
          %5 = air.channel.put async [%41]  @ab[] (%b[] [] []) : (memref<16x32xbf16, 2>)
          %da = air.execute [%5] {
            memref.dealloc %a : memref<16x64xbf16, 2>
          }
          %db = air.execute [%da] {
            memref.dealloc %b : memref<16x32xbf16, 2>
          }
          scf.yield %db : !air.async.token
        } else {
          %ta, %a = air.execute -> (memref<32x64xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x64xbf16, 2>
            air.execute_terminator %alloc : memref<32x64xbf16, 2>
          }
          %a_v = memref.subview %a[0, 0] [16, 64] [1, 1] : memref<32x64xbf16, 2> to memref<16x64xbf16, strided<[64, 1]>, 2>
          %tb, %b = air.execute -> (memref<16x32xbf16, 2>) {
            %alloc = memref.alloc() : memref<16x32xbf16, 2>
            air.execute_terminator %alloc : memref<16x32xbf16, 2>
          }
          %4 = air.channel.get async [%ta, %tb]  @ab[] (%a_v[] [] []) : (memref<16x64xbf16, strided<[64, 1]>, 2>)
          %ua = air.execute [%4] {
            func.call @use_a_view(%a_v) : (memref<16x64xbf16, strided<[64, 1]>, 2>) -> ()
          }
          %ta1, %a1 = air.execute -> (memref<32x64xbf16, 2>) {
            %alloc = memref.alloc() : memref<32x64xbf16, 2>
            air.execute_terminator %alloc : memref<32x64xbf16, 2>
          }
          %a1_v = memref.subview %a1[0, 0] [16, 64] [1, 1] : memref<32x64xbf16, 2> to memref<16x64xbf16, strided<[64, 1]>, 2>
          %41 = air.channel.get async [%ua, %ta1]  @ab[] (%a1_v[] [] []) : (memref<16x64xbf16, strided<[64, 1]>, 2>)
          %ua1 = air.execute [%41] {
            func.call @use_a_view(%a1_v) : (memref<16x64xbf16, strided<[64, 1]>, 2>) -> ()
          }
          %5 = air.channel.get async [%ua1]  @ab[] (%b[] [] []) : (memref<16x32xbf16, 2>)
          %ub = air.execute [%5] {
            func.call @use_b(%b) : (memref<16x32xbf16, 2>) -> ()
          }
          %o = air.execute [%ub] {
            func.call @other() : () -> ()
          }
          %da = air.execute [%o] {
            memref.dealloc %a : memref<32x64xbf16, 2>
          }
          %db = air.execute [%da] {
            memref.dealloc %b : memref<16x32xbf16, 2>
          }
          scf.yield %db : !air.async.token
        }
      }
    }
  }
  return
}

// -----

// The channel fills A twice from a fixed loop of its own, then B of the same
// type. A shared pair would start with two credits, and the DMA could fill A
// twice before the core reads the first fill. A buffer filled several times
// in a row keeps a pair of its own, and so does B.

// CHECK: aie.device
// CHECK-NOT: init = 2
// CHECK: aie.mem(%tile_2_3)
// CHECK: aie.use_lock(%[[AE:.*]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[A:.*]] : memref<8xbf16, 2>
// CHECK: aie.use_lock(%[[BE:.*]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[B:.*]] : memref<8xbf16, 2>
// CHECK: aie.core(%tile_2_3)
// CHECK: scf.for
// CHECK-NEXT: aie.use_lock(%[[AF:.*]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: func.call @use(%[[A]])
// CHECK-NEXT: aie.use_lock(%[[AE]], Release, %{{.*}})
// CHECK: aie.use_lock(%[[BF:.*]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: func.call @use(%[[B]])
// CHECK-NEXT: aie.use_lock(%[[BE]], Release, %{{.*}})

air.channel @c [1, 1]
func.func private @use(memref<8xbf16, 2>)
func.func @a_twice_in_row_then_b() {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) {
    air.segment @seg {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %t0, %a2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      %t1, %b2 = air.execute -> (memref<8xbf16, 1>) {
        %alloc = memref.alloc() : memref<8xbf16, 1>
        air.execute_terminator %alloc : memref<8xbf16, 1>
      }
      scf.for %j = %c0 to %c2 step %c1_0 {
        air.channel.put @c[] (%a2[] [] []) : (memref<8xbf16, 1>)
      }
      air.channel.put @c[] (%b2[] [] []) : (memref<8xbf16, 1>)
      %d2 = air.execute {memref.dealloc %a2 : memref<8xbf16, 1>}
      %d3 = air.execute {memref.dealloc %b2 : memref<8xbf16, 1>}
      air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {link_with = "use.o", x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c2_h = arith.constant 2 : index
        %t2, %a = air.execute -> (memref<8xbf16, 2>) {
          %alloc = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %alloc : memref<8xbf16, 2>
        }
        %t3, %b = air.execute -> (memref<8xbf16, 2>) {
          %alloc = memref.alloc() : memref<8xbf16, 2>
          air.execute_terminator %alloc : memref<8xbf16, 2>
        }
        scf.for %j = %c0_h to %c2_h step %c1_h {
          air.channel.get @c[] (%a[] [] []) : (memref<8xbf16, 2>)
          func.call @use(%a) : (memref<8xbf16, 2>) -> ()
        }
        air.channel.get @c[] (%b[] [] []) : (memref<8xbf16, 2>)
        func.call @use(%b) : (memref<8xbf16, 2>) -> ()
        %d0 = air.execute {memref.dealloc %a : memref<8xbf16, 2>}
        %d1 = air.execute {memref.dealloc %b : memref<8xbf16, 2>}
      }
    }
  }
  return
}

// -----

// The channel fills A and B whole, then the first half of B. A and the
// whole fill of B alone would be slots of one rotation, but B is also moved
// with another access pattern, so it rotates with no other buffer: A and B
// keep a pair each, and both fills of B take B's pair.

// CHECK: aie.device
// CHECK-NOT: init = 2
// CHECK: aie.mem(%tile_2_3)
// CHECK: aie.use_lock(%[[AE:.*]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[A:.*]] : memref<16xbf16, 2> offset = 0 len = 16)
// CHECK: aie.use_lock(%[[BE:.*]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[B:.*]] : memref<16xbf16, 2> offset = 0 len = 16)
// CHECK: aie.use_lock(%[[BE]], AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[B]] : memref<16xbf16, 2> offset = 0 len = 8)
// CHECK: aie.core(%tile_2_3)

air.channel @c [1, 1]
func.func private @use(memref<16xbf16, 2>)
func.func @b_whole_then_half() {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) {
    air.segment @seg {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %t0, %a2 = air.execute -> (memref<16xbf16, 1>) {
        %alloc = memref.alloc() : memref<16xbf16, 1>
        air.execute_terminator %alloc : memref<16xbf16, 1>
      }
      %t1, %b2 = air.execute -> (memref<16xbf16, 1>) {
        %alloc = memref.alloc() : memref<16xbf16, 1>
        air.execute_terminator %alloc : memref<16xbf16, 1>
      }
      air.channel.put @c[] (%a2[] [] []) : (memref<16xbf16, 1>)
      air.channel.put @c[] (%b2[] [] []) : (memref<16xbf16, 1>)
      air.channel.put @c[] (%b2[%c0] [%c8] [%c1_0]) : (memref<16xbf16, 1>)
      %d2 = air.execute {memref.dealloc %a2 : memref<16xbf16, 1>}
      %d3 = air.execute {memref.dealloc %b2 : memref<16xbf16, 1>}
      air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) attributes {link_with = "use.o", x_loc = 2 : i64, y_loc = 3 : i64} {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c8_h = arith.constant 8 : index
        %t2, %a = air.execute -> (memref<16xbf16, 2>) {
          %alloc = memref.alloc() : memref<16xbf16, 2>
          air.execute_terminator %alloc : memref<16xbf16, 2>
        }
        %t3, %b = air.execute -> (memref<16xbf16, 2>) {
          %alloc = memref.alloc() : memref<16xbf16, 2>
          air.execute_terminator %alloc : memref<16xbf16, 2>
        }
        air.channel.get @c[] (%a[] [] []) : (memref<16xbf16, 2>)
        func.call @use(%a) : (memref<16xbf16, 2>) -> ()
        air.channel.get @c[] (%b[] [] []) : (memref<16xbf16, 2>)
        func.call @use(%b) : (memref<16xbf16, 2>) -> ()
        air.channel.get @c[] (%b[%c0_h] [%c8_h] [%c1_h]) : (memref<16xbf16, 2>)
        func.call @use(%b) : (memref<16xbf16, 2>) -> ()
        %d0 = air.execute {memref.dealloc %a : memref<16xbf16, 2>}
        %d1 = air.execute {memref.dealloc %b : memref<16xbf16, 2>}
      }
    }
  }
  return
}
