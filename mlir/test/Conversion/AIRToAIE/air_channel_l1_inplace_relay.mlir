//===- air_channel_l1_inplace_relay.mlir -----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie='row-offset=2 col-offset=0 device=npu2' --split-input-file | FileCheck %s

// A herd relays four chunks in place through two L1 buffers (ping-pong): it
// gets each chunk into a buffer and puts it from there. Each buffer is chained
// through three locks: the S2MM takes "prod" and gives "cons", the core takes
// "cons" and gives "send", and the MM2S takes "send" and gives "prod". So the
// S2MM refills a buffer only after the MM2S has sent it; a core-side release
// of the S2MM's lock at the put would let it refill the buffer first.

// CHECK-LABEL: aie.device(npu2) @seg {
// CHECK-DAG: aie.lock(%{{.*}}) {init = 1 : i32, sym_name = "buf0_prod_lock"}
// CHECK-DAG: aie.lock(%{{.*}}) {init = 0 : i32, sym_name = "buf0_cons_lock"}
// CHECK-DAG: aie.lock(%{{.*}}) {init = 0 : i32, sym_name = "buf0_send_lock"}
// CHECK-DAG: aie.lock(%{{.*}}) {init = 1 : i32, sym_name = "buf1_prod_lock"}
// CHECK-DAG: aie.lock(%{{.*}}) {init = 0 : i32, sym_name = "buf1_cons_lock"}
// CHECK-DAG: aie.lock(%{{.*}}) {init = 0 : i32, sym_name = "buf1_send_lock"}

// CHECK: aie.mem(
// CHECK: aie.dma_start(MM2S, 0
// CHECK: aie.use_lock(%[[X:buf[0-9]+]]_send_lock, AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[X]] :
// CHECK-NEXT: aie.use_lock(%[[X]]_prod_lock, Release, %{{.*}})
// CHECK: aie.use_lock(%[[Y:buf[0-9]+]]_send_lock, AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[Y]] :
// CHECK-NEXT: aie.use_lock(%[[Y]]_prod_lock, Release, %{{.*}})
// CHECK: aie.dma_start(S2MM, 0
// CHECK: aie.use_lock(%[[X]]_prod_lock, AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[X]] :
// CHECK-NEXT: aie.use_lock(%[[X]]_cons_lock, Release, %{{.*}})
// CHECK: aie.use_lock(%[[Y]]_prod_lock, AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.dma_bd(%[[Y]] :
// CHECK-NEXT: aie.use_lock(%[[Y]]_cons_lock, Release, %{{.*}})

// CHECK: aie.core(
// CHECK: scf.for
// CHECK-NEXT: aie.use_lock(%[[X]]_cons_lock, AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.use_lock(%[[X]]_send_lock, Release, %{{.*}})
// CHECK-NEXT: aie.use_lock(%[[Y]]_cons_lock, AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.use_lock(%[[Y]]_send_lock, Release, %{{.*}})
// CHECK-NEXT: }

module {
  air.channel @In [1, 1]
  air.channel @Out [1, 1]
  func.func @relay(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx) in (%ls=%c1) args(%in=%arg0, %out=%arg1) : memref<64xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c1_0 = arith.constant 1 : index
      %t0 = air.wait_all async
      %puts = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%t = %t0) -> (!air.async.token) {
        %o = arith.muli %i, %c16 : index
        %p = air.channel.put async [%t] @In[%c0, %c0] (%in[%o] [%c16] [%c1_0]) {id = 1 : i32} : (memref<64xi32>)
        scf.yield %p : !air.async.token
      }
      %seg = air.segment @seg async attributes {x_loc = 0 : i64, x_size = 8 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c1_1 = arith.constant 1 : index
        %herd = air.herd @h async tile (%x, %y) in (%sx=%c1_1, %sy=%c1_1) attributes {x_loc = 0 : i64, y_loc = 2 : i64} {
          %c2 = arith.constant 2 : index
          %c0_2 = arith.constant 0 : index
          %c4_3 = arith.constant 4 : index
          %ta, %ping = air.execute -> (memref<16xi32, 2>) {
            %a = memref.alloc() : memref<16xi32, 2>
            air.execute_terminator %a : memref<16xi32, 2>
          }
          %tb, %pong = air.execute -> (memref<16xi32, 2>) {
            %b = memref.alloc() : memref<16xi32, 2>
            air.execute_terminator %b : memref<16xi32, 2>
          }
          %w0 = air.wait_all async [%ta, %tb]
          %w1 = air.wait_all async [%ta, %tb]
          %loop:4 = scf.for %j = %c0_2 to %c4_3 step %c2 iter_args(%t1 = %w1, %t2 = %tb, %t3 = %w0, %t4 = %w0) -> (!air.async.token, !air.async.token, !air.async.token, !air.async.token) {
            %g0 = air.channel.get async [%t1, %t4] @In[%x, %y] (%ping[] [] []) {id = 2 : i32} : (memref<16xi32, 2>)
            %p0 = air.channel.put async [%t3, %g0] @Out[%x, %y] (%ping[] [] []) {id = 3 : i32} : (memref<16xi32, 2>)
            %g1 = air.channel.get async [%t2, %g0] @In[%x, %y] (%pong[] [] []) {id = 4 : i32} : (memref<16xi32, 2>)
            %p1 = air.channel.put async [%p0, %g1] @Out[%x, %y] (%pong[] [] []) {id = 5 : i32} : (memref<16xi32, 2>)
            scf.yield %p0, %p1, %p1, %g1 : !air.async.token, !air.async.token, !air.async.token, !air.async.token
          }
          %da = air.execute [%loop#1] {
            memref.dealloc %ping : memref<16xi32, 2>
          }
          %db = air.execute [%loop#1] {
            memref.dealloc %pong : memref<16xi32, 2>
          }
        }
      }
      %t5 = air.wait_all async
      %gets = scf.for %i = %c0 to %c4 step %c1_0 iter_args(%t = %t5) -> (!air.async.token) {
        %o = arith.muli %i, %c16 : index
        %g = air.channel.get async [%t] @Out[%c0, %c0] (%out[%o] [%c16] [%c1_0]) {id = 6 : i32} : (memref<64xi32>)
        scf.yield %g : !air.async.token
      }
    }
    return
  }
}

// -----

// Two subviews of one buffer are relayed: they share the buffer's locks, so
// neither gets a lock cycle of its own (the cycle's lock names would clash).
// CHECK-LABEL: aie.device(npu2) @seg_views {
// CHECK-NOT: _send_lock
// CHECK: aie.core(

module {
  air.channel @In [1, 1]
  air.channel @Out [1, 1]
  func.func @views(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%lx) in (%ls=%c1) args(%in=%arg0, %out=%arg1) : memref<64xi32>, memref<64xi32> {
      air.channel.put @In[] (%in[] [] []) : (memref<64xi32>)
      air.segment @seg_views {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %a = memref.alloc() : memref<32xi32, 2>
          %s0 = memref.subview %a[0] [16] [1] : memref<32xi32, 2> to memref<16xi32, strided<[1]>, 2>
          %s1 = memref.subview %a[16] [16] [1] : memref<32xi32, 2> to memref<16xi32, strided<[1], offset: 16>, 2>
          air.channel.get @In[%x, %y] (%s0[] [] []) : (memref<16xi32, strided<[1]>, 2>)
          air.channel.put @Out[%x, %y] (%s0[] [] []) : (memref<16xi32, strided<[1]>, 2>)
          air.channel.get @In[%x, %y] (%s1[] [] []) : (memref<16xi32, strided<[1], offset: 16>, 2>)
          air.channel.put @Out[%x, %y] (%s1[] [] []) : (memref<16xi32, strided<[1], offset: 16>, 2>)
          memref.dealloc %a : memref<32xi32, 2>
        }
      }
      air.channel.get @Out[] (%out[] [] []) : (memref<64xi32>)
    }
    return
  }
}

// -----

// The channels also move a buffer the core writes: the channel's buffers
// share one counting lock pair, so the relay buffer keeps it too.
// CHECK-LABEL: aie.device(npu2) @seg_mixed {
// CHECK-NOT: _send_lock
// CHECK: aie.core(

module {
  air.channel @In [1, 1]
  air.channel @Out [1, 1]
  func.func @mixed(%arg0: memref<64xi32>, %arg1: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%lx) in (%ls=%c1) args(%in=%arg0, %out=%arg1) : memref<64xi32>, memref<64xi32> {
      air.channel.put @In[] (%in[] [] []) : (memref<64xi32>)
      air.segment @seg_mixed {
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%x, %y) in (%sx=%c1_0, %sy=%c1_0) {
          %c0 = arith.constant 0 : index
          %c4 = arith.constant 4 : index
          %c1_1 = arith.constant 1 : index
          %v = arith.constant 7 : i32
          scf.for %j = %c0 to %c4 step %c1_1 {
            %a = memref.alloc() : memref<16xi32, 2>
            %b = memref.alloc() : memref<16xi32, 2>
            air.channel.get @In[%x, %y] (%a[] [] []) : (memref<16xi32, 2>)
            air.channel.put @Out[%x, %y] (%a[] [] []) : (memref<16xi32, 2>)
            air.channel.get @In[%x, %y] (%b[] [] []) : (memref<16xi32, 2>)
            memref.store %v, %b[%c0] : memref<16xi32, 2>
            air.channel.put @Out[%x, %y] (%b[] [] []) : (memref<16xi32, 2>)
            memref.dealloc %a : memref<16xi32, 2>
            memref.dealloc %b : memref<16xi32, 2>
          }
        }
      }
      air.channel.get @Out[] (%out[] [] []) : (memref<64xi32>)
    }
    return
  }
}

// -----

// The relayed buffer arrives or leaves by cascade, not by the tile DMA: no
// DMA would take part in the cycle, so the buffer keeps its lock pair.
// CHECK-LABEL: aie.device(npu2) @seg_casc {
// CHECK-NOT: _send_lock
// CHECK: aie.core(

#set = affine_set<()[s0] : (s0 - 1 == 0)>
module {
  air.channel @In [1]
  air.channel @Out [1]
  air.channel @cascade [1] {channel_type = "npu_cascade"}
  func.func @casc(%arg0: memref<16xi32>, %arg1: memref<16xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%lx) in (%ls=%c1) args(%in=%arg0, %out=%arg1) : memref<16xi32>, memref<16xi32> {
      air.channel.put @In[] (%in[] [] []) : (memref<16xi32>)
      air.segment @seg_casc {
        %c2 = arith.constant 2 : index
        %c1_0 = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%sx=%c1_0, %sy=%c2) {
          %c0 = arith.constant 0 : index
          %c1_1 = arith.constant 1 : index
          %a = memref.alloc() : memref<16xi32, 2>
          affine.if #set()[%ty] {
            air.channel.get @In[] (%a[] [] []) : (memref<16xi32, 2>)
            air.channel.put @cascade[%c0] (%a[] [] []) : (memref<16xi32, 2>)
          } else {
            air.channel.get @cascade[%ty] (%a[] [] []) : (memref<16xi32, 2>)
            air.channel.put @Out[] (%a[] [] []) : (memref<16xi32, 2>)
          }
          memref.dealloc %a : memref<16xi32, 2>
        }
      }
      air.channel.get @Out[] (%out[] [] []) : (memref<16xi32>)
    }
    return
  }
}

