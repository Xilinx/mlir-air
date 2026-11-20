//===- air_channel_release_stream_two_buffers.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// One loop streams two buffers, the second's stream going on in a later
// loop from the first loop's results. Each buffer is free after the last
// loop that loads through a pointer into it: A after the first loop, B after
// the second.
// CHECK-LABEL: aie.core(%tile_2_4)
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %{{.*}})
// CHECK-NOT: aie.use_lock
// CHECK: } {loop_annotation
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK-NEXT: func.call @other
// CHECK-NEXT: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: scf.for
// CHECK-NOT: aie.use_lock
// CHECK: } {loop_annotation
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK-NEXT: func.call @other

air.channel @channel_0 [1, 1]
air.channel @channel_1 [1, 1]
func.func private @use(vector<64xi8>, vector<8xi8>)
func.func private @other()
func.func @two_streams() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) attributes {link_with = "w.o"} {
        %c0_inner = arith.constant 0 : index
        %cmp = arith.cmpi eq, %arg9, %c0_inner : index
        %3 = scf.if %cmp -> (!air.async.token) {
          %t, %a = air.execute -> (memref<576xi8, 2>) {
            %alloc = memref.alloc() : memref<576xi8, 2>
            air.execute_terminator %alloc : memref<576xi8, 2>
          }
          %4 = air.channel.put async [%t]  @channel_0[] (%a[] [] []) : (memref<576xi8, 2>)
          %5 = air.channel.put async [%4]  @channel_1[] (%a[] [] []) : (memref<576xi8, 2>)
          %6 = air.channel.put async [%5]  @channel_0[] (%a[] [] []) : (memref<576xi8, 2>)
          %d = air.execute [%6] {
            memref.dealloc %a : memref<576xi8, 2>
          }
          scf.yield %d : !air.async.token
        } else {
          %ta, %a = air.execute -> (memref<576xi8, 2>) {
            %alloc = memref.alloc() : memref<576xi8, 2>
            air.execute_terminator %alloc : memref<576xi8, 2>
          }
          %tb, %b = air.execute -> (memref<576xi8, 2>) {
            %alloc = memref.alloc() : memref<576xi8, 2>
            air.execute_terminator %alloc : memref<576xi8, 2>
          }
          %tc, %cbuf = air.execute -> (memref<576xi8, 2>) {
            %alloc = memref.alloc() : memref<576xi8, 2>
            air.execute_terminator %alloc : memref<576xi8, 2>
          }
          %4 = air.channel.get async [%ta]  @channel_0[] (%a[] [] []) : (memref<576xi8, 2>)
          %5 = air.channel.get async [%tb, %4]  @channel_1[] (%b[] [] []) : (memref<576xi8, 2>)
          %w, %rq, %rs, %rc = air.execute [%5] -> (!llvm.ptr, vector<32xi32>, i32) {
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %cs = arith.constant 1 : index
            %na, %sa, %ka = aievec.fifo_start %a[%c0] : memref<576xi8, 2> -> vector<32xi32>
            %nb, %sb, %kb = aievec.fifo_start %b[%c0] : memref<576xi8, 2> -> vector<32xi32>
            %r:6 = scf.for %i = %c0 to %c8 step %cs iter_args(%p = %na, %st = %sa, %ct = %ka, %q = %nb, %sq = %sb, %cq = %kb) -> (!llvm.ptr, vector<32xi32>, i32, !llvm.ptr, vector<32xi32>, i32) {
              %p2, %st2, %ct2, %m, %e = aievec.fifo_ld_pop_bfp16 %p, %st, %ct : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>
              %q2, %sq2, %cq2, %m2, %e2 = aievec.fifo_ld_pop_bfp16 %q, %sq, %cq : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>
              func.call @use(%m, %e) : (vector<64xi8>, vector<8xi8>) -> ()
              func.call @use(%m2, %e2) : (vector<64xi8>, vector<8xi8>) -> ()
              scf.yield %p2, %st2, %ct2, %q2, %sq2, %cq2 : !llvm.ptr, vector<32xi32>, i32, !llvm.ptr, vector<32xi32>, i32
            }
            func.call @other() : () -> ()
            air.execute_terminator %r#3, %r#4, %r#5 : !llvm.ptr, vector<32xi32>, i32
          }
          %6 = air.channel.get async [%w]  @channel_0[] (%cbuf[] [] []) : (memref<576xi8, 2>)
          %w2 = air.execute [%6] {
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %cs = arith.constant 1 : index
            %s:3 = scf.for %i = %c0 to %c8 step %cs iter_args(%q = %rq, %sq = %rs, %cq = %rc) -> (!llvm.ptr, vector<32xi32>, i32) {
              %q2, %sq2, %cq2, %m2, %e2 = aievec.fifo_ld_pop_bfp16 %q, %sq, %cq : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>
              func.call @use(%m2, %e2) : (vector<64xi8>, vector<8xi8>) -> ()
              scf.yield %q2, %sq2, %cq2 : !llvm.ptr, vector<32xi32>, i32
            }
          }
          %o = air.execute [%w2] {
            func.call @other() : () -> ()
          }
          %d = air.execute [%o] {
            memref.dealloc %a : memref<576xi8, 2>
            memref.dealloc %b : memref<576xi8, 2>
            memref.dealloc %cbuf : memref<576xi8, 2>
          }
          scf.yield %d : !air.async.token
        }
      }
    }
  }
  return
}
