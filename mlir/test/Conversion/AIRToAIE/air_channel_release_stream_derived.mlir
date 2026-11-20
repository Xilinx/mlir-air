//===- air_channel_release_stream_derived.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// The pointers into the buffer the core makes from the stream's: a GEP's,
// a call's result, the one a pop advances to, a loop's result, which is its
// init when the loop runs no trip though this body yields a null pointer,
// and a loop's result yielded from a pop in its body. Loads and stores
// through them are uses of it too. The buffer is in use until the last loop.
// CHECK: aie.core(%{{.*}})  {
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %{{.*}})
// CHECK-NOT: aie.use_lock
// CHECK: } {loop_annotation
// CHECK-NOT: aie.use_lock
// CHECK: } {loop_annotation
// CHECK-NOT: aie.use_lock
// CHECK: func.call @other
// CHECK-NEXT: scf.for
// CHECK-NOT: aie.use_lock
// CHECK: } {loop_annotation
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK-NEXT: func.call @other
// CHECK: aie.end

air.channel @channel_0 [1, 1]
func.func private @use(vector<64xi8>, vector<8xi8>)
func.func private @other()
func.func private @take(i8)
func.func private @adv(!llvm.ptr) -> !llvm.ptr
func.func @release_after_derived() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) attributes {link_with = "w.o"} {
        %c0_inner = arith.constant 0 : index
        %cmp = arith.cmpi eq, %arg9, %c0_inner : index
        %3 = scf.if %cmp -> (!air.async.token) {
          %t, %buf = air.execute -> (memref<576xi8, 2>) {
            %alloc = memref.alloc() : memref<576xi8, 2>
            air.execute_terminator %alloc : memref<576xi8, 2>
          }
          %4 = air.channel.put async [%t]  @channel_0[] (%buf[] [] []) : (memref<576xi8, 2>)
          %d = air.execute [%4] {
            memref.dealloc %buf : memref<576xi8, 2>
          }
          scf.yield %d : !air.async.token
        } else {
          %t, %buf = air.execute -> (memref<576xi8, 2>) {
            %alloc = memref.alloc() : memref<576xi8, 2>
            air.execute_terminator %alloc : memref<576xi8, 2>
          }
          %4 = air.channel.get async [%t]  @channel_0[] (%buf[] [] []) : (memref<576xi8, 2>)
          %w = air.execute [%4] {
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %cs = arith.constant 1 : index
            %n, %s, %k = aievec.fifo_start %buf[%c0] : memref<576xi8, 2> -> vector<32xi32>
            %g = llvm.getelementptr %n[1] : (!llvm.ptr) -> !llvm.ptr, i8
            %x = llvm.load %g : !llvm.ptr -> i8
            func.call @take(%x) : (i8) -> ()
            llvm.store %x, %g : i8, !llvm.ptr
            %h = func.call @adv(%g) : (!llvm.ptr) -> !llvm.ptr
            %n1, %s1, %k1, %m0, %e0 = aievec.fifo_ld_pop_bfp16 %h, %s, %k : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>
            func.call @use(%m0, %e0) : (vector<64xi8>, vector<8xi8>) -> ()
            func.call @other() : () -> ()
            %z = llvm.mlir.zero : !llvm.ptr
            %r:3 = scf.for %i = %c0 to %c8 step %cs iter_args(%p = %n1, %st = %s1, %ct = %k1) -> (!llvm.ptr, vector<32xi32>, i32) {
              %p2, %st2, %ct2, %m, %e = aievec.fifo_ld_pop_bfp16 %p, %st, %ct : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>
              func.call @use(%m, %e) : (vector<64xi8>, vector<8xi8>) -> ()
              scf.yield %z, %st2, %ct2 : !llvm.ptr, vector<32xi32>, i32
            }
            func.call @other() : () -> ()
            %q:3 = scf.for %i = %c0 to %c8 step %cs iter_args(%p = %z, %st = %r#1, %ct = %r#2) -> (!llvm.ptr, vector<32xi32>, i32) {
              %p2, %st2, %ct2, %m, %e = aievec.fifo_ld_pop_bfp16 %r#0, %st, %ct : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>
              func.call @use(%m, %e) : (vector<64xi8>, vector<8xi8>) -> ()
              scf.yield %p2, %st2, %ct2 : !llvm.ptr, vector<32xi32>, i32
            }
            func.call @other() : () -> ()
            %u:3 = scf.for %i = %c0 to %c8 step %cs iter_args(%p = %q#0, %st = %q#1, %ct = %q#2) -> (!llvm.ptr, vector<32xi32>, i32) {
              %p2, %st2, %ct2, %m, %e = aievec.fifo_ld_pop_bfp16 %p, %st, %ct : vector<32xi32> -> vector<32xi32>, vector<64xi8>, vector<8xi8>
              func.call @use(%m, %e) : (vector<64xi8>, vector<8xi8>) -> ()
              scf.yield %p2, %st2, %ct2 : !llvm.ptr, vector<32xi32>, i32
            }
          }
          %o = air.execute [%w] {
            func.call @other() : () -> ()
          }
          %d = air.execute [%o] {
            memref.dealloc %buf : memref<576xi8, 2>
          }
          scf.yield %d : !air.async.token
        }
      }
    }
  }
  return
}
