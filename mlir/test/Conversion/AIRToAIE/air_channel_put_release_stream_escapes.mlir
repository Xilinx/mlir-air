//===- air_channel_put_release_stream_escapes.mlir -------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// The core pushes into A through a stream whose pointer it also stores, then
// sends A and waits for C, which the other core sends only once it has A.
// The stored address says nothing of the core's use of A after the put: A's
// put releases A before the core waits for C.
// CHECK-LABEL: aie.core(%tile_2_4)
// CHECK: llvm.store
// CHECK-NOT: aie.use_lock
// CHECK: } {loop_annotation
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK-NEXT: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %{{.*}})
// CHECK-NEXT: func.call @usec

air.channel @ch0 [1, 1]
air.channel @ch2 [1, 1]
func.func private @mk() -> (vector<64xi8>, vector<8xi8>)
func.func private @usec(memref<576xi8, 2>)
func.func @put_after_stored_stream() {
  %c1 = arith.constant 1 : index
  %0 = air.launch async (%arg4, %arg5) in (%arg6=%c1, %arg7=%c1) {
    %1 = air.segment async {
      %c2 = arith.constant 2 : index
      %c1_0 = arith.constant 1 : index
      %2 = air.herd @herd_0 async tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c2) attributes {link_with = "w.o"} {
        %c0_inner = arith.constant 0 : index
        %cmp = arith.cmpi eq, %arg9, %c0_inner : index
        %3 = scf.if %cmp -> (!air.async.token) {
          %t, %x = air.execute -> (memref<576xi8, 2>) {
            %alloc = memref.alloc() : memref<576xi8, 2>
            air.execute_terminator %alloc : memref<576xi8, 2>
          }
          %5 = air.channel.get async [%t]  @ch0[] (%x[] [] []) : (memref<576xi8, 2>)
          %6 = air.channel.put async [%5]  @ch2[] (%x[] [] []) : (memref<576xi8, 2>)
          %d = air.execute [%6] {
            memref.dealloc %x : memref<576xi8, 2>
          }
          scf.yield %d : !air.async.token
        } else {
          %ta, %a = air.execute -> (memref<576xi8, 2>) {
            %alloc = memref.alloc() : memref<576xi8, 2>
            air.execute_terminator %alloc : memref<576xi8, 2>
          }
          %tc, %cbuf = air.execute -> (memref<576xi8, 2>) {
            %alloc = memref.alloc() : memref<576xi8, 2>
            air.execute_terminator %alloc : memref<576xi8, 2>
          }
          %w = air.execute [%ta] {
            %c0 = arith.constant 0 : index
            %c8 = arith.constant 8 : index
            %cs = arith.constant 1 : index
            %one = arith.constant 1 : i64
            %slot = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
            %na, %sa, %ka = aievec.fifo_start %a[%c0] : memref<576xi8, 2> -> vector<32xi32>
            llvm.store %na, %slot : !llvm.ptr, !llvm.ptr
            %r:3 = scf.for %i = %c0 to %c8 step %cs iter_args(%p = %na, %st = %sa, %ct = %ka) -> (!llvm.ptr, vector<32xi32>, i32) {
              %m, %e = func.call @mk() : () -> (vector<64xi8>, vector<8xi8>)
              %p2, %st2, %ct2 = aievec.fifo_st_push_bfp16 %p, %st, %ct, %m, %e : vector<32xi32>, vector<64xi8>, vector<8xi8> -> vector<32xi32>
              scf.yield %p2, %st2, %ct2 : !llvm.ptr, vector<32xi32>, i32
            }
          }
          %6 = air.channel.put async [%w]  @ch0[] (%a[] [] []) : (memref<576xi8, 2>)
          %7 = air.channel.get async [%6]  @ch2[] (%cbuf[] [] []) : (memref<576xi8, 2>)
          %w2 = air.execute [%7] {
            func.call @usec(%cbuf) : (memref<576xi8, 2>) -> ()
          }
          %d = air.execute [%w2] {
            memref.dealloc %a : memref<576xi8, 2>
            memref.dealloc %cbuf : memref<576xi8, 2>
          }
          scf.yield %d : !air.async.token
        }
      }
    }
  }
  return
}
