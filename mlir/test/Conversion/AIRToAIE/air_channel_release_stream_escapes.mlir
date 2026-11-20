//===- air_channel_release_stream_escapes.mlir -----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// The core reads the buffer by a call, then through a stream whose pointer
// is stored and loaded back before the loop pops from it. Loads through the
// reloaded pointer cannot be told from others, so the buffer is kept to the
// end of the block, past the call.
// CHECK-LABEL: aie.core(%tile_2_4)
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %{{.*}})
// CHECK-NOT: aie.use_lock
// CHECK: func.call @touch
// CHECK-NOT: aie.use_lock
// CHECK: } {loop_annotation
// CHECK-NEXT: func.call @other
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %{{.*}})
// CHECK-NEXT: aie.end

air.channel @channel_0 [1, 1]
func.func private @use(vector<64xi8>, vector<8xi8>)
func.func private @other()
func.func private @touch(memref<576xi8, 2>)
func.func private @cond() -> i1
func.func @stream_pointer_stored() {
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
            %one = arith.constant 1 : i64
            %slot = llvm.alloca %one x !llvm.ptr : (i64) -> !llvm.ptr
            func.call @touch(%buf) : (memref<576xi8, 2>) -> ()
            %n, %s, %k = aievec.fifo_start %buf[%c0] : memref<576xi8, 2> -> vector<32xi32>
            llvm.store %n, %slot : !llvm.ptr, !llvm.ptr
            func.call @other() : () -> ()
            %ld = llvm.load %slot : !llvm.ptr -> !llvm.ptr
            %r:3 = scf.for %i = %c0 to %c8 step %cs iter_args(%p = %ld, %st = %s, %ct = %k) -> (!llvm.ptr, vector<32xi32>, i32) {
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
