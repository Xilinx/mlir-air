//===- canonicalize_channel_get_barrier_dep.mlir ---------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -canonicalize -split-input-file %s | FileCheck %s

// A barrier dep injected on an async sync primitive (channel.get/put,
// dma_memcpy_nd, alloc execute) must survive canonicalization when its
// transitive token consumers — not the primitive itself — touch the source's
// memref. Issue #1559 race #2/#3.

// CHECK-LABEL: func.func @chan_get_barrier_dep_preserved
// CHECK: %[[FILL:[a-zA-Z0-9_]+]] = scf.for {{.*}} iter_args
// CHECK:{{ *}}memref.store {{.*}} : memref<32x32xi16, 2 : i32>
// CHECK: air.channel.get async [%[[FILL]]
// CHECK: air.channel.get async [%[[FILL]]

module {
  air.channel @ch_a [1, 1]
  air.channel @ch_b [1, 1]
  func.func @chan_get_barrier_dep_preserved() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
      %1 = air.segment @seg async {
        %c1_s = arith.constant 1 : index
        %tok_out, %out_buf = air.execute -> (memref<32x32xi16, 2 : i32>) {
          %a = memref.alloc() : memref<32x32xi16, 2 : i32>
          air.execute_terminator %a : memref<32x32xi16, 2 : i32>
        }
        %h = air.herd @h async [%tok_out] tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%argbuf=%out_buf) : memref<32x32xi16, 2 : i32> {
          %c0_i16 = arith.constant 0 : i16
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c32 = arith.constant 32 : index
          %wa_init = air.wait_all async
          %fill_loop = scf.for %i = %c0 to %c32 step %c1_h iter_args(%it = %wa_init) -> (!air.async.token) {
            %tk = air.execute [%it] {
              memref.store %c0_i16, %argbuf[%i, %i] : memref<32x32xi16, 2 : i32>
            }
            scf.yield %tk : !air.async.token
          }
          %tk_in1, %in_buf1 = air.execute -> (memref<32xi16, 2 : i32>) {
            %a = memref.alloc() : memref<32xi16, 2 : i32>
            air.execute_terminator %a : memref<32xi16, 2 : i32>
          }
          %tk_in2, %in_buf2 = air.execute -> (memref<32xi16, 2 : i32>) {
            %a = memref.alloc() : memref<32xi16, 2 : i32>
            air.execute_terminator %a : memref<32xi16, 2 : i32>
          }
          %g1 = air.channel.get async [%fill_loop, %tk_in1] @ch_a[%tx, %ty] (%in_buf1[] [] []) : (memref<32xi16, 2 : i32>)
          %g2 = air.channel.get async [%fill_loop, %tk_in2] @ch_b[%tx, %ty] (%in_buf2[] [] []) : (memref<32xi16, 2 : i32>)
          %wa = air.wait_all async [%g1, %g2]
          %loop = scf.for %k = %c0 to %c32 step %c1_h iter_args(%it = %wa) -> (!air.async.token) {
            %tk = air.execute [%it] {
              %v = memref.load %argbuf[%k, %k] : memref<32x32xi16, 2 : i32>
              %v1 = memref.load %in_buf1[%k] : memref<32xi16, 2 : i32>
              %v2 = memref.load %in_buf2[%k] : memref<32xi16, 2 : i32>
              %sum = arith.addi %v1, %v2 : i16
              %sum2 = arith.addi %sum, %v : i16
              memref.store %sum2, %argbuf[%k, %k] : memref<32x32xi16, 2 : i32>
            }
            scf.yield %tk : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// Sink is air.channel.put reading a fresh staging buffer; consumer loop reads %argbuf.

// CHECK-LABEL: func.func @chan_put_barrier_dep_preserved
// CHECK: %[[FILL:[a-zA-Z0-9_]+]] = scf.for {{.*}} iter_args
// CHECK:{{ *}}memref.store {{.*}} : memref<32x32xi16, 2 : i32>
// CHECK: air.channel.put async [%[[FILL]]

module {
  air.channel @ch_p [1, 1]
  func.func @chan_put_barrier_dep_preserved() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
      %1 = air.segment @seg async {
        %c1_s = arith.constant 1 : index
        %tok_out, %out_buf = air.execute -> (memref<32x32xi16, 2 : i32>) {
          %a = memref.alloc() : memref<32x32xi16, 2 : i32>
          air.execute_terminator %a : memref<32x32xi16, 2 : i32>
        }
        %h = air.herd @h async [%tok_out] tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%argbuf=%out_buf) : memref<32x32xi16, 2 : i32> {
          %c0_i16 = arith.constant 0 : i16
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c32 = arith.constant 32 : index
          %wa_init = air.wait_all async
          %fill_loop = scf.for %i = %c0 to %c32 step %c1_h iter_args(%it = %wa_init) -> (!air.async.token) {
            %tk = air.execute [%it] {
              memref.store %c0_i16, %argbuf[%i, %i] : memref<32x32xi16, 2 : i32>
            }
            scf.yield %tk : !air.async.token
          }
          %tk_stage, %stage_buf = air.execute -> (memref<32xi16, 2 : i32>) {
            %a = memref.alloc() : memref<32xi16, 2 : i32>
            air.execute_terminator %a : memref<32xi16, 2 : i32>
          }
          %p = air.channel.put async [%fill_loop, %tk_stage] @ch_p[%tx, %ty] (%stage_buf[] [] []) : (memref<32xi16, 2 : i32>)
          %loop = scf.for %k = %c0 to %c32 step %c1_h iter_args(%it = %p) -> (!air.async.token) {
            %tk = air.execute [%it] {
              %v = memref.load %argbuf[%k, %k] : memref<32x32xi16, 2 : i32>
              memref.store %v, %stage_buf[%k] : memref<32xi16, 2 : i32>
            }
            scf.yield %tk : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// Sink is air.dma_memcpy_nd between two fresh buffers; consumer loop reads %argbuf.

// CHECK-LABEL: func.func @dma_barrier_dep_preserved
// CHECK: %[[FILL:[a-zA-Z0-9_]+]] = scf.for {{.*}} iter_args
// CHECK:{{ *}}memref.store {{.*}} : memref<32x32xi16, 2 : i32>
// CHECK: air.dma_memcpy_nd async [%[[FILL]]

module {
  func.func @dma_barrier_dep_preserved() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
      %1 = air.segment @seg async {
        %c1_s = arith.constant 1 : index
        %tok_out, %out_buf = air.execute -> (memref<32x32xi16, 2 : i32>) {
          %a = memref.alloc() : memref<32x32xi16, 2 : i32>
          air.execute_terminator %a : memref<32x32xi16, 2 : i32>
        }
        %h = air.herd @h async [%tok_out] tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%argbuf=%out_buf) : memref<32x32xi16, 2 : i32> {
          %c0_i16 = arith.constant 0 : i16
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c32 = arith.constant 32 : index
          %wa_init = air.wait_all async
          %fill_loop = scf.for %i = %c0 to %c32 step %c1_h iter_args(%it = %wa_init) -> (!air.async.token) {
            %tk = air.execute [%it] {
              memref.store %c0_i16, %argbuf[%i, %i] : memref<32x32xi16, 2 : i32>
            }
            scf.yield %tk : !air.async.token
          }
          %tk_src, %src_buf = air.execute -> (memref<32xi16, 2 : i32>) {
            %a = memref.alloc() : memref<32xi16, 2 : i32>
            air.execute_terminator %a : memref<32xi16, 2 : i32>
          }
          %tk_dst, %dst_buf = air.execute -> (memref<32xi16, 2 : i32>) {
            %a = memref.alloc() : memref<32xi16, 2 : i32>
            air.execute_terminator %a : memref<32xi16, 2 : i32>
          }
          %dma = air.dma_memcpy_nd async [%fill_loop, %tk_src, %tk_dst] (%dst_buf[] [] [], %src_buf[] [] []) : (memref<32xi16, 2 : i32>, memref<32xi16, 2 : i32>)
          %loop = scf.for %k = %c0 to %c32 step %c1_h iter_args(%it = %dma) -> (!air.async.token) {
            %tk = air.execute [%it] {
              %v = memref.load %argbuf[%k, %k] : memref<32x32xi16, 2 : i32>
              memref.store %v, %dst_buf[%k] : memref<32xi16, 2 : i32>
            }
            scf.yield %tk : !air.async.token
          }
        }
      }
    }
    return
  }
}

// -----

// Sink is air.execute wrapping a pure alloc; consumer loop reads %argbuf.

// CHECK-LABEL: func.func @execute_barrier_dep_preserved
// CHECK: %[[FILL:[a-zA-Z0-9_]+]] = scf.for {{.*}} iter_args
// CHECK:{{ *}}memref.store {{.*}} : memref<32x32xi16, 2 : i32>
// CHECK: %{{.*}}, %{{.*}} = air.execute [%[[FILL]]]

module {
  func.func @execute_barrier_dep_preserved() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
      %1 = air.segment @seg async {
        %c1_s = arith.constant 1 : index
        %tok_out, %out_buf = air.execute -> (memref<32x32xi16, 2 : i32>) {
          %a = memref.alloc() : memref<32x32xi16, 2 : i32>
          air.execute_terminator %a : memref<32x32xi16, 2 : i32>
        }
        %h = air.herd @h async [%tok_out] tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%argbuf=%out_buf) : memref<32x32xi16, 2 : i32> {
          %c0_i16 = arith.constant 0 : i16
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c32 = arith.constant 32 : index
          %wa_init = air.wait_all async
          %fill_loop = scf.for %i = %c0 to %c32 step %c1_h iter_args(%it = %wa_init) -> (!air.async.token) {
            %tk = air.execute [%it] {
              memref.store %c0_i16, %argbuf[%i, %i] : memref<32x32xi16, 2 : i32>
            }
            scf.yield %tk : !air.async.token
          }
          %tk_scratch, %scratch = air.execute [%fill_loop] -> (memref<32xi16, 2 : i32>) {
            %a = memref.alloc() : memref<32xi16, 2 : i32>
            air.execute_terminator %a : memref<32xi16, 2 : i32>
          }
          %loop = scf.for %k = %c0 to %c32 step %c1_h iter_args(%it = %tk_scratch) -> (!air.async.token) {
            %tk = air.execute [%it] {
              %v = memref.load %argbuf[%k, %k] : memref<32x32xi16, 2 : i32>
              memref.store %v, %scratch[%k] : memref<32xi16, 2 : i32>
            }
            scf.yield %tk : !air.async.token
          }
        }
      }
    }
    return
  }
}
