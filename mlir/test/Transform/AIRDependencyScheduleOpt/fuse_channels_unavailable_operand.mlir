//===- fuse_channels_unavailable_operand.mlir ------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels="aggressive-mode=L1" | FileCheck %s

// A core runs two loops one after the other. Each computes into an L1 buffer
// and then puts it, the put waiting on the computation. Merging channel b into
// channel a would move b's put into a's loop, where the token it waits on is
// not defined. The merge is refused and both channels stay.
// CHECK-LABEL: @compute_then_put
// CHECK: air.herd
// CHECK: scf.for
// CHECK: %[[CA:.*]] = scf.for
// CHECK: air.channel.put async [%[[CA]]] @a_out
// CHECK: scf.for
// CHECK: %[[CB:.*]] = scf.for
// CHECK: air.channel.put async [%[[CB]]] @b_out
module {
  air.channel @a_out [1, 1]
  air.channel @b_out [1, 1]
  func.func @compute_then_put(%n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n) : index {
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %ga = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %ta2) -> (!air.async.token) {
          %g = air.channel.get async [%t] @a_out[] (%l2a[] [] []) : (memref<32xi32, 1>)
          scf.yield %g : !air.async.token
        }
        %gb = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %tb2) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_out[] (%l2b[] [] []) : (memref<32xi32, 1>)
          scf.yield %g : !air.async.token
        }
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%hn=%sn) : index {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_h = arith.constant 2 : index
          %v = arith.constant 7 : i32
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %tb, %bufb = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %la = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %ta) -> (!air.async.token) {
            %ca = scf.for %j = %c0_h to %c2_h step %c1_h iter_args(%u = %t) -> (!air.async.token) {
              %e = air.execute [%u] {
                memref.store %v, %bufa[%j] : memref<32xi32, 2>
              }
              scf.yield %e : !air.async.token
            }
            %p = air.channel.put async [%ca] @a_out[] (%bufa[] [] []) : (memref<32xi32, 2>)
            scf.yield %p : !air.async.token
          }
          %lb = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %la) -> (!air.async.token) {
            %cb = scf.for %j = %c0_h to %c2_h step %c1_h iter_args(%u = %t) -> (!air.async.token) {
              %e = air.execute [%u] {
                memref.store %v, %bufb[%j] : memref<32xi32, 2>
              }
              scf.yield %e : !air.async.token
            }
            %p = air.channel.put async [%cb] @b_out[] (%bufb[] [] []) : (memref<32xi32, 2>)
            scf.yield %p : !air.async.token
          }
        }
      }
    }
    return
  }
}
