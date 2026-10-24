//===- fuse_channels_nested_loops_kept.mlir --------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels="aggressive-mode=L1,L2" -split-input-file | FileCheck %s

// Nests of loops whose channels the pass must not fuse, or may only rename
// without moving. Each case changes one thing in @ab_chunks of
// fuse_channels_nested_loops.mlir.

// The memtile sends the two chunks of B and then A, with no loop around
// them, while the core takes A first. Renaming B onto A's channel would leave
// the memtile and the core in different orders, and nothing can move B's
// chunk loop after A without a loop that both share. The channels stay apart.

// CHECK-LABEL: func.func @b_chunks_before_a
// CHECK: air.segment
// CHECK: air.channel.put {{.*}} @b_l1[]
// CHECK: air.channel.put {{.*}} @a_l1[]
// CHECK: air.herd
// CHECK: air.channel.get {{.*}} @a_l1[
// CHECK: air.channel.get {{.*}} @b_l1[

module {
  air.channel @a_in [1, 1]
  air.channel @b_in [1, 1]
  air.channel @a_l1 [1, 1]
  air.channel @b_l1 [1, 1]
  func.func @b_chunks_before_a(%in_a: memref<64xi32>, %in_b: memref<64xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<64xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %pa3 = air.channel.put async [%w0] @a_in[] (%la[] [] []) : (memref<64xi32>)
      %pb3 = air.channel.put async [%pa3] @b_in[] (%lb[] [] []) : (memref<64xi32>)
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c32_s = arith.constant 32 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<64xi32, 1>) {
          %x = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %x : memref<64xi32, 1>
        }
        %w = air.wait_all async [%ta2, %tb2]
        %sgb = air.channel.get async [%w] @b_in[] (%l2b[] [] []) : (memref<64xi32, 1>)
        %spj = scf.for %j = %c0_s to %c2_s step %c1_s iter_args(%tj = %sgb) -> (!air.async.token) {
          %off = arith.muli %j, %c32_s : index
          %p = air.channel.put async [%tj] @b_l1[] (%l2b[%off] [%c32_s] [%c1_s]) : (memref<64xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %sga = air.channel.get async [%w] @a_in[] (%l2a[] [] []) : (memref<32xi32, 1>)
        %spa = air.channel.put async [%sga] @a_l1[] (%l2a[] [] []) : (memref<32xi32, 1>)
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%hn=%sn) : index {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_h = arith.constant 2 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %tb, %bufb = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %wh = air.wait_all async [%ta, %tb]
          %ga = air.channel.get async [%wh] @a_l1[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
          %g = scf.for %j = %c0_h to %c2_h step %c1_h iter_args(%tj = %ga) -> (!air.async.token) {
            %gb = air.channel.get async [%tj] @b_l1[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
            scf.yield %gb : !air.async.token
          }
          %da = air.execute [%g] {
            memref.dealloc %bufa : memref<32xi32, 2>
          }
          %db = air.execute [%g] {
            memref.dealloc %bufb : memref<32xi32, 2>
          }
        }
      }
    }
    return
  }
}

// -----

// As @ab_chunks, but B's chunk loop also carries the chunk offset as a
// result. Moving the loop after A would have to replace that result too, so
// the merge is not done and B keeps its channels.

// CHECK-LABEL: func.func @chunk_loop_index_result
// CHECK: air.segment
// CHECK: scf.for {{.*}} -> (!air.async.token, index)
// CHECK: air.channel.put {{.*}} @b_l1[]
// CHECK: air.herd
// CHECK: air.channel.get {{.*}} @b_l1[

module {
  air.channel @a_in [1, 1]
  air.channel @b_in [1, 1]
  air.channel @a_l1 [1, 1]
  air.channel @b_l1 [1, 1]
  func.func @chunk_loop_index_result(%in_a: memref<64xi32>, %in_b: memref<64xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<64xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @a_in[] (%la[] [] []) : (memref<64xi32>)
        %p2 = air.channel.put async [%p] @b_in[] (%lb[] [] []) : (memref<64xi32>)
        scf.yield %p2 : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c32_s = arith.constant 32 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<64xi32, 1>) {
          %x = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %x : memref<64xi32, 1>
        }
        %w = air.wait_all async [%ta2, %tb2]
        %loop_a = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @a_in[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @a_l1[] (%l2a[] [] []) : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in[] (%l2b[] [] []) : (memref<64xi32, 1>)
          %pj:2 = scf.for %j = %c0_s to %c2_s step %c1_s iter_args(%tj = %g, %off = %c0_s) -> (!air.async.token, index) {
            %p = air.channel.put async [%tj] @b_l1[] (%l2b[%off] [%c32_s] [%c1_s]) : (memref<64xi32, 1>)
            %next = arith.addi %off, %c32_s : index
            scf.yield %p, %next : !air.async.token, index
          }
          scf.yield %pj#0 : !air.async.token
        }
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%hn=%sn) : index {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_h = arith.constant 2 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %tb, %bufb = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %wh = air.wait_all async [%ta, %tb]
          %g = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %wh) -> (!air.async.token) {
            %ga = air.channel.get async [%t] @a_l1[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            %gj = scf.for %j = %c0_h to %c2_h step %c1_h iter_args(%tj = %ga) -> (!air.async.token) {
              %gb = air.channel.get async [%tj] @b_l1[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
              scf.yield %gb : !air.async.token
            }
            scf.yield %gj : !air.async.token
          }
          %da = air.execute [%g] {
            memref.dealloc %bufa : memref<32xi32, 2>
          }
          %db = air.execute [%g] {
            memref.dealloc %bufb : memref<32xi32, 2>
          }
        }
      }
    }
    return
  }
}

// -----

// As @ab_chunks, but the memtile copies B to a second buffer and the chunk
// loop sends from that copy. The chunk loop waits for the copy, which would
// stay behind in B's loop, so the chunk loop does not move and B keeps its
// channels.

// CHECK-LABEL: func.func @chunk_loop_waits_for_copy
// CHECK: air.segment
// CHECK: %[[COPY:.*]] = air.dma_memcpy_nd
// CHECK: scf.for {{.*}} iter_args(%{{.*}} = %[[COPY]])
// CHECK: air.channel.put {{.*}} @b_l1[]
// CHECK: air.herd
// CHECK: air.channel.get {{.*}} @b_l1[

module {
  air.channel @a_in [1, 1]
  air.channel @b_in [1, 1]
  air.channel @a_l1 [1, 1]
  air.channel @b_l1 [1, 1]
  func.func @chunk_loop_waits_for_copy(%in_a: memref<64xi32>, %in_b: memref<64xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<64xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @a_in[] (%la[] [] []) : (memref<64xi32>)
        %p2 = air.channel.put async [%p] @b_in[] (%lb[] [] []) : (memref<64xi32>)
        scf.yield %p2 : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c32_s = arith.constant 32 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<64xi32, 1>) {
          %x = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %x : memref<64xi32, 1>
        }
        %tc2, %l2c = air.execute -> (memref<64xi32, 1>) {
          %x = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %x : memref<64xi32, 1>
        }
        %w = air.wait_all async [%ta2, %tb2, %tc2]
        %loop_a = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @a_in[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @a_l1[] (%l2a[] [] []) : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in[] (%l2b[] [] []) : (memref<64xi32, 1>)
          %d = air.dma_memcpy_nd async [%g] (%l2c[] [] [], %l2b[] [] []) : (memref<64xi32, 1>, memref<64xi32, 1>)
          %pj = scf.for %j = %c0_s to %c2_s step %c1_s iter_args(%tj = %d) -> (!air.async.token) {
            %off = arith.muli %j, %c32_s : index
            %p = air.channel.put async [%tj] @b_l1[] (%l2c[%off] [%c32_s] [%c1_s]) : (memref<64xi32, 1>)
            scf.yield %p : !air.async.token
          }
          scf.yield %pj : !air.async.token
        }
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%hn=%sn) : index {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_h = arith.constant 2 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %tb, %bufb = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %wh = air.wait_all async [%ta, %tb]
          %g = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %wh) -> (!air.async.token) {
            %ga = air.channel.get async [%t] @a_l1[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
            %gj = scf.for %j = %c0_h to %c2_h step %c1_h iter_args(%tj = %ga) -> (!air.async.token) {
              %gb = air.channel.get async [%tj] @b_l1[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
              scf.yield %gb : !air.async.token
            }
            scf.yield %gj : !air.async.token
          }
          %da = air.execute [%g] {
            memref.dealloc %bufa : memref<32xi32, 2>
          }
          %db = air.execute [%g] {
            memref.dealloc %bufb : memref<32xi32, 2>
          }
        }
      }
    }
    return
  }
}

// -----

// A and B each sit in two levels of loops, and B also takes its data into
// L2 in the inner loop. B's inner loop starts its offset from a value
// computed in B's outer loop. Fusing the outer loops first would put that
// value after A's inner loop, which the fused inner loop needs to start from,
// so the nests are not fused. They have the same depth, so their channels
// still merge as they are.

// CHECK-LABEL: func.func @inner_init_from_outer_body
// CHECK: air.segment
// CHECK: scf.for
// CHECK-NEXT: scf.for
// CHECK-NEXT: air.channel.put {{.*}} @a_l1[]
// CHECK: scf.for
// CHECK-NEXT: arith.muli
// CHECK-NEXT: scf.for
// CHECK-NEXT: air.channel.get {{.*}} @b_in[]
// CHECK-NEXT: air.channel.put {{.*}} @a_l1[]

module {
  air.channel @b_in [1, 1]
  air.channel @a_l1 [1, 1]
  air.channel @b_l1 [1, 1]
  func.func @inner_init_from_outer_body(%in_a: memref<64xi32>, %in_b: memref<64xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<64xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_b = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p2 = air.channel.put async [%t] @b_in[] (%lb[] [] []) : (memref<64xi32>)
        scf.yield %p2 : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c32_s = arith.constant 32 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<64xi32, 1>) {
          %x = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %x : memref<64xi32, 1>
        }
        %w = air.wait_all async [%ta2, %tb2]
        %loop_a = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %pk = scf.for %k = %c0_s to %c2_s step %c1_s iter_args(%tk = %t) -> (!air.async.token) {
            %p = air.channel.put async [%tk] @a_l1[] (%l2a[] [] []) : (memref<32xi32, 1>)
            scf.yield %p : !air.async.token
          }
          scf.yield %pk : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %start = arith.muli %i, %c32_s : index
          %pk:2 = scf.for %k = %c0_s to %c2_s step %c1_s iter_args(%tk = %t, %off = %start) -> (!air.async.token, index) {
            %g = air.channel.get async [%tk] @b_in[] (%l2b[] [] []) : (memref<64xi32, 1>)
            %p = air.channel.put async [%g] @b_l1[] (%l2b[%off] [%c32_s] [%c1_s]) : (memref<64xi32, 1>)
            %next = arith.addi %off, %c32_s : index
            scf.yield %p, %next : !air.async.token, index
          }
          scf.yield %pk#0 : !air.async.token
        }
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%hn=%sn) : index {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_h = arith.constant 2 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %tb, %bufb = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %wh = air.wait_all async [%ta, %tb]
          %g = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %wh) -> (!air.async.token) {
            %gk = scf.for %k = %c0_h to %c2_h step %c1_h iter_args(%tk = %t) -> (!air.async.token) {
              %ga = air.channel.get async [%tk] @a_l1[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
              %gb = air.channel.get async [%ga] @b_l1[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
              scf.yield %gb : !air.async.token
            }
            scf.yield %gk : !air.async.token
          }
          %da = air.execute [%g] {
            memref.dealloc %bufa : memref<32xi32, 2>
          }
          %db = air.execute [%g] {
            memref.dealloc %bufb : memref<32xi32, 2>
          }
        }
      }
    }
    return
  }
}

// -----

// As @guarded_gets, but the gets of A and of B are guarded for different
// cores. A core that takes only one of them must not find the other on its
// channel, so the channels stay apart.

// CHECK-LABEL: func.func @guards_pick_other_cores
// CHECK: air.herd
// CHECK: air.channel.get {{.*}} @a_l1[
// CHECK: air.channel.get {{.*}} @b_l1[

#col0 = affine_set<()[s0, s1] : (s0 == 0)>
#row1 = affine_set<()[s0, s1] : (s1 - 1 == 0)>
module {
  air.channel @a_in [1, 1]
  air.channel @b_in [1, 1]
  air.channel @a_l1 [1, 1]
  air.channel @b_l1 [1, 1]
  func.func @guards_pick_other_cores(%in_a: memref<64xi32>, %in_b: memref<64xi32>, %n: index) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%ln=%n, %la=%in_a, %lb=%in_b) : index, memref<64xi32>, memref<64xi32> {
      %c0 = arith.constant 0 : index
      %c1_l = arith.constant 1 : index
      %w0 = air.wait_all async
      %put_ab = scf.for %i = %c0 to %ln step %c1_l iter_args(%t = %w0) -> (!air.async.token) {
        %p = air.channel.put async [%t] @a_in[] (%la[] [] []) : (memref<64xi32>)
        %p2 = air.channel.put async [%p] @b_in[] (%lb[] [] []) : (memref<64xi32>)
        scf.yield %p2 : !air.async.token
      }
      %1 = air.segment @seg async args(%sn=%ln) : index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c32_s = arith.constant 32 : index
        %ta2, %l2a = air.execute -> (memref<32xi32, 1>) {
          %x = memref.alloc() : memref<32xi32, 1>
          air.execute_terminator %x : memref<32xi32, 1>
        }
        %tb2, %l2b = air.execute -> (memref<64xi32, 1>) {
          %x = memref.alloc() : memref<64xi32, 1>
          air.execute_terminator %x : memref<64xi32, 1>
        }
        %w = air.wait_all async [%ta2, %tb2]
        %loop_a = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @a_in[] (%l2a[] [] []) : (memref<32xi32, 1>)
          %p = air.channel.put async [%g] @a_l1[] (%l2a[] [] []) : (memref<32xi32, 1>)
          scf.yield %p : !air.async.token
        }
        %loop_b = scf.for %i = %c0_s to %sn step %c1_s iter_args(%t = %w) -> (!air.async.token) {
          %g = air.channel.get async [%t] @b_in[] (%l2b[] [] []) : (memref<64xi32, 1>)
          %pj = scf.for %j = %c0_s to %c2_s step %c1_s iter_args(%tj = %g) -> (!air.async.token) {
            %off = arith.muli %j, %c32_s : index
            %p = air.channel.put async [%tj] @b_l1[] (%l2b[%off] [%c32_s] [%c1_s]) : (memref<64xi32, 1>)
            scf.yield %p : !air.async.token
          }
          scf.yield %pj : !air.async.token
        }
        %h = air.herd @herd_0 async tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) args(%hn=%sn) : index {
          %c0_h = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2_h = arith.constant 2 : index
          %ta, %bufa = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %tb, %bufb = air.execute -> (memref<32xi32, 2>) {
            %x = memref.alloc() : memref<32xi32, 2>
            air.execute_terminator %x : memref<32xi32, 2>
          }
          %wh = air.wait_all async [%ta, %tb]
          %g = scf.for %i = %c0_h to %hn step %c1_h iter_args(%t = %wh) -> (!air.async.token) {
            %ga = affine.if #col0()[%tx, %ty] -> !air.async.token {
              %get_a = air.channel.get async [%t] @a_l1[%tx, %ty] (%bufa[] [] []) : (memref<32xi32, 2>)
              affine.yield %get_a : !air.async.token
            } else {
              affine.yield %t : !air.async.token
            }
            %gj = scf.for %j = %c0_h to %c2_h step %c1_h iter_args(%tj = %ga) -> (!air.async.token) {
              %gb = affine.if #row1()[%tx, %ty] -> !air.async.token {
                %get_b = air.channel.get async [%tj] @b_l1[%tx, %ty] (%bufb[] [] []) : (memref<32xi32, 2>)
                affine.yield %get_b : !air.async.token
              } else {
                affine.yield %tj : !air.async.token
              }
              scf.yield %gb : !air.async.token
            }
            scf.yield %gj : !air.async.token
          }
          %da = air.execute [%g] {
            memref.dealloc %bufa : memref<32xi32, 2>
          }
          %db = air.execute [%g] {
            memref.dealloc %bufb : memref<32xi32, 2>
          }
        }
      }
    }
    return
  }
}
