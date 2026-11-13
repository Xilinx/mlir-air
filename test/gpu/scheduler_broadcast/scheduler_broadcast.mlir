//===- scheduler_broadcast.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// How Fleet actually makes the workgroups of a die agree on a task: it does not
// make them meet. Each die elects a scheduler, each worker owns a private
// queue, and the scheduler writes the same task id into every queue on its die
// (persistent_kernel.cuh:1644, "Broadcast the gang task to dispatch_count
// workers"). Workers pop from their own queue and never consult each other.
//
// test/gpu/gang_task does the same job the other way round -- a leader claims
// and the die rendezvouses -- and that works standalone but deadlocks once the
// workgroups of a die can be at different stages. This one cannot deadlock the
// same way, because after startup no workgroup ever waits on another.
//
// Role election follows Fleet's fused kernel (:1868-1882): thread 0 reads the
// die, tries to claim it with a compare-and-swap, and becomes that die's
// scheduler if it won.
//
//   ctl[0]        how many blocks have registered
//   ctl[8+d]      the block that claimed die d as scheduler, -1 if none
//   map[b]        the die block b is on, plus one; 0 means not yet registered
//   ready[b]      how many tasks have been published to block b
//   queue[b*Q+i]  block b's private queue
//   count[t]      how many workers executed task t
//   expect[t]     how many the scheduler published it to
//
//===------------------------------------------------------------------===//

module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %neg1 = arith.constant -1 : i32

    // ctl[0] registered, ctl[8+d] scheduler claim for die d
    %ctl = memref.alloc() : memref<64xi32>
    scf.for %i = %c0 to %c64 step %c1 {
      memref.store %neg1, %ctl[%i] : memref<64xi32>
    }
    memref.store %zero, %ctl[%c0] : memref<64xi32>

    %map = memref.alloc() : memref<32xi32>
    %ready = memref.alloc() : memref<32xi32>
    scf.for %i = %c0 to %c32 step %c1 {
      memref.store %zero, %map[%i] : memref<32xi32>
      memref.store %zero, %ready[%i] : memref<32xi32>
    }
    %queue = memref.alloc() : memref<256xi32>
    scf.for %i = %c0 to %c256 step %c1 {
      memref.store %zero, %queue[%i] : memref<256xi32>
    }
    %count = memref.alloc() : memref<64xi32>
    %expect = memref.alloc() : memref<64xi32>
    scf.for %i = %c0 to %c64 step %c1 {
      memref.store %zero, %count[%i] : memref<64xi32>
      memref.store %zero, %expect[%i] : memref<64xi32>
    }

    %dctl = gpu.alloc () : memref<64xi32>
    %dmap = gpu.alloc () : memref<32xi32>
    %dready = gpu.alloc () : memref<32xi32>
    %dqueue = gpu.alloc () : memref<256xi32>
    %dcount = gpu.alloc () : memref<64xi32>
    %dexpect = gpu.alloc () : memref<64xi32>
    gpu.memcpy %dctl, %ctl : memref<64xi32>, memref<64xi32>
    gpu.memcpy %dmap, %map : memref<32xi32>, memref<32xi32>
    gpu.memcpy %dready, %ready : memref<32xi32>, memref<32xi32>
    gpu.memcpy %dqueue, %queue : memref<256xi32>, memref<256xi32>
    gpu.memcpy %dcount, %count : memref<64xi32>, memref<64xi32>
    gpu.memcpy %dexpect, %expect : memref<64xi32>, memref<64xi32>

    call @roles(%dctl, %dmap, %dready, %dqueue, %dcount, %dexpect)
      : (memref<64xi32>, memref<32xi32>, memref<32xi32>, memref<256xi32>,
         memref<64xi32>, memref<64xi32>) -> ()

    gpu.memcpy %count, %dcount : memref<64xi32>, memref<64xi32>
    gpu.memcpy %expect, %dexpect : memref<64xi32>, memref<64xi32>
    gpu.memcpy %ctl, %dctl : memref<64xi32>, memref<64xi32>

    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %sched = scf.for %d = %c0 to %c16 step %c1
        iter_args(%n = %zero) -> (i32) {
      %idx = arith.addi %c8, %d : index
      %v = memref.load %ctl[%idx] : memref<64xi32>
      %claimed = arith.cmpi ne, %v, %neg1 : i32
      %inc = arith.select %claimed, %one, %zero : i32
      %n2 = arith.addi %n, %inc : i32
      scf.yield %n2 : i32
    }
    vector.print str "dies that elected a scheduler = "
    vector.print %sched : i32

    // Guard against a vacuous pass: if nothing was published, "counts match"
    // is true of a program that did nothing.
    %published = scf.for %t = %c0 to %c64 step %c1
        iter_args(%p = %zero) -> (i32) {
      %e = memref.load %expect[%t] : memref<64xi32>
      %p2 = arith.addi %p, %e : i32
      scf.yield %p2 : i32
    }
    vector.print str "worker-task pairs published = "
    vector.print %published : i32
    %none = arith.cmpi eq, %published, %zero : i32
    %vacuous = arith.select %none, %one, %zero : i32
    vector.print str "nothing was published (1 means this run proved nothing) = "
    vector.print %vacuous : i32

    // Every published task was executed by exactly the workers it went to.
    %bad = scf.for %t = %c0 to %c64 step %c1
        iter_args(%b = %zero) -> (i32) {
      %c = memref.load %count[%t] : memref<64xi32>
      %e = memref.load %expect[%t] : memref<64xi32>
      %ok = arith.cmpi eq, %c, %e : i32
      %inc = arith.select %ok, %zero, %one : i32
      %b2 = arith.addi %b, %inc : i32
      scf.yield %b2 : i32
    }
    vector.print str "tasks whose execution count differs from what was published = "
    vector.print %bad : i32
    return
  }

  func.func @roles(%ctl: memref<64xi32>, %map: memref<32xi32>,
                   %ready: memref<32xi32>, %queue: memref<256xi32>,
                   %count: memref<64xi32>, %expect: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    air.launch (%bx, %by) in (%nbx=%c32, %nby=%c1)
        args(%actl=%ctl, %amap=%map, %ardy=%ready, %aq=%queue, %ac=%count, %ae=%expect)
        : memref<64xi32>, memref<32xi32>, memref<32xi32>, memref<256xi32>,
          memref<64xi32>, memref<64xi32> {
      air.segment @block args(%sctl=%actl, %smap=%amap, %srdy=%ardy, %sq=%aq, %sc=%ac, %se=%ae)
          : memref<64xi32>, memref<32xi32>, memref<32xi32>, memref<256xi32>,
            memref<64xi32>, memref<64xi32> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c4_s = arith.constant 4 : index
        %c8_s = arith.constant 8 : index
        %c32_s = arith.constant 32 : index
        %one_s = arith.constant 1 : i32
        %zero_s = arith.constant 0 : i32
        %neg1_s = arith.constant -1 : i32
        %n32_s = arith.constant 32 : i32
        %c4_i = arith.constant 4 : i32
        %true = arith.constant true

        %die = air.chiplet_id
        %bid = gpu.block_id x

        %tx_s = gpu.thread_id x
        %ty_s = gpu.thread_id y
        %tz_s = gpu.thread_id z
        %t01 = arith.ori %tx_s, %ty_s : index
        %t012 = arith.ori %t01, %tz_s : index
        %isLead = arith.cmpi eq, %t012, %c0_s : index

        %cbase = memref.extract_aligned_pointer_as_index %sctl : memref<64xi32> -> index
        %cbi = arith.index_cast %cbase : index to i64
        %cp = llvm.inttoptr %cbi : i64 to !llvm.ptr
        %four = arith.constant 4 : i64

        scf.if %isLead {
          // Register which die this block is on, then announce it. Release, so
          // a scheduler that sees the count sees the map entry too.
          %d1 = arith.index_cast %die : index to i32
          %dplus = arith.addi %d1, %one_s : i32
          memref.store %dplus, %smap[%bid] : memref<32xi32>
          %reg = llvm.atomicrmw add %cp, %one_s syncscope("") release : !llvm.ptr, i32

          // Claim this die as its scheduler. Whoever gets there first wins;
          // nobody waits on the outcome.
          %slot = arith.addi %c8_s, %die : index
          %soff = arith.index_cast %slot : index to i64
          %sb = arith.muli %soff, %four : i64
          %sp = llvm.getelementptr %cp[%sb] : (!llvm.ptr, i64) -> !llvm.ptr, i8
          %bid_i = arith.index_cast %bid : index to i32
          %cas = llvm.cmpxchg %sp, %neg1_s, %bid_i acq_rel monotonic : !llvm.ptr, i32
          %won = llvm.extractvalue %cas[1] : !llvm.struct<(i32, i1)>

          scf.if %won {
            // Scheduler for this die. Wait once, at startup, for every block to
            // have registered -- the same wait Fleet's scheduler does at
            // persistent_kernel.cuh:1423 -- then read off which blocks share
            // this die.
            scf.while : () -> () {
              %seen = llvm.load %cp atomic syncscope("") acquire {alignment = 4 : i64} : !llvm.ptr -> i32
              %notYet = arith.cmpi ult, %seen, %n32_s : i32
              scf.condition(%notYet)
            } do {
              scf.yield
            }

            // Publish four gang tasks to every worker on this die. The task ids
            // are die-local so the check below can tell them apart.
            %dl = arith.index_cast %die : index to i32
            %tbase0 = arith.muli %dl, %c4_i : i32
            scf.for %j = %c0_s to %c4_s step %c1_s {
              %ji = arith.index_cast %j : index to i32
              %tid = arith.addi %tbase0, %ji : i32
              %tidx = arith.index_cast %tid : i32 to index
              // Count the workers it goes to, so the host can check that every
              // one of them ran it.
              %nw = scf.for %b = %c0_s to %c32_s step %c1_s
                  iter_args(%acc = %zero_s) -> (i32) {
                %mv = memref.load %smap[%b] : memref<32xi32>
                %sameDie = arith.cmpi eq, %mv, %dplus : i32
                %notMe = arith.cmpi ne, %b, %bid : index
                %both = arith.andi %sameDie, %notMe : i1
                %acc2 = scf.if %both -> i32 {
                  // Publish into this worker's own queue, then make it visible.
                  %qslot = arith.muli %b, %c8_s : index
                  %qpos = arith.addi %qslot, %j : index
                  memref.store %tid, %sq[%qpos] : memref<256xi32>
                  %bump = memref.atomic_rmw addi %one_s, %srdy[%b] : (i32, memref<32xi32>) -> i32
                  %a = arith.addi %acc, %one_s : i32
                  scf.yield %a : i32
                } else {
                  scf.yield %acc : i32
                }
                scf.yield %acc2 : i32
              }
              memref.store %nw, %se[%tidx] : memref<64xi32>
            }
          } else {
            // Worker. Poll its own queue and run what it finds. It never waits
            // on another workgroup.
            %done = scf.for %j = %c0_s to %c4_s step %c1_s
                iter_args(%acc = %zero_s) -> (i32) {
              %want = arith.index_cast %j : index to i32
              %want1 = arith.addi %want, %one_s : i32
              scf.while : () -> () {
                %r = memref.atomic_rmw addi %zero_s, %srdy[%bid] : (i32, memref<32xi32>) -> i32
                %notYet = arith.cmpi ult, %r, %want1 : i32
                scf.condition(%notYet)
              } do {
                scf.yield
              }
              %qslot = arith.muli %bid, %c8_s : index
              %qpos = arith.addi %qslot, %j : index
              %tid = memref.load %sq[%qpos] : memref<256xi32>
              %tidx = arith.index_cast %tid : i32 to index
              %old = memref.atomic_rmw addi %one_s, %sc[%tidx] : (i32, memref<64xi32>) -> i32
              %a = arith.addi %acc, %one_s : i32
              scf.yield %a : i32
            }
          }
        }

        air.herd @herd tile (%htx, %hty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
