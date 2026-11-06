//===- event_gated.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// A dependency between two stages of work, resolved inside the kernel rather
// than by returning to the host. This is the last piece of the machinery a
// megakernel needs: without it a task graph is just a bag of independent
// tasks, and every edge in the graph costs a launch.
//
// Stage one writes t+1 into element t. Stage two doubles it. Both stages are
// drained from queues by whichever worker gets there first, so the only thing
// keeping them apart is the event: a worker that finishes stage one adds its
// tally to the event counter and then waits for the counter to reach the
// number of stage-one tasks before touching stage two.
//
// Break the wait and the result says so. A stage-two task that runs early
// doubles a zero, and the stage-one write that follows leaves t+1 rather than
// 2*(t+1), so the final value is wrong for every element that raced.
//
//===------------------------------------------------------------------===//

module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32

    %q1h = memref.alloc() : memref<257xi32>
    %q2h = memref.alloc() : memref<257xi32>
    memref.store %zero, %q1h[%c0] : memref<257xi32>
    memref.store %zero, %q2h[%c0] : memref<257xi32>
    // Stage one in one order, stage two in another, so neither stage can be
    // reproduced from a block id.
    %c7 = arith.constant 7 : index
    %c13 = arith.constant 13 : index
    scf.for %i = %c0 to %c256 step %c1 {
      %slot = arith.addi %i, %c1 : index
      %m1 = arith.muli %i, %c7 : index
      %r1 = arith.remui %m1, %c256 : index
      %r1i = arith.index_cast %r1 : index to i32
      memref.store %r1i, %q1h[%slot] : memref<257xi32>
      %m2 = arith.muli %i, %c13 : index
      %r2 = arith.remui %m2, %c256 : index
      %r2i = arith.index_cast %r2 : index to i32
      memref.store %r2i, %q2h[%slot] : memref<257xi32>
    }
    %evh = memref.alloc() : memref<4xi32>
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      memref.store %zero, %evh[%i] : memref<4xi32>
    }
    %dh = memref.alloc() : memref<256xi32>
    scf.for %i = %c0 to %c256 step %c1 {
      memref.store %zero, %dh[%i] : memref<256xi32>
    }

    %q1 = gpu.alloc () : memref<257xi32>
    %q2 = gpu.alloc () : memref<257xi32>
    %ev = gpu.alloc () : memref<4xi32>
    %d  = gpu.alloc () : memref<256xi32>
    gpu.memcpy %q1, %q1h : memref<257xi32>, memref<257xi32>
    gpu.memcpy %q2, %q2h : memref<257xi32>, memref<257xi32>
    gpu.memcpy %ev, %evh : memref<4xi32>, memref<4xi32>
    gpu.memcpy %d,  %dh  : memref<256xi32>, memref<256xi32>

    call @pipeline(%q1, %q2, %ev, %d) : (memref<257xi32>, memref<257xi32>, memref<4xi32>, memref<256xi32>) -> ()

    gpu.memcpy %dh, %d : memref<256xi32>, memref<256xi32>
    gpu.memcpy %evh, %ev : memref<4xi32>, memref<4xi32>

    %signals = memref.load %evh[%c0] : memref<4xi32>
    %contribs = memref.load %evh[%c1] : memref<4xi32>
    vector.print str "stage-one tasks signalled = "
    vector.print %signals : i32
    vector.print str "workers that contributed a tally = "
    vector.print %contribs : i32

    %wrong = scf.for %i = %c0 to %c256 step %c1
        iter_args(%w = %zero) -> (i32) {
      %v = memref.load %dh[%i] : memref<256xi32>
      %ii = arith.index_cast %i : index to i32
      %tp1 = arith.addi %ii, %one : i32
      %want = arith.muli %tp1, %two : i32
      %ok = arith.cmpi eq, %v, %want : i32
      %inc = arith.select %ok, %zero, %one : i32
      %w2 = arith.addi %w, %inc : i32
      scf.yield %w2 : i32
    }
    vector.print str "elements not 2*(t+1) = "
    vector.print %wrong : i32
    return
  }

  func.func @pipeline(%q1: memref<257xi32>, %q2: memref<257xi32>,
                      %ev: memref<4xi32>, %data: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    air.launch (%bx, %by) in (%nbx=%c32, %nby=%c1)
        args(%a1=%q1, %a2=%q2, %ae=%ev, %ad=%data)
        : memref<257xi32>, memref<257xi32>, memref<4xi32>, memref<256xi32> {
      air.segment @worker args(%s1=%a1, %s2=%a2, %se=%ae, %sd=%ad)
          : memref<257xi32>, memref<257xi32>, memref<4xi32>, memref<256xi32> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c256_s = arith.constant 256 : index
        %one_s = arith.constant 1 : i32
        %two_s = arith.constant 2 : i32
        %tri = arith.constant 256 : i32
        %true = arith.constant true
        %zero_i = arith.constant 0 : i32
        %c3_s = arith.constant 3 : index

        %tx_s = gpu.thread_id x
        %ty_s = gpu.thread_id y
        %tz_s = gpu.thread_id z
        %t01 = arith.ori %tx_s, %ty_s : index
        %t012 = arith.ori %t01, %tz_s : index
        %isLead = arith.cmpi eq, %t012, %c0_s : index

        scf.if %isLead {
          // Stage one, counting what this worker did rather than signalling
          // per task: 32 atomics on the event instead of 256.
          %tallyR:2 = scf.while (%go = %true, %acc = %zero_i) : (i1, i32) -> (i1, i32) {
            scf.condition(%go) %go, %acc : i1, i32
          } do {
          ^bb0(%g: i1, %acc: i32):
            %claimed = memref.atomic_rmw addi %one_s, %s1[%c0_s] : (i32, memref<257xi32>) -> i32
            %idx = arith.index_cast %claimed : i32 to index
            %hasWork = arith.cmpi ult, %idx, %c256_s : index
            %acc2 = scf.if %hasWork -> i32 {
              %slot = arith.addi %idx, %c1_s : index
              %target = memref.load %s1[%slot] : memref<257xi32>
              %ti = arith.index_cast %target : i32 to index
              %val = arith.addi %target, %one_s : i32
              // Hold up the task that writes element 0. Without this the race
              // the event prevents is too narrow to ever lose: the workers all
              // finish at almost the same moment, so removing the wait still
              // happens to give the right answer, and the test would be
              // passing for no reason. The delay is real work -- atomics on a
              // scratch slot -- so it cannot be optimised away.
              %isSlow = arith.cmpi eq, %target, %zero_i : i32
              scf.if %isSlow {
                %c100k = arith.constant 200000 : index
                scf.for %k = %c0_s to %c100k step %c1_s {
                  %burn = memref.atomic_rmw addi %one_s, %se[%c3_s] : (i32, memref<4xi32>) -> i32
                }
              }
              memref.store %val, %sd[%ti] : memref<256xi32>
              %a = arith.addi %acc, %one_s : i32
              scf.yield %a : i32
            } else {
              scf.yield %acc : i32
            }
            scf.yield %hasWork, %acc2 : i1, i32
          }

          // The stage-one stores must be visible before the tally that
          // announces them, so the event add is a release at system scope.
          %evbase = memref.extract_aligned_pointer_as_index %se : memref<4xi32> -> index
          %evi = arith.index_cast %evbase : index to i64
          %evptr = llvm.inttoptr %evi : i64 to !llvm.ptr
          %prev = llvm.atomicrmw add %evptr, %tallyR#1 syncscope("") release : !llvm.ptr, i32
          %nc = memref.atomic_rmw addi %one_s, %se[%c1_s] : (i32, memref<4xi32>) -> i32

          // The edge itself: nobody touches stage two until the event says all
          // of stage one is done.
          scf.while : () -> () {
            %seen = llvm.load %evptr atomic syncscope("") acquire {alignment = 4 : i64} : !llvm.ptr -> i32
            %notYet = arith.cmpi ult, %seen, %tri : i32
            scf.condition(%notYet)
          } do {
            scf.yield
          }

          // Stage two.
          %done2 = scf.while (%go2 = %true) : (i1) -> i1 {
            scf.condition(%go2) %go2 : i1
          } do {
          ^bb0(%g2: i1):
            %claimed2 = memref.atomic_rmw addi %one_s, %s2[%c0_s] : (i32, memref<257xi32>) -> i32
            %idx2 = arith.index_cast %claimed2 : i32 to index
            %hasWork2 = arith.cmpi ult, %idx2, %c256_s : index
            scf.if %hasWork2 {
              %slot2 = arith.addi %idx2, %c1_s : index
              %target2 = memref.load %s2[%slot2] : memref<257xi32>
              %ti2 = arith.index_cast %target2 : i32 to index
              %cur = memref.load %sd[%ti2] : memref<256xi32>
              %dbl = arith.muli %cur, %two_s : i32
              memref.store %dbl, %sd[%ti2] : memref<256xi32>
            }
            scf.yield %hasWork2 : i1
          }
        }

        air.herd @herd tile (%tx, %ty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
