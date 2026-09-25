//===- task_queue.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// The dispatch loop a megakernel is built around, running: workgroups that
// stay resident and pull work out of a shared queue until it is empty, instead
// of the host launching a kernel per piece of work.
//
// What has to be true for that to be a real dispatch loop rather than a fixed
// partition: the work a workgroup gets is decided at run time, by whoever
// reaches the queue first, so the assignment is not a function of block id.
// Every task must still be executed exactly once, and every worker must leave.
//
//   queue[0]     the head, bumped atomically to claim a task
//   queue[1+i]   task i: the element it says to touch
//   out[t]       how many times element t was touched -- must be exactly 1
//
//===------------------------------------------------------------------===//

module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %c257 = arith.constant 257 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32

    // 256 tasks, task i asks for element (i * 7) mod 256 so the order in the
    // queue is not the order of the data -- a worker that ignored the queue
    // and used its block id would not reproduce this.
    %c7 = arith.constant 7 : index
    %qhost = memref.alloc() : memref<257xi32>
    memref.store %zero, %qhost[%c0] : memref<257xi32>
    scf.for %i = %c0 to %c256 step %c1 {
      %m = arith.muli %i, %c7 : index
      %r = arith.remui %m, %c256 : index
      %ri = arith.index_cast %r : index to i32
      %slot = arith.addi %i, %c1 : index
      memref.store %ri, %qhost[%slot] : memref<257xi32>
    }
    %ohost = memref.alloc() : memref<256xi32>
    scf.for %i = %c0 to %c256 step %c1 {
      memref.store %zero, %ohost[%i] : memref<256xi32>
    }

    %qdev = gpu.alloc () : memref<257xi32>
    gpu.memcpy %qdev, %qhost : memref<257xi32>, memref<257xi32>
    %odev = gpu.alloc () : memref<256xi32>
    gpu.memcpy %odev, %ohost : memref<256xi32>, memref<256xi32>

    call @dispatch(%qdev, %odev) : (memref<257xi32>, memref<256xi32>) -> ()

    gpu.memcpy %ohost, %odev : memref<256xi32>, memref<256xi32>
    gpu.memcpy %qhost, %qdev : memref<257xi32>, memref<257xi32>

    %head = memref.load %qhost[%c0] : memref<257xi32>
    vector.print str "queue head after drain = "
    vector.print %head : i32

    %wrong = scf.for %i = %c0 to %c256 step %c1
        iter_args(%w = %zero) -> (i32) {
      %v = memref.load %ohost[%i] : memref<256xi32>
      %ok = arith.cmpi eq, %v, %one : i32
      %inc = arith.select %ok, %zero, %one : i32
      %w2 = arith.addi %w, %inc : i32
      scf.yield %w2 : i32
    }
    vector.print str "tasks not executed exactly once = "
    vector.print %wrong : i32
    return
  }

  func.func @dispatch(%queue: memref<257xi32>, %out: memref<256xi32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    air.launch (%bx, %by) in (%nbx=%c32, %nby=%c1) args(%q=%queue, %o=%out) : memref<257xi32>, memref<256xi32> {
      air.segment @worker args(%sq=%q, %so=%o) : memref<257xi32>, memref<256xi32> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c256_s = arith.constant 256 : index
        %one_s = arith.constant 1 : i32
        %true = arith.constant true

        %tx_s = gpu.thread_id x
        %ty_s = gpu.thread_id y
        %tz_s = gpu.thread_id z
        %t01 = arith.ori %tx_s, %ty_s : index
        %t012 = arith.ori %t01, %tz_s : index
        %isLead = arith.cmpi eq, %t012, %c0_s : index

        scf.if %isLead {
          // Poll until the queue is empty. This is the loop a megakernel never
          // leaves until it is told to; here the sentinel is simply running out
          // of tasks.
          %done = scf.while (%go = %true) : (i1) -> i1 {
            scf.condition(%go) %go : i1
          } do {
          ^bb0(%g: i1):
            %claimed = memref.atomic_rmw addi %one_s, %sq[%c0_s] : (i32, memref<257xi32>) -> i32
            %idx = arith.index_cast %claimed : i32 to index
            %hasWork = arith.cmpi ult, %idx, %c256_s : index
            scf.if %hasWork {
              %slot = arith.addi %idx, %c1_s : index
              %target = memref.load %sq[%slot] : memref<257xi32>
              %ti = arith.index_cast %target : i32 to index
              %old = memref.atomic_rmw addi %one_s, %so[%ti] : (i32, memref<256xi32>) -> i32
            }
            scf.yield %hasWork : i1
          }
        }

        air.herd @herd tile (%tx, %ty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
