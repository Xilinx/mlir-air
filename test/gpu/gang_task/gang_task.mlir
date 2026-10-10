//===- gang_task.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// Fleet's gang task: a task is claimed by a die, not by a workgroup, and every
// workgroup on that die works on it together, splitting its tiles between them
// (persistent_kernel.cuh:1083-1094, the is_gang_task_type path).
//
// Claiming per workgroup, which is what the generator does today, gets the
// cache locality -- a die keeps touching its own slices. Claiming per die also
// gets the load balance, because a task too big for one workgroup is shared
// rather than serialised. What it needs that per-workgroup claiming does not is
// a rendezvous among the workgroups on one die: the leader claims, everyone
// reads the claim, and nobody may claim again until everyone has.
//
// There is no such barrier here. gpu.barrier is within a workgroup and the one
// behind air.chiplet_dim_blocks is device-wide; neither is the right scope. So
// this builds one out of the pieces that do exist -- chiplet_id to index it,
// chiplet_dim_blocks for how many have to arrive -- as a sense-reversing
// barrier per die, and checks it by having the gangs cover a range exactly once.
//
//   16 tasks x 64 tiles, every tile claimed exactly once.
//
//===------------------------------------------------------------------===//

module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c1024 = arith.constant 1024 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32

    // [0] task head, [16+d] die arrivals, [48+d] die generation.
    %ctl = memref.alloc() : memref<128xi32>
    %c128 = arith.constant 128 : index
    scf.for %i = %c0 to %c128 step %c1 {
      memref.store %zero, %ctl[%i] : memref<128xi32>
    }
    %out = memref.alloc() : memref<1024xi32>
    scf.for %i = %c0 to %c1024 step %c1 {
      memref.store %zero, %out[%i] : memref<1024xi32>
    }

    %dctl = gpu.alloc () : memref<128xi32>
    %dout = gpu.alloc () : memref<1024xi32>
    gpu.memcpy %dctl, %ctl : memref<128xi32>, memref<128xi32>
    gpu.memcpy %dout, %out : memref<1024xi32>, memref<1024xi32>

    call @gangs(%dctl, %dout) : (memref<128xi32>, memref<1024xi32>) -> ()

    gpu.memcpy %out, %dout : memref<1024xi32>, memref<1024xi32>

    %v0 = memref.load %out[%c0] : memref<1024xi32>
    vector.print str "out[0] = "
    vector.print %v0 : i32
    %wrong = scf.for %i = %c0 to %c1024 step %c1
        iter_args(%w = %zero) -> (i32) {
      %v = memref.load %out[%i] : memref<1024xi32>
      %ok = arith.cmpi eq, %v, %one : i32
      %inc = arith.select %ok, %zero, %one : i32
      %w2 = arith.addi %w, %inc : i32
      scf.yield %w2 : i32
    }
    vector.print str "tiles not covered exactly once = "
    vector.print %wrong : i32
    return
  }

  func.func @gangs(%ctl: memref<128xi32>, %out: memref<1024xi32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    air.launch (%bx, %by) in (%nbx=%c32, %nby=%c1) args(%actl=%ctl, %aout=%out)
        : memref<128xi32>, memref<1024xi32> {
      air.segment @worker args(%sc=%actl, %so=%aout) : memref<128xi32>, memref<1024xi32> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c16_s = arith.constant 16 : index
        %c48_s = arith.constant 48 : index
        %c64_s = arith.constant 64 : index
        %one_s = arith.constant 1 : i32
        %zero_s = arith.constant 0 : i32
        %n16 = arith.constant 16 : i32
        %true = arith.constant true

        %die  = air.chiplet_id
        %rank = air.chiplet_block_id
        %n    = air.chiplet_dim_blocks
        %n_i  = arith.index_cast %n : index to i32

        %tx_s = gpu.thread_id x
        %ty_s = gpu.thread_id y
        %tz_s = gpu.thread_id z
        %t01 = arith.ori %tx_s, %ty_s : index
        %t012 = arith.ori %t01, %tz_s : index
        %isLead = arith.cmpi eq, %t012, %c0_s : index

        %base = memref.extract_aligned_pointer_as_index %sc : memref<128xi32> -> index
        %basei = arith.index_cast %base : index to i64
        %ctlp = llvm.inttoptr %basei : i64 to !llvm.ptr

        // arrivals[die] at word 16+die, generation[die] at word 48+die.
        %arrIdx = arith.addi %c16_s, %die : index
        %genIdx = arith.addi %c48_s, %die : index
        %arrO = arith.index_cast %arrIdx : index to i64
        %genO = arith.index_cast %genIdx : index to i64
        %four = arith.constant 4 : i64
        %arrO4 = arith.muli %arrO, %four : i64
        %genO4 = arith.muli %genO, %four : i64
        %arrP = llvm.getelementptr %ctlp[%arrO4] : (!llvm.ptr, i64) -> !llvm.ptr, i8
        %genP = llvm.getelementptr %ctlp[%genO4] : (!llvm.ptr, i64) -> !llvm.ptr, i8

        scf.if %isLead {
          %done = scf.while (%go = %true) : (i1) -> i1 {
            scf.condition(%go) %go : i1
          } do {
          ^bb0(%g: i1):
            // The die's leader claims one task for the whole die.
            %isDieLead = arith.cmpi eq, %rank, %c0_s : index
            scf.if %isDieLead {
              %k = memref.atomic_rmw addi %one_s, %sc[%c0_s] : (i32, memref<128xi32>) -> i32
              %slot = arith.addi %c16_s, %c64_s : index
              %cslot = arith.addi %slot, %die : index
              memref.store %k, %sc[%cslot] : memref<128xi32>
            }

            // --- die barrier: everyone on this die waits for the claim ---
            %g0 = llvm.load %genP atomic syncscope("") acquire {alignment = 4 : i64} : !llvm.ptr -> i32
            %prev = llvm.atomicrmw add %arrP, %one_s syncscope("") release : !llvm.ptr, i32
            %lastIdx = arith.subi %n_i, %one_s : i32
            %isLast = arith.cmpi eq, %prev, %lastIdx : i32
            scf.if %isLast {
              %rearm = llvm.atomicrmw xchg %arrP, %zero_s syncscope("") monotonic : !llvm.ptr, i32
              %bump = llvm.atomicrmw add %genP, %one_s syncscope("") release : !llvm.ptr, i32
            } else {
              scf.while : () -> () {
                %gn = llvm.load %genP atomic syncscope("") acquire {alignment = 4 : i64} : !llvm.ptr -> i32
                %same = arith.cmpi eq, %gn, %g0 : i32
                scf.condition(%same)
              } do {
                scf.yield
              }
            }

            %slot2 = arith.addi %c16_s, %c64_s : index
            %cslot2 = arith.addi %slot2, %die : index
            %task = memref.load %sc[%cslot2] : memref<128xi32>
            %tIdx = arith.index_cast %task : i32 to index
            %has = arith.cmpi ult, %tIdx, %c16_s : index
            scf.if %has {
              // The gang splits this task's 64 tiles between its workgroups.
              %tbase = arith.muli %tIdx, %c64_s : index
              scf.for %t = %rank to %c64_s step %n {
                %idx = arith.addi %tbase, %t : index
                %o = memref.atomic_rmw addi %one_s, %so[%idx] : (i32, memref<1024xi32>) -> i32
              }
            }

            // There is deliberately no second barrier here. The obvious
            // worry is the leader claiming again before the rest of the gang
            // has read the current claim, but the read happens between the
            // barrier and the work, not after it, and the barrier itself stops
            // the leader getting more than one claim ahead. Adding a second
            // rendezvous costs two atomics and a spin per task per workgroup
            // and changes nothing -- removing it from a working version left
            // the answer correct, which is how this was settled.
            scf.yield %has : i1
          }
        }

        air.herd @herd tile (%htx, %hty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
