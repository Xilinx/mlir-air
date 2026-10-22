//===- two_level_signal.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// Fleet's two-level event counting (persistent_kernel.cuh:1226-1251), running.
//
// When a worker finishes a task it has to tell whoever is waiting. Done
// directly, every worker pays a device-scope atomic and the writeback that
// goes with it. The two-level version has the workers on a die accumulate into
// a counter only that die touches, and only the last one out flushes the die's
// whole share to the device counter.
//
// The instructions are the same either way -- the lit tests cover which ones
// get emitted. What changes is how many times the expensive one runs: once per
// die instead of once per worker. That is a control-flow property, and this
// measures it by counting the flushes.
//
//   out[0]   the device-wide event counter, flushed at system scope
//   out[1]   how many flushes happened
//   out[2+d] die d's local counter, touched only by workers on die d
//
//===------------------------------------------------------------------===//

module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %workers = arith.constant 32 : i32

    %host = memref.alloc() : memref<64xi32>
    scf.for %i = %c0 to %c64 step %c1 {
      memref.store %zero, %host[%i] : memref<64xi32>
    }
    %dev = gpu.alloc () : memref<64xi32>
    gpu.memcpy %dev, %host : memref<64xi32>, memref<64xi32>

    call @signal(%dev) : (memref<64xi32>) -> ()

    gpu.memcpy %host, %dev : memref<64xi32>, memref<64xi32>

    %total = memref.load %host[%c0] : memref<64xi32>
    %flushes = memref.load %host[%c1] : memref<64xi32>

    // Count the dies that reported, rather than assuming a part with 8 of them.
    %dies = scf.for %d = %c2 to %c64 step %c1
        iter_args(%acc = %zero) -> (i32) {
      %v = memref.load %host[%d] : memref<64xi32>
      %nz = arith.cmpi ne, %v, %zero : i32
      %inc = arith.select %nz, %one, %zero : i32
      %a = arith.addi %acc, %inc : i32
      scf.yield %a : i32
    }

    vector.print str "workers signalled = "
    vector.print %total : i32
    vector.print str "dies that reported = "
    vector.print %dies : i32
    vector.print str "device-scope flushes = "
    vector.print %flushes : i32

    // Nothing lost: every worker is accounted for in the device counter.
    %lostOk = arith.cmpi eq, %total, %workers : i32
    %lost = arith.select %lostOk, %zero, %one : i32
    vector.print str "signals lost = "
    vector.print %lost : i32

    // And the point of the arrangement: one flush per die, not one per worker.
    %flushOk = arith.cmpi eq, %flushes, %dies : i32
    %flushBad = arith.select %flushOk, %zero, %one : i32
    vector.print str "flushes not equal to die count = "
    vector.print %flushBad : i32
    return
  }

  func.func @signal(%out: memref<64xi32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    air.launch (%bx, %by) in (%nbx=%c32, %nby=%c1) args(%o=%out) : memref<64xi32> {
      air.segment @worker args(%so=%o) : memref<64xi32> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %one_s = arith.constant 1 : i32

        %xcd  = air.chiplet_id
        %n    = air.chiplet_dim_blocks

        %tx_s = gpu.thread_id x
        %ty_s = gpu.thread_id y
        %tz_s = gpu.thread_id z
        %t01 = arith.ori %tx_s, %ty_s : index
        %t012 = arith.ori %t01, %tz_s : index
        %isLead = arith.cmpi eq, %t012, %c0_s : index

        scf.if %isLead {
          // Level one: a counter only this die's workers touch, so it can live
          // in the die's own cache.
          %slot = arith.addi %xcd, %c2_s : index
          %prev = memref.atomic_rmw addi %one_s, %so[%slot] : (i32, memref<64xi32>) -> i32
          %mine = arith.addi %prev, %one_s : i32
          %n_i = arith.index_cast %n : index to i32
          %isLast = arith.cmpi eq, %mine, %n_i : i32

          scf.if %isLast {
            // Level two: the die's whole share, once, at system scope. This is
            // the atomic the arrangement exists to make rare.
            %base = memref.extract_aligned_pointer_as_index %so : memref<64xi32> -> index
            %basei = arith.index_cast %base : index to i64
            %ptr = llvm.inttoptr %basei : i64 to !llvm.ptr
            %old = llvm.atomicrmw add %ptr, %n_i syncscope("") release : !llvm.ptr, i32
            %prevf = memref.atomic_rmw addi %one_s, %so[%c1_s] : (i32, memref<64xi32>) -> i32
          }
        }

        air.herd @herd tile (%tx, %ty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
