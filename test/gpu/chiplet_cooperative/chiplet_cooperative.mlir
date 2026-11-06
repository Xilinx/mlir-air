//===- chiplet_cooperative.mlir -----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// The reason for knowing a workgroup's rank on its die is to split a task's
// tiles between the workgroups that share that die, so the tiles they touch
// stay in the die's own cache. That only works if the split is exact: a gap
// means a tile nobody computed, an overlap means two workgroups writing the
// same result.
//
// So check exactness directly. The workgroups on die 0 walk a shared range,
// worker r of n taking every n-th element from r, and each bumps a counter at
// the element it claims. Every counter must end at exactly 1 -- 0 is a gap, 2
// is an overlap. Workgroups on the other dies take part in the reporting
// barrier and then stand down, which is what makes the expected value exactly
// 1 and independent of how many dies the part has.
//
//===------------------------------------------------------------------===//

module {
  func.func private @printMemrefI32(memref<*xi32>)

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32

    %host = memref.alloc() : memref<1024xi32>
    scf.for %i = %c0 to %c1024 step %c1 {
      memref.store %zero, %host[%i] : memref<1024xi32>
    }
    %dev = gpu.alloc () : memref<1024xi32>
    gpu.memcpy %dev, %host : memref<1024xi32>, memref<1024xi32>

    call @cooperate(%dev) : (memref<1024xi32>) -> ()

    gpu.memcpy %host, %dev : memref<1024xi32>, memref<1024xi32>

    %v0 = memref.load %host[%c0] : memref<1024xi32>
    vector.print str "out[0] = "
    vector.print %v0 : i32

    // Every element claimed exactly once: count the ones that are not.
    %gaps = scf.for %i = %c0 to %c1024 step %c1
        iter_args(%g = %zero) -> (i32) {
      %v = memref.load %host[%i] : memref<1024xi32>
      %ok = arith.cmpi eq, %v, %one : i32
      %inc = arith.select %ok, %zero, %one : i32
      %g2 = arith.addi %g, %inc : i32
      scf.yield %g2 : i32
    }
    vector.print str "elements not claimed exactly once = "
    vector.print %gaps : i32
    return
  }

  func.func @cooperate(%out: memref<1024xi32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    air.launch (%bx, %by) in (%nbx=%c32, %nby=%c1) args(%o=%out) : memref<1024xi32> {
      air.segment @gang args(%so=%o) : memref<1024xi32> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c1024_s = arith.constant 1024 : index
        %one_s = arith.constant 1 : i32

        // Every workgroup reports in -- the barrier needs all of them --
        // before any of them decides whether it has work.
        %xcd  = air.chiplet_id
        %rank = air.chiplet_block_id
        %n    = air.chiplet_dim_blocks

        // An air.segment body is the workgroup's program, and on a GPU that
        // means every thread of the workgroup runs it -- there is no implicit
        // "once per workgroup". Claiming an element is a scalar side effect,
        // so it has to be guarded, the same way Fleet writes
        // `if (threadIdx.x == 0)`. Without the guard every counter lands on
        // blockDim instead of 1.
        %tx_s = gpu.thread_id x
        %ty_s = gpu.thread_id y
        %tz_s = gpu.thread_id z
        %t01 = arith.ori %tx_s, %ty_s : index
        %t012 = arith.ori %t01, %tz_s : index
        %isLead = arith.cmpi eq, %t012, %c0_s : index
        %isDieZero = arith.cmpi eq, %xcd, %c0_s : index
        %claims = arith.andi %isDieZero, %isLead : i1
        scf.if %claims {
          // Worker %rank of %n walks every %n-th element from its own rank.
          scf.for %t = %rank to %c1024_s step %n {
            %old = memref.atomic_rmw addi %one_s, %so[%t] : (i32, memref<1024xi32>) -> i32
          }
        }

        air.herd @herd tile (%tx, %ty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
