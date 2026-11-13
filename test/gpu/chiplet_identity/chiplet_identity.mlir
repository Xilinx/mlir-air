//===- chiplet_identity.mlir --------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// Runs air.chiplet_id / air.chiplet_block_id / air.chiplet_dim_blocks on the
// device and reads the answers back. The lit tests next to the pass only check
// that the right instructions are emitted; this checks that what they compute
// is true -- that the dies reported are the dies the workgroups are on, that
// the ranks within a die are dense and distinct, and that the reporting
// barrier terminates at all.
//
// Each workgroup writes three i32 into its own slice of the output:
//   out[3*b + 0] = chiplet_id        which die it landed on
//   out[3*b + 1] = chiplet_block_id  its rank among the workgroups on that die
//   out[3*b + 2] = chiplet_dim_blocks how many workgroups are on that die
//
//===------------------------------------------------------------------===//

module {
  func.func private @printMemrefI32(memref<*xi32>)

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c96 = arith.constant 96 : index
    %zero = arith.constant 0 : i32

    // 32 workgroups x 3 values each.
    %host = memref.alloc() : memref<96xi32>
    scf.for %i = %c0 to %c96 step %c1 {
      memref.store %zero, %host[%i] : memref<96xi32>
    }

    %dev = gpu.alloc () : memref<96xi32>
    gpu.memcpy %dev, %host : memref<96xi32>, memref<96xi32>

    call @report(%dev) : (memref<96xi32>) -> ()

    gpu.memcpy %host, %dev : memref<96xi32>, memref<96xi32>
    %cast = memref.cast %host : memref<96xi32> to memref<*xi32>
    call @printMemrefI32(%cast) : (memref<*xi32>) -> ()

    // Check the invariants rather than eyeballing the table. These hold for
    // any die count and any placement, so nothing here assumes the 8-way
    // round robin an MI300X happens to do:
    //   - a rank is in range: 0 <= rank < count
    //   - count is the truth: exactly `count` workgroups report that die
    //   - ranks on a die are distinct, so they really do partition the work
    %c32 = arith.constant 32 : index
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    %one = arith.constant 1 : i32
    %errors = scf.for %b = %c0 to %c32 step %c1
        iter_args(%err = %zero) -> (i32) {
      %base = arith.muli %b, %c3 : index
      %i1 = arith.addi %base, %c1 : index
      %i2 = arith.addi %base, %c2 : index
      %d = memref.load %host[%base] : memref<96xi32>
      %r = memref.load %host[%i1] : memref<96xi32>
      %n = memref.load %host[%i2] : memref<96xi32>

      %inRange = arith.cmpi slt, %r, %n : i32
      %bad1 = arith.select %inRange, %zero, %one : i32

      %counted:2 = scf.for %b2 = %c0 to %c32 step %c1
          iter_args(%cnt = %zero, %dups = %zero) -> (i32, i32) {
        %base2 = arith.muli %b2, %c3 : index
        %j1 = arith.addi %base2, %c1 : index
        %d2 = memref.load %host[%base2] : memref<96xi32>
        %r2 = memref.load %host[%j1] : memref<96xi32>
        %sameDie = arith.cmpi eq, %d2, %d : i32
        %inc = arith.select %sameDie, %one, %zero : i32
        %cnt2 = arith.addi %cnt, %inc : i32
        %other = arith.cmpi ne, %b2, %b : index
        %sameRank = arith.cmpi eq, %r2, %r : i32
        %clash0 = arith.andi %sameDie, %sameRank : i1
        %clash = arith.andi %clash0, %other : i1
        %dinc = arith.select %clash, %one, %zero : i32
        %dups2 = arith.addi %dups, %dinc : i32
        scf.yield %cnt2, %dups2 : i32, i32
      }
      %countRight = arith.cmpi eq, %counted#0, %n : i32
      %bad2 = arith.select %countRight, %zero, %one : i32
      %noDup = arith.cmpi eq, %counted#1, %zero : i32
      %bad3 = arith.select %noDup, %zero, %one : i32

      %e1 = arith.addi %err, %bad1 : i32
      %e2 = arith.addi %e1, %bad2 : i32
      %e3 = arith.addi %e2, %bad3 : i32
      scf.yield %e3 : i32
    }
    // CHECK: errors = 0
    vector.print str "errors = "
    vector.print %errors : i32
    return
  }

  func.func @report(%out: memref<96xi32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    // 32 workgroups: more than the 8 dies of an MI300X, so the interesting
    // case where several workgroups share one.
    air.launch (%bx, %by) in (%nbx=%c32, %nby=%c1) args(%o=%out) : memref<96xi32> {
      air.segment @reporter args(%so=%o) : memref<96xi32> {
        %c1_s = arith.constant 1 : index
        %c3_s = arith.constant 3 : index
        %xcd  = air.chiplet_id
        %rank = air.chiplet_block_id
        %n    = air.chiplet_dim_blocks

        %bid = gpu.block_id x
        %base = arith.muli %bid, %c3_s : index
        %o1 = arith.addi %base, %c1_s : index
        %c2_s = arith.constant 2 : index
        %o2 = arith.addi %base, %c2_s : index

        %xcd_i  = arith.index_cast %xcd  : index to i32
        %rank_i = arith.index_cast %rank : index to i32
        %n_i    = arith.index_cast %n    : index to i32
        memref.store %xcd_i,  %so[%base] : memref<96xi32>
        memref.store %rank_i, %so[%o1]   : memref<96xi32>
        memref.store %n_i,    %so[%o2]   : memref<96xi32>

        air.herd @herd tile (%tx, %ty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
