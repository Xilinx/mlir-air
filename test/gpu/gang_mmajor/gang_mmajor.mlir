//===- gang_mmajor.mlir -----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// The part of Fleet's Chiplet-task that only shows up when M > 1.
//
// A gang task is a GEMM split into (m_tile, n_tile) pieces and handed to the
// workgroups of one die. Which piece a workgroup takes is not the obvious
// order: gang_linear_mi300.cuh:75-76 sweeps M fast and N slow,
//
//   m_tile = first_row + (local % win_h)
//   n_tile = local / win_h
//
// so win_h consecutive workers take the same n_tile -- the same block of
// weight rows -- with different M rows of the input. The weight block is read
// once into the die's cache and serves win_h workers. Ordering the other way
// round, each worker would pull a different weight block and the die's cache
// would hold none of them long enough to matter.
//
// At M = 1, which is a decode step, m_tiles is 1 and the whole thing collapses
// to n_tile = local: every worker a different weight block, nothing shared.
// That is why the chain in megakernel_gen does not need this and a prefill
// would.
//
// Checked two ways: the GEMM matches a reference, and the traversal really
// does give win_h consecutive pieces the same n_tile.
//
//   C[64,64] = A[64,64] * B[64,64], 8x8 tiles -> m_tiles = n_tiles = 8
//
//===------------------------------------------------------------------===//

module {
  func.func private @printMemrefI32(memref<*xi32>)

  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c64 = arith.constant 64 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %fzero = arith.constant 0.0 : f32
    %c3f = arith.constant 3.0 : f32
    %c5f = arith.constant 5.0 : f32
    %inv8 = arith.constant 1.250000e-01 : f32

    %A = memref.alloc() : memref<64x64xf32>
    %B = memref.alloc() : memref<64x64xf32>
    %C = memref.alloc() : memref<64x64xf32>
    %R = memref.alloc() : memref<64x64xf32>
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        %ii = arith.index_cast %i : index to i32
        %jj = arith.index_cast %j : index to i32
        %fi = arith.sitofp %ii : i32 to f32
        %fj = arith.sitofp %jj : i32 to f32
        %a0 = arith.addf %fi, %fj : f32
        %a1 = arith.remf %a0, %c3f : f32
        %a2 = arith.mulf %a1, %inv8 : f32
        memref.store %a2, %A[%i, %j] : memref<64x64xf32>
        %b0 = arith.subf %fi, %fj : f32
        %b1 = arith.remf %b0, %c5f : f32
        %b2 = arith.mulf %b1, %inv8 : f32
        memref.store %b2, %B[%i, %j] : memref<64x64xf32>
        memref.store %fzero, %C[%i, %j] : memref<64x64xf32>
      }
    }
    scf.for %i = %c0 to %c64 step %c1 {
      scf.for %j = %c0 to %c64 step %c1 {
        %acc = scf.for %k = %c0 to %c64 step %c1
            iter_args(%s = %fzero) -> (f32) {
          %av = memref.load %A[%i, %k] : memref<64x64xf32>
          %bv = memref.load %B[%k, %j] : memref<64x64xf32>
          %m = arith.mulf %av, %bv : f32
          %s2 = arith.addf %s, %m : f32
          scf.yield %s2 : f32
        }
        memref.store %acc, %R[%i, %j] : memref<64x64xf32>
      }
    }

    // trace[p] = the n_tile that piece p mapped to, so the host can check the
    // traversal rather than take it on faith.
    %trace = memref.alloc() : memref<128xi32>
    %c128 = arith.constant 128 : index
    %neg1 = arith.constant -1 : i32
    scf.for %i = %c0 to %c128 step %c1 {
      memref.store %neg1, %trace[%i] : memref<128xi32>
    }
    %head = memref.alloc() : memref<16xi32>
    scf.for %i = %c0 to %c8 step %c1 {
      memref.store %zero, %head[%i] : memref<16xi32>
    }

    %dA = gpu.alloc () : memref<64x64xf32>
    %dB = gpu.alloc () : memref<64x64xf32>
    %dC = gpu.alloc () : memref<64x64xf32>
    %dT = gpu.alloc () : memref<128xi32>
    %dH = gpu.alloc () : memref<16xi32>
    gpu.memcpy %dA, %A : memref<64x64xf32>, memref<64x64xf32>
    gpu.memcpy %dB, %B : memref<64x64xf32>, memref<64x64xf32>
    gpu.memcpy %dC, %C : memref<64x64xf32>, memref<64x64xf32>
    gpu.memcpy %dT, %trace : memref<128xi32>, memref<128xi32>
    gpu.memcpy %dH, %head : memref<16xi32>, memref<16xi32>

    call @gang(%dA, %dB, %dC, %dT, %dH)
      : (memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>,
         memref<128xi32>, memref<16xi32>) -> ()

    gpu.memcpy %C, %dC : memref<64x64xf32>, memref<64x64xf32>
    gpu.memcpy %trace, %dT : memref<128xi32>, memref<128xi32>

    %tol = arith.constant 1.0e-3 : f32
    %fone = arith.constant 1.0 : f32
    %bad = scf.for %i = %c0 to %c64 step %c1
        iter_args(%b0 = %zero) -> (i32) {
      %bi = scf.for %j = %c0 to %c64 step %c1
          iter_args(%b1 = %b0) -> (i32) {
        %got = memref.load %C[%i, %j] : memref<64x64xf32>
        %want = memref.load %R[%i, %j] : memref<64x64xf32>
        %d = arith.subf %got, %want : f32
        %ad = math.absf %d : f32
        %aw = math.absf %want : f32
        %sc = arith.maxnumf %aw, %fone : f32
        %rel = arith.divf %ad, %sc : f32
        %ok = arith.cmpf ole, %rel, %tol : f32
        %inc = arith.select %ok, %zero, %one : i32
        %b2 = arith.addi %b1, %inc : i32
        scf.yield %b2 : i32
      }
      scf.yield %bi : i32
    }
    vector.print str "elements differing from the reference = "
    vector.print %bad : i32

    // The property that matters: every piece a die took used the same weight
    // block. If the traversal swept N fast instead of M, a die's pieces would
    // spread across weight blocks and the die's cache would hold none of them.
    %wrong = scf.for %p = %c0 to %c64 step %c1
        iter_args(%w = %zero) -> (i32) {
      %myn = memref.load %trace[%p] : memref<128xi32>
      %pd = arith.addi %p, %c64 : index
      %myd = memref.load %trace[%pd] : memref<128xi32>
      %ran = arith.cmpi ne, %myd, %neg1 : i32
      %bad_p = scf.if %ran -> i32 {
        %inner = scf.for %q = %c0 to %c64 step %c1
            iter_args(%b = %zero) -> (i32) {
          %qd0 = arith.addi %q, %c64 : index
          %qd = memref.load %trace[%qd0] : memref<128xi32>
          %qn = memref.load %trace[%q] : memref<128xi32>
          %same = arith.cmpi eq, %qd, %myd : i32
          %diff = arith.cmpi ne, %qn, %myn : i32
          %clash = arith.andi %same, %diff : i1
          %inc = arith.select %clash, %one, %zero : i32
          %b2 = arith.addi %b, %inc : i32
          scf.yield %b2 : i32
        }
        scf.yield %inner : i32
      } else {
        scf.yield %zero : i32
      }
      %w2 = arith.addi %w, %bad_p : i32
      scf.yield %w2 : i32
    }
    vector.print str "pieces sharing a die but not a weight block = "
    vector.print %wrong : i32
    return
  }

  func.func @gang(%A: memref<64x64xf32>, %B: memref<64x64xf32>,
                  %C: memref<64x64xf32>, %trace: memref<128xi32>,
                  %head: memref<16xi32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    air.launch (%bx, %by) in (%nbx=%c32, %nby=%c1)
        args(%aa=%A, %ab=%B, %ac=%C, %at=%trace, %ah=%head)
        : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>,
          memref<128xi32>, memref<16xi32> {
      air.segment @worker args(%sa=%aa, %sb=%ab, %sc=%ac, %st=%at, %sh=%ah)
          : memref<64x64xf32>, memref<64x64xf32>, memref<64x64xf32>,
            memref<128xi32>, memref<16xi32> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c8_s = arith.constant 8 : index
        %c64_s = arith.constant 64 : index
        %one_s = arith.constant 1 : i32
        %fzero_s = arith.constant 0.0 : f32
        %true = arith.constant true

        %die = air.chiplet_id

        %tx_s = gpu.thread_id x
        %ty_s = gpu.thread_id y
        %tz_s = gpu.thread_id z
        %t01 = arith.ori %tx_s, %ty_s : index
        %t012 = arith.ori %t01, %tz_s : index
        %isLead = arith.cmpi eq, %t012, %c0_s : index

        scf.if %isLead {
          // 64 pieces, claimed from this die's head. m_tiles = n_tiles = 8 and
          // the window is the full M, so win_h = 8.
          %done = scf.while (%go = %true) : (i1) -> i1 {
            scf.condition(%go) %go : i1
          } do {
          ^bb0(%g: i1):
            %hslot = arith.remui %die, %c8_s : index
            %cl = memref.atomic_rmw addi %one_s, %sh[%hslot] : (i32, memref<16xi32>) -> i32
            %pieces = arith.muli %c8_s, %c8_s : index
            %p0 = arith.index_cast %cl : i32 to index
            %dieoff = arith.muli %hslot, %c8_s : index
            %p = arith.addi %p0, %dieoff : index
            %pi = arith.remui %p, %c64_s : index
            %has = arith.cmpi ult, %p0, %c8_s : index
            scf.if %has {
              // Fleet's traversal: M fast, N slow.
              %mt = arith.remui %pi, %c8_s : index
              %nt = arith.divui %pi, %c8_s : index
              // Record both the weight block this piece used and the die that
              // used it, so the host can check they line up. Recording only
              // n_tile and comparing it against p/8 would be checking that the
              // host can divide.
              %nti = arith.index_cast %nt : index to i32
              memref.store %nti, %st[%pi] : memref<128xi32>
              %diei = arith.index_cast %die : index to i32
              %dslot = arith.addi %pi, %c64_s : index
              memref.store %diei, %st[%dslot] : memref<128xi32>
              %r0 = arith.muli %mt, %c8_s : index
              %c0t = arith.muli %nt, %c8_s : index
              scf.for %i = %c0_s to %c8_s step %c1_s {
                scf.for %j = %c0_s to %c8_s step %c1_s {
                  %gi = arith.addi %r0, %i : index
                  %gj = arith.addi %c0t, %j : index
                  %acc = scf.for %k = %c0_s to %c64_s step %c1_s
                      iter_args(%s = %fzero_s) -> (f32) {
                    %av = memref.load %sa[%gi, %k] : memref<64x64xf32>
                    %bv = memref.load %sb[%k, %gj] {nontemporal = true} : memref<64x64xf32>
                    %m = arith.mulf %av, %bv : f32
                    %s2 = arith.addf %s, %m : f32
                    scf.yield %s2 : f32
                  }
                  memref.store %acc, %sc[%gi, %gj] : memref<64x64xf32>
                }
              }
            }
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
