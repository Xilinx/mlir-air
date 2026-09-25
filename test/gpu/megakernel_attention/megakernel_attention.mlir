//===- megakernel_attention.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// Attention against a KV cache, as megakernel tasks. This is the last of the
// shapes a decode step is made of, and the one that is least like a matmul:
// a dot product per cache entry, then a softmax across all of them, then a
// weighted sum. The middle stage is a reduction over everything the first
// stage produced, so the two cannot be merged and the dependency cannot be
// expressed as disjoint tiles.
//
//   stage 0   score[i] = dot(q, K[i]) / sqrt(D)    one task per cache entry
//   stage 1   p = softmax(score)                   one task, reduces over all
//   stage 2   out[j] = sum_i p[i] * V[i,j]         one task per slice of out
//
// D = 64, cache length 32.
//
//===------------------------------------------------------------------===//

module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c32 = arith.constant 32 : index
    %c33 = arith.constant 33 : index
    %c64 = arith.constant 64 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %fzero = arith.constant 0.0 : f32
    %c3f = arith.constant 3.0 : f32
    %c5f = arith.constant 5.0 : f32
    %inv8 = arith.constant 1.250000e-01 : f32

    %q = memref.alloc() : memref<64xf32>
    %K = memref.alloc() : memref<32x64xf32>
    %V = memref.alloc() : memref<32x64xf32>
    %score = memref.alloc() : memref<32xf32>
    %prob = memref.alloc() : memref<32xf32>
    %out = memref.alloc() : memref<64xf32>
    %ref = memref.alloc() : memref<64xf32>
    %rs = memref.alloc() : memref<32xf32>

    scf.for %i = %c0 to %c64 step %c1 {
      %ii = arith.index_cast %i : index to i32
      %fi = arith.sitofp %ii : i32 to f32
      %v = arith.remf %fi, %c3f : f32
      %v2 = arith.mulf %v, %inv8 : f32
      memref.store %v2, %q[%i] : memref<64xf32>
      memref.store %fzero, %out[%i] : memref<64xf32>
    }
    scf.for %t = %c0 to %c32 step %c1 {
      memref.store %fzero, %score[%t] : memref<32xf32>
      memref.store %fzero, %prob[%t] : memref<32xf32>
      scf.for %i = %c0 to %c64 step %c1 {
        %tt = arith.index_cast %t : index to i32
        %ii = arith.index_cast %i : index to i32
        %ft = arith.sitofp %tt : i32 to f32
        %fi = arith.sitofp %ii : i32 to f32
        %s = arith.addf %ft, %fi : f32
        %kk = arith.remf %s, %c5f : f32
        %kn = arith.mulf %kk, %inv8 : f32
        memref.store %kn, %K[%t, %i] : memref<32x64xf32>
        %d = arith.subf %ft, %fi : f32
        %vv = arith.remf %d, %c3f : f32
        %vn = arith.mulf %vv, %inv8 : f32
        memref.store %vn, %V[%t, %i] : memref<32x64xf32>
      }
    }

    // Host reference.
    %invsqrtd = arith.constant 1.250000e-01 : f32   // 1/sqrt(64)
    scf.for %t = %c0 to %c32 step %c1 {
      %dot = scf.for %i = %c0 to %c64 step %c1
          iter_args(%s = %fzero) -> (f32) {
        %qv = memref.load %q[%i] : memref<64xf32>
        %kv = memref.load %K[%t, %i] : memref<32x64xf32>
        %m = arith.mulf %qv, %kv : f32
        %s2 = arith.addf %s, %m : f32
        scf.yield %s2 : f32
      }
      %sc = arith.mulf %dot, %invsqrtd : f32
      memref.store %sc, %rs[%t] : memref<32xf32>
    }
    %negbig = arith.constant -1.000000e30 : f32
    %mx = scf.for %t = %c0 to %c32 step %c1
        iter_args(%m = %negbig) -> (f32) {
      %v = memref.load %rs[%t] : memref<32xf32>
      %m2 = arith.maxnumf %m, %v : f32
      scf.yield %m2 : f32
    }
    %sum = scf.for %t = %c0 to %c32 step %c1
        iter_args(%s = %fzero) -> (f32) {
      %v = memref.load %rs[%t] : memref<32xf32>
      %d = arith.subf %v, %mx : f32
      %e = math.exp %d : f32
      memref.store %e, %rs[%t] : memref<32xf32>
      %s2 = arith.addf %s, %e : f32
      scf.yield %s2 : f32
    }
    scf.for %j = %c0 to %c64 step %c1 {
      %acc = scf.for %t = %c0 to %c32 step %c1
          iter_args(%a = %fzero) -> (f32) {
        %e = memref.load %rs[%t] : memref<32xf32>
        %p = arith.divf %e, %sum : f32
        %vv = memref.load %V[%t, %j] : memref<32x64xf32>
        %m = arith.mulf %p, %vv : f32
        %a2 = arith.addf %a, %m : f32
        scf.yield %a2 : f32
      }
      memref.store %acc, %ref[%j] : memref<64xf32>
    }

    %Q = memref.alloc() : memref<3x2xi32>
    scf.for %s = %c0 to %c1 step %c1 { }
    %c3 = arith.constant 3 : index
    %c2 = arith.constant 2 : index
    scf.for %s = %c0 to %c3 step %c1 {
      scf.for %k = %c0 to %c2 step %c1 {
        memref.store %zero, %Q[%s, %k] : memref<3x2xi32>
      }
    }
    %E = memref.alloc() : memref<4xi32>
    %c4 = arith.constant 4 : index
    scf.for %i = %c0 to %c4 step %c1 {
      memref.store %zero, %E[%i] : memref<4xi32>
    }

    %dq = gpu.alloc () : memref<64xf32>
    %dK = gpu.alloc () : memref<32x64xf32>
    %dV = gpu.alloc () : memref<32x64xf32>
    %ds = gpu.alloc () : memref<32xf32>
    %dp = gpu.alloc () : memref<32xf32>
    %do = gpu.alloc () : memref<64xf32>
    %dQq = gpu.alloc () : memref<3x2xi32>
    %dE = gpu.alloc () : memref<4xi32>
    gpu.memcpy %dq, %q : memref<64xf32>, memref<64xf32>
    gpu.memcpy %dK, %K : memref<32x64xf32>, memref<32x64xf32>
    gpu.memcpy %dV, %V : memref<32x64xf32>, memref<32x64xf32>
    gpu.memcpy %ds, %score : memref<32xf32>, memref<32xf32>
    gpu.memcpy %dp, %prob : memref<32xf32>, memref<32xf32>
    gpu.memcpy %do, %out : memref<64xf32>, memref<64xf32>
    gpu.memcpy %dQq, %Q : memref<3x2xi32>, memref<3x2xi32>
    gpu.memcpy %dE, %E : memref<4xi32>, memref<4xi32>

    call @attention(%dQq, %dE, %dq, %dK, %dV, %ds, %dp, %do)
      : (memref<3x2xi32>, memref<4xi32>, memref<64xf32>, memref<32x64xf32>,
         memref<32x64xf32>, memref<32xf32>, memref<32xf32>, memref<64xf32>) -> ()

    gpu.memcpy %out, %do : memref<64xf32>, memref<64xf32>

    %tol = arith.constant 1.0e-4 : f32
    %fone = arith.constant 1.0 : f32
    %bad = scf.for %i = %c0 to %c64 step %c1
        iter_args(%b = %zero) -> (i32) {
      %got = memref.load %out[%i] : memref<64xf32>
      %want = memref.load %ref[%i] : memref<64xf32>
      %d = arith.subf %got, %want : f32
      %ad = math.absf %d : f32
      %aw = math.absf %want : f32
      %sc = arith.maxnumf %aw, %fone : f32
      %rel = arith.divf %ad, %sc : f32
      %ok = arith.cmpf ole, %rel, %tol : f32
      %inc = arith.select %ok, %zero, %one : i32
      %b2 = arith.addi %b, %inc : i32
      scf.yield %b2 : i32
    }
    vector.print str "attention output elements differing from the reference = "
    vector.print %bad : i32
    return
  }

  func.func @attention(%Qq: memref<3x2xi32>, %E: memref<4xi32>,
                       %q: memref<64xf32>, %K: memref<32x64xf32>,
                       %V: memref<32x64xf32>, %score: memref<32xf32>,
                       %prob: memref<32xf32>, %out: memref<64xf32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    air.launch (%bx, %by) in (%nbx=%c32, %nby=%c1)
        args(%aq=%Qq, %ae=%E, %aqq=%q, %ak=%K, %av=%V, %as=%score, %ap=%prob, %ao=%out)
        : memref<3x2xi32>, memref<4xi32>, memref<64xf32>, memref<32x64xf32>,
          memref<32x64xf32>, memref<32xf32>, memref<32xf32>, memref<64xf32> {
      air.segment @worker args(%sq=%aq, %se=%ae, %sqq=%aqq, %sk=%ak, %sv=%av, %ss=%as, %sp=%ap, %so=%ao)
          : memref<3x2xi32>, memref<4xi32>, memref<64xf32>, memref<32x64xf32>,
            memref<32x64xf32>, memref<32xf32>, memref<32xf32>, memref<64xf32> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c3_s = arith.constant 3 : index
        %c4_s = arith.constant 4 : index
        %c8_s = arith.constant 8 : index
        %c32_s = arith.constant 32 : index
        %c64_s = arith.constant 64 : index
        %one_s = arith.constant 1 : i32
        %zero_s = arith.constant 0 : i32
        %n32 = arith.constant 32 : i32
        %n8 = arith.constant 8 : i32
        %n1 = arith.constant 1 : i32
        %fzero_s = arith.constant 0.0 : f32
        %invsqrtd_s = arith.constant 1.250000e-01 : f32
        %negbig_s = arith.constant -1.000000e30 : f32
        %true = arith.constant true

        %tx_s = gpu.thread_id x
        %ty_s = gpu.thread_id y
        %tz_s = gpu.thread_id z
        %t01 = arith.ori %tx_s, %ty_s : index
        %t012 = arith.ori %t01, %tz_s : index
        %isLead = arith.cmpi eq, %t012, %c0_s : index

        %evbase = memref.extract_aligned_pointer_as_index %se : memref<4xi32> -> index
        %evi = arith.index_cast %evbase : index to i64
        %evptr = llvm.inttoptr %evi : i64 to !llvm.ptr
        %p1 = llvm.getelementptr %evptr[4] : (!llvm.ptr) -> !llvm.ptr, i8
        %p2 = llvm.getelementptr %evptr[8] : (!llvm.ptr) -> !llvm.ptr, i8

        scf.if %isLead {
          // ---- stage 0: one dot product per cache entry.
          %t0:2 = scf.while (%go = %true, %acc = %zero_s) : (i1, i32) -> (i1, i32) {
            scf.condition(%go) %go, %acc : i1, i32
          } do {
          ^bb0(%g: i1, %acc: i32):
            %cl = memref.atomic_rmw addi %one_s, %sq[%c0_s, %c0_s] : (i32, memref<3x2xi32>) -> i32
            %ix = arith.index_cast %cl : i32 to index
            %has = arith.cmpi ult, %ix, %c32_s : index
            %acc2 = scf.if %has -> i32 {
              %dot = scf.for %i = %c0_s to %c64_s step %c1_s
                  iter_args(%s = %fzero_s) -> (f32) {
                %qv = memref.load %sqq[%i] : memref<64xf32>
                %kv = memref.load %sk[%ix, %i] : memref<32x64xf32>
                %m = arith.mulf %qv, %kv : f32
                %s2 = arith.addf %s, %m : f32
                scf.yield %s2 : f32
              }
              %sc = arith.mulf %dot, %invsqrtd_s : f32
              // Hold up one cache entry. Without this the race the event
              // prevents is too narrow to ever lose -- 32 workers each doing
              // one dot product finish within microseconds of each other -- and
              // deleting the gate would still give the right answer, so the
              // test would be passing for no reason. Atomics on a spare event
              // slot, so it cannot be optimised away.
              %slow = arith.cmpi eq, %ix, %c0_s : index
              scf.if %slow {
                %burnN = arith.constant 200000 : index
                scf.for %bk = %c0_s to %burnN step %c1_s {
                  %b = memref.atomic_rmw addi %one_s, %se[%c3_s] : (i32, memref<4xi32>) -> i32
                }
              }
              memref.store %sc, %ss[%ix] : memref<32xf32>
              %n = arith.addi %acc, %one_s : i32
              scf.yield %n : i32
            } else {
              scf.yield %acc : i32
            }
            scf.yield %has, %acc2 : i1, i32
          }
          %e0 = llvm.atomicrmw add %evptr, %t0#1 syncscope("") release : !llvm.ptr, i32
          scf.while : () -> () {
            %seen = llvm.load %evptr atomic syncscope("") acquire {alignment = 4 : i64} : !llvm.ptr -> i32
            %notYet = arith.cmpi ult, %seen, %n32 : i32
            scf.condition(%notYet)
          } do {
            scf.yield
          }

          // ---- stage 1: softmax. A reduction over everything stage 0 wrote,
          // so exactly one worker does it and everyone waits.
          %cl1 = memref.atomic_rmw addi %one_s, %sq[%c1_s, %c0_s] : (i32, memref<3x2xi32>) -> i32
          %mine = arith.cmpi eq, %cl1, %zero_s : i32
          scf.if %mine {
            %mx = scf.for %t = %c0_s to %c32_s step %c1_s
                iter_args(%m = %negbig_s) -> (f32) {
              %v = memref.load %ss[%t] : memref<32xf32>
              %m2 = arith.maxnumf %m, %v : f32
              scf.yield %m2 : f32
            }
            %sum = scf.for %t = %c0_s to %c32_s step %c1_s
                iter_args(%s = %fzero_s) -> (f32) {
              %v = memref.load %ss[%t] : memref<32xf32>
              %d = arith.subf %v, %mx : f32
              %e = math.exp %d : f32
              memref.store %e, %sp[%t] : memref<32xf32>
              %s2 = arith.addf %s, %e : f32
              scf.yield %s2 : f32
            }
            scf.for %t = %c0_s to %c32_s step %c1_s {
              %e = memref.load %sp[%t] : memref<32xf32>
              %p = arith.divf %e, %sum : f32
              memref.store %p, %sp[%t] : memref<32xf32>
            }
            %e1 = llvm.atomicrmw add %p1, %n1 syncscope("") release : !llvm.ptr, i32
          }
          scf.while : () -> () {
            %seen = llvm.load %p1 atomic syncscope("") acquire {alignment = 4 : i64} : !llvm.ptr -> i32
            %notYet = arith.cmpi ult, %seen, %n1 : i32
            scf.condition(%notYet)
          } do {
            scf.yield
          }

          // ---- stage 2: weighted sum of V.
          %t2:2 = scf.while (%go2 = %true, %acc2i = %zero_s) : (i1, i32) -> (i1, i32) {
            scf.condition(%go2) %go2, %acc2i : i1, i32
          } do {
          ^bb0(%g2: i1, %acc2i: i32):
            %cl2 = memref.atomic_rmw addi %one_s, %sq[%c2_s, %c0_s] : (i32, memref<3x2xi32>) -> i32
            %ix2 = arith.index_cast %cl2 : i32 to index
            %has2 = arith.cmpi ult, %ix2, %c8_s : index
            %acc3 = scf.if %has2 -> i32 {
              %j0 = arith.muli %ix2, %c8_s : index
              scf.for %jj = %c0_s to %c8_s step %c1_s {
                %j = arith.addi %j0, %jj : index
                %a = scf.for %t = %c0_s to %c32_s step %c1_s
                    iter_args(%s = %fzero_s) -> (f32) {
                  %p = memref.load %sp[%t] : memref<32xf32>
                  %vv = memref.load %sv[%t, %j] : memref<32x64xf32>
                  %m = arith.mulf %p, %vv : f32
                  %s2 = arith.addf %s, %m : f32
                  scf.yield %s2 : f32
                }
                memref.store %a, %so[%j] : memref<64xf32>
              }
              %n = arith.addi %acc2i, %one_s : i32
              scf.yield %n : i32
            } else {
              scf.yield %acc2i : i32
            }
            scf.yield %has2, %acc3 : i1, i32
          }
          %e2 = llvm.atomicrmw add %p2, %t2#1 syncscope("") release : !llvm.ptr, i32
        }

        air.herd @herd tile (%htx, %hty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
