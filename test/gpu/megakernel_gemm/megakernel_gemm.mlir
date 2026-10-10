//===- megakernel_gemm.mlir ---------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// Real arithmetic inside the megakernel: a tiled GEMM whose output tiles are
// tasks pulled from a queue by resident workgroups, rather than a grid the host
// dispatches.
//
// The tiles are handed out at run time, so which workgroup computes which tile
// is not a function of block id, and the result still has to match a reference
// exactly. That is the property a task-graph runtime has to preserve before any
// of the scheduling around it is worth anything.
//
//   C[128,128] = A[128,128] * B[128,128], in 32x32 tiles -> 16 tasks
//
//===------------------------------------------------------------------===//

module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c17 = arith.constant 17 : index
    %c128 = arith.constant 128 : index
    %zero = arith.constant 0 : i32
    %fzero = arith.constant 0.0 : f32
    %one = arith.constant 1 : i32

    %A = memref.alloc() : memref<128x128xf32>
    %B = memref.alloc() : memref<128x128xf32>
    %C = memref.alloc() : memref<128x128xf32>
    %R = memref.alloc() : memref<128x128xf32>

    // Values that make a wrong tile obvious rather than plausibly close.
    %c7f = arith.constant 7.0 : f32
    %c13f = arith.constant 13.0 : f32
    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c128 step %c1 {
        %ii = arith.index_cast %i : index to i32
        %jj = arith.index_cast %j : index to i32
        %fi = arith.sitofp %ii : i32 to f32
        %fj = arith.sitofp %jj : i32 to f32
        %a = arith.addf %fi, %fj : f32
        %a2 = arith.remf %a, %c7f : f32
        memref.store %a2, %A[%i, %j] : memref<128x128xf32>
        %b = arith.subf %fi, %fj : f32
        %b2 = arith.remf %b, %c13f : f32
        memref.store %b2, %B[%i, %j] : memref<128x128xf32>
        memref.store %fzero, %C[%i, %j] : memref<128x128xf32>
      }
    }
    // Reference, on the host.
    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c128 step %c1 {
        %acc = scf.for %k = %c0 to %c128 step %c1
            iter_args(%s = %fzero) -> (f32) {
          %a = memref.load %A[%i, %k] : memref<128x128xf32>
          %b = memref.load %B[%k, %j] : memref<128x128xf32>
          %m = arith.mulf %a, %b : f32
          %s2 = arith.addf %s, %m : f32
          scf.yield %s2 : f32
        }
        memref.store %acc, %R[%i, %j] : memref<128x128xf32>
      }
    }

    // Queue: head at [0], then 16 tile ids in a scrambled order.
    %qh = memref.alloc() : memref<17xi32>
    memref.store %zero, %qh[%c0] : memref<17xi32>
    %c11 = arith.constant 11 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %m = arith.muli %i, %c11 : index
      %r = arith.remui %m, %c16 : index
      %ri = arith.index_cast %r : index to i32
      %slot = arith.addi %i, %c1 : index
      memref.store %ri, %qh[%slot] : memref<17xi32>
    }

    %dA = gpu.alloc () : memref<128x128xf32>
    %dB = gpu.alloc () : memref<128x128xf32>
    %dC = gpu.alloc () : memref<128x128xf32>
    %dQ = gpu.alloc () : memref<17xi32>
    gpu.memcpy %dA, %A : memref<128x128xf32>, memref<128x128xf32>
    gpu.memcpy %dB, %B : memref<128x128xf32>, memref<128x128xf32>
    gpu.memcpy %dC, %C : memref<128x128xf32>, memref<128x128xf32>
    gpu.memcpy %dQ, %qh : memref<17xi32>, memref<17xi32>

    call @megakernel(%dQ, %dA, %dB, %dC)
        : (memref<17xi32>, memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32>) -> ()

    gpu.memcpy %C, %dC : memref<128x128xf32>, memref<128x128xf32>

    %tol = arith.constant 1.0e-3 : f32
    %bad = scf.for %i = %c0 to %c128 step %c1
        iter_args(%b0 = %zero) -> (i32) {
      %bi = scf.for %j = %c0 to %c128 step %c1
          iter_args(%b1 = %b0) -> (i32) {
        %got = memref.load %C[%i, %j] : memref<128x128xf32>
        %want = memref.load %R[%i, %j] : memref<128x128xf32>
        %d = arith.subf %got, %want : f32
        %ad = math.absf %d : f32
        %ok = arith.cmpf ole, %ad, %tol : f32
        %inc = arith.select %ok, %zero, %one : i32
        %b2 = arith.addi %b1, %inc : i32
        scf.yield %b2 : i32
      }
      scf.yield %bi : i32
    }
    vector.print str "elements differing from the reference = "
    vector.print %bad : i32
    return
  }

  func.func @megakernel(%queue: memref<17xi32>, %A: memref<128x128xf32>,
                        %B: memref<128x128xf32>, %C: memref<128x128xf32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    air.launch (%bx, %by) in (%nbx=%c32, %nby=%c1)
        args(%q=%queue, %a=%A, %b=%B, %c=%C)
        : memref<17xi32>, memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32> {
      air.segment @worker args(%sq=%q, %sa=%a, %sb=%b, %sc=%c)
          : memref<17xi32>, memref<128x128xf32>, memref<128x128xf32>, memref<128x128xf32> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c4_s = arith.constant 4 : index
        %c16_s = arith.constant 16 : index
        %c32_s = arith.constant 32 : index
        %c128_s = arith.constant 128 : index
        %one_s = arith.constant 1 : i32
        %fzero_s = arith.constant 0.0 : f32
        %true = arith.constant true

        %tx_s = gpu.thread_id x
        %ty_s = gpu.thread_id y
        %tz_s = gpu.thread_id z
        %t01 = arith.ori %tx_s, %ty_s : index
        %t012 = arith.ori %t01, %tz_s : index
        %isLead = arith.cmpi eq, %t012, %c0_s : index

        scf.if %isLead {
          %done = scf.while (%go = %true) : (i1) -> i1 {
            scf.condition(%go) %go : i1
          } do {
          ^bb0(%g: i1):
            %claimed = memref.atomic_rmw addi %one_s, %sq[%c0_s] : (i32, memref<17xi32>) -> i32
            %idx = arith.index_cast %claimed : i32 to index
            %hasWork = arith.cmpi ult, %idx, %c16_s : index
            scf.if %hasWork {
              %slot = arith.addi %idx, %c1_s : index
              %tile = memref.load %sq[%slot] : memref<17xi32>
              %ti = arith.index_cast %tile : i32 to index
              %tr = arith.divui %ti, %c4_s : index
              %tc = arith.remui %ti, %c4_s : index
              %r0 = arith.muli %tr, %c32_s : index
              %c0t = arith.muli %tc, %c32_s : index
              scf.for %i = %c0_s to %c32_s step %c1_s {
                scf.for %j = %c0_s to %c32_s step %c1_s {
                  %gi = arith.addi %r0, %i : index
                  %gj = arith.addi %c0t, %j : index
                  %acc = scf.for %k = %c0_s to %c128_s step %c1_s
                      iter_args(%s = %fzero_s) -> (f32) {
                    %av = memref.load %sa[%gi, %k] : memref<128x128xf32>
                    %bv = memref.load %sb[%k, %gj] : memref<128x128xf32>
                    %m = arith.mulf %av, %bv : f32
                    %s2 = arith.addf %s, %m : f32
                    scf.yield %s2 : f32
                  }
                  memref.store %acc, %sc[%gi, %gj] : memref<128x128xf32>
                }
              }
            }
            scf.yield %hasWork : i1
          }
        }

        air.herd @herd tile (%htx, %hty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
