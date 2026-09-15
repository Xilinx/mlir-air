//===- megakernel_layers.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// A chain of layers in one launch. This is the shape a transformer decode step
// has -- each layer consumes the previous layer's output -- and the reason a
// megakernel exists: done conventionally every arrow in that chain is a kernel
// boundary, and at decode sizes the launch costs more than the arithmetic.
//
// Four layers of X <- X * W, 128x128x128 each, tiles handed out from a per-layer
// queue and the layer boundary enforced by an event rather than by returning to
// the host. One launch for the whole chain; the host sees the input and the
// final output and nothing in between.
//
//   X[0] input, X[l+1] = X[l] * W[l]
//   queue[l][0] head for layer l, queue[l][1+i] tile i
//   ev[l]       tiles of layer l completed; the gate for layer l+1
//
//===------------------------------------------------------------------===//

module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %fzero = arith.constant 0.0 : f32
    %c7f = arith.constant 7.0 : f32
    %c5f = arith.constant 5.0 : f32
    %c1f = arith.constant 1.0 : f32
    %cwscale = arith.constant 1.250000e-01 : f32
    %c2 = arith.constant 2 : index

    %X = memref.alloc() : memref<5x128x128xf32>
    %W = memref.alloc() : memref<4x128x128xf32>
    %R = memref.alloc() : memref<2x128x128xf32>

    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c128 step %c1 {
        %ii = arith.index_cast %i : index to i32
        %jj = arith.index_cast %j : index to i32
        %fi = arith.sitofp %ii : i32 to f32
        %fj = arith.sitofp %jj : i32 to f32
        %s = arith.addf %fi, %fj : f32
        %x0 = arith.remf %s, %c7f : f32
        %scaled = arith.mulf %x0, %c1f : f32
        memref.store %scaled, %X[%c0, %i, %j] : memref<5x128x128xf32>
        memref.store %scaled, %R[%c0, %i, %j] : memref<2x128x128xf32>
        scf.for %l = %c0 to %c4 step %c1 {
          %ll = arith.index_cast %l : index to i32
          %fl = arith.sitofp %ll : i32 to f32
          %d = arith.subf %fi, %fj : f32
          %d2 = arith.addf %d, %fl : f32
          %w = arith.remf %d2, %c5f : f32
          %wn = arith.mulf %w, %cwscale : f32
          memref.store %wn, %W[%l, %i, %j] : memref<4x128x128xf32>
        }
      }
    }

    // Host reference for the whole chain, ping-ponging between R[0] and R[1].
    scf.for %l = %c0 to %c4 step %c1 {
      %src = arith.remui %l, %c2 : index
      %dst = arith.subi %c1, %src : index
      scf.for %i = %c0 to %c128 step %c1 {
        scf.for %j = %c0 to %c128 step %c1 {
          %acc = scf.for %k = %c0 to %c128 step %c1
              iter_args(%s = %fzero) -> (f32) {
            %a = memref.load %R[%src, %i, %k] : memref<2x128x128xf32>
            %b = memref.load %W[%l, %k, %j] : memref<4x128x128xf32>
            %m = arith.mulf %a, %b : f32
            %s2 = arith.addf %s, %m : f32
            scf.yield %s2 : f32
          }
          memref.store %acc, %R[%dst, %i, %j] : memref<2x128x128xf32>
        }
      }
    }

    %Q = memref.alloc() : memref<4x17xi32>
    %c11 = arith.constant 11 : index
    scf.for %l = %c0 to %c4 step %c1 {
      memref.store %zero, %Q[%l, %c0] : memref<4x17xi32>
      scf.for %i = %c0 to %c16 step %c1 {
        %m = arith.muli %i, %c11 : index
        %r = arith.remui %m, %c16 : index
        %ri = arith.index_cast %r : index to i32
        %slot = arith.addi %i, %c1 : index
        memref.store %ri, %Q[%l, %slot] : memref<4x17xi32>
      }
    }
    %E = memref.alloc() : memref<8xi32>
    %c8 = arith.constant 8 : index
    scf.for %i = %c0 to %c8 step %c1 {
      memref.store %zero, %E[%i] : memref<8xi32>
    }

    %dX = gpu.alloc () : memref<5x128x128xf32>
    %dW = gpu.alloc () : memref<4x128x128xf32>
    %dQ = gpu.alloc () : memref<4x17xi32>
    %dE = gpu.alloc () : memref<8xi32>
    gpu.memcpy %dX, %X : memref<5x128x128xf32>, memref<5x128x128xf32>
    gpu.memcpy %dW, %W : memref<4x128x128xf32>, memref<4x128x128xf32>
    gpu.memcpy %dQ, %Q : memref<4x17xi32>, memref<4x17xi32>
    gpu.memcpy %dE, %E : memref<8xi32>, memref<8xi32>

    call @layers(%dQ, %dE, %dX, %dW)
        : (memref<4x17xi32>, memref<8xi32>, memref<5x128x128xf32>, memref<4x128x128xf32>) -> ()

    gpu.memcpy %X, %dX : memref<5x128x128xf32>, memref<5x128x128xf32>

    %tol = arith.constant 1.0e-2 : f32
    %bad = scf.for %i = %c0 to %c128 step %c1
        iter_args(%b0 = %zero) -> (i32) {
      %bi = scf.for %j = %c0 to %c128 step %c1
          iter_args(%b1 = %b0) -> (i32) {
        %got = memref.load %X[%c4, %i, %j] : memref<5x128x128xf32>
        %want = memref.load %R[%c0, %i, %j] : memref<2x128x128xf32>
        %d = arith.subf %got, %want : f32
        %ad = math.absf %d : f32
        %aw = math.absf %want : f32
        %scale = arith.maxnumf %aw, %c1f : f32
        %rel = arith.divf %ad, %scale : f32
        %ok = arith.cmpf ole, %rel, %tol : f32
        %inc = arith.select %ok, %zero, %one : i32
        %b2 = arith.addi %b1, %inc : i32
        scf.yield %b2 : i32
      }
      scf.yield %bi : i32
    }
    vector.print str "elements of the final layer differing from the reference = "
    vector.print %bad : i32
    return
  }

  func.func @layers(%queue: memref<4x17xi32>, %ev: memref<8xi32>,
                    %X: memref<5x128x128xf32>, %W: memref<4x128x128xf32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    air.launch (%bx, %by) in (%nbx=%c32, %nby=%c1)
        args(%q=%queue, %e=%ev, %x=%X, %w=%W)
        : memref<4x17xi32>, memref<8xi32>, memref<5x128x128xf32>, memref<4x128x128xf32> {
      air.segment @worker args(%sq=%q, %se=%e, %sx=%x, %sw=%w)
          : memref<4x17xi32>, memref<8xi32>, memref<5x128x128xf32>, memref<4x128x128xf32> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c4_s = arith.constant 4 : index
        %c16_s = arith.constant 16 : index
        %c32_s = arith.constant 32 : index
        %c128_s = arith.constant 128 : index
        %one_s = arith.constant 1 : i32
        %zero_s = arith.constant 0 : i32
        %tiles = arith.constant 16 : i32
        %fzero_s = arith.constant 0.0 : f32
        %true = arith.constant true

        %tx_s = gpu.thread_id x
        %ty_s = gpu.thread_id y
        %tz_s = gpu.thread_id z
        %t01 = arith.ori %tx_s, %ty_s : index
        %t012 = arith.ori %t01, %tz_s : index
        %isLead = arith.cmpi eq, %t012, %c0_s : index

        %evbase = memref.extract_aligned_pointer_as_index %se : memref<8xi32> -> index
        %evi = arith.index_cast %evbase : index to i64
        %evptr = llvm.inttoptr %evi : i64 to !llvm.ptr

        scf.if %isLead {
          // The chain, all of it, without leaving the kernel.
          scf.for %l = %c0_s to %c4_s step %c1_s {
            // Wait for the previous layer, except for the first.
            %isFirst = arith.cmpi eq, %l, %c0_s : index
            scf.if %isFirst {
            } else {
              %prev = arith.subi %l, %c1_s : index
              %off = arith.muli %prev, %c4_s : index
              %offi = arith.index_cast %off : index to i64
              %pptr = llvm.getelementptr %evptr[%offi] : (!llvm.ptr, i64) -> !llvm.ptr, i8
              scf.while : () -> () {
                %seen = llvm.load %pptr atomic syncscope("") acquire {alignment = 4 : i64} : !llvm.ptr -> i32
                %notYet = arith.cmpi ult, %seen, %tiles : i32
                scf.condition(%notYet)
              } do {
                scf.yield
              }
            }

            %lnext = arith.addi %l, %c1_s : index
            %tallyR:2 = scf.while (%go = %true, %acc = %zero_s) : (i1, i32) -> (i1, i32) {
              scf.condition(%go) %go, %acc : i1, i32
            } do {
            ^bb0(%g: i1, %acc: i32):
              %claimed = memref.atomic_rmw addi %one_s, %sq[%l, %c0_s] : (i32, memref<4x17xi32>) -> i32
              %idx = arith.index_cast %claimed : i32 to index
              %hasWork = arith.cmpi ult, %idx, %c16_s : index
              %acc2 = scf.if %hasWork -> i32 {
                %slot = arith.addi %idx, %c1_s : index
                %tile = memref.load %sq[%l, %slot] : memref<4x17xi32>
                %ti = arith.index_cast %tile : i32 to index
                %tr = arith.divui %ti, %c4_s : index
                %tc = arith.remui %ti, %c4_s : index
                %r0 = arith.muli %tr, %c32_s : index
                %c0t = arith.muli %tc, %c32_s : index
                scf.for %i = %c0_s to %c32_s step %c1_s {
                  scf.for %j = %c0_s to %c32_s step %c1_s {
                    %gi = arith.addi %r0, %i : index
                    %gj = arith.addi %c0t, %j : index
                    %accv = scf.for %k = %c0_s to %c128_s step %c1_s
                        iter_args(%s = %fzero_s) -> (f32) {
                      %av = memref.load %sx[%l, %gi, %k] : memref<5x128x128xf32>
                      %bv = memref.load %sw[%l, %k, %gj] : memref<4x128x128xf32>
                      %m = arith.mulf %av, %bv : f32
                      %s2 = arith.addf %s, %m : f32
                      scf.yield %s2 : f32
                    }
                    memref.store %accv, %sx[%lnext, %gi, %gj] : memref<5x128x128xf32>
                  }
                }
                %a = arith.addi %acc, %one_s : i32
                scf.yield %a : i32
              } else {
                scf.yield %acc : i32
              }
              scf.yield %hasWork, %acc2 : i1, i32
            }

            // Announce this layer's share. Release, so the tiles this worker
            // wrote are visible to whoever the gate lets through next.
            %loff = arith.muli %l, %c4_s : index
            %loffi = arith.index_cast %loff : index to i64
            %lptr = llvm.getelementptr %evptr[%loffi] : (!llvm.ptr, i64) -> !llvm.ptr, i8
            %pv = llvm.atomicrmw add %lptr, %tallyR#1 syncscope("") release : !llvm.ptr, i32
          }
        }

        air.herd @herd tile (%htx, %hty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
