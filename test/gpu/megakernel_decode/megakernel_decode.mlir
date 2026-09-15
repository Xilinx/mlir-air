//===- megakernel_decode.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// A decode step's worth of layer, as megakernel tasks. Decode is one token at a
// time, so the matmuls are matrix-vector and there is very little arithmetic
// per launch -- which is exactly why the launches are what costs, and why
// putting the whole chain in one kernel is the point.
//
// Two layers, each three stages with an event between them, because each stage
// needs all of the previous one:
//
//   stage 0  r = x / sqrt(mean(x^2) + eps)   a reduction, one task
//   stage 1  h = r @ W1                      8 tasks, one per slice of h
//   stage 2  y = relu(h) @ W2                8 tasks, needs all of h
//
// The point of including stage 0 and the activation is that a task body is not
// only a matmul. A reduction has to finish before anything reads it, and an
// elementwise stage reads its whole input, so both lean on the event rather
// than on tiles being independent.
//
// ReLU rather than SiLU: SiLU needs exp from the device math library, which
// this pipeline does not link. The shape of the dependency is the same.
//
//===------------------------------------------------------------------===//

module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %fzero = arith.constant 0.0 : f32
    %eps = arith.constant 1.0e-6 : f32
    %fdim = arith.constant 1.280000e+02 : f32
    %c3f = arith.constant 3.0 : f32
    %c5f = arith.constant 5.0 : f32
    %scale = arith.constant 1.250000e-01 : f32

    // x, and per-layer W1/W2.
    %X = memref.alloc() : memref<128xf32>
    %W1 = memref.alloc() : memref<2x128x128xf32>
    %W2 = memref.alloc() : memref<2x128x128xf32>
    // Device scratch: r, h, y.
    %Rv = memref.alloc() : memref<128xf32>
    %Hv = memref.alloc() : memref<128xf32>
    %Yv = memref.alloc() : memref<128xf32>
    // Host reference state.
    %ref = memref.alloc() : memref<128xf32>
    %refh = memref.alloc() : memref<128xf32>

    scf.for %i = %c0 to %c128 step %c1 {
      %ii = arith.index_cast %i : index to i32
      %fi = arith.sitofp %ii : i32 to f32
      %v = arith.remf %fi, %c3f : f32
      %v2 = arith.addf %v, %scale : f32
      memref.store %v2, %X[%i] : memref<128xf32>
      memref.store %v2, %ref[%i] : memref<128xf32>
      memref.store %fzero, %Rv[%i] : memref<128xf32>
      memref.store %fzero, %Hv[%i] : memref<128xf32>
      memref.store %fzero, %Yv[%i] : memref<128xf32>
      scf.for %l = %c0 to %c2 step %c1 {
        scf.for %j = %c0 to %c128 step %c1 {
          %jj = arith.index_cast %j : index to i32
          %fj = arith.sitofp %jj : i32 to f32
          %ll = arith.index_cast %l : index to i32
          %fl = arith.sitofp %ll : i32 to f32
          %d = arith.subf %fi, %fj : f32
          %d2 = arith.addf %d, %fl : f32
          %w = arith.remf %d2, %c5f : f32
          %wn = arith.mulf %w, %scale : f32
          memref.store %wn, %W1[%l, %i, %j] : memref<2x128x128xf32>
          %w3 = arith.addf %d2, %c3f : f32
          %w4 = arith.remf %w3, %c5f : f32
          %wn2 = arith.mulf %w4, %scale : f32
          memref.store %wn2, %W2[%l, %i, %j] : memref<2x128x128xf32>
        }
      }
    }

    // Host reference for both layers.
    scf.for %l = %c0 to %c2 step %c1 {
      %ss = scf.for %i = %c0 to %c128 step %c1
          iter_args(%s = %fzero) -> (f32) {
        %v = memref.load %ref[%i] : memref<128xf32>
        %sq = arith.mulf %v, %v : f32
        %s2 = arith.addf %s, %sq : f32
        scf.yield %s2 : f32
      }
      %mean = arith.divf %ss, %fdim : f32
      %me = arith.addf %mean, %eps : f32
      %rms = math.sqrt %me : f32
      scf.for %j = %c0 to %c128 step %c1 {
        %acc = scf.for %i = %c0 to %c128 step %c1
            iter_args(%s = %fzero) -> (f32) {
          %xv = memref.load %ref[%i] : memref<128xf32>
          %rv = arith.divf %xv, %rms : f32
          %wv = memref.load %W1[%l, %i, %j] : memref<2x128x128xf32>
          %m = arith.mulf %rv, %wv : f32
          %s2 = arith.addf %s, %m : f32
          scf.yield %s2 : f32
        }
        memref.store %acc, %refh[%j] : memref<128xf32>
      }
      scf.for %j = %c0 to %c128 step %c1 {
        %acc = scf.for %i = %c0 to %c128 step %c1
            iter_args(%s = %fzero) -> (f32) {
          %hv = memref.load %refh[%i] : memref<128xf32>
          %rl = arith.maxnumf %hv, %fzero : f32
          %wv = memref.load %W2[%l, %i, %j] : memref<2x128x128xf32>
          %m = arith.mulf %rl, %wv : f32
          %s2 = arith.addf %s, %m : f32
          scf.yield %s2 : f32
        }
        memref.store %acc, %Yv[%j] : memref<128xf32>
      }
      scf.for %j = %c0 to %c128 step %c1 {
        %v = memref.load %Yv[%j] : memref<128xf32>
        memref.store %v, %ref[%j] : memref<128xf32>
      }
    }
    scf.for %i = %c0 to %c128 step %c1 {
      memref.store %fzero, %Yv[%i] : memref<128xf32>
    }

    // Queues: 2 layers x 3 stages, head at [l][s][0].
    // Zero all of it. Slot [l][0][1] is used as a claim for the copy at the end
    // of the layer, so leaving any of it uninitialised means nobody claims and
    // everyone waits forever.
    %Q = memref.alloc() : memref<2x3x9xi32>
    %c3 = arith.constant 3 : index
    scf.for %l = %c0 to %c2 step %c1 {
      scf.for %st = %c0 to %c3 step %c1 {
        scf.for %k = %c0 to %c9 step %c1 {
          memref.store %zero, %Q[%l, %st, %k] : memref<2x3x9xi32>
        }
      }
    }
    %E = memref.alloc() : memref<8xi32>
    scf.for %i = %c0 to %c8 step %c1 {
      memref.store %zero, %E[%i] : memref<8xi32>
    }

    %dX = gpu.alloc () : memref<128xf32>
    %dR = gpu.alloc () : memref<128xf32>
    %dH = gpu.alloc () : memref<128xf32>
    %dY = gpu.alloc () : memref<128xf32>
    %dW1 = gpu.alloc () : memref<2x128x128xf32>
    %dW2 = gpu.alloc () : memref<2x128x128xf32>
    %dQ = gpu.alloc () : memref<2x3x9xi32>
    %dE = gpu.alloc () : memref<8xi32>
    gpu.memcpy %dX, %X : memref<128xf32>, memref<128xf32>
    gpu.memcpy %dR, %Rv : memref<128xf32>, memref<128xf32>
    gpu.memcpy %dH, %Hv : memref<128xf32>, memref<128xf32>
    gpu.memcpy %dY, %Yv : memref<128xf32>, memref<128xf32>
    gpu.memcpy %dW1, %W1 : memref<2x128x128xf32>, memref<2x128x128xf32>
    gpu.memcpy %dW2, %W2 : memref<2x128x128xf32>, memref<2x128x128xf32>
    gpu.memcpy %dQ, %Q : memref<2x3x9xi32>, memref<2x3x9xi32>
    gpu.memcpy %dE, %E : memref<8xi32>, memref<8xi32>

    call @decode(%dQ, %dE, %dX, %dR, %dH, %dY, %dW1, %dW2)
      : (memref<2x3x9xi32>, memref<8xi32>, memref<128xf32>, memref<128xf32>,
         memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32>) -> ()

    gpu.memcpy %X, %dX : memref<128xf32>, memref<128xf32>

    %tol = arith.constant 2.0e-2 : f32
    %fone = arith.constant 1.0 : f32
    %bad = scf.for %i = %c0 to %c128 step %c1
        iter_args(%b = %zero) -> (i32) {
      %got = memref.load %X[%i] : memref<128xf32>
      %want = memref.load %ref[%i] : memref<128xf32>
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
    vector.print str "decode output elements differing from the reference = "
    vector.print %bad : i32
    return
  }

  func.func @decode(%Q: memref<2x3x9xi32>, %E: memref<8xi32>,
                    %X: memref<128xf32>, %R: memref<128xf32>,
                    %H: memref<128xf32>, %Y: memref<128xf32>,
                    %W1: memref<2x128x128xf32>, %W2: memref<2x128x128xf32>) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    air.launch (%bx, %by) in (%nbx=%c32, %nby=%c1)
        args(%q=%Q, %e=%E, %x=%X, %r=%R, %h=%H, %y=%Y, %w1=%W1, %w2=%W2)
        : memref<2x3x9xi32>, memref<8xi32>, memref<128xf32>, memref<128xf32>,
          memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32> {
      air.segment @worker args(%sq=%q, %se=%e, %sx=%x, %sr=%r, %sh=%h, %sy=%y, %sw1=%w1, %sw2=%w2)
          : memref<2x3x9xi32>, memref<8xi32>, memref<128xf32>, memref<128xf32>,
            memref<128xf32>, memref<128xf32>, memref<2x128x128xf32>, memref<2x128x128xf32> {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c2_s = arith.constant 2 : index
        %c3_s = arith.constant 3 : index
        %c4_s = arith.constant 4 : index
        %c8_s = arith.constant 8 : index
        %c16_s = arith.constant 16 : index
        %c128_s = arith.constant 128 : index
        %one_s = arith.constant 1 : i32
        %zero_s = arith.constant 0 : i32
        %eight = arith.constant 8 : i32
        %oneTask = arith.constant 1 : i32
        %fzero_s = arith.constant 0.0 : f32
        %eps_s = arith.constant 1.0e-6 : f32
        %fdim_s = arith.constant 1.280000e+02 : f32
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
          scf.for %l = %c0_s to %c2_s step %c1_s {
            // ---- stage 0: the reduction. One task, so only one worker does it.
            %ev0 = arith.muli %l, %c3_s : index
            %c0off = arith.muli %ev0, %c4_s : index
            %c0offi = arith.index_cast %c0off : index to i64
            %p0 = llvm.getelementptr %evptr[%c0offi] : (!llvm.ptr, i64) -> !llvm.ptr, i8
            %claim0 = memref.atomic_rmw addi %one_s, %sq[%l, %c0_s, %c0_s] : (i32, memref<2x3x9xi32>) -> i32
            %mine0 = arith.cmpi eq, %claim0, %zero_s : i32
            scf.if %mine0 {
              %ss = scf.for %i = %c0_s to %c128_s step %c1_s
                  iter_args(%s = %fzero_s) -> (f32) {
                %v = memref.load %sx[%i] : memref<128xf32>
                %sq2 = arith.mulf %v, %v : f32
                %s2 = arith.addf %s, %sq2 : f32
                scf.yield %s2 : f32
              }
              %mean = arith.divf %ss, %fdim_s : f32
              %me = arith.addf %mean, %eps_s : f32
              %rms = math.sqrt %me : f32
              scf.for %i = %c0_s to %c128_s step %c1_s {
                %v = memref.load %sx[%i] : memref<128xf32>
                %n = arith.divf %v, %rms : f32
                memref.store %n, %sr[%i] : memref<128xf32>
              }
              %p = llvm.atomicrmw add %p0, %oneTask syncscope("") release : !llvm.ptr, i32
            }
            scf.while : () -> () {
              %seen = llvm.load %p0 atomic syncscope("") acquire {alignment = 4 : i64} : !llvm.ptr -> i32
              %notYet = arith.cmpi ult, %seen, %oneTask : i32
              scf.condition(%notYet)
            } do {
              scf.yield
            }

            // ---- stage 1: h = r @ W1, one task per 16-wide slice of h.
            %ev1 = arith.addi %ev0, %c1_s : index
            %c1off = arith.muli %ev1, %c4_s : index
            %c1offi = arith.index_cast %c1off : index to i64
            %p1 = llvm.getelementptr %evptr[%c1offi] : (!llvm.ptr, i64) -> !llvm.ptr, i8
            %t1:2 = scf.while (%go = %true, %acc = %zero_s) : (i1, i32) -> (i1, i32) {
              scf.condition(%go) %go, %acc : i1, i32
            } do {
            ^bb0(%g: i1, %acc: i32):
              %cl = memref.atomic_rmw addi %one_s, %sq[%l, %c1_s, %c0_s] : (i32, memref<2x3x9xi32>) -> i32
              %ix = arith.index_cast %cl : i32 to index
              %has = arith.cmpi ult, %ix, %c8_s : index
              %acc2 = scf.if %has -> i32 {
                %j0 = arith.muli %ix, %c16_s : index
                scf.for %jj = %c0_s to %c16_s step %c1_s {
                  %j = arith.addi %j0, %jj : index
                  %a = scf.for %i = %c0_s to %c128_s step %c1_s
                      iter_args(%s = %fzero_s) -> (f32) {
                    %rv = memref.load %sr[%i] : memref<128xf32>
                    %wv = memref.load %sw1[%l, %i, %j] : memref<2x128x128xf32>
                    %m = arith.mulf %rv, %wv : f32
                    %s2 = arith.addf %s, %m : f32
                    scf.yield %s2 : f32
                  }
                  memref.store %a, %sh[%j] : memref<128xf32>
                }
                %n = arith.addi %acc, %one_s : i32
                scf.yield %n : i32
              } else {
                scf.yield %acc : i32
              }
              scf.yield %has, %acc2 : i1, i32
            }
            %pp1 = llvm.atomicrmw add %p1, %t1#1 syncscope("") release : !llvm.ptr, i32
            scf.while : () -> () {
              %seen = llvm.load %p1 atomic syncscope("") acquire {alignment = 4 : i64} : !llvm.ptr -> i32
              %notYet = arith.cmpi ult, %seen, %eight : i32
              scf.condition(%notYet)
            } do {
              scf.yield
            }

            // ---- stage 2: y = relu(h) @ W2. Reads all of h, hence the gate.
            %ev2 = arith.addi %ev0, %c2_s : index
            %c2off = arith.muli %ev2, %c4_s : index
            %c2offi = arith.index_cast %c2off : index to i64
            %p2 = llvm.getelementptr %evptr[%c2offi] : (!llvm.ptr, i64) -> !llvm.ptr, i8
            %t2:2 = scf.while (%go2 = %true, %acc2i = %zero_s) : (i1, i32) -> (i1, i32) {
              scf.condition(%go2) %go2, %acc2i : i1, i32
            } do {
            ^bb0(%g2: i1, %acc2i: i32):
              %cl2 = memref.atomic_rmw addi %one_s, %sq[%l, %c2_s, %c0_s] : (i32, memref<2x3x9xi32>) -> i32
              %ix2 = arith.index_cast %cl2 : i32 to index
              %has2 = arith.cmpi ult, %ix2, %c8_s : index
              %acc3 = scf.if %has2 -> i32 {
                %j0 = arith.muli %ix2, %c16_s : index
                scf.for %jj = %c0_s to %c16_s step %c1_s {
                  %j = arith.addi %j0, %jj : index
                  %a = scf.for %i = %c0_s to %c128_s step %c1_s
                      iter_args(%s = %fzero_s) -> (f32) {
                    %hv = memref.load %sh[%i] : memref<128xf32>
                    %rl = arith.maxnumf %hv, %fzero_s : f32
                    %wv = memref.load %sw2[%l, %i, %j] : memref<2x128x128xf32>
                    %m = arith.mulf %rl, %wv : f32
                    %s2 = arith.addf %s, %m : f32
                    scf.yield %s2 : f32
                  }
                  memref.store %a, %sy[%j] : memref<128xf32>
                }
                %n = arith.addi %acc2i, %one_s : i32
                scf.yield %n : i32
              } else {
                scf.yield %acc2i : i32
              }
              scf.yield %has2, %acc3 : i1, i32
            }
            %pp2 = llvm.atomicrmw add %p2, %t2#1 syncscope("") release : !llvm.ptr, i32
            scf.while : () -> () {
              %seen = llvm.load %p2 atomic syncscope("") acquire {alignment = 4 : i64} : !llvm.ptr -> i32
              %notYet = arith.cmpi ult, %seen, %eight : i32
              scf.condition(%notYet)
            } do {
              scf.yield
            }

            // y becomes the next layer's x. One worker copies; the stage-2 gate
            // above already made y complete, and the stage-0 claim below
            // serialises the next layer behind this.
            %claimC = memref.atomic_rmw addi %one_s, %sq[%l, %c0_s, %c1_s] : (i32, memref<2x3x9xi32>) -> i32
            %mineC = arith.cmpi eq, %claimC, %zero_s : i32
            scf.if %mineC {
              scf.for %i = %c0_s to %c128_s step %c1_s {
                %v = memref.load %sy[%i] : memref<128xf32>
                memref.store %v, %sx[%i] : memref<128xf32>
              }
              %pc = llvm.atomicrmw add %p2, %oneTask syncscope("") release : !llvm.ptr, i32
            }
            %nine = arith.constant 9 : i32
            scf.while : () -> () {
              %seen = llvm.load %p2 atomic syncscope("") acquire {alignment = 4 : i64} : !llvm.ptr -> i32
              %notYet = arith.cmpi ult, %seen, %nine : i32
              scf.condition(%notYet)
            } do {
              scf.yield
            }
          }
        }

        air.herd @herd tile (%htx, %hty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
