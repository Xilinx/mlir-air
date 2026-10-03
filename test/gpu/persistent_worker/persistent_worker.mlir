//===- persistent_worker.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===------------------------------------------------------------------===//
//
// The megakernel shape, running. One launch, workgroups that stay resident,
// and the work loop inside them rather than around them.
//
// What that buys is the thing this checks: chiplet identity is discovered once,
// above the loop, and stays true for every iteration after. If it did not --
// if the binding had to be re-established per iteration -- the loop would pay
// a device-wide barrier every time round, and the die's cache would be useless
// across iterations, which is the entire reason for organising work this way.
//
// Eight iterations, each claiming every element of a shared range once, so
// every counter must end at exactly 8. A rank that drifted between iterations
// shows up as a counter that is not 8.
//
//===------------------------------------------------------------------===//

module {
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c1024 = arith.constant 1024 : index
    %c8 = arith.constant 8 : index
    %zero = arith.constant 0 : i32
    %eight = arith.constant 8 : i32
    %one = arith.constant 1 : i32

    %host = memref.alloc() : memref<1024xi32>
    scf.for %i = %c0 to %c1024 step %c1 {
      memref.store %zero, %host[%i] : memref<1024xi32>
    }
    %dev = gpu.alloc () : memref<1024xi32>
    gpu.memcpy %dev, %host : memref<1024xi32>, memref<1024xi32>

    call @persistent(%dev, %c8) : (memref<1024xi32>, index) -> ()

    gpu.memcpy %host, %dev : memref<1024xi32>, memref<1024xi32>

    %v0 = memref.load %host[%c0] : memref<1024xi32>
    vector.print str "out[0] = "
    vector.print %v0 : i32

    %wrong = scf.for %i = %c0 to %c1024 step %c1
        iter_args(%w = %zero) -> (i32) {
      %v = memref.load %host[%i] : memref<1024xi32>
      %ok = arith.cmpi eq, %v, %eight : i32
      %inc = arith.select %ok, %zero, %one : i32
      %w2 = arith.addi %w, %inc : i32
      scf.yield %w2 : i32
    }
    vector.print str "elements not claimed exactly 8 times = "
    vector.print %wrong : i32
    return
  }

  func.func @persistent(%out: memref<1024xi32>, %iters: index) {
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    air.launch (%bx, %by) in (%nbx=%c32, %nby=%c1) args(%o=%out, %ni=%iters) : memref<1024xi32>, index {
      air.segment @worker args(%so=%o, %sn=%ni) : memref<1024xi32>, index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c1024_s = arith.constant 1024 : index
        %one_s = arith.constant 1 : i32

        // Discovered once, above the loop. Everything after depends on this
        // staying true.
        %xcd  = air.chiplet_id
        %rank = air.chiplet_block_id
        %n    = air.chiplet_dim_blocks

        %tx_s = gpu.thread_id x
        %ty_s = gpu.thread_id y
        %tz_s = gpu.thread_id z
        %t01 = arith.ori %tx_s, %ty_s : index
        %t012 = arith.ori %t01, %tz_s : index
        %isLead = arith.cmpi eq, %t012, %c0_s : index
        %isDieZero = arith.cmpi eq, %xcd, %c0_s : index
        %claims = arith.andi %isDieZero, %isLead : i1

        // The work loop lives here, inside the resident workgroup.
        scf.for %iter = %c0_s to %sn step %c1_s {
          scf.if %claims {
            scf.for %t = %rank to %c1024_s step %n {
              %old = memref.atomic_rmw addi %one_s, %so[%t] : (i32, memref<1024xi32>) -> i32
            }
          }
        }

        air.herd @herd tile (%tx, %ty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
