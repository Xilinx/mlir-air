//===- core_put_acquire_before_write.mlir ----------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-to-aie="use-lock-race-condition-fix-v2=true row-offset=3 col-offset=2 device=xcve2802" | FileCheck %s

// AN OUTBOUND PUT'S ACQUIRE MUST PRECEDE THE OPS THAT WRITE THE BUFFER.
//
// A producing core is `write buffer; put buffer`. Placing the acquire on the
// put lets the write for iteration N+1 land while iteration N's transfer is
// still in flight -- the core only blocks afterwards, having already destroyed
// the bytes the DMA is reading. It is invisible whenever the consumer keeps up,
// and deterministic as soon as anything downstream stalls.
//
// Found on qwen3-4b's LM head, where a memtile relay's fill boundary stalls the
// stream for exactly one transfer: every fill after the first lost its row 0,
// and token 0's logits came back a blend of rows 0 and 1 (corr 0.845 to each,
// against 0.992/0.681 when correct) while rows 1..7 stayed bit-identical. With
// the acquire hoisted the logits are bit-identical to the reference build.
//
// THE SHAPE MATTERS, which is why this test has all of it. Two puts of the same
// buffer on ONE channel is what selects the interleaved lock placement, and the
// producing call sits behind an air.execute with an air.wait_all between it and
// the put. air.wait_all is token plumbing with no memory effect of its own but
// does not advertise itself as effect-free, so a naive backward scan stops
// there and the hoist silently does nothing -- which is exactly what happened.

// The PRODUCING core (x_loc=2, y_loc=3). Both loops must read
// acquire-then-write, not write-then-acquire.
// CHECK: aie.core(%tile_2_3)
// CHECK: scf.for
// CHECK-NEXT: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: func.call @kern(%{{.*}})
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %c1_i32)
// CHECK: scf.for
// CHECK-NEXT: aie.use_lock(%{{.*}}, AcquireGreaterEqual, %c1_i32)
// CHECK-NEXT: func.call @kern(%{{.*}})
// CHECK-NEXT: aie.use_lock(%{{.*}}, Release, %c1_i32)

air.channel @drain [1, 1]
func.func private @kern(memref<32xbf16, 2>) attributes {link_with = "k.o", llvm.emit_c_interface}
func.func @core_put_acquire_before_write() {
  %c1 = arith.constant 1 : index
  air.launch (%a, %b) in (%c=%c1, %d=%c1) {
    air.segment @seg {
      %c1_0 = arith.constant 1 : index
      air.herd @hp tile (%tx, %ty) in (%sx=%c1_0, %sy=%c1_0)
            attributes {x_loc = 2 : i64, y_loc = 3 : i64} {
        %c1_h = arith.constant 1 : index
        %c0_h = arith.constant 0 : index
        %c4_h = arith.constant 4 : index
        %tok, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %aa = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %aa : memref<32xbf16, 2>
        }
        scf.for %i = %c0_h to %c4_h step %c1_h {
          %e0 = air.execute {
            func.call @kern(%l1) : (memref<32xbf16, 2>) -> ()
          }
          air.channel.put @drain[] (%l1[] [] []) : (memref<32xbf16, 2>)
        }
        scf.for %i2 = %c0_h to %c4_h step %c1_h {
          %e1 = air.execute {
            func.call @kern(%l1) : (memref<32xbf16, 2>) -> ()
          }
          air.channel.put @drain[] (%l1[] [] []) : (memref<32xbf16, 2>)
        }
        %da = air.execute {memref.dealloc %l1 : memref<32xbf16, 2>}
      }
      air.herd @hr tile (%txr, %tyr) in (%sxr=%c1_0, %syr=%c1_0)
            attributes {x_loc = 4 : i64, y_loc = 3 : i64} {
        %c1_r = arith.constant 1 : index
        %c0_r = arith.constant 0 : index
        %c4_r = arith.constant 4 : index
        %tok, %l1 = air.execute -> (memref<32xbf16, 2>) {
          %aa = memref.alloc() : memref<32xbf16, 2>
          air.execute_terminator %aa : memref<32xbf16, 2>
        }
        scf.for %i = %c0_r to %c4_r step %c1_r {
          air.channel.get @drain[] (%l1[] [] []) : (memref<32xbf16, 2>)
        }
        scf.for %i2 = %c0_r to %c4_r step %c1_r {
          air.channel.get @drain[] (%l1[] [] []) : (memref<32xbf16, 2>)
        }
        %dr = air.execute {memref.dealloc %l1 : memref<32xbf16, 2>}
      }
    }
  }
  return
}
