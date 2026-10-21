// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
// RUN: air-opt %s -air-fuse-alloc-dealloc --split-input-file | FileCheck %s

// Keep only the marked buffer outside the K loop; default sinking still works.
// CHECK-LABEL: @mixed_raw
// CHECK: %[[PIN:.*]] = memref.alloc() {air.disable_alloc_sink}
// CHECK: scf.for
// CHECK-NOT: air.disable_alloc_sink
// CHECK: %[[MOVE:.*]] = memref.alloc()
// CHECK: func.call @use(%[[PIN]], %[[MOVE]])
// CHECK: memref.dealloc %[[MOVE]]
// CHECK: }
// CHECK: memref.dealloc %[[PIN]]
func.func private @use(memref<64xbf16, 2>, memref<64xbf16, 2>)
func.func @mixed_raw() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %pin = memref.alloc() {air.disable_alloc_sink} : memref<64xbf16, 2>
  %move = memref.alloc() : memref<64xbf16, 2>
  scf.for %i = %c0 to %c8 step %c1 {
    func.call @use(%pin, %move) : (memref<64xbf16, 2>, memref<64xbf16, 2>) -> ()
  }
  memref.dealloc %move : memref<64xbf16, 2>
  memref.dealloc %pin : memref<64xbf16, 2>
  return
}

// -----

// air-dependency's execute wrapper must not hide the placement constraint.
// CHECK-LABEL: @wrapped
// CHECK: air.execute
// CHECK: memref.alloc() {air.disable_alloc_sink}
// CHECK: air.execute_terminator
// CHECK: scf.for
// CHECK-NOT: memref.alloc
// CHECK: air.channel.put
// CHECK: scf.yield
// CHECK: }
// CHECK: memref.dealloc
air.channel @c [1, 1]
func.func @wrapped() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %t, %pin = air.execute -> (memref<64xbf16, 2>) {
    %a = memref.alloc() {air.disable_alloc_sink} : memref<64xbf16, 2>
    air.execute_terminator %a : memref<64xbf16, 2>
  }
  %r = scf.for %i = %c0 to %c8 step %c1 iter_args(%tok = %t) -> (!air.async.token) {
    %put = air.channel.put async [%tok] @c[] (%pin[] [] []) : (memref<64xbf16, 2>)
    scf.yield %put : !air.async.token
  }
  %done = air.execute [%r] {
    memref.dealloc %pin : memref<64xbf16, 2>
  }
  return
}
