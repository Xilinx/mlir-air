//===- scf_if_consumer_order.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency | FileCheck %s

// An op after a branch op that accesses what the branch accesses must wait
// for the branch op's token. Before, the dependency on the op inside the
// branch was dropped (its token cannot dominate the op after the branch) and
// nothing replaced it, so the dealloc below could free the buffer while the
// call used it.

func.func private @kernel(memref<48xi32, 2>)

// CHECK-LABEL: func.func @consumer_after_if
// CHECK: %[[A:.*]], %[[BUF:.*]] = air.execute -> (memref<48xi32, 2>)
// CHECK: %[[IN:.*]] = air.wait_all async [%[[A]]]
// CHECK: %[[IF:.*]] = scf.if
// CHECK:   func.call @kernel
// CHECK: } else {
// The empty region still yields what the branch op waits for.
// CHECK:   %[[E:.*]] = air.wait_all async [%[[IN]]]
// CHECK:   scf.yield %[[E]]
// CHECK: air.execute [%[[IF]]]
// CHECK-NEXT: memref.dealloc %[[BUF]]
func.func @consumer_after_if(%c: i1) {
  %c1 = arith.constant 1 : index
  air.herd @h tile (%x, %y) in (%sx = %c1, %sy = %c1) args(%cc = %c) : i1 {
    %a = memref.alloc() : memref<48xi32, 2>
    scf.if %cc {
      func.call @kernel(%a) : (memref<48xi32, 2>) -> ()
    }
    memref.dealloc %a : memref<48xi32, 2>
  }
  return
}

// A branch op nested in another: its token is yielded by the outer one.
// CHECK-LABEL: func.func @nested_if
// CHECK: %[[OUTER:.*]] = scf.if
// CHECK: } else {
// CHECK:   %[[INNER:.*]] = scf.if
// CHECK:     func.call @kernel
// CHECK:   %[[Y:.*]] = air.wait_all async [{{.*}}%[[INNER]]{{.*}}]
// CHECK:   scf.yield %[[Y]]
// CHECK: air.execute [%[[OUTER]]]
// CHECK-NEXT: memref.dealloc
func.func @nested_if(%c: i1, %d: i1) {
  %c1 = arith.constant 1 : index
  air.herd @h tile (%x, %y) in (%sx = %c1, %sy = %c1) args(%cc = %c, %dd = %d) : i1, i1 {
    %a = memref.alloc() : memref<48xi32, 2>
    scf.if %cc {
    } else {
      scf.if %dd {
        func.call @kernel(%a) : (memref<48xi32, 2>) -> ()
      }
    }
    memref.dealloc %a : memref<48xi32, 2>
  }
  return
}

// memref.copy inside a branch op is an async event like any other access.
// CHECK-LABEL: func.func @copy_in_if
// CHECK: %[[IF:.*]] = scf.if
// CHECK:   air.execute [{{.*}}] {
// CHECK-NEXT: memref.copy
// CHECK: air.execute [%[[IF]]]
// CHECK-NEXT: memref.dealloc
// CHECK: air.execute [%[[IF]]]
// CHECK-NEXT: memref.dealloc
func.func @copy_in_if(%c: i1) {
  %c1 = arith.constant 1 : index
  air.herd @h tile (%x, %y) in (%sx = %c1, %sy = %c1) args(%cc = %c) : i1 {
    %a = memref.alloc() : memref<48xi32, 2>
    %b = memref.alloc() : memref<48xi32, 2>
    scf.if %cc {
      memref.copy %a, %b : memref<48xi32, 2> to memref<48xi32, 2>
    }
    memref.dealloc %a : memref<48xi32, 2>
    memref.dealloc %b : memref<48xi32, 2>
  }
  return
}

// Two branch ops in a row accessing one buffer: the second one's token
// follows the first one's whichever region runs, so the dealloc after both
// waits for the first call even when the second condition is false.
// CHECK-LABEL: func.func @consecutive_ifs
// CHECK: %[[IF1:.*]] = scf.if
// CHECK:   func.call @kernel
// CHECK: %[[IF2:.*]] = scf.if
// CHECK:   air.execute [%[[IF1]]]
// CHECK-NEXT: func.call @kernel
// CHECK: } else {
// CHECK:   %[[E:.*]] = air.wait_all async [%[[IF1]]]
// CHECK:   scf.yield %[[E]]
// CHECK: air.execute [%[[IF2]]]
// CHECK-NEXT: memref.dealloc
func.func @consecutive_ifs(%c: i1, %d: i1) {
  %c1 = arith.constant 1 : index
  air.herd @h tile (%x, %y) in (%sx = %c1, %sy = %c1) args(%cc = %c, %dd = %d) : i1, i1 {
    %a = memref.alloc() : memref<48xi32, 2>
    scf.if %cc {
      func.call @kernel(%a) : (memref<48xi32, 2>) -> ()
    }
    scf.if %dd {
      func.call @kernel(%a) : (memref<48xi32, 2>) -> ()
    }
    memref.dealloc %a : memref<48xi32, 2>
  }
  return
}
