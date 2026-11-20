//===- rerun_core_loop.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt -air-to-aie="emit-while-loop=false row-offset=2 col-offset=0 device=npu2" -split-input-file %s | FileCheck %s

// A core that reruns its herd's body (here, a launch in a loop of the host
// function) loops even without emit-while-loop; one that runs it once ends.

// CHECK-LABEL: aie.device(npu2) @rep
// CHECK: aie.core
// CHECK: ^bb1:
// CHECK: cf.br ^bb2
// CHECK: ^bb2:
// CHECK-NEXT: cf.br ^bb1
// CHECK-NEXT: } {{{.*}}air.reruns
func.func @in_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    air.launch (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
      air.segment @rep {
        %c1_s = arith.constant 1 : index
        air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) {
          %a = memref.alloc() : memref<16xi32, 2>
          memref.dealloc %a : memref<16xi32, 2>
        }
      }
    }
  }
  return
}

// -----

// CHECK-LABEL: aie.device(npu2) @once
// CHECK: aie.core
// CHECK: ^bb2:
// CHECK-NEXT: aie.end
// CHECK-NOT: air.reruns
func.func @single() {
  %c1 = arith.constant 1 : index
  air.launch (%lx, %ly) in (%lsx=%c1, %lsy=%c1) {
    air.segment @once {
      %c1_s = arith.constant 1 : index
      air.herd @h tile (%tx, %ty) in (%sx=%c1_s, %sy=%c1_s) {
        %a = memref.alloc() : memref<16xi32, 2>
        memref.dealloc %a : memref<16xi32, 2>
      }
    }
  }
  return
}
