//===- air_nontemporal.mlir -------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl | FileCheck %s
// RUN: air-opt %s -air-to-rocdl -air-gpu-outlining \
// RUN:   | air-opt --pass-pipeline='builtin.module(rocdl-attach-target{chip=gfx942 O=3},gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=gfx942 runtime=HIP},reconcile-unrealized-casts))' \
// RUN:   | mlir-opt --pass-pipeline='builtin.module(gpu-module-to-binary{format=isa})' \
// RUN:   | FileCheck %s --check-prefix=ISA

// Streaming data -- read once, never reused -- should not evict what the die's
// L2 is holding for everyone else. On AMDGPU that is the `nt` bit, and the
// existing `nontemporal` attribute already reaches it: this test exists to say
// AIR does not need a cache-policy attribute of its own.
//
// Together with the agent-scope atomics in air_chiplet_rank.mlir, this covers
// both halves of the aux operand a chiplet-aware lowering needs -- `nt` from
// nontemporal, `sc1` from syncscope -- with vocabulary that is already there.

// CHECK-LABEL: func.func @stream
// CHECK:       gpu.launch
// CHECK:         memref.load %{{.*}} {nontemporal = true}
// CHECK:         memref.store %{{.*}} {nontemporal = true}

// ISA: global_store{{[a-z0-9_]*}} {{.*}} nt

module {
  func.func @stream(%arg0: memref<64xf32>, %arg1: memref<64xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%bx, %by) in (%nbx=%c1, %nby=%c1) args(%in=%arg0, %out=%arg1) : memref<64xf32>, memref<64xf32> {
      air.segment @seg args(%s0=%in, %s1=%out) : memref<64xf32>, memref<64xf32> {
        %c1_s = arith.constant 1 : index
        %c0 = arith.constant 0 : index
        %v = memref.load %s0[%c0] {nontemporal = true} : memref<64xf32>
        memref.store %v, %s1[%c0] {nontemporal = true} : memref<64xf32>
        air.herd @herd tile (%tx, %ty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
