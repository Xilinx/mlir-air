//===- air_chiplet_rank.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl | FileCheck %s
//
// And all the way down. The scope on these atomics is the whole point -- it is
// what keeps the coordination on the die instead of crossing the fabric -- so
// check the instructions it becomes, not just the MLIR attribute.
// RUN: %if amdgpu-isa %{ air-opt %s -air-to-rocdl -air-gpu-outlining \
// RUN:   | air-opt --pass-pipeline='builtin.module(rocdl-attach-target{chip=gfx942 O=3},gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=gfx942 runtime=HIP},reconcile-unrealized-casts))' \
// RUN:   | mlir-opt --pass-pipeline='builtin.module(gpu-module-to-binary{format=isa})' \
// RUN:   | FileCheck %s --check-prefix=ISA %}
//
// The die is read from the hardware register, as in air_chiplet_id.mlir.
// ISA-DAG: s_getreg_b32 {{s[0-9]+}}, hwreg(HW_REG_XCC_ID, 0, 16)
//
// syncscope("agent") becomes sc1 and nothing else: the writeback and the
// invalidate reach the other dies on this device and stop there. System scope
// would be `sc0 sc1` and would push the traffic off-chip -- that is what Fleet
// measured 14.5x of, and what the two-level protocol exists to avoid.
// ISA-DAG: buffer_wbl2 sc1
// ISA-DAG: buffer_inv sc1
// ISA-DAG: flat_atomic_add

// air.chiplet_block_id and air.chiplet_dim_blocks answer "which of the
// workgroups on my die am I, and how many of us are there". Neither is
// readable from a register: the dispatcher decides the placement, so the
// workgroups have to tell each other. This is Fleet's protocol
// (persistent_kernel.cuh:1083-1094):
//
//   map[block_id] = chiplet_id        one store per workgroup
//   <everyone has stored>             device-wide barrier
//   rank  = |{w : map[w] == mine, w < me}|
//   count = |{w : map[w] == mine}|
//
// Both ops come from one run of it, so a program asking for both pays for one
// barrier and one scan.

// The reporting array and the barrier state are module-scope globals, zeroed.
// CHECK-DAG:   memref.global "private" @__air_chiplet_map : memref<4096xi32> = dense<0>
// CHECK-DAG:   memref.global "private" @__air_chiplet_arrivals : memref<1xi32> = dense<0>
// CHECK-DAG:   memref.global "private" @__air_chiplet_generation : memref<1xi32> = dense<0>

// CHECK-LABEL: func.func @coop
// CHECK:       gpu.launch

// Each workgroup reports the die it landed on into its own slot.
// CHECK:         %[[XCD:.*]] = llvm.call_intrinsic "llvm.amdgcn.s.getreg"
// CHECK:         %[[MAP:.*]] = memref.get_global @__air_chiplet_map

// Only thread 0 stores and runs the barrier; the rest of the workgroup waits
// on gpu.barrier instead of duplicating the arrival.
// CHECK:         scf.if
// CHECK:           memref.store %[[XCD]], %[[MAP]]

// The barrier is sense-reversing rather than a one-shot count, so that a
// kernel launched twice does not find the counter already past the mark, skip
// the wait, and scan a half-filled map. Read the generation on entry...
// CHECK:           %[[GEN0:.*]] = llvm.load %{{.*}} atomic syncscope("agent") acquire
// ...then announce arrival. The slot store must reach the other dies before
// the arrival that announces it, hence release here and acquire on every poll,
// both at agent scope -- device wide, since the workgroups being waited on are
// on other dies.
// CHECK:           %[[PREV:.*]] = llvm.atomicrmw add %{{.*}} syncscope("agent") release
// CHECK:           arith.cmpi eq, %[[PREV]]
// CHECK:           scf.if
// The last one in rearms the counter, then opens the gate.
// CHECK:             llvm.atomicrmw xchg %{{.*}} syncscope("agent") monotonic
// CHECK:             llvm.atomicrmw add %{{.*}} syncscope("agent") release
// CHECK:           } else {
// Everyone else waits for the generation to move off the one they came in on.
// CHECK:             scf.while
// CHECK:               %[[GEN:.*]] = llvm.load %{{.*}} atomic syncscope("agent") acquire
// CHECK:               arith.cmpi eq, %[[GEN]], %[[GEN0]]
// CHECK:         gpu.barrier

// The scan is a pure function of (map, block_id, chiplet_id), so every thread
// recomputes it and no broadcast through LDS is needed.
// CHECK:         scf.for
// CHECK:           memref.load %[[MAP]]
// CHECK:           arith.cmpi eq
// CHECK:           arith.cmpi ult

// Two questions, one answer: the barrier and the scan run once for the pair,
// not once per op. A second register read here would mean a second barrier.
// CHECK-NOT:   llvm.amdgcn.s.getreg
// CHECK-NOT:   air.chiplet

module {
  func.func @coop(%arg0: memref<8xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%bx, %by) in (%nbx=%c1, %nby=%c1) args(%out=%arg0) : memref<8xf32> {
      air.segment @seg args(%s0=%out) : memref<8xf32> {
        %c1_s = arith.constant 1 : index
        // Cooperative tiling: worker %r of %n takes every %n-th tile from %r.
        %r = air.chiplet_block_id
        %n = air.chiplet_dim_blocks
        %ri = arith.index_cast %r : index to i32
        %rf = arith.sitofp %ri : i32 to f32
        memref.store %rf, %s0[%n] : memref<8xf32>
        air.herd @herd tile (%tx, %ty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
