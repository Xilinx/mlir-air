//===- air_chiplet_id.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// REQUIRES: gpu
// RUN: air-opt %s -air-to-rocdl | FileCheck %s
//
// And all the way down: the point of the op is the instruction it becomes, so
// compile the outlined kernel to gfx942 and look for it in the ISA.
// RUN: air-opt %s -air-to-rocdl -air-gpu-outlining \
// RUN:   | air-opt --pass-pipeline='builtin.module(rocdl-attach-target{chip=gfx942 O=3},gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl{chipset=gfx942 runtime=HIP},reconcile-unrealized-casts))' \
// RUN:   | mlir-opt --pass-pipeline='builtin.module(gpu-module-to-binary{format=isa})' \
// RUN:   | FileCheck %s --check-prefix=ISA
//
// ISA: s_getreg_b32 {{s[0-9]+}}, hwreg(HW_REG_XCC_ID, 0, 16)

// air.chiplet_id reports which accelerator die the workgroup landed on. ROCDL
// has no s_getreg op, so it lowers to the llvm.amdgcn.s.getreg intrinsic with
// the packed hwreg descriptor for HW_REG_XCC_ID read as bits [0, 16):
//   id | (offset << 6) | ((size - 1) << 11) = 20 | (15 << 11) = 30740
// which llc -mcpu=gfx942 prints back as hwreg(HW_REG_XCC_ID, 0, 16) -- the
// same operand Fleet writes by hand in persistent_kernel.cuh:188.

// CHECK-LABEL: func.func @chiplet_id
// CHECK:       gpu.launch
// CHECK:         %[[DESC:.*]] = arith.constant 30740 : i32
// CHECK:         %[[RAW:.*]] = llvm.call_intrinsic "llvm.amdgcn.s.getreg"(%[[DESC]])
// CHECK:         %[[ID:.*]] = arith.index_cast %[[RAW]] : i32 to index
// CHECK:         memref.store %{{.*}}, %{{.*}}[%[[ID]]]
// CHECK-NOT:   air.chiplet_id

module {
  func.func @chiplet_id(%arg0: memref<8xf32>) {
    %c1 = arith.constant 1 : index
    air.launch (%bx, %by) in (%nbx=%c1, %nby=%c1) args(%out=%arg0) : memref<8xf32> {
      // Read in the segment body: a segment is one workgroup, and chiplet
      // identity is a property of where that workgroup was placed.
      air.segment @seg args(%s0=%out) : memref<8xf32> {
        %c1_s = arith.constant 1 : index
        %xcd = air.chiplet_id
        %one = arith.constant 1.0 : f32
        // Each workgroup stamps a marker into the slot for its own die.
        memref.store %one, %s0[%xcd] : memref<8xf32>
        air.herd @herd tile (%tx, %ty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
