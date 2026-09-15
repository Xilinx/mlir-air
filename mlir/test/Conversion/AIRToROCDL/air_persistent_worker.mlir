//===- air_persistent_worker.mlir -------------------------------*- MLIR -*-===//
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

// The shape a Fleet-style megakernel has, written in AIR: one launch sized to
// the dies, one segment per workgroup that stays resident, and the work loop
// inside it. A segment instance here is not "one unit of work" -- it is a
// worker that lives for the whole run, which is the opposite of how a segment
// is instantiated on AIE.
//
// The nesting is load-bearing. Chiplet identity has to be discovered outside
// the loop: putting it inside would re-run the barrier every iteration, and
// the point of knowing the die is that the data it cached stays useful across
// iterations. This test pins the order -- discovery, then loop -- because
// nothing downstream would notice if it flipped.

// CHECK-LABEL: func.func @worker

// One launch, one grid, sized to the dies rather than to the problem.
// CHECK:       gpu.launch blocks
// CHECK-SAME:    %{{.*}} = %c8

// Discovery first, and only once: one register read and one barrier, both
// above the loop.
// CHECK:         llvm.call_intrinsic "llvm.amdgcn.s.getreg"
// CHECK:         llvm.atomicrmw add %{{.*}} syncscope("agent") release
// CHECK:         gpu.barrier
// CHECK:         %[[COOP:.*]]:2 = scf.for

// Then the work loop, inside the kernel body...
// CHECK:         scf.for %{{.*}} = %{{.*}} to %{{.*}} step
// ...with the tiles split between the workers sharing this die: worker
// %[[COOP]]#0 of %[[COOP]]#1 walks every %[[COOP]]#1-th tile from its own rank.
// CHECK:           scf.for %{{.*}} = %[[COOP]]#0 to %{{.*}} step %[[COOP]]#1

// CHECK-NOT:   air.

// Down at the instruction level, the die is read once for the whole kernel
// rather than once per iteration -- the prose above says "outside the loop",
// this is what checks it.
// ISA: s_getreg_b32 {{s[0-9]+}}, hwreg(HW_REG_XCC_ID, 0, 16)
// ISA-NOT: s_getreg_b32 {{s[0-9]+}}, hwreg(HW_REG_XCC_ID

module {
  func.func @worker(%w: memref<1024xf32>, %o: memref<1024xf32>, %niters: index) {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    air.launch (%bx, %by) in (%nbx=%c8, %nby=%c1) args(%aw=%w, %ao=%o, %ni=%niters) : memref<1024xf32>, memref<1024xf32>, index {
      air.segment @persistent_worker args(%sw=%aw, %so=%ao, %sn=%ni) : memref<1024xf32>, memref<1024xf32>, index {
        %c0_s = arith.constant 0 : index
        %c1_s = arith.constant 1 : index
        %c64_s = arith.constant 64 : index
        // Discovered once, outside the work loop: the die binding has to
        // outlive every iteration.
        %rank = air.chiplet_block_id
        %n    = air.chiplet_dim_blocks
        scf.for %iter = %c0_s to %sn step %c1_s {
          // Cooperative tiling: worker %rank of %n walks every %n-th tile.
          scf.for %t = %rank to %c64_s step %n {
            %v = memref.load %sw[%t] : memref<1024xf32>
            %a = arith.addf %v, %v : f32
            memref.store %a, %so[%t] : memref<1024xf32>
          }
        }
        air.herd @herd tile (%tx, %ty) in (%ntx=%c1_s, %nty=%c1_s) {
        }
      }
    }
    return
  }
}
