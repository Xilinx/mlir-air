//===- named_op_costs.mlir ----------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-runner %s -f test -m %S/arch.json -g core | FileCheck %s

// Three ops of the same kind and the same shape, costing three different
// things.
//
// The model's kernels are keyed by op name, which is enough only while an op
// name identifies the work. It often does not: the projections of a
// transformer layer are all a matvec over the same activation, and they cost
// differently. Keyed by name they would share one entry and there would be no
// way to tell them apart -- the reason to reach for air.custom, which gives up
// an op that says what it computes in exchange for a symbol to hang a number
// on.
//
// `air.op_cost` names the entry instead, so an op can be both.
//
//     proj_a  ceildiv(volume0, 4096) * 40 +  20 =  60
//     proj_b  ceildiv(volume0, 4096) * 40 +  60 = 100
//     proj_c  ceildiv(volume0, 4096) * 40 + 140 = 180

// CHECK: "name": "LinalgOp(linalg.matvec)",
// CHECK: "ph": "B",
// CHECK: "ts": 0.00[[#%d,T0:]],
// CHECK: "name": "LinalgOp(linalg.matvec)",
// CHECK: "ph": "E",
// CHECK: "ts": 0.0[[#T0 + 60]],

// CHECK: "name": "LinalgOp(linalg.matvec)",
// CHECK: "ph": "B",
// CHECK: "ts": 0.0[[#%d,T1:]],
// CHECK: "name": "LinalgOp(linalg.matvec)",
// CHECK: "ph": "E",
// CHECK: "ts": 0.[[#T1 + 100]],

// CHECK: "name": "LinalgOp(linalg.matvec)",
// CHECK: "ph": "B",
// CHECK: "ts": 0.[[#%d,T2:]],
// CHECK: "name": "LinalgOp(linalg.matvec)",
// CHECK: "ph": "E",
// CHECK: "ts": 0.[[#T2 + 180]],

// CHECK: "name": "LaunchTerminator",
// CHECK: "ph": "E",

module {
  func.func @test() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) attributes {id = 1 : i32} {
      %1 = air.segment @seg async attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c1_0 = arith.constant 1 : index
        %2 = air.herd @herd_0 async tile (%hx, %hy) in (%hsx=%c1_0, %hsy=%c1_0) attributes {id = 3 : i32, x_loc = 0 : i64, y_loc = 0 : i64} {
          // Weights are allocated and never written: on a weight-stationary
          // machine they are already resident, so the alloc is the residency
          // and there is no fill to model.
          %tw, %w = air.execute -> (memref<64x64xi8, 2>) {
            %alloc = memref.alloc() : memref<64x64xi8, 2>
            air.execute_terminator %alloc : memref<64x64xi8, 2>
          }
          %ta, %act = air.execute -> (memref<64xi8, 2>) {
            %alloc = memref.alloc() : memref<64xi8, 2>
            air.execute_terminator %alloc : memref<64xi8, 2>
          }
          %e0 = air.execute [%tw, %ta] {
            linalg.matvec {air.op_cost = "proj_a"} ins(%w, %act : memref<64x64xi8, 2>, memref<64xi8, 2>) outs(%act : memref<64xi8, 2>)
          }
          %e1 = air.execute [%e0] {
            linalg.matvec {air.op_cost = "proj_b"} ins(%w, %act : memref<64x64xi8, 2>, memref<64xi8, 2>) outs(%act : memref<64xi8, 2>)
          }
          %e2 = air.execute [%e1] {
            linalg.matvec {air.op_cost = "proj_c"} ins(%w, %act : memref<64x64xi8, 2>, memref<64xi8, 2>) outs(%act : memref<64xi8, 2>)
          }
        }
      }
    }
    return
  }
}
