//===- metadata_before_compute.mlir ----------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// An air.execute region is priced by the op in it that computes, not by
// whichever op happens to come first.
//
// A region wraps whatever its producer put there, and only some of that has a
// cost: a linalg op or an air.custom does, while memref.expand_shape, subview
// and cast are metadata that lower to no instructions. air-dependency
// legitimately sinks a reshape into the region ahead of the compute it feeds,
// which used to leave the region costing nothing at all.
//
// Both herds below run one linalg.matvec priced at 200 cycles by the arch. The
// only difference is that the second one has an expand_shape in front of it,
// so if the region were priced by its first op the second herd would be free.
// Both must take the same time.

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// CHECK: "name": "LinalgOp(linalg.matvec)",
// CHECK: "ph": "B",
// CHECK: "ts": 0.00[[#%d,BARE:]],
// CHECK: "name": "LinalgOp(linalg.matvec)",
// CHECK: "ph": "E",
// CHECK: "ts": 0.[[#BARE + 200]],

// CHECK: "name": "LinalgOp(linalg.matvec)",
// CHECK: "ph": "B",
// CHECK: "ts": 0.[[#%d,BEHIND_RESHAPE:]],
// CHECK: "name": "LinalgOp(linalg.matvec)",
// CHECK: "ph": "E",
// CHECK: "ts": 0.[[#BEHIND_RESHAPE + 200]],

module {
  func.func @test(%arg0: memref<64xi8>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%la=%arg0) : memref<64xi8> attributes {id = 1 : i32} {
      %1 = air.segment async args(%sa=%la) : memref<64xi8> attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 1 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c1_s = arith.constant 1 : index

        // The compute is the first op in its region.
        %2 = air.herd @bare async tile (%hx, %hy) in (%hsx=%c1_s, %hsy=%c1_s) attributes {id = 3 : i32} {
          %tok_w, %w = air.execute -> (memref<8x8xi8, 2>) {
            %alloc = memref.alloc() : memref<8x8xi8, 2>
            air.execute_terminator %alloc : memref<8x8xi8, 2>
          }
          %tok_v, %v = air.execute -> (memref<8xi8, 2>) {
            %alloc = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %alloc : memref<8xi8, 2>
          }
          %tok_o, %o = air.execute -> (memref<8xi8, 2>) {
            %alloc = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %alloc : memref<8xi8, 2>
          }
          %tok_c = air.execute [%tok_w, %tok_v, %tok_o] {
            linalg.matvec {air.op_cost = "priced"} ins(%w, %v : memref<8x8xi8, 2>, memref<8xi8, 2>) outs(%o : memref<8xi8, 2>)
          }
        }

        // The same compute, with metadata ahead of it in the region.
        %3 = air.herd @behind_reshape async [%2] tile (%hx2, %hy2) in (%hsx2=%c1_s, %hsy2=%c1_s) attributes {id = 4 : i32} {
          %tok_f, %f = air.execute -> (memref<64xi8, 2>) {
            %alloc = memref.alloc() : memref<64xi8, 2>
            air.execute_terminator %alloc : memref<64xi8, 2>
          }
          %tok_v2, %v2 = air.execute -> (memref<8xi8, 2>) {
            %alloc = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %alloc : memref<8xi8, 2>
          }
          %tok_o2, %o2 = air.execute -> (memref<8xi8, 2>) {
            %alloc = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %alloc : memref<8xi8, 2>
          }
          %tok_c2 = air.execute [%tok_f, %tok_v2, %tok_o2] {
            %w2 = memref.expand_shape %f [[0, 1]] output_shape [8, 8] : memref<64xi8, 2> into memref<8x8xi8, 2>
            linalg.matvec {air.op_cost = "priced"} ins(%w2, %v2 : memref<8x8xi8, 2>, memref<8xi8, 2>) outs(%o2 : memref<8xi8, 2>)
          }
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}
