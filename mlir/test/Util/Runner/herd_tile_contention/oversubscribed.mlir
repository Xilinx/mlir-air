//===- oversubscribed.mlir                          -*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// The same two independent herds, five tiles each, on a hierarchy of eight.
// They cannot both be resident, so the second waits for the first to release
// and the two serialise -- twice the latency of fits.mlir.
//
// A herd that cannot get its tiles used to be a hard error: the runner emitted
// 'isn't allocated with enough resources to run' and refused the design, while
// the channel ops beside it deferred and retried. Only a herd that cannot fit
// the hierarchy AT ALL is a configuration error; one that fits but finds its
// tiles busy is contending with a sibling, and two operators time-sharing one
// compute hierarchy is a mapping rather than a mistake.

// RUN: air-runner %s -f test -m %S/arch.json | FileCheck %s

// CHECK: Latency (all-iterations mode): 0.414us

module {
  func.func @test(%arg0: memref<64xi8>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%tx, %ty) in (%sx=%c1, %sy=%c1) args(%la=%arg0) : memref<64xi8> attributes {id = 1 : i32} {
      %1 = air.segment async args(%sa=%la) : memref<64xi8> attributes {id = 2 : i32, x_loc = 0 : i64, x_size = 8 : i64, y_loc = 0 : i64, y_size = 1 : i64} {
        %c1_s = arith.constant 1 : index
        %cn_s = arith.constant 5 : index
        %2 = air.herd @first async tile (%hx, %hy) in (%hsx=%c1_s, %hsy=%cn_s) attributes {id = 3 : i32} {
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
        %3 = air.herd @second async tile (%hx2, %hy2) in (%hsx2=%c1_s, %hsy2=%cn_s) attributes {id = 4 : i32} {
          %tok_w2, %w2 = air.execute -> (memref<8x8xi8, 2>) {
            %alloc = memref.alloc() : memref<8x8xi8, 2>
            air.execute_terminator %alloc : memref<8x8xi8, 2>
          }
          %tok_v2, %v2 = air.execute -> (memref<8xi8, 2>) {
            %alloc = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %alloc : memref<8xi8, 2>
          }
          %tok_o2, %o2 = air.execute -> (memref<8xi8, 2>) {
            %alloc = memref.alloc() : memref<8xi8, 2>
            air.execute_terminator %alloc : memref<8xi8, 2>
          }
          %tok_c2 = air.execute [%tok_w2, %tok_v2, %tok_o2] {
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
