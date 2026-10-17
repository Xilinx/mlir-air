//===- shrink_memref_sizes_by_access_slots.mlir ----------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-shrink-memref-sizes-by-access | FileCheck %s

// A memtile buffer of four slots, each transfer moving one slot at a constant
// offset. The buffer keeps its four slots (shrinking it to one transfer's
// size would put three of them past its end).

// CHECK-LABEL: func.func @constant
// CHECK: memref.alloc() : memref<4x2x64xi16, 1>
// CHECK-NOT: memref<1x2x64xi16, 1>

module {
  air.channel @in [1, 1]
  air.channel @out [1, 1]
  func.func @constant() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%l) in (%ls=%c1) {
      %1 = air.segment @seg2 async {
        %t, %buf = air.execute -> (memref<4x2x64xi16, 1>) {
          %a = memref.alloc() : memref<4x2x64xi16, 1>
          air.execute_terminator %a : memref<4x2x64xi16, 1>
        }
        %g0 = air.channel.get async [%t] @in[] (%buf[0, 0, 0] [1, 2, 64] [128, 64, 1]) : (memref<4x2x64xi16, 1>)
        %g1 = air.channel.get async [%g0] @in[] (%buf[1, 0, 0] [1, 2, 64] [128, 64, 1]) : (memref<4x2x64xi16, 1>)
        %g2 = air.channel.get async [%g1] @in[] (%buf[2, 0, 0] [1, 2, 64] [128, 64, 1]) : (memref<4x2x64xi16, 1>)
        %g3 = air.channel.get async [%g2] @in[] (%buf[3, 0, 0] [1, 2, 64] [128, 64, 1]) : (memref<4x2x64xi16, 1>)
        %p0 = air.channel.put async [%g3] @out[] (%buf[0, 0, 0] [1, 2, 64] [128, 64, 1]) : (memref<4x2x64xi16, 1>)
        %p1 = air.channel.put async [%p0] @out[] (%buf[1, 0, 0] [1, 2, 64] [128, 64, 1]) : (memref<4x2x64xi16, 1>)
        %p2 = air.channel.put async [%p1] @out[] (%buf[2, 0, 0] [1, 2, 64] [128, 64, 1]) : (memref<4x2x64xi16, 1>)
        %p3 = air.channel.put async [%p2] @out[] (%buf[3, 0, 0] [1, 2, 64] [128, 64, 1]) : (memref<4x2x64xi16, 1>)
        %4 = air.execute [%p3] {
          memref.dealloc %buf : memref<4x2x64xi16, 1>
        }
      }
    }
    return
  }
}
