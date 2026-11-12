//===- host_readback_after_loop.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-dependency -air-dma-to-channel | FileCheck %s

// Herd @w writes a host buffer in a loop (a segment's output drained to host
// memory), and herd @r of the same launch reads part of it back. Hoisted to
// the launch, the read's put depends on the nest of the writes' gets: with no
// edge it could be moved ahead of the writes it reads.

// CHECK-LABEL: func.func @readback
// CHECK: %[[W:.*]] = scf.parallel
// CHECK: scf.for
// CHECK: air.channel.get
// CHECK: air.channel.put async [%[[W]], %{{.*}}]

module {
  func.func @readback(%buf: memref<128xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%lx, %ly) in (%lsx=%c1, %lsy=%c1) args(%b=%buf) : memref<128xi32> {
      air.segment @seg args(%sb=%b) : memref<128xi32> {
        %c1_s = arith.constant 1 : index
        air.herd @w tile (%x, %y) in (%sx=%c1_s, %sy=%c1_s) args(%hb=%sb) : memref<128xi32> {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c2 = arith.constant 2 : index
          %c64 = arith.constant 64 : index
          %l1 = memref.alloc() : memref<64xi32, 2>
          scf.for %i = %c0 to %c2 step %c1_h {
            %o = arith.muli %i, %c64 : index
            air.dma_memcpy_nd (%hb[%o] [%c64] [%c1_h], %l1[] [] []) : (memref<128xi32>, memref<64xi32, 2>)
          }
          memref.dealloc %l1 : memref<64xi32, 2>
        }
        air.herd @r tile (%x, %y) in (%sx=%c1_s, %sy=%c1_s) args(%hb=%sb) : memref<128xi32> {
          %c0 = arith.constant 0 : index
          %c1_h = arith.constant 1 : index
          %c64 = arith.constant 64 : index
          %l1 = memref.alloc() : memref<64xi32, 2>
          air.dma_memcpy_nd (%l1[] [] [], %hb[%c0] [%c64] [%c1_h]) : (memref<64xi32, 2>, memref<128xi32>)
          memref.dealloc %l1 : memref<64xi32, 2>
        }
      }
    }
    return
  }
}
