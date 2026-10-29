//===- runtime_repeat_count_one_bd_fixed_inner.mlir ------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -split-input-file -air-to-aie="row-offset=2 col-offset=0 device=npu2" | FileCheck %s

// Core 0 fills %src once, then sends %out five times per trip of a run-time
// loop from a fixed loop of its own. Under a condition it fills %src again
// and sends %out from fixed loops only. Every transfer on each of its DMA
// channels moves the same buffer, so each channel is one BD that loops to
// itself, whatever loops the transfers sit in.

// CHECK: %[[T0:.*]] = aie.tile(0, 2)
// CHECK: %[[SRC:.*]] = aie.buffer(%[[T0]]) {{.*}} : memref<320xbf16, 2>
// CHECK: %[[OUT:.*]] = aie.buffer(%[[T0]]) {{.*}} : memref<64xbf16, 2>
// CHECK: aie.mem(%[[T0]])
// CHECK: aie.dma_start(MM2S, 0, ^[[PUT:.*]], ^
// CHECK: ^[[PUT]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[OUT]] : memref<64xbf16, 2>
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[PUT]]
// CHECK: aie.dma_start(S2MM, 0, ^[[GET:.*]], ^
// CHECK: ^[[GET]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[SRC]] : memref<320xbf16, 2>
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[GET]]

air.channel @fill [1, 1]
air.channel @out [1, 1]
func.func @one_bd_fixed_inner(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c1_0 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c2) args(%hn=%sn) : i32 {
        %c0 = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c3 = arith.constant 3 : index
        %c5 = arith.constant 5 : index
        %c0_i32 = arith.constant 0 : i32
        %n0 = arith.index_cast %hn : i32 to index
        %again = arith.cmpi ne, %hn, %c0_i32 : i32
        %t0, %src = air.execute -> (memref<320xbf16, 2>) {
          %alloc = memref.alloc() : memref<320xbf16, 2>
          air.execute_terminator %alloc : memref<320xbf16, 2>
        }
        %t1, %out = air.execute -> (memref<64xbf16, 2>) {
          %alloc = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %alloc : memref<64xbf16, 2>
        }
        %is0 = arith.cmpi eq, %ty, %c0 : index
        scf.if %is0 {
          air.channel.get @fill[] (%src[] [] []) : (memref<320xbf16, 2>)
          scf.for %i = %c0 to %n0 step %c1_h {
            scf.for %j = %c0 to %c5 step %c1_h {
              air.channel.put @out[] (%out[] [] []) : (memref<64xbf16, 2>)
            }
          }
          scf.if %again {
            air.channel.get @fill[] (%src[] [] []) : (memref<320xbf16, 2>)
            scf.for %i = %c0 to %c3 step %c1_h {
              scf.for %j = %c0 to %c5 step %c1_h {
                air.channel.put @out[] (%out[] [] []) : (memref<64xbf16, 2>)
              }
            }
          }
        } else {
          air.channel.put @fill[] (%src[] [] []) : (memref<320xbf16, 2>)
          scf.for %i = %c0 to %n0 step %c1_h {
            scf.for %j = %c0 to %c5 step %c1_h {
              air.channel.get @out[] (%out[] [] []) : (memref<64xbf16, 2>)
            }
          }
          scf.if %again {
            air.channel.put @fill[] (%src[] [] []) : (memref<320xbf16, 2>)
            scf.for %i = %c0 to %c3 step %c1_h {
              scf.for %j = %c0 to %c5 step %c1_h {
                air.channel.get @out[] (%out[] [] []) : (memref<64xbf16, 2>)
              }
            }
          }
        }
        %d0 = air.execute {memref.dealloc %src : memref<320xbf16, 2>}
        %d1 = air.execute {memref.dealloc %out : memref<64xbf16, 2>}
      }
    }
  }
  return
}

// -----

// The same from a memtile: an L2 buffer sent five times per trip of a
// run-time loop from a fixed loop of its own, then under a condition from
// fixed loops only, and the core receiving it the same way. Both channels
// are one BD that loops to itself.

// CHECK: %[[T1:.*]] = aie.tile(0, 2)
// CHECK: %[[L1:.*]] = aie.buffer(%[[T1]]) {{.*}} : memref<64xbf16, 2>
// CHECK: aie.mem(%[[T1]])
// CHECK: aie.dma_start(S2MM, 0, ^[[CGET:.*]], ^
// CHECK: ^[[CGET]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[L1]] : memref<64xbf16, 2>
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[CGET]]
// CHECK: %[[L2:.*]] = aie.buffer(%{{.*}}) {{.*}} : memref<64xbf16, 1>
// CHECK: aie.memtile_dma
// CHECK: aie.dma_start(MM2S, 0, ^[[MPUT:.*]], ^
// CHECK: ^[[MPUT]]:
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.dma_bd(%[[L2]] : memref<64xbf16, 1>
// CHECK-NEXT: aie.use_lock
// CHECK-NEXT: aie.next_bd ^[[MPUT]]

air.channel @l2out [1, 1]
func.func @memtile_one_bd(%n: i32) {
  %c1 = arith.constant 1 : index
  air.launch (%x, %y) in (%sx=%c1, %sy=%c1) args(%ln=%n) : i32 {
    air.segment @seg args(%sn=%ln) : i32 {
      %c0 = arith.constant 0 : index
      %c1_0 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      %c5 = arith.constant 5 : index
      %c0_i32 = arith.constant 0 : i32
      %k = arith.index_cast %sn : i32 to index
      %again = arith.cmpi ne, %sn, %c0_i32 : i32
      %t, %l2 = air.execute -> (memref<64xbf16, 1>) {
        %alloc = memref.alloc() : memref<64xbf16, 1>
        air.execute_terminator %alloc : memref<64xbf16, 1>
      }
      scf.for %i = %c0 to %k step %c1_0 {
        scf.for %j = %c0 to %c5 step %c1_0 {
          air.channel.put @l2out[] (%l2[] [] []) : (memref<64xbf16, 1>)
        }
      }
      scf.if %again {
        scf.for %i = %c0 to %c3 step %c1_0 {
          scf.for %j = %c0 to %c5 step %c1_0 {
            air.channel.put @l2out[] (%l2[] [] []) : (memref<64xbf16, 1>)
          }
        }
      }
      %d = air.execute {memref.dealloc %l2 : memref<64xbf16, 1>}
      air.herd @h tile (%tx, %ty) in (%hx=%c1_0, %hy=%c1_0) args(%hn=%sn) : i32 {
        %c0_h = arith.constant 0 : index
        %c1_h = arith.constant 1 : index
        %c3_h = arith.constant 3 : index
        %c5_h = arith.constant 5 : index
        %c0_i32_h = arith.constant 0 : i32
        %kh = arith.index_cast %hn : i32 to index
        %ag = arith.cmpi ne, %hn, %c0_i32_h : i32
        %t1, %l1 = air.execute -> (memref<64xbf16, 2>) {
          %alloc = memref.alloc() : memref<64xbf16, 2>
          air.execute_terminator %alloc : memref<64xbf16, 2>
        }
        scf.for %i = %c0_h to %kh step %c1_h {
          scf.for %j = %c0_h to %c5_h step %c1_h {
            air.channel.get @l2out[] (%l1[] [] []) : (memref<64xbf16, 2>)
          }
        }
        scf.if %ag {
          scf.for %i = %c0_h to %c3_h step %c1_h {
            scf.for %j = %c0_h to %c5_h step %c1_h {
              air.channel.get @l2out[] (%l1[] [] []) : (memref<64xbf16, 2>)
            }
          }
        }
        %d1 = air.execute {memref.dealloc %l1 : memref<64xbf16, 2>}
      }
    }
  }
  return
}
