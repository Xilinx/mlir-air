//===- fuse_alloc_dealloc_across_loop.mlir ---------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-alloc-dealloc -split-input-file | FileCheck %s

// Moving an alloc into a loop, or across one into a hierarchy op, gives every
// iteration a fresh buffer. It is done only when no iteration reads what an
// earlier one left in the buffer.

// Each iteration writes one element and sends the whole buffer, including what
// earlier iterations wrote: the alloc stays outside the loop.

// CHECK-LABEL: func.func @carried_in_loop
// CHECK: memref.alloc()
// CHECK: scf.for
// CHECK-NOT: memref.alloc()
// CHECK: memref.store
// CHECK: air.channel.put
air.channel @c [1]
func.func @carried_in_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %a = memref.alloc() : memref<4xi32, 2 : i32>
  scf.for %i = %c0 to %c4 step %c1 {
    %v = arith.index_cast %i : index to i32
    memref.store %v, %a[%i] : memref<4xi32, 2 : i32>
    air.channel.put @c[] (%a[] [] []) : (memref<4xi32, 2 : i32>)
  }
  memref.dealloc %a : memref<4xi32, 2 : i32>
  return
}

// -----

// Each iteration receives the whole buffer before it reads it: the alloc moves
// into the loop.

// CHECK-LABEL: func.func @written_whole_in_loop
// CHECK: scf.for
// CHECK-NEXT: memref.alloc()
// CHECK: air.channel.get
// CHECK: air.channel.put
// CHECK: memref.dealloc
air.channel @c [1]
air.channel @d [1]
func.func @written_whole_in_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %a = memref.alloc() : memref<4xi32, 2 : i32>
  scf.for %i = %c0 to %c4 step %c1 {
    air.channel.get @d[] (%a[] [] []) : (memref<4xi32, 2 : i32>)
    air.channel.put @c[] (%a[] [] []) : (memref<4xi32, 2 : i32>)
  }
  memref.dealloc %a : memref<4xi32, 2 : i32>
  return
}

// -----

// A herd in a loop writes one element of a buffer it keeps from one iteration
// to the next: the alloc stays outside the loop.

// CHECK-LABEL: func.func @carried_across_loop
// CHECK: memref.alloc()
// CHECK: scf.for
// CHECK: air.herd {{.*}} args({{.*}}=%{{.*}}, {{.*}}) : memref<4xi32, 2 : i32>, index
// CHECK-NOT: memref.alloc()
air.channel @c [1]
func.func @carried_across_loop() {
  %c1 = arith.constant 1 : index
  air.segment @seg {
    %c0 = arith.constant 0 : index
    %c1s = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %a = memref.alloc() : memref<4xi32, 2 : i32>
    scf.for %i = %c0 to %c3 step %c1s {
      air.herd @h tile (%tx, %ty) in (%hx=%c1s, %hy=%c1s) args(%b=%a, %j=%i) : memref<4xi32, 2 : i32>, index {
        %v = arith.index_cast %j : index to i32
        memref.store %v, %b[%j] : memref<4xi32, 2 : i32>
        air.channel.put @c[] (%b[] [] []) : (memref<4xi32, 2 : i32>)
      }
    }
    memref.dealloc %a : memref<4xi32, 2 : i32>
  }
  return
}

// -----

// The herd receives the whole buffer before it reads it: the alloc moves into
// the herd.

// CHECK-LABEL: func.func @written_whole_across_loop
// CHECK: scf.for
// CHECK: air.herd
// CHECK: memref.alloc()
// CHECK: air.channel.get
air.channel @c [1]
air.channel @d [1]
func.func @written_whole_across_loop() {
  %c1 = arith.constant 1 : index
  air.segment @seg {
    %c0 = arith.constant 0 : index
    %c1s = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %a = memref.alloc() : memref<4xi32, 2 : i32>
    scf.for %i = %c0 to %c3 step %c1s {
      air.herd @h tile (%tx, %ty) in (%hx=%c1s, %hy=%c1s) args(%b=%a) : memref<4xi32, 2 : i32> {
        air.channel.get @d[] (%b[] [] []) : (memref<4xi32, 2 : i32>)
        air.channel.put @c[] (%b[] [] []) : (memref<4xi32, 2 : i32>)
      }
    }
    memref.dealloc %a : memref<4xi32, 2 : i32>
  }
  return
}

// -----

// Each arm of a branch receives the whole buffer from its own channel, with a
// pattern that visits the dims in another order: the alloc moves into the
// loop.

// CHECK-LABEL: func.func @written_whole_in_every_arm
// CHECK: scf.for
// CHECK-NEXT: memref.alloc()
air.channel @c [1]
air.channel @d [1]
air.channel @e [1]
func.func @written_whole_in_every_arm(%p: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %a = memref.alloc() : memref<2x4xi32, 2 : i32>
  scf.for %i = %c0 to %c4 step %c1 {
    scf.if %p {
      air.channel.get @d[] (%a[%c0, %c0] [%c4, %c2] [%c1, %c4]) : (memref<2x4xi32, 2 : i32>)
    } else {
      air.channel.get @e[] (%a[] [] []) : (memref<2x4xi32, 2 : i32>)
    }
    air.channel.put @c[] (%a[] [] []) : (memref<2x4xi32, 2 : i32>)
  }
  memref.dealloc %a : memref<2x4xi32, 2 : i32>
  return
}

// -----

// Only one arm writes the buffer, and the put after the branch reads it: on
// the other arm it sends what an earlier iteration wrote. The alloc stays.

// CHECK-LABEL: func.func @written_in_one_arm
// CHECK: memref.alloc()
// CHECK: scf.for
// CHECK-NOT: memref.alloc()
air.channel @c [1]
air.channel @d [1]
func.func @written_in_one_arm(%p: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %a = memref.alloc() : memref<2x4xi32, 2 : i32>
  scf.for %i = %c0 to %c4 step %c1 {
    scf.if %p {
      air.channel.get @d[] (%a[] [] []) : (memref<2x4xi32, 2 : i32>)
    }
    air.channel.put @c[] (%a[] [] []) : (memref<2x4xi32, 2 : i32>)
  }
  memref.dealloc %a : memref<2x4xi32, 2 : i32>
  return
}

// -----

// A herd receives the buffer under an affine.if on its tile ids that every
// tile of the 4x1 herd satisfies, so every iteration writes it whole: the
// alloc moves into the loop.

// CHECK-LABEL: func.func @every_tile_writes
// CHECK: air.herd
// CHECK: scf.for
// CHECK-NEXT: memref.alloc()
air.channel @c [4, 1]
air.channel @d [4, 1]
func.func @every_tile_writes() {
  %c4 = arith.constant 4 : index
  %c1 = arith.constant 1 : index
  air.herd @h tile (%tx, %ty) in (%sx=%c4, %sy=%c1) {
    %c0 = arith.constant 0 : index
    %c1_0 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %a = memref.alloc() : memref<8xi32, 2 : i32>
    scf.for %i = %c0 to %c8 step %c1_0 {
      affine.if affine_set<()[s0, s1] : (s0 >= 0, -s0 + 3 >= 0, s1 == 0)>()[%tx, %ty] {
        air.channel.get @d[%tx, %ty] (%a[] [] []) : (memref<8xi32, 2 : i32>)
      }
      air.channel.put @c[%tx, %ty] (%a[] [] []) : (memref<8xi32, 2 : i32>)
    }
    memref.dealloc %a : memref<8xi32, 2 : i32>
  }
  return
}

// -----

// Four gets fill the four slices of the buffer before the put reads it: the
// alloc moves into the loop.

// CHECK-LABEL: func.func @written_whole_in_pieces
// CHECK: scf.for
// CHECK-NEXT: memref.alloc()
air.channel @c [1]
air.channel @d [4]
func.func @written_whole_in_pieces() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  %c8 = arith.constant 8 : index
  %a = memref.alloc() : memref<4x8xi32, 1 : i32>
  scf.for %i = %c0 to %c4 step %c1 {
    air.channel.get @d[%c0] (%a[%c0, %c0] [%c1, %c8] [%c8, %c1]) : (memref<4x8xi32, 1 : i32>)
    air.channel.get @d[%c1] (%a[%c1, %c0] [%c1, %c8] [%c8, %c1]) : (memref<4x8xi32, 1 : i32>)
    air.channel.get @d[%c2] (%a[%c2, %c0] [%c1, %c8] [%c8, %c1]) : (memref<4x8xi32, 1 : i32>)
    air.channel.get @d[%c3] (%a[%c3, %c0] [%c1, %c8] [%c8, %c1]) : (memref<4x8xi32, 1 : i32>)
    air.channel.put @c[] (%a[] [] []) : (memref<4x8xi32, 1 : i32>)
  }
  memref.dealloc %a : memref<4x8xi32, 1 : i32>
  return
}

// -----

// A transposed vector write: each row step writes one column element, so the
// steps cover column 0 only and the put reads what earlier iterations left in
// the rest. The alloc stays outside the loop.

// CHECK-LABEL: func.func @transposed_write_in_steps
// CHECK: memref.alloc()
// CHECK-NEXT: scf.for
// CHECK-NOT: memref.alloc()
// CHECK: air.channel.put
air.channel @c [1]
func.func @transposed_write_in_steps(%v: vector<1x8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %a = memref.alloc() : memref<4x8xi32, 2 : i32>
  scf.for %i = %c0 to %c4 step %c1 {
    scf.for %j = %c0 to %c4 step %c1 {
      vector.transfer_write %v, %a[%j, %c0] {permutation_map = affine_map<(d0, d1) -> (d1, d0)>} : vector<1x8xi32>, memref<4x8xi32, 2 : i32>
    }
    air.channel.put @c[] (%a[] [] []) : (memref<4x8xi32, 2 : i32>)
  }
  memref.dealloc %a : memref<4x8xi32, 2 : i32>
  return
}

// -----

// A vector with as many elements as the buffer but another shape writes only
// the first two columns: the alloc stays outside the loop.

// CHECK-LABEL: func.func @vector_of_other_shape
// CHECK: memref.alloc()
// CHECK-NEXT: scf.for
// CHECK-NOT: memref.alloc()
// CHECK: air.channel.put
air.channel @c [1]
func.func @vector_of_other_shape(%v: vector<4x2xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %a = memref.alloc() : memref<2x4xi32, 2 : i32>
  scf.for %i = %c0 to %c4 step %c1 {
    vector.transfer_write %v, %a[%c0, %c0] : vector<4x2xi32>, memref<2x4xi32, 2 : i32>
    air.channel.put @c[] (%a[] [] []) : (memref<2x4xi32, 2 : i32>)
  }
  memref.dealloc %a : memref<2x4xi32, 2 : i32>
  return
}

// -----

// An execute that fills the buffer only when %p holds leaves it untouched
// otherwise, and the put after it then reads the previous iteration's
// contents: the alloc stays outside the loop.

// CHECK-LABEL: func.func @conditional_fill_in_execute
// CHECK: memref.alloc()
// CHECK-NEXT: scf.for
// CHECK-NOT: memref.alloc()
// CHECK: air.channel.put
air.channel @c [1]
func.func @conditional_fill_in_execute(%p: i1) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %z = arith.constant 0 : i32
  %a = memref.alloc() : memref<4xi32, 2 : i32>
  scf.for %i = %c0 to %c4 step %c1 {
    %t = air.execute {
      scf.if %p {
        linalg.fill ins(%z : i32) outs(%a : memref<4xi32, 2 : i32>)
      }
    }
    air.channel.put @c[] (%a[] [] []) : (memref<4xi32, 2 : i32>)
  }
  memref.dealloc %a : memref<4xi32, 2 : i32>
  return
}

// -----

// Each iteration receives one column block, at an offset that is the loop's
// induction variable, and sends that block on, in another shape: no iteration
// reads what another wrote, so the alloc moves into the loop.

// CHECK-LABEL: func.func @slice_per_iteration
// CHECK: scf.for
// CHECK-NEXT: memref.alloc()
air.channel @c [1]
air.channel @d [1]
func.func @slice_per_iteration() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  %c32 = arith.constant 32 : index
  %c128 = arith.constant 128 : index
  %a = memref.alloc() : memref<8x32xi32, 1 : i32>
  scf.for %i = %c0 to %c32 step %c8 {
    air.channel.get @d[] (%a[%c0, %i] [%c8, %c8] [%c32, %c1]) : (memref<8x32xi32, 1 : i32>)
    air.channel.put @c[] (%a[%c0, %c0, %i] [%c2, %c4, %c8] [%c128, %c32, %c1]) : (memref<8x32xi32, 1 : i32>)
  }
  memref.dealloc %a : memref<8x32xi32, 1 : i32>
  return
}

// -----

// The same, but the put sends the whole buffer, column blocks earlier
// iterations wrote included: the alloc stays outside the loop.

// CHECK-LABEL: func.func @slice_per_iteration_read_whole
// CHECK: memref.alloc()
// CHECK-NEXT: scf.for
air.channel @c [1]
air.channel @d [1]
func.func @slice_per_iteration_read_whole() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c32 = arith.constant 32 : index
  %a = memref.alloc() : memref<8x32xi32, 1 : i32>
  scf.for %i = %c0 to %c32 step %c8 {
    air.channel.get @d[] (%a[%c0, %i] [%c8, %c8] [%c32, %c1]) : (memref<8x32xi32, 1 : i32>)
    air.channel.put @c[] (%a[] [] []) : (memref<8x32xi32, 1 : i32>)
  }
  memref.dealloc %a : memref<8x32xi32, 1 : i32>
  return
}
